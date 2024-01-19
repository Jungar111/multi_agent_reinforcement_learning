"""Main file for the SAC implementation for the project."""
from __future__ import print_function

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import trange

import wandb
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.algos.sac import SAC
from multi_agent_reinforcement_learning.algos.sac_gnn_parser import GNNParser
from multi_agent_reinforcement_learning.data_models.actor_data import (
    ActorData,
    ModelLog,
)
from multi_agent_reinforcement_learning.data_models.city_enum import City
from multi_agent_reinforcement_learning.data_models.config import SACConfig
from multi_agent_reinforcement_learning.data_models.model_data_pair import ModelDataPair
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.envs.scenario import Scenario
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.price_utils import value_of_time
from multi_agent_reinforcement_learning.utils.sac_argument_parser import args_to_config

logger = init_logger()


def main(config: SACConfig, run_name: str):
    """Run main training loop of SAC."""
    number_of_cars = int(config.no_cars / config.no_actors)
    actor_data = [
        ActorData(
            name=f"RL_{i+1}_SAC",
            no_cars=number_of_cars,
        )
        for i in range(config.no_actors)
    ]

    wandb.init(
        mode=config.wandb_mode,
        project="master2023",
        name=f"{run_name} ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
    )

    logging_dict = {}
    logger.info("Running main training loop for SAC.")
    scenario = Scenario(
        config=config,
        json_file=str(config.json_file),
        sd=config.seed,
        demand_ratio=config.demand_ratio[config.city],
        json_hr=config.json_hr[config.city],
        json_tstep=config.json_tstep,
        actor_data=actor_data,
    )

    env = AMoD(
        beta=config.beta[config.city],
        scenario=scenario,
        config=config,
        actor_data=actor_data,
    )
    # Timehorizon T=6 (K in paper)
    parser = GNNParser(env, T=6, json_file=f"data/scenario_{config.city}.json")
    # Initialise SAC
    rl_actors = [
        SAC(
            env=env,
            config=config,
            actor_data=actor_data[i],
            input_size=13,
            hidden_size=config.hidden_size,
            p_lr=config.p_lr,
            q_lr=config.q_lr,
            alpha=config.alpha,
            batch_size=config.batch_size,
            use_automatic_entropy_tuning=False,
            clip=config.clip,
        )
        for i in range(config.no_actors)
    ]
    model_data_pairs = [
        ModelDataPair(rl_actors[i], actor_data[i]) for i in range(config.no_actors)
    ]
    # model_data_pairs[0].model.load_checkpoint(
    #     path=f"saved_files/ckpt/{config.path}/{model_data_pairs[0].actor_data.name}_last.pth"
    # )
    train_episodes = config.max_episodes
    # T = config.max_steps
    epochs = trange(train_episodes)
    best_reward = -np.inf
    # best_reward_test = -np.inf

    with open(str(env.config.json_file)) as file:
        data = json.load(file)

    if config.include_price:
        df = pd.DataFrame(data["demand"])
        df["converted_time_stamp"] = (
            df["time_stamp"] - config.json_hr[config.city] * 60
        ) // config.json_tstep
        travel_time_dict = (
            df.groupby(["origin", "destination", "converted_time_stamp"])["travel_time"]
            .mean()
            .to_dict()
        )

        logger.info(
            f"VOT {value_of_time(df.price, df.travel_time, demand_ratio=config.demand_ratio[config.city]):.2f}"
        )

        # Used for price diff
        init_price_dict = df.groupby(["origin", "destination"]).price.mean().to_dict()
        init_price_mean = df.price.mean()

    wandb_config_log = {**vars(config)}
    for model in model_data_pairs:
        wandb_config_log[f"test_{model.actor_data.name}"] = model.actor_data.no_cars

    for model_data_pair in model_data_pairs:
        model_data_pair.model.train()

    for i_episode in epochs:
        for model_data_pair in model_data_pairs:
            model_data_pair.actor_data.model_log = ModelLog()

        env.reset(model_data_pairs)  # initialize environment
        episode_reward = [0 for _ in range(config.no_actors)]
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        done = False
        step = 0
        o = [None for _ in range(config.no_actors)]
        action_rl = [None for _ in range(config.no_actors)]
        obs_list = [None for _ in range(config.no_actors)]
        if config.include_price:
            prices = {i: [] for i in range(config.no_actors)}
        while not done:
            # take matching step (Step 1 in paper)
            if step > 0:
                for i in range(config.no_actors):
                    obs_list[i] = copy.deepcopy(o[i])

            done = env.pax_step(
                model_data_pairs=model_data_pairs,
                cplex_path=config.cplex_path,
                path=config.path,
            )
            for idx, model_data_pair in enumerate(model_data_pairs):
                o[idx] = parser.parse_obs(model_data_pair.actor_data.graph_state)
                episode_reward[idx] += model_data_pair.actor_data.rewards.pax_reward
                if step > 0:
                    # store transition in memory
                    rl_reward = (
                        model_data_pair.actor_data.rewards.pax_reward
                        + model_data_pair.actor_data.rewards.reb_reward
                    )
                    if config.include_price:
                        model_data_pair.model.replay_buffer.store(
                            obs_list[idx],
                            action_rl[idx],
                            config.rew_scale * rl_reward,
                            o[idx],
                            prices[idx],
                        )
                    else:
                        model_data_pair.model.replay_buffer.store(
                            obs_list[idx],
                            action_rl[idx],
                            config.rew_scale * rl_reward,
                            o[idx],
                            None,
                        )

                if config.include_price:
                    action_rl[idx], price = model_data_pair.model.select_action(o[idx])

                    for i in range(config.n_regions[config.city]):
                        for j in range(config.n_regions[config.city]):
                            tt = travel_time_dict.get(
                                (i, j, step * config.json_tstep), 1
                            )
                            model_data_pair.actor_data.flow.value_of_time[i, j][
                                step + 1
                            ] = price[0][0]

                            model_data_pair.actor_data.flow.travel_time[i, j][
                                step + 1
                            ] = tt

                            model_data_pair.actor_data.graph_state.price[i, j][
                                step + 1
                            ] =  init_price_dict.get((i,j), init_price_mean) + price[0][0] #max((price[0][0] * tt), 10)
                    prices[idx].append(price)
                else:
                    action_rl[idx] = model_data_pair.model.select_action(o[idx])
                    for i in range(config.n_regions[config.city]):
                        for j in range(config.n_regions[config.city]):
                            tt = travel_time_dict.get(
                                (i, j, step * config.json_tstep), 1
                            )

                            model_data_pair.actor_data.flow.travel_time[i, j][
                                step + 1
                            ] = tt
            for idx, model in enumerate(model_data_pairs):
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                model.actor_data.flow.desired_acc = {
                    env.region[i]: int(
                        action_rl[idx][i]
                        * dictsum(model.actor_data.graph_state.acc, env.time + 1)
                    )
                    for i in range(len(env.region))
                }

            # solve minimum rebalancing distance problem (Step 3 in paper)
            solveRebFlow(
                env,
                config.path,
                config.cplex_path,
                model_data_pairs,
            )
            # Take action in environment
            done = env.reb_step(model_data_pairs)

            for idx, model in enumerate(model_data_pairs):
                episode_reward[idx] += model.actor_data.rewards.reb_reward
            # track performance over episode
            for model in model_data_pairs:
                episode_served_demand += model.actor_data.info.served_demand
                episode_rebalancing_cost += model.actor_data.info.rebalancing_cost

            # track performance over episode
            for model_data_pair in model_data_pairs:
                reward_for_episode = (
                    model_data_pair.actor_data.rewards.pax_reward
                    + model_data_pair.actor_data.rewards.reb_reward
                )
                model_data_pair.actor_data.model_log.reward += reward_for_episode

                model_data_pair.actor_data.model_log.revenue_reward += (
                    model_data_pair.actor_data.rewards.pax_reward
                )
                model_data_pair.actor_data.model_log.rebalancing_reward += (
                    model_data_pair.actor_data.rewards.reb_reward
                )

                model_data_pair.actor_data.model_log.served_demand += (
                    model_data_pair.actor_data.info.served_demand
                )
                model_data_pair.actor_data.model_log.rebalancing_cost += (
                    model_data_pair.actor_data.info.rebalancing_cost
                )
                model_data_pair.model.rewards.append(reward_for_episode)
            step += 1
            for model in model_data_pairs:
                if i_episode > 10:
                    # sample from memory and update model
                    batch = model.model.replay_buffer.sample_batch(
                        config.batch_size, norm=False
                    )
                    model.model.update(data=batch)

        description = (
            f"Episode {i_episode+1} | "
            f"ServedDemand: {episode_served_demand:.2f} | "
            f"Reward 1: {episode_reward[0]:.2f} | "
            # f"Reb. Cost: {episode_rebalancing_cost:.2f} | "
            f"Mean price 1: {np.mean(prices[0]) if config.include_price else 0:.2f} | "
        )
        if config.no_actors > 1:
            description += (
                f"Reward 2: {episode_reward[1]:.2f} |"
                f" Mean price 2: {np.mean(prices[1]) if config.include_price else 0:.2f}"
            )
        epochs.set_description(description)
        # Checkpoint best performing model
        if np.sum(episode_reward) >= best_reward:
            ckpt_paths = [
                str(
                    Path(
                        "saved_files",
                        "ckpt",
                        f"{config.path}",
                        f"{model_data_pair.actor_data.name}.pth",
                    )
                )
                for model_data_pair in model_data_pairs
            ]

            for model in model_data_pairs:
                model.model.save_checkpoint(
                    path=f"saved_files/ckpt/{config.path}/{model.actor_data.name}.pth"
                )

            for ckpt_path in ckpt_paths:
                wandb.save(ckpt_path)

            best_reward = np.sum(episode_reward)
            logging_dict.update({"Best Reward": best_reward})

        for idx, model_data_pair in enumerate(model_data_pairs):
            logging_dict.update(
                model_data_pair.actor_data.model_log.dict(
                    model_data_pair.actor_data.name
                )
            )
            if config.include_price:
                logging_dict.update(
                    {
                        f"{model_data_pair.actor_data.name} Mean Price": np.mean(
                            prices[idx]
                        )
                    }
                )
        if i_episode + 1 == train_episodes:
            ckpt_paths = [
                str(
                    Path(
                        "saved_files",
                        "ckpt",
                        f"{config.path}",
                        f"{model_data_pair.actor_data.name}_last.pth",
                    )
                )
                for model_data_pair in model_data_pairs
            ]

            for model in model_data_pairs:
                model.model.save_checkpoint(
                    path=f"saved_files/ckpt/{config.path}/{model.actor_data.name}_last.pth"
                )

            for ckpt_path in ckpt_paths:
                wandb.save(ckpt_path)
        wandb.log(logging_dict)


if __name__ == "__main__":
    torch.manual_seed(42)
    city = City.san_francisco
    config = args_to_config(city, cuda=True)
    config.tf = 20

    if config.run_name == "":
        config.run_name = "Mean price for OD-pair + price difference for state (eq. 3.37 in pdf) - RUN 2"

    config.max_episodes = 1000
    config.include_price = True
    config.dynamic_scaling = False
    # config.test = True
    # config.wandb_mode = "disabled"
    main(config, run_name=config.run_name)
