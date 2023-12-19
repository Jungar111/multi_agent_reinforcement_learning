"""Main file for project."""
from __future__ import print_function

import json
import typing as T
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

import wandb
from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.data_models.actor_data import (
    ActorData,
    ModelLog,
)
from multi_agent_reinforcement_learning.data_models.city_enum import City
from multi_agent_reinforcement_learning.data_models.config import BaseConfig
from multi_agent_reinforcement_learning.data_models.model_data_pair import ModelDataPair
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.envs.scenario import Scenario
from multi_agent_reinforcement_learning.utils.argument_parser import args_to_config
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.price_utils import value_of_time
from multi_agent_reinforcement_learning.utils.setup_grid import setup_dummy_grid

logger = init_logger()


def _train_loop(
    n_episodes: int,
    env: AMoD,
    model_data_pairs: T.List[ModelDataPair],
    n_actions: int,
    episode_length: int,
    travel_time_dict: dict,
    training: bool = True,
):
    """General train loop.

    Used both for testing and training, by setting training.
    """
    best_reward = -np.inf
    data = None
    if env.config.json_file is not None:
        with open(env.config.json_file) as json_file:
            data = json.load(json_file)

    epochs = trange(n_episodes)
    for i_episode in epochs:
        for model_data_pair in model_data_pairs:
            model_data_pair.actor_data.model_log = ModelLog()
        env.reset(model_data_pairs=model_data_pairs)  # initialize environment

        all_actions = np.zeros(
            (
                len(model_data_pairs),
                episode_length,
                np.max(list(model_data_pairs[0].actor_data.flow.pax_flow.keys())) + 1,
            )
        )

        for step in range(episode_length):
            # take matching step (Step 1 in paper)
            done = env.pax_step(
                model_data_pairs=model_data_pairs,
                cplex_path=config.cplex_path,
                path=config.path,
            )
            for model_data_pair in model_data_pairs:
                model_data_pair.actor_data.model_log.reward += (
                    model_data_pair.actor_data.rewards.pax_reward
                )
                model_data_pair.actor_data.model_log.revenue_reward += (
                    model_data_pair.actor_data.rewards.pax_reward
                )
            # use GNN-RL policy (Step 2 in paper)

            actions = []
            prices = []
            for idx, model_data_pair in enumerate(model_data_pairs):
                if config.include_price:
                    action, price = model_data_pair.model.select_action(
                        obs=model_data_pair.actor_data.graph_state,
                        probabilistic=training,
                        data=data,
                    )

                    for i in range(config.n_regions[config.city]):
                        for j in range(config.n_regions[config.city]):
                            tt = travel_time_dict.get(
                                (i, j, step * config.json_tstep), 1
                            )
                            model_data_pair.actor_data.flow.value_of_time[i, j][
                                step + 1
                            ] = price[0][0][0]
                            model_data_pair.actor_data.graph_state.price[i, j][
                                step + 1
                            ] = (price[0][0][0] * tt)

                    prices.append(price)
                else:
                    action = model_data_pair.model.select_action(
                        obs=model_data_pair.actor_data.graph_state,
                        probabilistic=training,
                        data=data,
                    )
                    for i in range(config.n_regions[config.city]):
                        for j in range(config.n_regions[config.city]):
                            tt = travel_time_dict.get(
                                (i, j, step * config.json_tstep), 1
                            )

                            model_data_pair.actor_data.flow.travel_time[i, j][
                                step + 1
                            ] = tt

                actions.append(action)

            for idx, action in enumerate(actions):
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                model_data_pairs[idx].actor_data.flow.desired_acc = {
                    env.region[i]: int(
                        action[i]
                        * dictsum(
                            model_data_pairs[idx].actor_data.graph_state.acc,
                            env.time + 1,
                        )
                    )
                    for i in range(n_actions)
                }

                all_actions[idx, step, :] = list(
                    model_data_pairs[idx].actor_data.flow.desired_acc.values()
                )

            # solve minimum rebalancing distance problem (Step 3 in paper)

            solveRebFlow(
                env,
                config.path,
                config.cplex_path,
                model_data_pairs=model_data_pairs,
            )

            done = env.reb_step(model_data_pairs=model_data_pairs)

            # track performance over episode
            for model_data_pair in model_data_pairs:
                model_data_pair.actor_data.model_log.reward += (
                    model_data_pair.actor_data.rewards.reb_reward
                )
                model_data_pair.actor_data.model_log.served_demand += (
                    model_data_pair.actor_data.info.served_demand
                )
                model_data_pair.actor_data.model_log.rebalancing_cost += (
                    model_data_pair.actor_data.info.rebalancing_cost
                )
                model_data_pair.model.rewards.append(
                    model_data_pair.actor_data.rewards.pax_reward
                    + model_data_pair.actor_data.rewards.reb_reward
                )

            # stop episode if terminating conditions are met
            if done:
                break

        if training:
            # perform on-policy backprop
            for model_data_pair in model_data_pairs:
                model_data_pair.model.training_step()

        # Send current statistics to screen
        all_prices = [
            list(p.values())
            for p in list(model_data_pairs[0].actor_data.graph_state.price.values())
        ]
        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {model_data_pairs[0].actor_data.model_log.reward:.2f} "
            f"| ServedDemand: {model_data_pairs[0].actor_data.model_log.served_demand:.2f} "
            f"| Reb. Cost: {model_data_pairs[0].actor_data.model_log.rebalancing_cost:.2f} "
            f"| Mean price: {np.mean(all_prices):.2f} "
        )

        # Checkpoint best performing model
        logging_dict = {}
        if training:
            if (
                sum(
                    [
                        model_data_pair.actor_data.model_log.reward
                        for model_data_pair in model_data_pairs
                    ]
                )
                > best_reward
            ):
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

                for ckpt_path in ckpt_paths:
                    wandb.save(ckpt_path)

                for model_data_pair in model_data_pairs:
                    model_data_pair.model.save_checkpoint(
                        path=f"./{config.directory}/ckpt/{config.path}/a2c_{model_data_pair.actor_data.name}.pth"
                    )
                    best_reward = sum(
                        [
                            model_data_pair.actor_data.model_log.reward
                            for model_data_pair in model_data_pairs
                        ]
                    )
                    logging_dict.update({"Best Reward": best_reward})
        # Log KPIs on weights and biases
        for idx, model_data_pair in enumerate(model_data_pairs):
            logging_dict.update(
                model_data_pair.actor_data.model_log.dict(
                    model_data_pair.actor_data.name
                )
            )
            logging_dict.update(
                {f"{model_data_pair.actor_data.name} Mean Price": np.mean(all_prices)}
            )

            overall_sum = sum(
                value
                for inner_dict in model_data_pair.actor_data.unmet_demand.values()
                for value in inner_dict.values()
            )

            logging_dict.update(
                {f"{model_data_pair.actor_data.name} Unmet Demand": overall_sum}
            )

        wandb.log(logging_dict)

        if not training:
            return all_actions


def main(config: BaseConfig):
    """Run main training loop."""
    logger.info("Running main loop.")

    advesary_number_of_cars = int(config.total_number_of_cars / 2)

    actor_data = [
        ActorData(
            name="RL_1_SAC",
            no_cars=config.total_number_of_cars - advesary_number_of_cars,
        ),
        ActorData(
            name="RL_2_SAC",
            no_cars=advesary_number_of_cars,
        ),
    ]

    wandb_config_log = {**vars(config)}
    for actor in actor_data:
        wandb_config_log[f"test_{actor.name}"] = actor.no_cars

    wandb.init(
        mode=config.wandb_mode,
        project="master2023",
        name=f"Test ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        if config.test
        else f"A2C with bus ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        config=wandb_config_log,
    )

    # Define AMoD Simulator Environment
    if config.json_file is None:
        # Define variable for environment
        demand_ratio, demand_input = setup_dummy_grid(config, determ=True)
        scenario = Scenario(
            config=config,
            demand_ratio=demand_ratio,
            demand_input=demand_input,
            actor_data=actor_data,
        )
    else:
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
        scenario=scenario,
        beta=config.beta[config.city],
        actor_data=actor_data,
        config=config,
    )
    # Initialize A2C-GNN
    rl1_actor = ActorCritic(env=env, input_size=21, config=config)
    rl2_actor = ActorCritic(env=env, input_size=21, config=config)

    model_data_pairs = [
        ModelDataPair(rl1_actor, actor_data[0]),
        ModelDataPair(rl2_actor, actor_data[1]),
    ]
    episode_length = config.max_steps  # set episode length
    n_actions = len(env.region)

    with open(str(env.config.json_file)) as file:
        data = json.load(file)

    df = pd.DataFrame(data["demand"])
    df["converted_time_stamp"] = (
        df["time_stamp"] - config.json_hr[config.city] * 60
    ) // config.json_tstep
    travel_time_dict = (
        df.groupby(["origin", "destination", "converted_time_stamp"])["travel_time"]
        .mean()
        .to_dict()
    )

    logger.info(f"VOT {value_of_time(df.price, df.travel_time, demand_ratio=2):.2f}")

    if not config.test:
        train_episodes = config.max_episodes  # set max number of training episodes
        for model_data_pair in model_data_pairs:
            model_data_pair.model.train()

        _train_loop(
            train_episodes,
            env,
            model_data_pairs,
            n_actions,
            episode_length,
            travel_time_dict,
            training=True,
        )

    wandb.finish()


if __name__ == "__main__":
    city = City.san_francisco
    config = args_to_config(city, cuda=False)
    # config.wandb_mode = "disabled"
    config.max_episodes = 300
    # config.test = True
    # config.max_episodes = 11
    # config.json_file = None
    # config.grid_size_x = 2
    # config.grid_size_y = 3
    config.tf = 20
    config.include_price = False
    main(config)
