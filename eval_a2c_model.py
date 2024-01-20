"""Main file for project."""
from __future__ import print_function

import json
import typing as T
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import trange

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
from multi_agent_reinforcement_learning.evaluation.actor_evaluation import (
    get_summary_stats,
)
from multi_agent_reinforcement_learning.utils.argument_parser import args_to_config
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
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
    data = None
    if env.config.json_file is not None:
        with open(env.config.json_file) as json_file:
            data = json.load(json_file)

    epoch_served_demand = []
    epoch_cancelled_demand = []
    epoch_rewards = defaultdict(list)

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
        episode_served_demand = defaultdict(list)
        episode_cancelled_demand = defaultdict(list)

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
            for idx, model in enumerate(model_data_pairs):
                episode_served_demand[idx].append(model.actor_data.info.served_demand)
                episode_cancelled_demand[idx].append(
                    model.actor_data.model_log.bus_unmet_demand
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

        # Send current statistics to screen
        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {model_data_pairs[0].actor_data.model_log.reward:.2f}"
        )
        for idx in range(len(model_data_pairs)):
            epoch_rewards[idx].append(model_data_pairs[idx].actor_data.model_log.reward)

        epoch_served_demand.append(episode_served_demand)
        epoch_cancelled_demand.append(episode_cancelled_demand)
    get_summary_stats(
        {},
        epoch_rewards,
        epoch_served_demand,
        epoch_cancelled_demand,
        "A2C_2_actor_no_cancel_no_price",
        config,
    )


def main(config: BaseConfig):
    """Run main training loop."""
    logger.info("Running main loop.")

    advesary_number_of_cars = int(config.no_cars / 2)

    actor_data = [
        ActorData(
            name="RL_1",
            no_cars=config.no_cars - advesary_number_of_cars,
        ),
        ActorData(
            name="RL_2",
            no_cars=advesary_number_of_cars,
        ),
    ]

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
    for idx, model_data_pair in enumerate(model_data_pairs):
        logger.info(
            f"Loading from saved_files/ckpt/{config.path}/{model_data_pair.actor_data.name}.pth"
        )
        model_data_pair.model.load_checkpoint(
            path=f"saved_files/ckpt/{config.path}/{model_data_pair.actor_data.name}.pth"
        )
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

    train_episodes = config.max_episodes  # set max number of training episodes

    _train_loop(
        train_episodes,
        env,
        model_data_pairs,
        n_actions,
        episode_length,
        travel_time_dict,
        training=False,
    )


if __name__ == "__main__":
    city = City.san_francisco
    config = args_to_config(city, cuda=False)
    config.wandb_mode = "disabled"
    config.max_episodes = 10
    config.test = True
    # config.max_episodes = 11
    # config.json_file = None
    # config.grid_size_x = 2
    # config.grid_size_y = 3
    config.tf = 20
    config.include_price = False
    config.cancellation = False
    main(config)
