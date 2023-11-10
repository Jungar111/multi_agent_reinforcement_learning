"""Main file for project."""
from __future__ import print_function
from pathlib import Path

from datetime import datetime
import json
import typing as T

import numpy as np
from tqdm import trange

import wandb
from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.data_models.actor_data import ActorData
from multi_agent_reinforcement_learning.data_models.model_data_pair import ModelDataPair
from multi_agent_reinforcement_learning.data_models.config import Config
from multi_agent_reinforcement_learning.data_models.logs import ModelLog
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.envs.scenario import Scenario
from multi_agent_reinforcement_learning.evaluation.actor_evaluation import (
    ActorEvaluator,
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
    episode_mean_price = {
        model_data_pair.actor_data.name: [] for model_data_pair in model_data_pairs
    }
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
                path="scenario_nyc4",
            )
            for model_data_pair in model_data_pairs:
                model_data_pair.actor_data.model_log.reward += (
                    model_data_pair.actor_data.rewards.pax_reward
                )
            # use GNN-RL policy (Step 2 in paper)

            actions = []
            prices = []
            for idx, model_data_pair in enumerate(model_data_pairs):
                model_data_pair.model.train_log.reward += (
                    model_data_pair.actor_data.rewards.pax_reward
                )
                action, price = model_data_pair.model.select_action(
                    obs=model_data_pair.actor_data.graph_state,
                    probabilistic=training,
                    data=data,
                )
                actions.append(action)
                prices.append(price)

                # @TODO please optimise this. Must be very slow.
                num_cells = config.grid_size_y * config.grid_size_y
                for i in range(num_cells):
                    for j in range(num_cells):
                        model_data_pair.actor_data.graph_state.price[i, j][
                            step + 1
                        ] = price[i][j]

                actions.append(action)
                prices.append(price)
                episode_mean_price[model_data_pair.actor_data.name].append(price.mean())

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
                "scenario_nyc4",
                config.cplex_path,
                model_data_pairs=model_data_pairs,
            )

            done = env.reb_step(model_data_pairs=model_data_pairs)

            for model_data_pair in model_data_pairs:
                model_data_pair.model.train_log.reward += (
                    model_data_pair.actor_data.rewards.reb_reward
                )
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

                model_data_pair.model.train_log.served_demand += (
                    model_data_pair.actor_data.info.served_demand
                )
                model_data_pair.model.train_log.rebalancing_cost += (
                    model_data_pair.actor_data.info.rebalancing_cost
                )
            # stop episode if terminating conditions are met
            if done:
                break

        if training:
            # perform on-policy backprop
            for model_data_pair in model_data_pairs:
                model_data_pair.model.training_step()

        # Send current statistics to screen
        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {model_data_pairs[0].actor_data.model_log.reward:.2f} |"
            f"ServedDemand: {model_data_pairs[0].actor_data.model_log.served_demand:.2f} "
            f"| Reb. Cost: {model_data_pairs[0].actor_data.model_log.rebalancing_cost:.2f}"
        )

        # Checkpoint best performing model
        logging_dict = {}
        if training:
            if (  # TODO fix
                sum(
                    [
                        model_data_pair.actor_data.model_log.reward
                        for model_data_pair in model_data_pairs
                    ]
                )
                > best_reward
            ):
                for model_data_pair in model_data_pairs:
                    model_data_pair.model.save_checkpoint(
                        path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_{model_data_pair.actor_data.name}.pth"
                    )
                    best_reward = sum(
                        [
                            model_data_pair.actor_data.model_log.reward
                            for model_data_pair in model_data_pairs
                        ]
                    )
                    logging_dict.update({"Best Reward": best_reward})
        # Log KPIs on weights and biases
        for model_data_pair in model_data_pairs:
            logging_dict.update(
                model_data_pair.actor_data.model_log.dict(
                    model_data_pair.actor_data.name
                )
            )
        wandb.log(logging_dict)

        if not training:
            return all_actions


def main(config: Config):
    """Run main training loop."""
    logger.info("Running main loop.")

    advesary_number_of_cars = int(config.total_number_of_cars / 2)

    actor_data = [
        ActorData(
            name="RL_1_with_price",
            no_cars=config.total_number_of_cars - advesary_number_of_cars,
        ),
        ActorData(name="RL_2_with_price", no_cars=advesary_number_of_cars),
    ]

    wandb_config_log = {**vars(config)}
    for actor in actor_data:
        wandb_config_log[f"test_{actor.name}"] = actor.no_cars

    wandb.init(
        mode=config.wandb_mode,
        project="master2023",
        name=f"test_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        if config.test
        else f"train_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
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
            demand_ratio=config.demand_ratio,
            json_hr=config.json_hr[config.city],
            json_tstep=config.json_tsetp,
            actor_data=actor_data,
        )

    env = AMoD(
        scenario=scenario, beta=config.beta, actor_data=actor_data, config=config
    )
    # Initialize A2C-GNN
    rl1_actor = ActorCritic(
        env=env, input_size=21, config=config, actor_data=actor_data[0]
    )
    rl2_actor = ActorCritic(
        env=env, input_size=21, config=config, actor_data=actor_data[1]
    )

    model_data_pairs = [
        ModelDataPair(rl1_actor, actor_data[0]),
        ModelDataPair(rl2_actor, actor_data[1]),
    ]
    episode_length = config.max_steps  # set episode length
    n_actions = len(env.region)

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
            training=True,
        )

        ckpt_paths = [
            str(
                Path(
                    "saved_files",
                    "ckpt",
                    "nyc4",
                    f"a2c_gnn_{model_data_pair.actor_data.name}.pth",
                )
            )
            for model_data_pair in model_data_pairs
        ]

        for ckpt_path in ckpt_paths:
            wandb.save(ckpt_path)

    else:
        # Load pre-trained model
        rl1_actor.load_checkpoint(
            path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_RL_1_big_run.pth"
        )
        rl2_actor.load_checkpoint(
            path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_RL_2_big_run.pth"
        )

        test_episodes = 1
        episode_length = config.max_steps

        all_actions = _train_loop(
            test_episodes,
            env,
            model_data_pairs,
            n_actions,
            episode_length,
            training=False,
        )

        actor_evaluator = ActorEvaluator()
        actor_evaluator.plot_average_distribution(
            actions=np.array(all_actions),
            T=episode_length,
            models=model_data_pairs,
        )

        # actor_evaluator.plot_distribution_at_time_step_t(
        #     actions=np.array(all_actions), models=models
        # )

    wandb.finish()


if __name__ == "__main__":
    config = args_to_config()
    # config.wandb_mode = "disabled"
    # config.test = True
    # config.max_episodes = 3
    # config.json_file = None
    # config.grid_size_x = 2
    # config.grid_size_y = 3
    # config.tf = 20
    main(config)
