"""Main file for project."""
from __future__ import print_function

from datetime import datetime

import numpy as np
from tqdm import trange

import wandb
from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.data_models.actor_data import ActorData
from multi_agent_reinforcement_learning.data_models.config import Config
from multi_agent_reinforcement_learning.data_models.logs import ModelLog
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.envs.scenario import Scenario
from multi_agent_reinforcement_learning.utils.argument_parser import args_to_config
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.setup_grid import setup_dummy_grid
from multi_agent_reinforcement_learning.plots.map_plot import (
    make_map_plot,
    images_to_gif,
)
from multi_agent_reinforcement_learning.evaluation.actor_evaluation import (
    ActorEvaluator,
)

logger = init_logger()


def _train_loop():
    pass


def main(config: Config):
    """Run main training loop."""
    logger.info("Running main loop.")

    advesary_number_of_cars = int(1408 / 2)

    actor_data = [
        ActorData(name="RL", no_cars=1408 - advesary_number_of_cars),
        ActorData(name="Uniform", no_cars=advesary_number_of_cars),
    ]

    wandb_config_log = {**vars(config)}
    for actor in actor_data:
        wandb_config_log[f"no_cars_{actor.name}"] = actor.no_cars

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
            json_file=config.json_file,
            sd=config.seed,
            demand_ratio=config.demand_ratio,
            json_hr=config.json_hr,
            json_tstep=config.json_tsetp,
            actor_data=actor_data,
        )

    env = AMoD(scenario=scenario, beta=config.beta, actor_data=actor_data)
    # Initialize A2C-GNN
    rl1_actor = ActorCritic(
        env=env, input_size=21, config=config, actor_data=actor_data[0]
    )
    rl2_actor = ActorCritic(
        env=env, input_size=21, config=config, actor_data=actor_data[1]
    )

    models = [rl1_actor, rl2_actor]

    if not config.test:
        #######################################
        #############Training Loop#############
        #######################################

        # Initialize lists for logging
        train_episodes = config.max_episodes  # set max number of training episodes
        T = config.max_steps  # set episode length
        epochs = trange(train_episodes)  # epoch iterator
        best_reward = -np.inf  # set best reward
        for model in models:
            model.train()
        n_actions = len(env.region)

        for i_episode in epochs:
            rl1_train_log = ModelLog()
            rl2_train_log = ModelLog()
            env.reset()  # initialize environment
            for step in range(T):
                # take matching step (Step 1 in paper)
                actor_data, done, ext_done = env.pax_step(
                    cplex_path=config.cplex_path, path="scenario_nyc4"
                )
                # use GNN-RL policy (Step 2 in paper)
                actions = []
                for model in models:
                    model.train_log.reward += model.actor_data.pax_reward
                    actions.append(model.select_action(model.actor_data.obs))

                for idx, action in enumerate(actions):
                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    actor_data[idx].desired_acc = {
                        env.region[i]: int(
                            action[i] * dictsum(actor_data[idx].acc, env.time + 1)
                        )
                        for i in range(n_actions)
                    }

                # solve minimum rebalancing distance problem (Step 3 in paper)

                solveRebFlow(env, "scenario_nyc4", config.cplex_path)

                actor_data, done = env.reb_step()

                for model in models:
                    model.train_log.reward += model.actor_data.reb_reward
                    model.rewards.append(
                        model.actor_data.pax_reward + model.actor_data.reb_reward
                    )

                    model.train_log.served_demand += model.actor_data.info.served_demand
                    model.train_log.rebalancing_cost += (
                        model.actor_data.info.rebalancing_cost
                    )

                # stop episode if terminating conditions are met
                if done:
                    break

            # perform on-policy backprop
            for model in models:
                model.training_step()

            # Send current statistics to screen
            epochs.set_description(
                f"Episode {i_episode+1} | Reward: {rl1_train_log.reward:.2f} |"
                f"ServedDemand: {rl1_train_log.served_demand:.2f} | Reb. Cost: {rl1_train_log.rebalancing_cost:.2f}"
            )
            # Checkpoint best performing model
            if rl1_train_log.reward >= best_reward:
                rl1_actor.save_checkpoint(
                    path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_test.pth"
                )
                best_reward = rl1_train_log.reward
            # Log KPIs on weights and biases
            wandb.log(
                {
                    **rl1_train_log.dict("reinforcement"),
                    **rl2_train_log.dict("reinforcement_2"),
                }
            )
    else:
        # Load pre-trained model
        rl1_actor.load_checkpoint(
            path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_test.pth"
        )
        rl2_actor.load_checkpoint(
            path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_test.pth"
        )
        model_data = actor_data[0]
        test_episodes = config.max_episodes  # set max number of training episodes
        T = config.max_steps  # set episode length
        epochs = trange(test_episodes)  # epoch iterator
        n_actions = len(env.region)
        # Initialize lists for logging
        for episode in epochs:
            test_log = ModelLog()
            env.reset()
            done = False
            k = 0
            rl_train_log = ModelLog()
            uniform_train_log = ModelLog()
            while not done:
                # take matching step (Step 1 in paper)
                actor_data, done, ext_done = env.pax_step(
                    cplex_path=config.cplex_path, path="scenario_nyc4_test"
                )
                test_log.reward += model_data.pax_reward
                # use GNN-RL policy (Step 2 in paper)
                action_rl = rl1_actor.select_action(model_data.obs)
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                actor_data[0].desired_acc = {
                    env.region[i]: int(
                        action_rl[i] * dictsum(actor_data[0].acc, env.time + 1)
                    )
                    for i in range(n_actions)
                }

                # solve minimum rebalancing distance problem (Step 3 in paper)
                solveRebFlow(env, "scenario_nyc4_test", config.cplex_path)
                # Take action in environment
                actor_data, done = env.reb_step()
                # Take action in environment
                rl_train_log.reward += actor_data[0].reb_reward
                uniform_train_log.reward = actor_data[1].reb_reward

                rl1_actor.rewards.append(
                    actor_data[0].pax_reward + actor_data[0].reb_reward
                )
                # track performance over episode
                rl_train_log.served_demand += actor_data[0].info.served_demand
                rl_train_log.rebalancing_cost += actor_data[0].info.rebalancing_cost
                uniform_train_log.served_demand += actor_data[1].info.served_demand
                uniform_train_log.rebalancing_cost += actor_data[
                    1
                ].info.rebalancing_cost

                k += 1
                # Create map if at last episode
                if episode == epochs.iterable[0]:
                    if k == 1:
                        logger.info("Making map plot.")
                    make_map_plot(env.G, actor_data[0], k, T, config)
            # Make images to gif, and cleanup
            if episode == epochs.iterable[0]:
                images_to_gif()

            # Send current statistics to screen
            epochs.set_description(
                f"Episode {episode+1} | Reward: {rl_train_log.reward:.2f} |"
                f"ServedDemand: {rl_train_log.served_demand:.2f} | Reb. Cost: {rl_train_log.rebalancing_cost:.2f}"
            )
            actor_evaluator = ActorEvaluator(actor_data=actor_data)
            actor_evaluator.plot_average_distribution()
            # Log KPIs on weights and biases
            wandb.log({**dict(test_log)})
            break
        wandb.finish()


if __name__ == "__main__":
    config = args_to_config()
    config.wandb_mode = "disabled"
    config.test = True
    # config.max_episodes = 4
    # config.json_file = None
    # config.grid_size_x = 2
    # config.grid_size_y = 3
    # config.tf = 20
    # config.ninit = 80
    main(config)
