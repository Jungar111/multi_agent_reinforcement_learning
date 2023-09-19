"""Main file for project."""
from __future__ import print_function

from datetime import datetime

import numpy as np
from tqdm import trange

import wandb
from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow

from multi_agent_reinforcement_learning.algos.uniform_actor import UniformActor
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

logger = init_logger()


def main(config: Config):
    """Run main training loop."""
    logger.info("Running main loop.")
    wandb.init(
        mode=config.wandb_mode,
        project="master2023",
        name=f"test_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        if config.test
        else f"train_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        config={**vars(config)},
    )

    uniform_number_of_cars = int(1408 / 3)

    actor_data = [
        ActorData(name="RL", no_cars=1408 - uniform_number_of_cars),
        ActorData(name="Uniform", no_cars=uniform_number_of_cars),
    ]

    # Define AMoD Simulator Environment
    if config.json_file is None:
        # Define variable for environment
        tf, demand_ratio, demand_input, ninit = setup_dummy_grid(config)
        scenario = Scenario(
            json_file=config.json_file,
            tf=tf,
            demand_ratio=demand_ratio,
            demand_input=demand_input,
            ninit=ninit,
            actor_data=actor_data,
        )
    else:
        scenario = Scenario(
            json_file=config.json_file,
            sd=config.seed,
            demand_ratio=config.demand_ratio,
            json_hr=config.json_hr,
            json_tstep=config.json_tsetp,
            actor_data=actor_data,
        )

    env = AMoD(scenario=scenario, beta=config.beta, actor_data=actor_data)
    # Initialize A2C-GNN
    model = ActorCritic(env=env, input_size=21, config=config)
    uniform_actor = UniformActor()

    if not config.test:
        #######################################
        #############Training Loop#############
        #######################################

        # Initialize lists for logging
        train_episodes = config.max_episodes  # set max number of training episodes
        T = config.max_steps  # set episode length
        epochs = trange(train_episodes)  # epoch iterator
        best_reward = -np.inf  # set best reward
        model.train()  # set model in train mode
        n_actions = len(env.region)

        for i_episode in epochs:
            rl_train_log = ModelLog()
            uniform_train_log = ModelLog()
            env.reset()  # initialize environment
            for step in range(T):
                # take matching step (Step 1 in paper)
                actor_data, done, ext_done = env.pax_step(
                    cplex_path=config.cplex_path, path="scenario_nyc4"
                )
                rl_train_log.reward += actor_data[0].pax_reward
                uniform_train_log.reward += actor_data[1].pax_reward
                # use GNN-RL policy (Step 2 in paper)
                action_rl = model.select_action(actor_data[0].obs)
                action_uniform = uniform_actor.select_action(
                    n_regions=config.grid_size_x * config.grid_size_y
                )

                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                actor_data[0].desired_acc = {
                    env.region[i]: int(
                        action_rl[i] * dictsum(actor_data[0].acc, env.time + 1)
                    )
                    for i in range(n_actions)
                }

                actor_data[1].desired_acc = {
                    env.region[i]: int(
                        action_uniform[i] * dictsum(actor_data[1].acc, env.time + 1)
                    )
                    for i in range(n_actions)
                }

                # solve minimum rebalancing distance problem (Step 3 in paper)

                solveRebFlow(env, "scenario_nyc4", config.cplex_path)

                actor_data, done = env.reb_step()

                # Take action in environment
                rl_train_log.reward += actor_data[0].reb_reward
                uniform_train_log.reward = actor_data[1].reb_reward

                model.rewards.append(
                    actor_data[0].pax_reward + actor_data[0].reb_reward
                )
                # track performance over episode
                rl_train_log.served_demand += actor_data[0].info.served_demand
                rl_train_log.rebalancing_cost += actor_data[0].info.rebalancing_cost
                uniform_train_log.served_demand += actor_data[1].info.served_demand
                uniform_train_log.rebalancing_cost += actor_data[
                    1
                ].info.rebalancing_cost
                # stop episode if terminating conditions are met
                if done:
                    break
                # Create map if at last episode
                if i_episode == epochs.iterable[-1]:
                    if step == 0:
                        logger.info("Making map plot.")
                    make_map_plot(env.G, actor_data[0].obs, step, T, env, config)
            # Make images to gif, and cleanup
            if i_episode == epochs.iterable[-1]:
                images_to_gif()

            # perform on-policy backprop
            model.training_step()

            # Send current statistics to screen
            epochs.set_description(
                f"Episode {i_episode+1} | Reward: {rl_train_log.reward:.2f} |"
                f"ServedDemand: {rl_train_log.served_demand:.2f} | Reb. Cost: {rl_train_log.rebalancing_cost:.2f}"
            )
            # Checkpoint best performing model
            if rl_train_log.reward >= best_reward:
                model.save_checkpoint(
                    path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_test.pth"
                )
                best_reward = rl_train_log.reward
            # Log KPIs on weights and biases
            wandb.log(
                {
                    **rl_train_log.dict("reninforcement"),
                    # **uniform_train_log.dict("uniform"),
                }
            )
    else:
        # Load pre-trained model
        model.load_checkpoint(path=f"./{config.directory}/ckpt/nyc4/a2c_gnn.pth")
        model_data = actor_data[0]
        test_episodes = config.max_episodes  # set max number of training episodes
        T = config.max_steps  # set episode length
        epochs = trange(test_episodes)  # epoch iterator
        # Initialize lists for logging
        for episode in epochs:
            test_log = ModelLog()
            env.reset()
            done = False
            k = 0
            while not done:
                # take matching step (Step 1 in paper)
                actor_data, done, ext_done = env.pax_step(
                    cplex_path=config.cplex_path, path="scenario_nyc4_test"
                )
                test_log.reward += model_data.pax_reward
                # use GNN-RL policy (Step 2 in paper)
                action_rl = model.select_action(model_data.obs)
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desired_acc_rl = {
                    env.region[i]: int(
                        action_rl[i] * dictsum(model_data.acc, env.time + 1)
                    )
                    for i in range(len(env.region))
                }
                # solve minimum rebalancing distance problem (Step 3 in paper)
                solveRebFlow(
                    env, "scenario_nyc4_test", desired_acc_rl, config.cplex_path
                )
                # Take action in environment
                actor_data, done = env.reb_step()
                test_log.reward += model_data.reb_reward
                # track performance over episode
                test_log.served_demand += model_data.info.served_demand
                test_log.rebalancing_cost += model_data.info.rebalancing_cost
                k += 1
            # Send current statistics to screen
            epochs.set_description(
                f"Episode {episode+1} | Reward: {test_log.reward:.2f} | ServedDemand:"
                f"{test_log.served_demand:.2f} | Reb. Cost: {test_log.rebalancing_cost}"
            )
            # Log KPIs on weights and biases
            wandb.log({**dict(test_log)})
            break
        wandb.finish()


if __name__ == "__main__":
    config = args_to_config()
    config.wandb_mode = "disabled"
    config.test = False
    # config.json_file = None
    # config.grid_size_x = 2
    # config.grid_size_y = 3
    # config.tf = 20
    # config.ninit = 10
    main(config)
