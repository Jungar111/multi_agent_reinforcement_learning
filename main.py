"""Main file for project."""
from __future__ import print_function

from datetime import datetime

import numpy as np
from tqdm import trange

import wandb
from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.data_models.logs import ModelLog
from multi_agent_reinforcement_learning.envs.amod_env import AMoD, Scenario
from multi_agent_reinforcement_learning.misc.utils import dictsum
from multi_agent_reinforcement_learning.utils.argument_parser import args_to_config
from multi_agent_reinforcement_learning.data_models.config import Config
from multi_agent_reinforcement_learning.plots.mapPlot import makeMapPlot


def main(config: Config):
    """Run main training loop."""
    wandb.init(
        project="master2023",
        name=f"test_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        if config.test
        else f"train_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        config={**vars(config)},
    )

    # Define AMoD Simulator Environment
    scenario = Scenario(
        json_file="data/scenario_nyc4x4.json",
        sd=config.seed,
        demand_ratio=config.demand_ratio,
        json_hr=config.json_hr,
        json_tstep=config.json_tsetp,
    )
    env = AMoD(scenario, beta=config.beta)
    # Initialize A2C-GNN
    model = ActorCritic(env=env, input_size=21, config=config)

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

        for i_episode in epochs:
            train_log = ModelLog()
            obs = env.reset()  # initialize environment
            for step in range(T):
                # take matching step (Step 1 in paper)
                obs, paxreward, done, info = env.pax_step(
                    CPLEXPATH=config.cplex_path, PATH="scenario_nyc4"
                )
                train_log.reward += paxreward
                # use GNN-RL policy (Step 2 in paper)
                action_rl = model.select_action(
                    obs
                )  # Selects actions given the observations
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(
                    env, "scenario_nyc4", desiredAcc, config.cplex_path
                )
                # Take action in environment
                new_obs, rebreward, done, info = env.reb_step(rebAction)
                train_log.reward += rebreward
                # Store the transition in memory
                model.rewards.append(paxreward + rebreward)
                # track performance over episode
                train_log.served_demand += info["served_demand"]
                train_log.rebalancing_cost += info["rebalancing_cost"]
                # stop episode if terminating conditions are met
                if done:
                    break
                # plotting map
                makeMapPlot(scenario.G, obs, rebAction, new_obs, step)

            # perform on-policy backprop
            model.training_step()

            # Send current statistics to screen
            epochs.set_description(
                f"Episode {i_episode+1} | Reward: {train_log.reward:.2f} |"
                f"ServedDemand: {train_log.served_demand:.2f} | Reb. Cost: {train_log.rebalancing_cost:.2f}"
            )
            # Checkpoint best performing model
            if train_log.reward >= best_reward:
                model.save_checkpoint(
                    path=f"./{config.directory}/ckpt/nyc4/a2c_gnn_test.pth"
                )
                best_reward = train_log.reward
            # Log KPIs on weights and biases
            wandb.log({**dict(train_log)})
    else:
        # Load pre-trained model
        model.load_checkpoint(path=f"./{config.directory}/ckpt/nyc4/a2c_gnn.pth")
        test_episodes = config.max_episodes  # set max number of training episodes
        T = config.max_steps  # set episode length
        epochs = trange(test_episodes)  # epoch iterator
        # Initialize lists for logging
        for episode in epochs:
            test_log = ModelLog()
            obs = env.reset()
            done = False
            k = 0
            while not done:
                # take matching step (Step 1 in paper)
                obs, paxreward, done, info = env.pax_step(
                    CPLEXPATH=config.cplex_path, PATH="scenario_nyc4_test"
                )
                test_log.reward += paxreward
                # use GNN-RL policy (Step 2 in paper)
                action_rl = model.select_action(obs)
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(
                    env, "scenario_nyc4_test", desiredAcc, config.cplex_path
                )
                # Take action in environment
                new_obs, rebreward, done, info = env.reb_step(rebAction)
                test_log.reward += rebreward
                # track performance over episode
                test_log.served_demand += info["served_demand"]
                test_log.rebalancing_cost += info["rebalancing_cost"]
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
    args = args_to_config()
    main(args)
