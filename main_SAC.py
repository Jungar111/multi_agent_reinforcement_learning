"""Main file for the SAC implementation for the project."""
from __future__ import print_function

# from datetime import datetime
# import typing as T

from tqdm import trange

# import wandb

import numpy as np
from multi_agent_reinforcement_learning.envs.sac_amod import AMoD
from multi_agent_reinforcement_learning.envs.sac_scenario import Scenario
from multi_agent_reinforcement_learning.algos.sac import SAC
from multi_agent_reinforcement_learning.algos.sac_reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.init_logger import init_logger
from multi_agent_reinforcement_learning.algos.sac_gnn_parser import GNNParser
from multi_agent_reinforcement_learning.utils.sac_argument_parser import args_to_config
from multi_agent_reinforcement_learning.data_models.config import SACConfig
import copy

logger = init_logger()


def main(config: SACConfig):
    """Main loop for training and testing."""
    if not config.test:
        """Run main training loop."""
        logger.info("Running main training loop for SAC.")

        scenario = Scenario(
            json_file=config.json_file,
            demand_ratio=config.demand_ratio[config.city],
            json_hr=config.json_hr[config.city],
            sd=config.seed,
            json_tstep=config.json_tstep,
            tf=config.max_steps,
        )
        env = AMoD(scenario, beta=config.beta[config.city])
        # Timehorizon T=6 (K in paper)
        parser = GNNParser(env, T=6, json_file=f"data/scenario_{config.city}.json")

        model = SAC(
            env=env,
            input_size=13,
            hidden_size=config.hidden_size,
            p_lr=config.p_lr,
            q_lr=config.q_lr,
            alpha=config.alpha,
            batch_size=config.batch_size,
            use_automatic_entropy_tuning=False,
            clip=config.clip,
            critic_version=config.critic_version,
        ).to(config.device)

        train_episodes = config.max_episodes
        # T = config.max_steps
        epochs = trange(train_episodes)
        best_reward = -np.inf
        best_reward_test = -np.inf
        model.train()

        for i_episode in epochs:
            obs = env.reset()  # initialize environment
            episode_reward = 0
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            # actions = []

            # current_eps = []
            done = False
            step = 0
            o = None
            rebreward = None
            action_rl = None
            while not done:
                # take matching step (Step 1 in paper)
                if step > 0:
                    obs1 = copy.deepcopy(o)

                obs, paxreward, done, info, _, _ = env.pax_step(
                    CPLEXPATH=config.cplex_path,
                    PATH=config.path,
                    directory=config.directory,
                )

                o = parser.parse_obs(obs=obs)
                episode_reward += paxreward
                if step > 0:
                    # store transition in memroy
                    rl_reward = paxreward + rebreward
                    model.replay_buffer.store(
                        obs1, action_rl, config.rew_scale * rl_reward, o
                    )

                action_rl = model.select_action(o)

                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(
                    env,
                    config.path,
                    desiredAcc,
                    config.cplex_path,
                    directory=config.directory,
                )
                # Take action in environment
                new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward
                # track performance over episode
                episode_served_demand += info["served_demand"]
                episode_rebalancing_cost += info["rebalancing_cost"]

                step += 1
                if i_episode > 10:
                    # sample from memory and update model
                    batch = model.replay_buffer.sample_batch(
                        config.batch_size, norm=False
                    )
                    model.update(data=batch)

            epochs.set_description(
                f"Episode {i_episode+1} | "
                f"Reward: {episode_reward:.2f} | "
                f"ServedDemand: {episode_served_demand:.2f} | "
                f"Reb. Cost: {episode_rebalancing_cost:.2f}"
            )
            # Checkpoint best performing model
            if episode_reward >= best_reward:
                model.save_checkpoint(
                    path=f"saved_files/ckpt/{config.path}/{config.checkpoint_path}_sample.pth"
                )
                best_reward = episode_reward
            model.save_checkpoint(
                path=f"saved_files/ckpt/{config.path}/{config.checkpoint_path}_running.pth"
            )
            if i_episode % 10 == 0:
                (
                    test_reward,
                    test_served_demand,
                    test_rebalancing_cost,
                ) = model.test_agent(
                    1, env, config.cplex_path, config.directory, parser=parser
                )
                if test_reward >= best_reward_test:
                    best_reward_test = test_reward
                    model.save_checkpoint(
                        path=f"saved_files/ckpt/{config.path}/{config.checkpoint_path}_test.pth"
                    )
    else:
        """Run main testing loop."""
        logger.info("Running main testing loop for SAC.")

        scenario = Scenario(
            json_file=f"data/scenario_{config.city}.json",
            demand_ratio=config.demand_ratio[config.city],
            json_hr=config.json_hr[config.city],
            sd=config.seed,
            json_tstep=config.json_tstep,
            tf=config.max_steps,
        )

        env = AMoD(scenario, beta=config.beta[config.city])
        parser = GNNParser(env, T=6, json_file=f"data/scenario_{config.city}.json")

        model = SAC(
            env=env,
            input_size=13,
            hidden_size=256,
            p_lr=1e-3,
            q_lr=1e-3,
            alpha=0.3,
            batch_size=100,
            use_automatic_entropy_tuning=False,
            critic_version=config.critic_version,
        ).to(config.device)

        print("load model")
        model.load_checkpoint(
            path=f"saved_files/ckpt/{config.path}/{config.checkpoint_path}_test.pth"
        )

        test_episodes = config.max_episodes  # set max number of training episodes
        # T = config.max_steps  # set episode length
        epochs = trange(test_episodes)  # epoch iterator
        # Initialize lists for logging
        # log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}

        rewards = []
        demands = []
        costs = []

        for episode in range(10):
            episode_reward = 0
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            obs = env.reset()
            done = False
            k = 0
            pax_reward = 0
            while not done:
                # take matching step (Step 1 in paper)
                obs, paxreward, done, info, _, _ = env.pax_step(
                    CPLEXPATH=config.cplexpath,
                    PATH=config.path,
                    directory=config.directory,
                )

                episode_reward += paxreward
                pax_reward += paxreward
                # use GNN-RL policy (Step 2 in paper)
                o = parser.parse_obs(obs=obs)
                action_rl = model.select_action(o, deterministic=True)

                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(
                    env, config.path, desiredAcc, config.cplexpath, config.directory
                )

                _, rebreward, done, info, _, _ = env.reb_step(rebAction)

                episode_reward += rebreward
                # track performance over episode
                episode_served_demand += info["served_demand"]
                episode_rebalancing_cost += info["rebalancing_cost"]
                k += 1
            # Send current statistics to screen
            epochs.set_description(
                f"Episode {episode + 1} | "
                f"Reward: {episode_reward:.2f} | "
                f"ServedDemand: {episode_served_demand:.2f} | "
                f"Reb. Cost: {episode_rebalancing_cost:.2f}"
            )
            # Log KPIs

            rewards.append(episode_reward)
            demands.append(episode_served_demand)
            costs.append(episode_rebalancing_cost)

        print("Rewards (mean, std):", np.mean(rewards), np.std(rewards))
        print("Served demand (mean, std):", np.mean(demands), np.std(demands))
        print("Rebalancing cost (mean, std):", np.mean(costs), np.std(costs))


if __name__ == "__main__":
    config = args_to_config()
    main(config)
