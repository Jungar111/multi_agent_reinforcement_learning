"""Main file for project."""
from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
import wandb
import platform

from multi_agent_reinforcement_learning.envs.amod_env import Scenario, AMoD
from multi_agent_reinforcement_learning.algos.a2c_gnn import ActorCritic
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.misc.utils import dictsum
from multi_agent_reinforcement_learning.algos.uniform_actor import UniformActor


def main(args):
    """Run main training loop."""
    device = torch.device("cuda" if args.cuda else "cpu")
    wandb.init(project="master2023", name="making_stuff_work", config={**vars(args)})

    # Define AMoD Simulator Environment
    scenario = Scenario(
        json_file="data/scenario_nyc4x4.json",
        sd=args.seed,
        demand_ratio=args.demand_ratio,
        json_hr=args.json_hr,
        json_tstep=args.json_tsetp,
    )
    env = AMoD(scenario, beta=args.beta)
    # Initialize ActorCritic-GNN
    model = ActorCritic(env=env, input_size=21, device=device).to(device)
    uniform_actor = UniformActor(model.critic, 10)

    if not args.test:
        #######################################
        #############Training Loop#############
        #######################################

        # Initialize lists for logging
        log = {"train_reward": [], "train_served_demand": [], "train_reb_cost": []}
        train_episodes = args.max_episodes  # set max number of training episodes
        T = args.max_steps  # set episode length
        epochs = trange(train_episodes)  # epoch iterator
        best_reward = -np.inf  # set best reward
        model.train()  # set model in train mode
        n_actions = len(env.region)

        for i_episode in epochs:
            obs = env.reset()  # initialize environment
            episode_reward = 0
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            episode_reward_uniform = 0
            episode_served_demand_uniform = 0
            episode_rebalancing_cost_uniform = 0
            for step in range(T):
                # take matching step (Step 1 in paper)
                obs, pax_reward, done, info = env.pax_step(
                    CPLEXPATH=args.cplexpath, PATH="scenario_nyc4"
                )
                episode_reward += pax_reward
                # use GNN-RL policy (Step 2 in paper)
                action_rl = model.select_action(obs)
                action_uniform = uniform_actor.select_action(n_actions=n_actions)
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)

                n_vehicles = dictsum(env.acc, env.time + 1)

                desired_acc = {
                    env.region[i]: int(action_rl[i] * n_vehicles)
                    for i in range(n_actions)
                }

                desired_acc_uniform = {
                    env.region[i]: int(action_uniform[i] * n_vehicles)
                    for i in range(n_actions)
                }

                # solve minimum rebalancing distance problem (Step 3 in paper)
                reb_action = solveRebFlow(
                    env, "scenario_nyc4", desired_acc, args.cplexpath
                )

                reb_action_uniform = solveRebFlow(
                    env, "scenario_nyc4", desired_acc_uniform, args.cplexpath
                )

                _, reb_reward_uniform, _, info_uniform = env.reb_step(
                    reb_action_uniform
                )
                episode_reward_uniform += reb_reward_uniform

                # Take action in environment
                _, reb_reward, done, info = env.reb_step(reb_action)
                episode_reward += reb_reward
                # Store the transition in memory
                model.rewards.append(pax_reward + reb_reward)
                # track performance over episode
                episode_served_demand += info["served_demand"]
                episode_rebalancing_cost += info["rebalancing_cost"]
                episode_served_demand_uniform += info_uniform["served_demand"]
                episode_rebalancing_cost_uniform += info_uniform["rebalancing_cost"]
                # stop episode if terminating conditions are met
                if done:
                    break
            # perform on-policy backprop
            model.training_step()

            # Send current statistics to screen
            epochs.set_description(
                f"Episode {i_episode+1} | Reward: {episode_reward:.2f} |"
                f"ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
            )
            # Checkpoint best performing model
            if episode_reward >= best_reward:
                model.save_checkpoint(
                    path=f"./{args.directory}/ckpt/nyc4/a2c_gnn_test.pth"
                )
                best_reward = episode_reward
            # Log KPIs
            log["train_reward"].append(episode_reward)
            log["train_served_demand"].append(episode_served_demand)
            log["train_reb_cost"].append(episode_rebalancing_cost)
            model.log(log, path=f"./{args.directory}/rl_logs/nyc4/a2c_gnn_test.pth")
            wandb.log(
                {
                    "test_reward": episode_reward,
                    "test_served_demand": episode_served_demand,
                    "test_reb_cost": episode_rebalancing_cost,
                    "test_reward_uniform": episode_reward_uniform,
                    "test_served_demand_uniform": episode_served_demand_uniform,
                    "test_reb_cost_uniform": episode_rebalancing_cost_uniform,
                }
            )
    else:
        # Load pre-trained model
        model.load_checkpoint(path=f"./{args.directory}/ckpt/nyc4/a2c_gnn.pth")
        test_episodes = args.max_episodes  # set max number of training episodes
        T = args.max_steps  # set episode length
        epochs = trange(test_episodes)  # epoch iterator
        # Initialize lists for logging
        log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}
        for episode in epochs:
            episode_reward = 0
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            obs = env.reset()
            done = False
            k = 0
            while not done:
                # take matching step (Step 1 in paper)
                obs, pax_reward, done, info = env.pax_step(
                    CPLEXPATH=args.cplexpath, PATH="scenario_nyc4_test"
                )
                episode_reward += pax_reward
                # use GNN-RL policy (Step 2 in paper)
                action_rl = model.select_action(obs)
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desired_acc = {
                    env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                # solve minimum rebalancing distance problem (Step 3 in paper)
                reb_action = solveRebFlow(
                    env, "scenario_nyc4_test", desired_acc, args.cplexpath
                )
                # Take action in environment
                new_obs, reb_reward, done, info = env.reb_step(reb_action)
                episode_reward += reb_reward
                # track performance over episode
                episode_served_demand += info["served_demand"]
                episode_rebalancing_cost += info["rebalancing_cost"]
                k += 1
            # Send current statistics to screen
            epochs.set_description(
                f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand:"
                f"{episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
            )
            # Log KPIs
            log["test_reward"].append(episode_reward)
            log["test_served_demand"].append(episode_served_demand)
            log["test_reb_cost"].append(episode_rebalancing_cost)
            model.log(log, path=f"./{args.directory}/rl_logs/nyc4/a2c_gnn_test.pth")
            break

        wandb.finish()


if __name__ == "__main__":
    cplex_path = ""
    if platform.system() == "Linux":
        cplex_path = "/opt/ibm/ILOG/CPLEX_Studio2211/opl/bin/x86-64_linux/"
    elif platform.system() == "Windows":
        cplex_path = (
            r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\\opl\\bin\\x64_win64\\"
        )
    else:
        raise NotImplementedError()

    parser = argparse.ArgumentParser(description="ActorCritic-GNN")

    # Simulator parameters
    parser.add_argument(
        "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
    )
    parser.add_argument(
        "--demand_ratio",
        type=int,
        default=0.5,
        metavar="S",
        help="demand_ratio (default: 0.5)",
    )
    parser.add_argument(
        "--json_hr", type=int, default=7, metavar="S", help="json_hr (default: 7)"
    )
    parser.add_argument(
        "--json_tsetp",
        type=int,
        default=3,
        metavar="S",
        help="minutes per timestep (default: 3min)",
    )
    parser.add_argument(
        "--beta",
        type=int,
        default=0.5,
        metavar="S",
        help="cost of rebalancing (default: 0.5)",
    )

    # Model parameters
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="activates test mode for agent evaluation",
    )
    parser.add_argument(
        "--cplexpath",
        type=str,
        default=cplex_path,
        help="defines directory of the CPLEX installation",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="saved_files",
        help="defines directory where to save files",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=16000,
        metavar="N",
        help="number of episodes to train agent (default: 16k)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=60,
        metavar="N",
        help="number of steps per episode (default: T=60)",
    )
    parser.add_argument(
        "--no-cuda", type=bool, default=False, help="disables CUDA training"
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
