"""Main file for project."""
from __future__ import print_function
import argparse
from datetime import datetime
from tqdm import trange
import numpy as np
import torch
import wandb
import platform

from multi_agent_reinforcement_learning.envs.amod_env import Scenario, AMoD
from multi_agent_reinforcement_learning.algos.a2c_gnn import A2C
from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.misc.utils import dictsum
from multi_agent_reinforcement_learning.logs import ModelLog


def main(args):
    """Run main training loop."""
    device = torch.device("cuda" if args.cuda else "cpu")
    wandb.init(
        project="master2023",
        name=f"test_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        if args.test
        else f"train_log ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        config={**vars(args)},
    )

    # # Define AMoD Simulator Environment
    # scenario = Scenario(
    #     json_file="data/scenario_nyc4x4.json",
    #     sd=args.seed,
    #     demand_ratio=args.demand_ratio,
    #     json_hr=args.json_hr,
    #     json_tstep=args.json_tsetp,
    # )

    # Define AMoD Simulator Environment
    scenario = Scenario(
        json_file=None,
        tf=20,
        demand_ratio={
            0: [1, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2, 2],
            1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            2: [1, 1, 1, 2, 2, 3, 4, 4, 2, 1, 1, 1],
            3: [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1],
            4: [1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1],
            5: [1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 2, 2],
        },
        demand_input={
            (5, 1): 3,
            (5, 3): 4,
            (5, 2): 5,
            (2, 3): 6,
            (2, 4): 5,
            (2, 0): 4,
            (1, 4): 3,
            (0, 3): 4,
            "default": 1,
        },
        ninit=80,
    )

    env = AMoD(scenario, beta=args.beta)
    # Initialize A2C-GNN
    model = A2C(env=env, input_size=21, device=device).to(device)

    if not args.test:
        #######################################
        #############Training Loop#############
        #######################################

        # Initialize lists for logging
        train_episodes = args.max_episodes  # set max number of training episodes
        T = args.max_steps  # set episode length
        epochs = trange(train_episodes)  # epoch iterator
        best_reward = -np.inf  # set best reward
        model.train()  # set model in train mode

        for i_episode in epochs:
            train_log = ModelLog()
            obs = env.reset()  # initialize environment
            for step in range(T):
                # take matching step (Step 1 in paper)
                obs, paxreward, done, info, ext_reward, ext_done = env.pax_step(
                    CPLEXPATH=args.cplexpath, PATH="scenario_nyc4"
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
                    env, "scenario_nyc4", desiredAcc, args.cplexpath
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
                    path=f"./{args.directory}/ckpt/nyc4/a2c_gnn_test.pth"
                )
                best_reward = train_log.reward
            # Log KPIs on weights and biases
            wandb.log({**dict(train_log)})
    else:
        # Load pre-trained model
        model.load_checkpoint(path=f"./{args.directory}/ckpt/nyc4/a2c_gnn.pth")
        test_episodes = args.max_episodes  # set max number of training episodes
        T = args.max_steps  # set episode length
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
                    CPLEXPATH=args.cplexpath, PATH="scenario_nyc4_test"
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
                    env, "scenario_nyc4_test", desiredAcc, args.cplexpath
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
    cplex_path = ""
    if platform.system() == "Linux":
        cplex_path = "/opt/ibm/ILOG/CPLEX_Studio2211/opl/bin/x86-64_linux/"
    elif platform.system() == "Windows":
        cplex_path = (
            r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\\opl\\bin\\x64_win64\\"
        )
    else:
        raise NotImplementedError()

    parser = argparse.ArgumentParser(description="A2C-GNN")

    # Simulator parameters
    parser.add_argument(
        "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
    )
    parser.add_argument(
        "--demand_ratio",
        type=float,
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
        type=float,
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
