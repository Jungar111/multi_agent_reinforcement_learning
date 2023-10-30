"""Argument parser."""

import argparse
import platform
import torch

from multi_agent_reinforcement_learning.data_models.config import A2CConfig


def parse_arguments():
    """Parse passed arguments."""
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
        "--cplex_path",
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

    args.device = torch.device("cpu")

    if args.cuda:
        args.device = torch.device("cuda:0")

    return args


def args_to_config():
    """Convert args to Pydantic model."""
    args = parse_arguments()
    return A2CConfig(**vars(args))


if __name__ == "__main__":
    print(args_to_config())
