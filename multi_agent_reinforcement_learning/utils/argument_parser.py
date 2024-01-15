"""Argument parser."""

import argparse
import platform
from pathlib import Path

import torch

from multi_agent_reinforcement_learning.data_models.city_enum import City
from multi_agent_reinforcement_learning.data_models.config import A2CConfig


def parse_arguments(cuda: bool):
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
        "--json_tstep",
        type=int,
        default=3,
        metavar="S",
        help="minutes per timestep (default: 3min)",
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
        "--no-cuda", type=bool, default=not cuda, help="disables CUDA training"
    )

    parser.add_argument(
        "--rew_scale",
        type=float,
        default=0.1,
        help="reward scaling factor (default: 0.1)",
    )

    parser.add_argument(
        "--no-cars",
        type=int,
        default=374,
        help="Number of cars in total",
    )

    parser.add_argument(
        "--no-actors",
        type=int,
        default=2,
        help="Number of actors",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Name for the run on W&B",
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = torch.device("cpu")

    if args.cuda:
        args.device = torch.device("cuda:0")

    return args


def args_to_config(city: City, cuda=False):
    """Convert args to Pydantic model."""
    args = parse_arguments(cuda)
    return A2CConfig(
        **vars(args),
        city=city.value,
        path=f"scenario_{city.value}",
        json_file=Path("data", f"scenario_{city.value}.json"),
    )


if __name__ == "__main__":
    city = City.new_york
    print(args_to_config(city))
