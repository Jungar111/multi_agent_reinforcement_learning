"""Argument parser for the SAC implementation."""
import argparse
import platform
from pathlib import Path

import torch

from multi_agent_reinforcement_learning.data_models.city_enum import City
from multi_agent_reinforcement_learning.data_models.config import SACConfig


def parse_arguments(cuda):
    """Parse arguments for SAC."""
    cplex_path = ""
    if platform.system() == "Linux":
        cplex_path = "/opt/ibm/ILOG/CPLEX_Studio2211/opl/bin/x86-64_linux/"
    elif platform.system() == "Windows":
        cplex_path = (
            r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\\opl\\bin\\x64_win64\\"
        )
    else:
        raise NotImplementedError()

    parser = argparse.ArgumentParser(description="SAC-GNN")

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
        default=10000,
        metavar="N",
        help="number of episodes to train agent (default: 16k)",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        metavar="N",
        help="number of steps per episode (default: T=20)",
    )

    parser.add_argument(
        "--no-cuda", type=bool, default=not cuda, help="disables CUDA training"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="batch size for training (default: 100)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="entropy coefficient (default: 0.3)",
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="hidden size of neural networks (default: 256)",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="SAC",
        help="name of checkpoint file to save/load (default: SAC)",
    )

    parser.add_argument(
        "--clip",
        type=int,
        default=500,
        help="clip value for gradient clipping (default: 500)",
    )

    parser.add_argument(
        "--p_lr",
        type=float,
        default=1e-3,
        help="learning rate for policy network (default: 1e-4)",
    )

    parser.add_argument(
        "--q_lr",
        type=float,
        default=1e-3,
        help="learning rate for Q networks (default: 4e-3)",
    )

    parser.add_argument(
        "--rew_scale",
        type=float,
        default=0.1,
        help="reward scaling factor (default: 0.1)",
    )

    parser.add_argument(
        "--critic_version",
        type=int,
        default=4,
        help="critic version (default: 4)",
    )

    parser.add_argument(
        "-f", "--fff", help="a dummy argument to fool ipython", default="1"
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cpu")

    if args.cuda:
        args.device = torch.device("cuda:0")

    return args


def args_to_config(city: City, cuda=False):
    """Convert args to pydantic model."""
    args = parse_arguments(cuda)
    return SACConfig(
        **vars(args),
        city=city.value,
        path=f"scenario_{city.value}",
        json_file=Path("data", f"scenario_{city.value}.json"),
    )
