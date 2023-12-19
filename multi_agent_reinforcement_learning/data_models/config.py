"""Config for the entire project."""
from pydantic import BaseModel, ConfigDict
import torch
from pathlib import Path
import typing as T


class BaseConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    city: str
    path: str
    json_file: T.Optional[Path]
    cplex_path: str
    seed: int
    json_tstep: int
    test: bool
    directory: str
    max_episodes: int
    max_steps: int
    no_cuda: bool
    render: bool = True
    city: str = "san_francisco"
    device: torch.device
    grid_size_x: int
    grid_size_y: int
    tf: int
    total_number_of_cars: int
    wandb_mode: str = "online"
    gamma: float = 0.97
    include_price: bool = True

    demand_ratio: T.Dict[str, float] = {
        "san_francisco": 2.0,
        "washington_dc": 4.2,
        "nyc_brooklyn": 9.0,
        "shenzhen_downtown_west": 2.5,
        "nyc4x4": 0.5,
    }
    json_hr: T.Dict[str, int] = {
        "san_francisco": 19,
        "washington_dc": 19,
        "nyc_brooklyn": 19,
        "shenzhen_downtown_west": 8,
        "nyc4x4": 7,
    }
    beta: T.Dict[str, float] = {
        "san_francisco": 0.2,
        "washington_dc": 0.5,
        "nyc_brooklyn": 0.5,
        "shenzhen_downtown_west": 0.5,
        "nyc4x4": 0.5,
    }
    grid_size_x: T.Dict[str, int] = {
        "san_francisco": 2,
        "washington_dc": 3,
        "nyc_brooklyn": 2,
        "shenzhen_downtown_west": 3,
        "nyc4x4": 4,
    }
    grid_size_y: T.Dict[str, int] = {
        "san_francisco": 5,
        "washington_dc": 6,
        "nyc_brooklyn": 7,
        "shenzhen_downtown_west": 6,
        "nyc4x4": 4,
    }
    n_regions: T.Dict[str, int] = {
        "san_francisco": 10,
        "washington_dc": 18,
        "nyc_brooklyn": 14,
        "shenzhen_downtown_west": 17,
        "nyc4x4": 12,
    }


class A2CConfig(BaseConfig):
    """Config class for the A2C method."""

    total_number_of_cars: int = 1408
    log_interval: int = 10
    tf: int = 60


class SACConfig(BaseConfig):
    """Config class for the SAC implementation."""

    test: bool
    hidden_size: int
    p_lr: float
    q_lr: float
    alpha: float
    batch_size: int
    critic_version: int
    rew_scale: float
    checkpoint_path: str
    clip: int

    dynamic_scaling: bool = True
    tf: int = 20
    total_number_of_cars: int = 374
