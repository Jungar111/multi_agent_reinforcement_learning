"""Config for the entire project."""
from pydantic import BaseModel, ConfigDict, validator
import torch
from pathlib import Path
import typing as T


class BaseConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    city: str
    cplex_path: str
    seed: int
    json_tstep: int
    test: bool
    directory: str
    max_episodes: int
    max_steps: int
    no_cuda: bool
    render: bool = True
    device: torch.device
    grid_size_x: int
    grid_size_y: int
    n_regions: int
    tf: int
    total_number_of_cars: int
    wandb_mode: str = "online"
    gamma: float = 0.97

    path: str
    json_file: T.Optional[Path]

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
        "nyc4x4": 2.0,
    }


class A2CConfig(BaseConfig):
    """Config class for the A2C method."""

    grid_size_x: int = 4
    grid_size_y: int = 4
    n_regions: int = 4 * 4
    total_number_of_cars: int = 1408
    log_interval: int = 10
    tf: int = 60

    @validator("tf", pre=True)
    @classmethod
    def check_tf(cls, value):
        if value < 10:
            raise ValueError("tf must be at least 10. WE THINK! Depends on grid size.")
        return value

    @validator("n_regions")
    def update_n_regions(cls, value, values):
        value = values.grid_size_x * values.grid_size_y
        return value


class SACConfig(BaseModel):
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

    tf: int = 20
    total_number_of_cars: int = 1408
    grid_size_x: int = 2
    grid_size_y: int = 4
    path: str = "scenario_nyc_brooklyn"
    json_file: T.Optional[Path] = Path("data", "scenario_nyc_brooklyn.json")
