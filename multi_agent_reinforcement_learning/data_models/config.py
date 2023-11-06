"""Config for the entire project."""
from pydantic import BaseModel, ConfigDict, validator
import torch
from pathlib import Path
import typing as T


class A2CConfig(BaseModel):
    """Config class for the A2C method."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    cplex_path: str
    seed: int
    demand_ratio: float
    json_hr: int
    json_tstep: int
    beta: float
    test: bool
    directory: str
    max_episodes: int
    max_steps: int
    no_cuda: bool
    render: bool = True
    device: torch.device
    grid_size_x: int = 4
    grid_size_y: int = 4
    n_regions: int = 4 * 4
    tf: int = 60
    total_number_of_cars: int = 1408
    wandb_mode: str = "online"
    gamma: float = 0.97
    path: str = "scenario_nyc4x4"
    json_file: T.Optional[Path] = Path("data", "scenario_nyc4x4.json")
    log_interval: int = 10

    json_hr: T.Dict[str, int] = {
        "nyc_brooklyn": 19,
        "nyc4x4": 7,
    }

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

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    device: torch.device

    test: bool

    city: str
    seed: int
    json_tstep: int
    max_steps: int
    hidden_size: int
    p_lr: float
    q_lr: float
    alpha: float
    batch_size: int
    critic_version: int
    max_episodes: int
    cplex_path: str
    directory: str
    rew_scale: float
    checkpoint_path: str
    clip: int

    tf: int = 20
    total_number_of_cars: int = 1408
    wandb_mode: str = "online"
    grid_size_x: int = 2
    grid_size_y: int = 4
    path: str = "scenario_nyc_brooklyn"
    json_file: T.Optional[Path] = Path("data", "scenario_nyc_brooklyn.json")
    demand_ratio: T.Dict[str, float] = {
        "san_francisco": 2,
        "washington_dc": 4.2,
        "nyc_brooklyn": 9,
        "shenzhen_downtown_west": 2.5,
    }
    json_hr: T.Dict[str, int] = {
        "san_francisco": 19,
        "washington_dc": 19,
        "nyc_brooklyn": 19,
        "shenzhen_downtown_west": 8,
    }
    beta: T.Dict[str, float] = {
        "san_francisco": 0.2,
        "washington_dc": 0.5,
        "nyc_brooklyn": 0.5,
        "shenzhen_downtown_west": 0.5,
    }
