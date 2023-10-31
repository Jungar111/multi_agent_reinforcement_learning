"""Config for the entire project."""
from pydantic import BaseModel, ConfigDict, validator
import torch
from pathlib import Path
import typing as T


class Config(BaseModel):
    """Config class."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    cplex_path: str
    seed: int
    demand_ratio: float
    json_hr: int
    json_tsetp: int
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
    json_file: T.Optional[Path] = Path("data", "scenario_nyc_brooklyn.json")
    log_interval: int = 10

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
