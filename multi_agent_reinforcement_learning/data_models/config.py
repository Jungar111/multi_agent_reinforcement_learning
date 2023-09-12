"""Config for the entire project."""
from pydantic import BaseModel, ConfigDict
import torch
from pathlib import Path


class Config(BaseModel):
    """Config class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    tf: int = 60
    ninit: int = 80

    gamma: float = 0.97
    json_file: str = Path("data", "scenario_nyc4x4.json")
    log_interval: int = 10

    # @model_validator
    # def check_tf(cls, values):
    #     tf = values.get("tf")
    #     if tf <= 10:
    #         raise ValueError("tf must be at least 11.")
    #     return values

    # @model_validator
    # def check_grid(cls, values):
    #     n1 = values.get("grid_size_x")
    #     n2 = values.get("grid_size_y")
    #     if n1 < 2 | n2 < 3:
    #         raise ValueError("n1 must be at least 2 and n2 must be at least 3.")
    #     return values
