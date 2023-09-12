"""Config for the entire project."""
from pydantic import BaseModel, ConfigDict
import torch


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
    log_interval: int = 10
