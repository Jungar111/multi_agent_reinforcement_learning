"""Function that produces dummy grid for quicker training and testing."""
from multi_agent_reinforcement_learning.data_models.config import Config
import numpy as np


def setup_dummy_grid(config: Config):
    """Setup dummy grid."""
    np.random.seed(1337)
    n1 = config.grid_size_x
    n2 = config.grid_size_y
    demand_ratio = {
        i: np.random.randint(1, 5, n1 * n2 * 2).tolist() for i in range(n1 * n2)
    }
    demand_input = {
        (
            np.random.randint(0, n1 * n2 - 1),
            np.random.randint(0, n1 * n2 - 1),
        ): np.random.randint(3, 7)
        for _ in range(n1 * n2 + n1 + n2)
    }
    demand_input["default"] = 1

    return config.tf, demand_ratio, demand_input, config.ninit
