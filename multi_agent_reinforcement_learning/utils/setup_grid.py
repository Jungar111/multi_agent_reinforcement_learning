"""Function that produces dummy grid for quicker training and testing."""
from multi_agent_reinforcement_learning.data_models.config import Config
import numpy as np


def setup_dummy_grid(config: Config, determ: bool = True):
    """Setup dummy grid."""
    if determ:
        demand_input = {
            (5, 1): 3,
            (5, 3): 4,
            (5, 2): 5,
            (2, 3): 6,
            (2, 4): 5,
            (2, 0): 4,
            (1, 4): 3,
            (0, 3): 4,
        }
        demand_ratio = {
            0: [1, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2, 2],
            1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            2: [1, 1, 1, 2, 2, 3, 4, 4, 2, 1, 1, 1],
            3: [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1],
            4: [1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1],
            5: [1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 2, 2],
        }

    else:
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

    return demand_ratio, demand_input
