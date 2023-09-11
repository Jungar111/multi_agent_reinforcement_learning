"""Parser for GNN."""

import typing as T

import torch
from torch_geometric.data import Data
from torch_geometric.utils import grid

from multi_agent_reinforcement_learning.envs.amod_env import AMoD


class GNNParser:
    """Parser converting raw environment observations to agent inputs (s_t)."""

    def __init__(
        self,
        env: AMoD,
        T: int = 10,
        grid_h: int = 4,
        grid_w: int = 4,
        scale_factor: float = 0.01,
    ):
        """Initialise GNN Parser."""
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.grid_h = grid_h
        self.grid_w = grid_w

    def parse_obs(self, obs: T.Tuple[dict, int, dict, dict]):
        """Parse observations.

        Return the data object called 'data' which is used in the Actors and critc forward pass.
        """
        first_t = torch.tensor(
            [obs[0][n][self.env.time + 1] * self.s for n in self.env.region]
        )
        second_t = torch.tensor(
            [
                [
                    (obs[0][n][self.env.time + 1] + self.env.dacc[n][t]) * self.s
                    for n in self.env.region
                ]
                for t in range(self.env.time + 1, self.env.time + self.T + 1)
            ]
        )
        third_t = torch.tensor(
            [
                [
                    sum(
                        [
                            (self.env.scenario.demand_input[i, j][t])
                            * (self.env.price[i, j][t])
                            * self.s
                            for j in self.env.region
                        ]
                    )
                    for i in self.env.region
                ]
                for t in range(self.env.time + 1, self.env.time + self.T + 1)
            ]
        )
        x = (
            torch.cat(
                (
                    first_t.view(1, 1, self.env.nregion).float(),
                    second_t.view(1, self.T, self.env.nregion).float(),
                    third_t.view(1, self.T, self.env.nregion).float(),
                ),
                dim=1,
            )
            .squeeze(0)
            .view(21, self.env.nregion)
            .T
        )
        # Define width and height of the grid.
        edge_index, pos_coord = grid(height=self.grid_h, width=self.grid_w)
        data = Data(x, edge_index)
        return data
