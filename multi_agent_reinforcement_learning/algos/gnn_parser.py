"""Parser for GNN."""

import typing as T

import torch
from torch_geometric.data import Data
from torch_geometric.utils import grid

from multi_agent_reinforcement_learning.data_models.actor_data import (
    GraphState,
)
from multi_agent_reinforcement_learning.data_models.config import A2CConfig
from multi_agent_reinforcement_learning.envs.amod import AMoD


class GNNParser:
    """Parser converting raw environment observations to agent inputs (s_t)."""

    def __init__(
        self,
        env: AMoD,
        config: A2CConfig,
        T: int = 10,
        scale_factor: float = 0.01,
    ):
        """Initialise GNN Parser."""
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.grid_size_x = config.grid_size_x
        self.grid_size_y = config.grid_size_y
        self.demand_input = self.env.scenario.demand_input
        if config.json_file is None:
            self.demand_input = self.env.scenario.demand_input2

    def parse_obs(self, data: T.Optional[T.Dict], obs: GraphState, config: A2CConfig):
        """Parse observations.

        Return the data object called 'data' which is used in the Actors and critc forward pass.
        """
        first_t = torch.tensor(
            [obs.acc[n][self.env.time + 1] * self.s for n in self.env.region]
        )
        second_t = torch.tensor(
            [
                [
                    (obs.acc[n][self.env.time + 1] + obs.dacc[n][t]) * self.s
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
                            (self.demand_input[i, j][t])
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
        if config.json_file is not None and "4x4" not in str(config.json_file):
            edge_index = torch.vstack(
                (
                    torch.tensor([edge["i"] for edge in data["topology_graph"]]).view(
                        1, -1
                    ),
                    torch.tensor([edge["j"] for edge in data["topology_graph"]]).view(
                        1, -1
                    ),
                )
            ).long()
        else:
            edge_index, _ = grid(height=self.grid_size_x, width=self.grid_size_y)
        data = Data(x, edge_index)
        return data
