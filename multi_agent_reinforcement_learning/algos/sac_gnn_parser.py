"""Temporary document for parser for GNN in the SAC implementation."""

import json

import torch
from torch_geometric.data import Data

from multi_agent_reinforcement_learning.data_models.actor_data import (
    # ActorData,
    GraphState,
)


class GNNParser:
    """Parser converting raw environment observations to agent inputs (s_t)."""

    def __init__(self, env, T=10, json_file=None, scale_factor=0.01):
        """Init for the parser."""
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs: GraphState):
        """Parse observations to provide GNN input."""
        x = (
            torch.cat(
                (
                    torch.tensor(
                        [
                            obs.acc[n][self.env.time + 1] * self.s
                            for n in self.env.region
                        ]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    torch.tensor(
                        [
                            [
                                (obs.acc[n][self.env.time + 1] + obs.dacc[n][t])
                                * self.s
                                for n in self.env.region
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                    torch.tensor(
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
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                ),
                dim=1,
            )
            .squeeze(0)
            .view(1 + self.T + self.T, self.env.nregion)
            .T
        )
        if self.json_file is not None:
            edge_index = torch.vstack(
                (
                    torch.tensor(
                        [edge["i"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                    torch.tensor(
                        [edge["j"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                )
            ).long()
        else:
            edge_index = torch.cat(
                (
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                ),
                dim=0,
            ).long()
        data = Data(x, edge_index)
        return data
