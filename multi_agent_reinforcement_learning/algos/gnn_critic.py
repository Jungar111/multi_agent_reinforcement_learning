"""GNNCritic module."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GNNCritic(nn.Module):
    """Critic parametrizing the value function estimator V(s_t)."""

    def __init__(self, in_channels: int, out_channels: int):
        """Init method for GNNCritic.

        Defines the architecture of the network which is different form the actor network.
        """
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)

    def forward(self, data: Data):
        """Make a forward pass. With data from GNNParser and returns x."""
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
