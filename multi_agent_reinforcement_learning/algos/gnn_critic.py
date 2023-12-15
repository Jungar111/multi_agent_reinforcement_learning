"""GNNCritic module."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GNNCritic(nn.Module):
    """Critic parametrizing the value function estimator V(s_t)."""

    def __init__(self, in_channels: int):
        """Init method for GNNCritic.

        Defines the architecture of the network which is different form the actor network.
        """
        super().__init__()
        self.act_dim = 10
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + 2, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)

    def forward(self, data: Data, action, price=None):
        """Make a forward pass. With data from GNNParser and returns x."""
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        # x = torch.sum(x, dim=0)
        # x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        # x = self.lin3(x)
        # x = x.reshape(self.act_dim, self.in_channels)

        if price is not None:
            concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)
            concat = torch.concat(
                [concat, price[0][0][0].repeat(1, action.size(0)).T], dim=-1
            )
        else:
            concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=0)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x

    def _forward(self, data, state, edge_index, action, price=None):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        # (B,N,22)
        # (10,2)
        # Do this v
        # Action  price
        # [0.723, 1.23]
        # [0.251, 1.23]
        # [0.451, 1.23]
        # [0.281, 1.23]
        # [0.851, 1.23]
        # [0.251, 1.23]
        # [0.251, 1.23]
        # [0.251, 1.23]
        # [0.251, 1.23]
        # [0.251, 1.23]
        if price is not None:
            concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)
            concat = torch.concat(
                [concat, price.repeat(1, action.size(1)).unsqueeze(-1)], dim=-1
            )
        else:
            concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        # Add price here maybe?
        x = self.lin3(x).squeeze(-1)  # (B)
        return x
