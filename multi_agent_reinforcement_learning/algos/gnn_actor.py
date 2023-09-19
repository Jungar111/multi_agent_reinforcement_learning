"""GNNActor module."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GNNActor(nn.Module):
    """Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        device: torch.device = torch.device("cuda:0"),
    ):
        """Init method for an GNNActor. Defining the model architecture."""
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)
        self.device = device

    def forward(self, data: Data):
        """Take one forward pass in the model defined in init and return x."""
        out = F.relu(
            self.conv1(data.x.to(self.device), data.edge_index.to(self.device))
        ).to(self.device)
        x = out + data.x.to(self.device)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
