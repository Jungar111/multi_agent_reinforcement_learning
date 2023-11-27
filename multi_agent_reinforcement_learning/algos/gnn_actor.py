"""GNNActor module."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn.init import normal_, constant_


class GNNActor(nn.Module):
    """Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy."""

    def __init__(
        self,
        in_channels: int,
        device: torch.device = torch.device("cuda:0"),
    ):
        """Init method for an GNNActor. Defining the model architecture."""
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.dirichlet_concentration_layer = nn.Linear(32, 1)
        self.price_lin_mu = nn.Linear(32, 1)
        self.price_lin_std = nn.Linear(32, 1)
        self.device = device

        normal_(self.price_lin_std.weight, mean=0, std=0.1)
        constant_(self.price_lin_std.bias, 0)

        normal_(self.price_lin_mu.weight, mean=0, std=1)
        constant_(self.price_lin_mu.bias, 1)

    def forward(self, data: Data):
        """Take one forward pass in the model defined in init and return x."""
        out = F.relu(
            self.conv1(data.x.to(self.device), data.edge_index.to(self.device))
        ).to(self.device)
        x = out + data.x.to(self.device)
        x = F.relu(self.lin1(x))
        last_hidden_layer = F.relu(self.lin2(x))

        dirichlet_concentration = self.dirichlet_concentration_layer(last_hidden_layer)

        price_pool = global_mean_pool(last_hidden_layer, data.batch)

        # outputs mu and sigma for a lognormal distribution
        mu = self.price_lin_mu(price_pool)
        sigma = F.softplus(self.price_lin_std(price_pool))

        return dirichlet_concentration, mu, sigma
