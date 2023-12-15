"""GNNActor module."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch.distributions import Normal, Dirichlet
import numpy as np

from multi_agent_reinforcement_learning.data_models.config import A2CConfig
from multi_agent_reinforcement_learning.utils.price_utils import map_to_price


class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(
        self,
        config: A2CConfig,
        in_channels,
        hidden_size=32,
        act_dim=6,
        device: torch.device = torch.device("cuda:0"),
    ):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.dirichlet_concentration_layer = nn.Linear(hidden_size, 1)
        if self.config.include_price:
            self.price_upper_bound = 10
            self.price_lower_bound = 0
            self.log_std_min = -20
            self.log_std_max = 2
            self.price_lin_mu = nn.Linear(hidden_size, 1)
            self.price_lin_std = nn.Linear(hidden_size, 1)
            self.device = device

    def forward(self, data: Data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        last_hidden_layer = F.leaky_relu(self.lin2(x))

        a_out = self.dirichlet_concentration_layer(last_hidden_layer)
        a_out = F.softplus(a_out).reshape(-1) + 1e-20
        m = Dirichlet(a_out)
        action = m.rsample()
        logp_a = m.log_prob(action)

        if self.config.include_price:
            price_pool = global_mean_pool(
                last_hidden_layer,
                torch.tensor(
                    [0 for i in range(int(self.config.n_regions[self.config.city]))]
                ),
            )
            # outputs mu and sigma for a normal distribution
            mu = self.price_lin_mu(price_pool)  # [-1,1]
            log_std = torch.clamp(
                self.price_lin_std(price_pool), self.log_std_min, self.log_std_max
            )
            sigma = torch.exp(log_std)

            p = Normal(mu, sigma)
            pi_action_p = p.rsample()
            log_prob_p = p.log_prob(pi_action_p).sum(axis=-1)
            # Correction formula for Tanh squashing see: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
            # and appendix C in original SAC paper.
            log_prob_p -= 2 * (
                np.log(2) - pi_action_p[0] - F.softplus(-2 * pi_action_p[0])
            )
            price_tanh = torch.tanh(pi_action_p)

            price = map_to_price(
                price_tanh,
                lower=self.config.price_lower_bound,
                upper=self.config.price_upper_bound,
            )

            return a_out, action, logp_a, price, log_prob_p

        return a_out
