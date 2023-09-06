"""A2C-GNN.

-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from collections import namedtuple

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])
args = namedtuple("args", ("render", "gamma", "log_interval"))
args.render = True
args.gamma = 0.97
args.log_interval = 10

#########################################
############## PARSER ###################
#########################################


class GNNParser:
    """Parser converting raw environment observations to agent inputs (s_t)."""

    def __init__(self, env, T=10, grid_h=4, grid_w=4, scale_factor=0.01):
        """Initialise GNN Parser."""
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.grid_h = grid_h
        self.grid_w = grid_w

    def parse_obs(self, obs):
        """Parse observations."""
        x = (
            torch.cat(
                (
                    torch.tensor(
                        [obs[0][n][self.env.time + 1] * self.s for n in self.env.region]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    torch.tensor(
                        [
                            [
                                (obs[0][n][self.env.time + 1] + self.env.dacc[n][t])
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
            .view(21, self.env.nregion)
            .T
        )
        edge_index, pos_coord = grid(height=self.grid_h, width=self.grid_w)
        data = Data(x, edge_index)
        return data


#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy."""

    def __init__(self, in_channels, out_channels, device="cuda:0"):
        """Init method for an GNNActor."""
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)
        self.device = device

    def forward(self, data):
        """Take one forward pass."""
        out = F.relu(
            self.conv1(data.x.to(self.device), data.edge_index.to(self.device))
        ).to(self.device)
        x = out + data.x.to(self.device)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


#########################################
############## CRITIC ###################
#########################################


class GNNCritic(nn.Module):
    """Critic parametrizing the value function estimator V(s_t)."""

    def __init__(self, in_channels, out_channels):
        """Init method for GNNCritic."""
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)

    def forward(self, data):
        """Make a forward pass."""
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


#########################################
############## A2C AGENT ################
#########################################


class A2C(nn.Module):
    """Advantage Actor Critic algorithm for the AMoD control problem."""

    def __init__(
        self,
        env,
        input_size,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cuda:0"),
    ):
        """Init method for A2C."""
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device

        self.actor_1 = GNNActor(
            self.input_size, self.hidden_size, device=self.device
        ).to(self.device)

        self.actor_2 = GNNActor(
            self.input_size, self.hidden_size, device=self.device
        ).to(self.device)

        self.critic = GNNCritic(self.input_size, self.hidden_size).to(self.device)
        self.obs_parser = GNNParser(self.env)

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions_1 = []
        self.saved_actions_2 = []
        self.rewards_1 = []
        self.rewards_2 = []
        self.to(self.device)

    def forward(self, obs, jitter=1e-20):
        """Forward of both actor and critic."""
        # parse raw environment data in model format
        x = self.parse_obs(obs).to(self.device)

        # actor: computes concentration parameters of a Dirichlet distribution
        a_out_1 = self.actor_1(x)
        a_out_2 = self.actor_2(x)

        concentration_1 = F.softplus(a_out_1).reshape(-1) + jitter
        concentration_2 = F.softplus(a_out_2).reshape(-1) + jitter

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration_1, concentration_2, value

    def parse_obs(self, obs):
        """Parse observations."""
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, obs):
        """Select an action."""
        concentration_1, concentration_2, value = self.forward(obs)

        m_1 = Dirichlet(concentration_1)
        m_2 = Dirichlet(concentration_2)

        action_1 = m_1.sample()
        action_2 = m_2.sample()

        self.saved_actions_1.append(SavedAction(m_1.log_prob(action_1), value))
        self.saved_actions_2.append(SavedAction(m_2.log_prob(action_2), value))

        return list(action_1.cpu().numpy()), list(action_2.cpu().numpy())

    def training_step(self):
        """Take one training step."""
        R_1 = 0
        R_2 = 0
        saved_actions_1 = self.saved_actions_1
        saved_actions_2 = self.saved_actions_2
        policy_losses_1 = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns_1 = []  # list to save the true values
        policy_losses_2 = []  # list to save actor (policy) loss
        returns_2 = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards_1[::-1]:
            # calculate the discounted value
            R_1 = r + args.gamma * R_1
            returns_1.insert(0, R_1)

        for r in self.rewards_2[::-1]:
            # calculate the discounted value
            R_2 = r + args.gamma * R_2
            returns_2.insert(0, R_2)

        returns_1 = torch.tensor(returns_1)
        returns_1 = (returns_1 - returns_1.mean()) / (returns_1.std() + self.eps)

        returns_2 = torch.tensor(returns_2)
        returns_2 = (returns_2 - returns_2.mean()) / (returns_2.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions_1, returns_1):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses_1.append(-log_prob * advantage)

        for (log_prob, value), R in zip(saved_actions_2, returns_2):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses_2.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(
                F.smooth_l1_loss(value, torch.tensor([R]).to(self.device))
            )

        # take gradient steps
        self.optimizers["a_optimizer_1"].zero_grad()
        self.optimizers["a_optimizer_2"].zero_grad()
        a_loss_1 = torch.stack(policy_losses_1).sum()
        a_loss_2 = torch.stack(policy_losses_2).sum()
        a_loss_1.backward()
        a_loss_2.backward()
        self.optimizers["a_optimizer_1"].step()
        self.optimizers["a_optimizer_2"].step()

        self.optimizers["c_optimizer"].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers["c_optimizer"].step()

        # reset rewards and action buffer
        del self.rewards_1[:]
        del self.rewards_2[:]
        del self.saved_actions_1[:]
        del self.saved_actions_2[:]

    def configure_optimizers(self):
        """Configure the optimisers for the GNN."""
        optimizers = dict()
        actor_params_1 = list(self.actor_1.parameters())
        actor_params_2 = list(self.actor_2.parameters())
        critic_params = list(self.critic.parameters())
        optimizers["a_optimizer_1"] = torch.optim.Adam(actor_params_1, lr=1e-3)
        optimizers["a_optimizer_2"] = torch.optim.Adam(actor_params_2, lr=1e-3)
        optimizers["c_optimizer"] = torch.optim.Adam(critic_params, lr=1e-3)
        return optimizers

    def save_checkpoint(self, path="ckpt.pth"):
        """Save checkpoints during training."""
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model"])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        """Log params."""
        torch.save(log_dict, path)
