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
from multi_agent_reinforcement_learning.envs.amod_env import AMoD
import typing as T

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

    def __init__(
        self,
        env: AMoD,
        T: int = 10,
        grid_h: int = 4,
        grid_w: int = 4,
        scale_factor: int = 0.01,
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


#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy."""

    def __init__(self, in_channels: int, out_channels: int, device: str = "cuda:0"):
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


#########################################
############## CRITIC ###################
#########################################


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


#########################################
############## A2C AGENT ################
#########################################


class A2C(nn.Module):
    """Advantage Actor Critic algorithm for the AMoD control problem."""

    # Defines env, input size, episodes and device
    def __init__(
        self,
        env: AMoD,
        input_size: int,
        eps: int = np.finfo(np.float32).eps.item(),
        device=torch.device("cuda:0"),
    ):
        """Init method for A2C. Sets up the desired attributes including.

        actor: Defined by GNNActor
        critic: Defined by GNNCritc
        obs_parser: Defined by GNNParser
        optimizer: Defines the optimizer
        """
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device

        self.actor = GNNActor(self.input_size, self.hidden_size, device=self.device).to(
            self.device
        )
        self.critic = GNNCritic(self.input_size, self.hidden_size).to(self.device)
        self.obs_parser = GNNParser(self.env)

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def forward(self, obs: T.Tuple[dict, int, dict, dict], jitter: float = 1e-20):
        """Forward of both actor and critic in the current enviorenment defined by data.

        softplus used on the actor along with 'jitter'.
        concentration: input for the Dirichlet distribution.
        value: The objective value of the current state?
        returns: concentration, value
        """
        # parse raw environment data in model format
        x = self.parse_obs(obs).to(self.device)

        # actor: computes concentration parameters of a Dirichlet distribution
        a_out = self.actor(x)
        concentration = F.softplus(a_out).reshape(-1) + jitter

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, value

    def parse_obs(self, obs: T.Tuple[dict, int, dict, dict]):
        """Parse observations.

        state: current state of the enviorenment
        returns: state
        """
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, obs: T.Tuple[dict, int, dict, dict]):
        """Select an action based on the distribution of the vehicles with Dirichlet.

        Saves the log of the new action along with the value computed by the critic
        obs: observation of the current distribution of vehicles.
        return: List of the next actions
        """
        concentration, value = self.forward(obs)

        m = Dirichlet(concentration)

        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), value))
        return list(action.cpu().numpy())

    def training_step(self):
        """Take one training step."""
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (
            returns.std() + self.eps
        )  # Standadize the returns object

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()  # Don't know what this is?

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(
                F.smooth_l1_loss(value, torch.tensor([R]).to(self.device))
            )

        # take gradient steps a = actor, c = critic
        self.optimizers["a_optimizer"].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        self.optimizers["a_optimizer"].step()

        self.optimizers["c_optimizer"].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers["c_optimizer"].step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def configure_optimizers(self):
        """Configure the optimisers for the GNN using adam as optimizer."""
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=1e-3)
        optimizers["c_optimizer"] = torch.optim.Adam(critic_params, lr=1e-3)
        return optimizers

    def save_checkpoint(self, path: str = "ckpt.pth"):
        """Save checkpoints during training."""
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str = "ckpt.pth"):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model"])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict: dict, path: str = "log.pth"):
        """Log params."""
        torch.save(log_dict, path)
