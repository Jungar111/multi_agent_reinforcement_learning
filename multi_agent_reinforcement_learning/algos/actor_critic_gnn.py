"""A2C-GNN.

A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import typing as T

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Dirichlet, Normal

from multi_agent_reinforcement_learning.algos.gnn_actor import GNNActor
from multi_agent_reinforcement_learning.algos.gnn_critic import GNNCritic
from multi_agent_reinforcement_learning.algos.gnn_parser import GNNParser
from multi_agent_reinforcement_learning.data_models.actor_critic_data import SavedAction
from multi_agent_reinforcement_learning.data_models.actor_data import GraphState
from multi_agent_reinforcement_learning.data_models.config import A2CConfig
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.utils.price_utils import map_to_price


class ActorCritic(nn.Module):
    """Advantage Actor Critic algorithm for the AMoD control problem."""

    # Defines env, input size, episodes and device
    def __init__(
        self,
        env: AMoD,
        input_size: int,
        config: A2CConfig,
        eps: float = np.finfo(np.float32).eps.item(),
    ):
        """Init method for A2C. Sets up the desired attributes including.

        actor: Defined by GNNActor
        critic: Defined by GNNCritc
        obs_parser: Defined by GNNParser
        optimizer: Defines the optimizer
        """
        super(ActorCritic, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.config = config

        self.actor = GNNActor(
            in_channels=self.input_size,
            device=self.config.device,
            config=self.config,
            act_dim=self.config.n_regions[self.config.city],
        ).to(self.config.device)

        self.critic = GNNCritic(self.input_size).to(self.config.device)

        self.obs_parser = GNNParser(self.env, self.config)
        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.config.device)

    def _forward_with_price(
        self, data: T.Optional[T.Dict], obs: GraphState, jitter: float = 1e-20
    ):
        """Forward of both actor and critic in the current enviorenment defined by data.

        softplus used on the actor along with 'jitter'.
        concentration: input for the Dirichlet distribution.
        value: The objective value of the current state, given an action
        returns: concentration, value
        """
        # parse raw environment data in model format
        x = self.parse_obs(obs, data).to(self.config.device)

        # actor: computes concentration parameters of a Dirichlet distribution
        a_out, mu, std = self.actor(x)

        concentration = F.softplus(a_out).reshape(-1) + jitter

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, mu, std, value

    def _forward_without_price(
        self, data: T.Optional[T.Dict], obs: GraphState, jitter: float = 1e-20
    ):
        """Forward of both actor and critic in the current enviorenment defined by data.

        softplus used on the actor along with 'jitter'.
        concentration: input for the Dirichlet distribution.
        value: The objective value of the current state, given an action
        returns: concentration, value
        """
        # parse raw environment data in model format
        x = self.parse_obs(obs, data).to(self.config.device)

        # actor: computes concentration parameters of a Dirichlet distribution
        a_out = self.actor(x)

        concentration = F.softplus(a_out).reshape(-1) + jitter

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, value

    def forward(
        self,
        data: T.Optional[T.Dict],
        obs: GraphState,
        jitter: float = 1e-20,
        include_price=True,
    ):
        """Forward factory method for the actor critic."""
        if include_price:
            return self._forward_with_price(data, obs, jitter)
        else:
            return self._forward_without_price(data, obs, jitter)

    def parse_obs(self, obs: GraphState, data: T.Optional[T.Dict]):
        """Parse observations.

        state: current state of the enviorenment
        returns: state
        """
        state = self.obs_parser.parse_obs(obs=obs, data=data, config=self.config)

        return state

    def select_action(
        self, obs: GraphState, data: T.Optional[T.Dict], probabilistic: bool
    ):
        """Select an action based on the distribution of the vehicles with Dirichlet.

        Saves the log of the new action along with the value computed by the critic
        obs: observation of the current distribution of vehicles.
        return: List of the next actions
        """
        if self.config.include_price:
            concentration, mu, std, value = self.forward(
                obs=obs, data=data, include_price=True
            )
        else:
            concentration, value = self.forward(obs=obs, data=data, include_price=False)

        if probabilistic:
            m = Dirichlet(concentration)
            action = m.rsample()
            if self.config.include_price:
                p = Normal(mu, std)
                price = p.rsample()
                log_prob_p = p.log_prob(price).sum(axis=1)
                log_prob_p -= 2 * (np.log(2) - price[0] - F.softplus(-2 * price[0]))
                price_tanh = torch.tanh(price)

                price = map_to_price(
                    price_tanh,
                    lower=self.config.price_lower_bound,
                    upper=self.config.price_upper_bound,
                )
                self.saved_actions.append(
                    SavedAction(
                        log_prob=m.log_prob(action),
                        value=value,
                        log_prob_price=log_prob_p,
                    )
                )
            else:
                self.saved_actions.append(
                    SavedAction(log_prob=m.log_prob(action), value=value)
                )
        else:
            action = (concentration) / (concentration.sum() + 1e-20)
            action = action.detach()
            price = mu.detach()

        if self.config.include_price:
            return (
                list(action.detach().cpu().numpy()),
                price.detach().cpu().numpy(),
            )

        return list(action.detach().cpu().numpy())

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
            R = r + self.config.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(np.array(returns)).to(self.config.device)
        returns = (returns - returns.mean()) / (
            returns.std() + self.eps
        )  # Standadize the returns object

        for saved_action, R in zip(saved_actions, returns):
            advantage = R - saved_action.value.item()

            # calculate actor (policy) loss
            policy_losses.append(
                -saved_action.log_prob.to(self.config.device) * advantage
            )

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(
                F.smooth_l1_loss(
                    saved_action.value.to(self.config.device),
                    torch.tensor([R]).to(self.config.device),
                )
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
