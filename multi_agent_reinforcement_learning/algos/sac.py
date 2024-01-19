"""Module for the SAC algorithm."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Dirichlet, Normal
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from multi_agent_reinforcement_learning.algos.reb_flow_solver import solveRebFlow
from multi_agent_reinforcement_learning.algos.sac_gnn_parser import (
    GNNParser as SACGNNParser,
)
from multi_agent_reinforcement_learning.data_models.actor_data import (
    ActorData,
    GraphState,
    ModelLog,
)
from multi_agent_reinforcement_learning.data_models.config import SACConfig
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.utils.minor_utils import dictsum
from multi_agent_reinforcement_learning.utils.price_utils import map_to_price


class PairData(Data):
    """Class holding elements in the replay buffer."""

    def __init__(
        self,
        edge_index_s=None,
        x_s=None,
        reward=None,
        action=None,
        edge_index_t=None,
        x_t=None,
        price=None,
        device: torch.device = torch.device("cuda:0"),
    ):
        """Initialise the element."""
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.price = price
        self.device = device

    def __inc__(self, key, value, *args, **kwargs):
        """Increment the data."""
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ReplayData:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, device):
        """Initialise the buffer."""
        self.device = device
        self.data_list = []
        self.rewards = []

    def store(self, data1, action, reward, data2, price):
        """Store data in the buffer."""
        self.data_list.append(
            PairData(
                edge_index_s=data1.edge_index,
                x_s=data1.x,
                reward=torch.as_tensor(reward),
                action=torch.as_tensor(action),
                edge_index_t=data2.edge_index,
                x_t=data2.x,
                price=torch.as_tensor(price[0]) if price != None else None,
            )
        )
        self.rewards.append(reward)

    def size(self):
        """Return the size of the buffer."""
        return len(self.data_list)

    def sample_batch(self, batch_size=32, norm=False):
        """Sample a batch from the replay buffer."""
        data = random.sample(self.data_list, batch_size)
        if norm:
            mean = np.mean(self.rewards)
            std = np.std(self.rewards)
            batch = Batch.from_data_list(data, follow_batch=["x_s", "x_t"])
            batch.reward = (batch.reward - mean) / (std + 1e-16)
            return batch.to(self.device)
        else:
            return Batch.from_data_list(data, follow_batch=["x_s", "x_t"]).to(
                self.device
            )


class Scalar(nn.Module):
    """Defines a scalar in torch."""

    def __init__(self, init_value):
        """Init for the scalar."""
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        """Return the scalar."""
        return self.constant


#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    r"""Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy."""

    def __init__(
        self,
        config: SACConfig,
        in_channels,
        hidden_size=32,
        act_dim=6,
        device: torch.device = torch.device("cuda:0"),
    ):
        """Init method for the actor."""
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.dirichlet_concentration_layer = nn.Linear(hidden_size, 1)
        if self.config.include_price:
            self.log_std_min = -20
            self.log_std_max = 2
            self.price_lin_mu = nn.Linear(hidden_size, 1)
            self.price_lin_std = nn.Linear(hidden_size, 1)
            self.device = device

    def forward(self, state, edge_index, deterministic=False):
        """Forward pass for the actor."""
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        last_hidden_layer = F.leaky_relu(self.lin2(x))

        concentration = F.softplus(
            self.dirichlet_concentration_layer(last_hidden_layer)
        ).squeeze(-1)

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

        if deterministic:
            action = (concentration) / (concentration.sum() + 1e-20)
            log_prob = None
            if self.config.include_price:
                pi_action_p = mu[0, 0].detach()
        else:
            m = Dirichlet(concentration + 1e-20)
            action = m.rsample()
            log_prob = m.log_prob(action)
            if self.config.include_price:
                p = Normal(mu, sigma)
                pi_action_p = p.rsample()
                log_prob_p = p.log_prob(pi_action_p).sum(axis=-1)
        if self.config.include_price:
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

            return action, log_prob, price, log_prob_p
        return action, log_prob


#########################################
############## CRITIC ###################
#########################################


class GNNCritic(nn.Module):
    """Architecture: GNN, Concatenation, FC, Readout."""

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        """Init for the critic."""
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + 2, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action, price=None):
        """Forward pass for the critic."""
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
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


#########################################
############## A2C AGENT ################
#########################################


class SAC(nn.Module):
    """Advantage Actor Critic algorithm for the AMoD control problem."""

    def __init__(
        self,
        env: AMoD,
        actor_data: ActorData,
        config: SACConfig,
        input_size: int,
        hidden_size: int = 32,
        alpha: float = 0.2,
        gamma: float = 0.99,
        polyak: float = 0.995,
        batch_size: int = 128,
        p_lr: float = 3e-4,
        q_lr: float = 1e-3,
        use_automatic_entropy_tuning=False,
        lagrange_thresh: int = -1,
        min_q_weight: int = 1,
        deterministic_backup=False,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
        min_q_version: int = 3,
        clip: int = 200,
    ):
        """Init method for the SAC algorithm."""
        super(SAC, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nregion
        self.config = config
        self.actor_data = actor_data
        self.train_log = ModelLog()

        # SAC parameters
        self.alpha = alpha
        self.polyak = polyak
        self.env = env
        self.BATCH_SIZE = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.min_q_version = min_q_version
        self.clip = clip

        # Parser
        self.obs_parser = SACGNNParser(self.env, T=6, json_file=self.config.json_file)

        # conservative Q learning parameters
        self.num_random = 10
        self.temp = 1.0
        self.min_q_weight = min_q_weight
        if lagrange_thresh == -1:
            self.with_lagrange = False
        else:
            print("using lagrange")
            self.with_lagrange = True
        self.deterministic_backup = deterministic_backup
        self.step = 0
        self.nodes = env.nregion

        self.replay_buffer = ReplayData(device=device)
        # nnets
        self.actor = GNNActor(
            self.config, self.input_size, self.hidden_size, act_dim=self.act_dim
        )

        self.critic1 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim
        )
        self.critic2 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim
        )
        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim
        )
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh  # lagrange treshhold
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = torch.optim.Adam(
                self.log_alpha_prime.parameters(),
                lr=self.p_lr,
            )

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.act_dim).item()
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=1e-3
            )

    def parse_obs(self, obs: GraphState):
        """Parse observations from graph."""
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, data, deterministic=False):
        """Select action with actor."""
        with torch.no_grad():
            if self.config.include_price:
                a, _, price, _ = self.actor(data.x, data.edge_index, deterministic)
            else:
                a, _ = self.actor(data.x, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy()[0]
        if self.config.include_price:
            price = price.squeeze(-1).detach().cpu().numpy()
            return list(a), price.tolist()
        return list(a)

    def compute_loss_q(self, data):
        """Loss for the critic."""
        (
            state_batch,
            edge_index,
            next_state_batch,
            edge_index2,
            reward_batch,
            action_batch,
        ) = (
            data.x_s,
            data.edge_index_s,
            data.x_t,
            data.edge_index_t,
            data.reward,
            data.action.reshape(-1, self.nodes),
        )
        if self.config.include_price:
            price = data.price
            q1 = self.critic1(state_batch, edge_index, action_batch, price)
            q2 = self.critic2(state_batch, edge_index, action_batch, price)
        else:
            q1 = self.critic1(state_batch, edge_index, action_batch)
            q2 = self.critic2(state_batch, edge_index, action_batch)
        with torch.no_grad():
            # Target actions come from *current* policy
            if self.config.include_price:
                a2, logp_a2, _, logp_p = self.actor(
                    next_state_batch,
                    edge_index2,
                )
                if self.config.dynamic_scaling:
                    logp_a2 *= float(torch.abs(logp_p.mean() / logp_a2.mean()))
                else:
                    logp_a2 *= 0.1
                q1_pi_targ = self.critic1_target(
                    next_state_batch, edge_index2, a2, price
                )
                q2_pi_targ = self.critic2_target(
                    next_state_batch, edge_index2, a2, price
                )
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = reward_batch + self.gamma * (
                    q_pi_targ - self.alpha * (logp_a2 + logp_p)
                )
            else:
                a2, logp_a2 = self.actor(next_state_batch, edge_index2)
                q1_pi_targ = self.critic1_target(next_state_batch, edge_index2, a2)
                q2_pi_targ = self.critic2_target(next_state_batch, edge_index2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = reward_batch + self.gamma * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        return loss_q1, loss_q2

    def compute_loss_pi(self, data):
        """Loss for the actor."""
        state_batch, edge_index = (
            data.x_s,
            data.edge_index_s,
        )
        actor_val = 0
        if self.config.include_price:
            actions, logp_a, price, logp_p = self.actor(state_batch, edge_index)
            price = price[:, :, 0]
            if self.config.dynamic_scaling:
                logp_a *= float(torch.abs(logp_p.mean() / logp_a.mean()))
            else:
                logp_a *= 0.1
            # @TODO Investigate the magnitude of the logprobs. Maybe find magic number.
            # maybe TODO Look into alpha, may make a difference - 0.1 to 0.3
            actor_val = self.alpha * (logp_a + logp_p)
            # @TODO hallo pris
            q1_1 = self.critic1(state_batch, edge_index, actions, price)
            q2_a = self.critic2(state_batch, edge_index, actions, price)
            q_a = torch.min(q1_1, q2_a)
        else:
            actions, logp_a = self.actor(state_batch, edge_index)
            actor_val = self.alpha * logp_a
            q1_1 = self.critic1(state_batch, edge_index, actions)
            q2_a = self.critic2(state_batch, edge_index, actions)
            q_a = torch.min(q1_1, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (logp_a + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        loss_pi = (actor_val - q_a).mean()

        return loss_pi

    def update(self, data):
        """Update the model weights and biases."""
        loss_q1, loss_q2 = self.compute_loss_q(data)

        self.optimizers["c1_optimizer"].zero_grad()

        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        loss_q1.backward()
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.optimizers["c2_optimizer"].step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # one gradient descent step for policy network
        self.optimizers["a_optimizer"].zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.optimizers["a_optimizer"].step()

        # Unfreeze Q-networks
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

    def configure_optimizers(self):
        """Configure optimizers for the sac algorithm."""
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.q_lr)

        return optimizers

    def test_agent(self, test_episodes, env, cplexpath, directory, parser):
        """Legacy method for testing the agent. Not in use."""
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            obs = env.reset()
            actions = []
            done = False
            while not done:
                obs, paxreward, done, info, _, _ = env.pax_step(
                    CPLEXPATH=cplexpath,
                    PATH="scenario_sac_brooklyn",
                    directory=directory,
                )
                eps_reward += paxreward

                o = parser.parse_obs(obs)

                action_rl, price = self.select_action(o, deterministic=True)
                actions.append(action_rl)

                desiredAcc = {
                    env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }

                rebAction = solveRebFlow(
                    env, "scenario_sac_brooklyn", desiredAcc, cplexpath, directory
                )

                _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                eps_reward += rebreward
                eps_served_demand += info["served_demand"]
                eps_rebalancing_cost += info["rebalancing_cost"]
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
        )

    def save_checkpoint(self, path="ckpt.pth"):
        """Save model weights."""
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["model"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        """Log the model."""
        torch.save(log_dict, path)
