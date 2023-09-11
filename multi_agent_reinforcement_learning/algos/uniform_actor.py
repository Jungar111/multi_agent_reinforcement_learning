"""Uniform actor."""

import torch
from torch.distributions import Uniform


class UniformActor:
    """Actor which operates under the uniform distribution.

    This actor is purely rule based.
    """

    def __init__(
        self,
        n_cars: int,
        device: torch.device = torch.device("cuda:0"),
    ):
        """Init method for the uniform actor."""
        self.n_cars = n_cars
        self.saved_actions = []
        self.device = device

    def select_action(self, n_actions: int):
        """Select a random action based on the uniform distribution.

        @TODO 1/regions for the dirichlet, for actions.
        """
        m = Uniform(0, 0.05)
        action = m.sample(sample_shape=torch.Size([n_actions]))

        self.saved_actions.append(m.log_prob(action))

        return list(action.cpu().numpy())
