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
        """Select a random action based on the uniform distribution."""
        m = Uniform(0, 1)
        print(n_actions)
        action = m.sample(sample_shape=torch.Size([n_actions]))
        print(action)

        self.saved_actions.append(m.log_prob(action))

        return list(action.cpu().numpy())
