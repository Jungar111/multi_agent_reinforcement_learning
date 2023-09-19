"""Uniform actor."""
import torch


class UniformActor:
    """Actor which operates under the uniform distribution.

    This actor is purely rule based.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda:0"),
    ):
        """Init method for the uniform actor."""
        self.saved_actions = []
        self.device = device

    def select_action(self, n_regions: int):
        """Select a random action based on the uniform distribution."""
        action = torch.Tensor([1 / n_regions for _ in range(n_regions)])

        self.saved_actions.append(action)

        return list(action.cpu().numpy())
