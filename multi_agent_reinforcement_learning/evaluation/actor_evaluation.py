"""Module for testing and evaluating actor performance."""
import matplotlib.pyplot as plt
import typing as T
import numpy as np


class ActorEvaluator:
    """Evaluate actor actions."""

    def __init__(self) -> None:
        """Initialise the class."""
        pass

    def plot_average_distribution(self, actions: np.ndarray, names: T.List[str]):
        """Plot average distribution for the actors."""
        fig, ax = plt.subplots(1, len(actions))
        for idx, reb_action in enumerate(actions):
            ax[idx].hist(reb_action.mean(axis=0))
            ax[idx].set_title(f"Actor: {names[idx]}")

        plt.show()
