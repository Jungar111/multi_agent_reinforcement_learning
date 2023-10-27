"""Module for testing and evaluating actor performance."""
import multi_agent_reinforcement_learning  # noqa: F401
import matplotlib.pyplot as plt
import typing as T
import numpy as np

from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic


class ActorEvaluator:
    """Evaluate actor actions."""

    def __init__(self) -> None:
        """Initialise the class."""
        pass

    def plot_distribution_at_time_step_t(
        self,
        actions: np.ndarray,
        models: T.List[ActorCritic],
    ):
        """Plot average distribution for the actors."""
        no_time_steps = 6
        fig, ax = plt.subplots(no_time_steps, len(actions))
        for idx, model in enumerate(models):
            for t in range(no_time_steps):
                chosen_time_step = t * 11
                actor_actions = actions[idx, chosen_time_step, :].reshape(4, 4)
                for i in range(4):
                    for j in range(4):
                        ax[t, idx].text(
                            i,
                            j,
                            # int(actor_actions[j, i]),
                            model.actor_data.demand[j, i][chosen_time_step + 1],
                            color="red",
                        )
                pos = ax[t, idx].matshow(actor_actions)
                fig.colorbar(pos, ax=ax[t, idx])
            ax[0, idx].set_title(f"Actor: {model.actor_data.name}")

        plt.show()

    def plot_average_distribution(
        self,
        actions: np.ndarray,
        T: int,
        models: T.List[ActorCritic],
    ):
        """Plot average distribution for the actors."""
        fig, ax = plt.subplots(1, len(actions))
        for idx, model in enumerate(models):
            actor_actions = actions[idx, :, :].reshape(T, 4, 4)
            for i in range(4):
                for j in range(4):
                    ax[idx].text(
                        i,
                        j,
                        np.mean(list(model.actor_data.demand[j, i].values())).round(2),
                        color="red",
                    )
            pos = ax[idx].matshow(actor_actions.mean(axis=0))
            fig.colorbar(pos, ax=ax[idx])
            ax[idx].set_title(f"Actor: {model.actor_data.name}")

        plt.show()
