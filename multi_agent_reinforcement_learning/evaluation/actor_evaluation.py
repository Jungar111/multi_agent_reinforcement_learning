"""Module for testing and evaluating actor performance."""
import multi_agent_reinforcement_learning  # noqa: F401
import matplotlib.pyplot as plt
import typing as T
import numpy as np

from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
from multi_agent_reinforcement_learning.data_models.model_data_pair import ModelDataPair


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
        model_data_pairs: T.List[ModelDataPair],
    ):
        """Plot average distribution for the actors."""
        max_values_for_cbar = np.array(
            (
                actions[0, :, :].mean(axis=0).max().astype(int),
                actions[1, :, :].mean(axis=0).max().astype(int),
            )
        )
        norm = plt.cm.colors.Normalize(vmin=0, vmax=np.max(max_values_for_cbar))
        sc = plt.cm.ScalarMappable(norm=norm)
        fig, ax = plt.subplots(1, len(actions))
        no_grids = actions[0, :, :].shape[1]
        for idx, model in enumerate(model_data_pairs):
            if no_grids < 16:
                actor_actons = np.pad(
                    actions[idx, :, :],
                    pad_width=((0, 0), (0, 16 - no_grids)),
                )
            actor_actions = actor_actons.reshape(T, 4, 4)
            for i in range(4):
                for j in range(4):
                    # Computes demand from i to all other grids
                    demand_from_grid = np.array(
                        [
                            np.array(
                                list(
                                    model.actor_data.graph_state.demand[
                                        i * 4 + j, k
                                    ].values()
                                )
                            )
                            for k in range(no_grids)
                        ]
                    ).sum()
                    unmet_demand_from_grid = np.array(
                        [
                            np.array(
                                list(
                                    model.actor_data.unmet_demand[i * 4 + j, k].values()
                                )
                            )
                            for k in range(no_grids)
                        ]
                    ).sum()
                    ax[idx].text(
                        j,
                        i,
                        f"{demand_from_grid.round(2)}/{unmet_demand_from_grid.round(2)}",
                        color="White",
                    )
            ax[idx].matshow(actor_actions.mean(axis=0), norm=norm)
            ax[idx].set_title(f"Actor: {model.actor_data.name}")
        fig.subplots_adjust(right=0.8)
        fig.colorbar(sc, ax=ax.ravel().tolist())

        plt.show()
