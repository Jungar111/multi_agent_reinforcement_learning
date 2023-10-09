import matplotlib.pyplot as plt
from multi_agent_reinforcement_learning.data_models.actor_data import ActorData
import typing as T


class ActorEvaluator:
    def __init__(self, actor_data: T.List[ActorData]) -> None:
        self.actor_data = actor_data

    def plot_average_distribution(self):
        fig, ax = plt.subplots(1, len(self.actor_data))
        for idx, actor in enumerate(self.actor_data):
            ax[idx].hist(actor.reb_action)
            ax.set_title(f"Actor: {actor.name}")

        plt.show()
