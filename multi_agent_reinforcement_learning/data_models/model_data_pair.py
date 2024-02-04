"""Module for the main data structure."""

from multi_agent_reinforcement_learning.data_models.actor_data import ActorData


class ModelDataPair:
    """An object holding both the actor_data and the model."""

    def __init__(self, model, actor_data: ActorData):
        """Init for the object."""
        self.model = model
        self.actor_data: ActorData = actor_data
