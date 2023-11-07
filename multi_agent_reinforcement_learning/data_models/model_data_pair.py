from multi_agent_reinforcement_learning.data_models.actor_data import ActorData


class ModelDataPair:
    def __init__(self, model, actor_data: ActorData):
        self.model = model
        self.actor_data = actor_data
