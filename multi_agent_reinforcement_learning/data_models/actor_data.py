from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import typing as T
from multi_agent_reinforcement_learning.data_models.logs import ModelLog


@dataclass
class PaxStepInfo:
    served_demand: int = 0
    operating_cost: int = 0
    revenue: int = 0
    rebalancing_cost: int = 0


@dataclass
class ActorData:
    name: str
    no_cars: int
    model_log: ModelLog = ModelLog()
    best_reward: float = -np.inf
    reb_action: T.Optional[T.List[float]] = None
    pax_action: T.Optional[T.List[int]] = None
    ext_reward: T.Optional[np.ndarray] = None
    obs: T.Optional[T.Tuple[defaultdict, int, defaultdict, defaultdict]] = None
    pax_reward: float = 0
    reb_reward: float = 0
    price: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    demand: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    acc: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    dacc: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    reb_flow: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    pax_flow: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    served_demand: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    info: PaxStepInfo = PaxStepInfo()
    desired_acc: T.Optional[T.Dict[int, int]] = None
    acc_actor_tuple: T.Optional[T.List[T.Tuple[int, int]]] = None
    acc_init_tuple: T.Optional[T.List[T.Tuple[int, int]]] = None
    action: T.Optional[T.List[float]] = None
