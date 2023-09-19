from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import typing as T


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
    reb_action: T.Optional[T.List[float]] = None
    pax_action: T.Optional[T.List[int]] = None
    ext_reward: T.Optional[np.ndarray] = None
    obs: T.Optional[T.Tuple[defaultdict, int, defaultdict, defaultdict]] = None
    pax_reward: float = 0
    reb_reward: float = 0
    demand: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    acc: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    dacc: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    reb_flow: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    pax_flow: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    served_demand: defaultdict[dict] = field(default_factory=lambda: defaultdict(dict))
    info: PaxStepInfo = PaxStepInfo()
    desired_acc: T.Optional[T.Dict[int, int]] = None
    acc_actor_tuple: T.Optional[T.List[T.Tuple[int, int]]] = None
    acc_init_tuple: T.Optional[T.List[T.Tuple[int, int]]] = None
    action: T.Optional[T.List[float]] = None
