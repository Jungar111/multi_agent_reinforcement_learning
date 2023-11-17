from collections import defaultdict
from dataclasses import dataclass, field
import typing as T
from multi_agent_reinforcement_learning.data_models.logs import ModelLog


@dataclass
class PaxStepInfo:
    served_demand: int = 0
    operating_cost: int = 0
    revenue: int = 0
    rebalancing_cost: int = 0


@dataclass
class GraphState:
    time: int = 0
    demand: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    acc: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    dacc: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    price: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@dataclass
class Actions:
    reb_action: T.Optional[T.List[float]] = None
    pax_action: T.Optional[T.List[int]] = None


@dataclass
class Rewards:
    pax_reward: float = 0
    reb_reward: float = 0


@dataclass
class Flow:
    reb_flow: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    pax_flow: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    desired_acc: T.Optional[T.Dict[int, int]] = None
    served_demand: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@dataclass
class CplexData:
    acc_actor_tuple: T.Optional[T.List[T.Tuple[int, int]]] = None
    acc_init_tuple: T.Optional[T.List[T.Tuple[int, int]]] = None


@dataclass
class ActorData:
    name: str
    no_cars: int
    unmet_demand: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    graph_state: GraphState = field(default_factory=GraphState)
    actions: Actions = field(default_factory=Actions)
    flow: Flow = field(default_factory=Flow)
    model_log: ModelLog = field(default_factory=ModelLog)
    rewards: Rewards = field(default_factory=Rewards)
    info: PaxStepInfo = field(default_factory=PaxStepInfo)
    cplex_data: CplexData = field(default_factory=CplexData)
