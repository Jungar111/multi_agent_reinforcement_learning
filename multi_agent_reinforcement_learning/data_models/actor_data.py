"""Module for holding all data associated with an actor."""
import typing as T
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field


class ModelLog(BaseModel):
    """Pydantic basemodel to log training/test."""

    model_config = ConfigDict(validate_assignment=True)

    reward: float = Field(default=0.0)
    served_demand: int = Field(default=0)
    rebalancing_cost: float = Field(default=0.0)
    bus_unmet_demand: float = Field(default=0.0)
    overflow_unmet_demand: float = Field(default=0.0)
    total_unmet_demand: float = Field(default=0.0)
    revenue_reward: float = Field(default=0.0)
    rebalancing_reward: float = Field(default=0.0)

    def dict(self, name):
        """Return logs as a dict."""
        return {f"{name}_{key}": val for key, val in dict(self).items()}


@dataclass
class PaxStepInfo:
    """Summary stats about an episode run for an actor."""

    served_demand: int = 0
    operating_cost: int = 0
    revenue: float = 0
    rebalancing_cost: int = 0


@dataclass
class GraphState:
    """Input to the GNNs."""

    price: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
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


@dataclass
class Actions:
    """A list of actions which the actor takes."""

    reb_action: T.List[float] = field(default_factory=list)
    pax_action: T.List[int] = field(default_factory=list)


@dataclass
class Rewards:
    """Rewards for the actor."""

    pax_reward: float = 0
    reb_reward: float = 0


@dataclass
class Flow:
    """Describes the flow, and other stats for the actor."""

    reb_flow: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    pax_flow: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    desired_acc: T.Dict[int, int] = field(default_factory=dict)
    served_demand: defaultdict[T.Tuple[int, int], T.Dict[int, int]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    market_share: float = 0
    value_of_time: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    travel_time: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@dataclass
class CplexData:
    """Data associated with the optimisations."""

    acc_actor_tuple: T.Optional[T.List[T.Tuple[int, int]]] = None
    acc_init_tuple: T.Optional[T.List[T.Tuple[int, int]]] = None


@dataclass
class ActorData:
    """Main class holding all the data."""

    name: str
    no_cars: int
    graph_state: GraphState = field(default_factory=GraphState)
    unmet_demand: defaultdict[T.Tuple[int, int], T.Dict[int, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    actions: Actions = field(default_factory=Actions)
    flow: Flow = field(default_factory=Flow)
    model_log: ModelLog = field(default_factory=ModelLog)
    rewards: Rewards = field(default_factory=Rewards)
    info: PaxStepInfo = field(default_factory=PaxStepInfo)
    cplex_data: CplexData = field(default_factory=CplexData)
