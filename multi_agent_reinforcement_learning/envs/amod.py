"""This is the Autonomous Mobility On Demand (AMoD) environment."""

import os
import subprocess
import typing as T
from collections import defaultdict
from copy import deepcopy

import numpy as np
from multi_agent_reinforcement_learning.data_models.actor_data import (
    ActorData,
    PaxStepInfo,
)
from multi_agent_reinforcement_learning.envs.scenario import Scenario

from multi_agent_reinforcement_learning.utils.minor_utils import mat2str
from multi_agent_reinforcement_learning.data_models.config import Config


class AMoD:
    """Class for the Autonomous Mobility On Demand."""

    # initialization
    def __init__(
        self,
        actor_data: T.List[ActorData],
        scenario: Scenario,
        config: Config,
        beta: float = 0.2,
    ):
        """Initialise env.

        scenario: The current scenario
        beta: cost of rebalancing
        """
        # updated to take scenario and beta (cost for rebalancing) as input
        self.config = config
        self.actor_data = actor_data
        self.scenario = deepcopy(
            scenario
        )  # I changed it to deep copy so that the scenario input is not modified by env
        self.G = (
            scenario.G
        )  # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.demand_time = self.scenario.demand_time
        self.reb_time = self.scenario.reb_time
        self.time = 0  # current time
        self.tf = scenario.tf  # final time
        self.demand = defaultdict(dict)  # demand
        self.depDemand = dict()
        self.arrDemand = dict()
        self.region = list(self.G)  # set of regions
        for i in self.region:
            self.depDemand[i] = defaultdict(float)
            self.arrDemand[i] = defaultdict(float)

        self.price = defaultdict(dict)  # price
        for i, j, t, d, p in scenario.tripAttr:
            # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            self.depDemand[i][t] += d
            self.arrDemand[i][t + self.demand_time[i, j][t]] += d
        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions
        for i in self.G:  # Append all nodes to itself so staying is a possibility
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [
            len(self.G.out_edges(n)) + 1 for n in self.region
        ]  # number of edges leaving each region
        for i, j in self.G.edges:
            self.G.edges[i, j]["time"] = self.reb_time[i, j][
                self.time
            ]  # Append the time it takes to rebalance to all edges
        for actor in actor_data:
            for n in self.region:
                actor.acc[n][0] = self.G.nodes[n][f"acc_init_{actor.name}"]
                actor.dacc[n] = defaultdict(float)
        self.beta = beta * scenario.tstep  # Rebalancing cost
        t = self.time

        self.N = len(self.region)  # total number of cells

    def matching(
        self,
        demand: defaultdict,
        acc: defaultdict,
        name: str,
        CPLEXPATH: T.Optional[str] = None,
        PATH: str = "",
        platform: str = "linux",
    ) -> T.List[int]:
        """Match in the class Matches passengers with vehicles.

        return: paxAction
        """
        t = self.time
        demandAttr = [
            (i, j, demand[i, j][t], self.price[i, j][t])
            for i, j in demand
            if t in demand[i, j] and demand[i, j][t] > 1e-3
        ]  # Setup demand and price at time t.
        accTuple = [(n, acc[n][t + 1]) for n in acc]
        modPath = (
            os.getcwd().replace("\\", "/")
            + "/multi_agent_reinforcement_learning/cplex_mod/"
        )
        matchingPath = (
            os.getcwd().replace("\\", "/")
            + "/saved_files/cplex_logs/matching/"
            + PATH
            + "/"
        )
        if not os.path.exists(matchingPath):
            os.makedirs(matchingPath)
        datafile = matchingPath + f"data_{name}_{t}.dat"
        resfile = matchingPath + f"res_{name}_{t}.dat"
        with open(datafile, "w") as file:
            file.write('path="' + resfile + '";\r\n')
            file.write("demandAttr=" + mat2str(demandAttr) + ";\r\n")
            file.write("accInitTuple=" + mat2str(accTuple) + ";\r\n")
        modfile = modPath + "matching.mod"
        if CPLEXPATH is None:
            CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        if platform == "mac":
            my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
        else:
            my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file = matchingPath + f"out_{name}_{t}.dat"
        with open(out_file, "w") as output_f:
            subprocess.check_call(
                [CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env
            )
        output_f.close()
        flow = defaultdict(float)
        with open(resfile, "r", encoding="utf8") as file:
            for row in file:
                item = row.replace("e)", ")").strip().strip(";").split("=")
                if item[0] == "flow":
                    values = item[1].strip(")]").strip("[(").split(")(")
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f = v.split(",")
                        flow[int(i), int(j)] = float(f)
        paxAction = [flow[i, j] if (i, j) in flow else 0 for i, j in self.edges]
        return paxAction

    def distribute_hypergeometric(
        self,
        distribution_in_area_for_actor: T.List[T.Union[int, float]],
        no_customers: int,
        origin: int,
        dest: int,
        t: int,
    ):
        # PCG64 produces a random integer stream that the generator needs
        # will always procuce same stream given seed
        demand_distribution_to_actors = np.random.Generator(
            np.random.PCG64(self.config.seed)
        ).multivariate_hypergeometric(
            np.array(distribution_in_area_for_actor), no_customers
        )
        for idx, demand in enumerate(demand_distribution_to_actors):
            self.actor_data[idx].demand[origin, dest][t] = demand

    def distribute_based_on_price(
        self,
        price: T.List[float],
        no_customers: int,
        origin: int,
        dest: int,
        t: int,
        cars_in_area_for_each_company: T.List[int],
    ):
        rand = np.random.dirichlet(np.array(price) + 1e-2, size=no_customers)
        values, counts = np.unique(np.argmax(rand, axis=1), return_counts=True)
        chosen_company = {val: co for val, co in zip(values, counts)}
        actor_full = {}
        for actor_idx in range(2):
            no_cars = min(
                cars_in_area_for_each_company[actor_idx],
                chosen_company.get(actor_idx, 0),
            )
            self.actor_data[actor_idx].demand[origin, dest][t] = no_cars
            actor_full[actor_idx] = {
                "full": no_cars == cars_in_area_for_each_company[actor_idx],
                "excess": chosen_company.get(actor_idx, None)
                - cars_in_area_for_each_company[actor_idx],
            }

        for actor_idx, data in actor_full.items():
            if data["full"]:
                if actor_idx == 0:
                    self.actor_data[1].demand[origin, dest][t] += data["excess"]
                if actor_idx == 1:
                    self.actor_data[0].demand[origin, dest][t] += data["excess"]

    # pax step
    def pax_step(
        self,
        pax_action: T.Optional[list] = None,
        cplex_path: T.Optional[str] = None,
        path: str = "",
        platform: str = "linux",
    ) -> T.Tuple[T.List[ActorData], bool, T.List[bool]]:
        """Take one pax step.

        paxAction: Passenger actions for timestep
        returns: self.obs, max(0, self.reward), done, self.info
        """
        t = self.time

        for data in self.actor_data:
            data.info = PaxStepInfo()
            data.pax_reward = 0

        # Distributing customers stochastic given presence in area.
        for (origin, dest), area_demand in self.demand.items():
            no_customers = area_demand[t]
            cars_in_area_for_each_company = [
                int(data.acc[origin][t]) for data in self.actor_data
            ]

            if sum(cars_in_area_for_each_company) < no_customers:
                for idx, data in enumerate(self.actor_data):
                    data.demand[origin, dest][t] = cars_in_area_for_each_company[idx]
            else:
                prices = [actor.price[origin, dest][t] for actor in self.actor_data]
                self.distribute_based_on_price(
                    price=prices,
                    no_customers=no_customers,
                    cars_in_area_for_each_company=cars_in_area_for_each_company,
                    origin=origin,
                    dest=dest,
                    t=t,
                )

        self.ext_reward = np.zeros(self.nregion)
        for i in self.region:
            for actor in self.actor_data:
                actor.acc[i][t + 1] = actor.acc[i][t]

        if pax_action is None:
            # default matching algorithm used if isMatching is True, matching method will need the
            # information of self.acc[t+1], therefore this part cannot be put forward
            for actor in self.actor_data:
                actor.pax_action = self.matching(
                    CPLEXPATH=cplex_path,
                    PATH=path,
                    platform=platform,
                    demand=actor.demand,
                    acc=actor.acc,
                    name=actor.name,
                )

        for actor in self.actor_data:
            # serving passengers, if vehicle is in same section
            for k in range(len(self.edges)):
                i, j = self.edges[k]
                if (
                    (i, j) not in actor.demand
                    or t not in actor.demand[i, j]
                    or actor.pax_action[k] < 1e-3
                ):
                    continue
                # I moved the min operator above, since we want paxFlow to be consistent with paxAction
                actor.pax_action[k] = min(actor.acc[i][t + 1], actor.pax_action[k])
                assert actor.pax_action[k] < actor.acc[i][t + 1] + 1e-3
                # define servedDemand as the current passenger action
                actor.served_demand[i, j][t] = actor.pax_action[k]
                actor.pax_flow[i, j][t + self.demand_time[i, j][t]] = actor.pax_action[
                    k
                ]
                actor.info.operating_cost += (
                    self.demand_time[i, j][t] * self.beta * actor.pax_action[k]
                )  # define the cost of picking of the current passenger
                actor.acc[i][t + 1] -= actor.pax_action[k]
                # Add to served_demand
                actor.info.served_demand += actor.served_demand[i, j][t]
                actor.dacc[j][t + self.demand_time[i, j][t]] += actor.pax_flow[i, j][
                    t + self.demand_time[i, j][t]
                ]
                # add to reward
                actor.pax_reward += actor.pax_action[k] * (
                    actor.price[i, j][t] - self.demand_time[i, j][t] * self.beta
                )
                # Add passenger action * price to revenue
                # actor.ext_reward[i] += max(
                #     0,
                #     actor.pax_action[k]
                #     * (self.price[i, j][t] - self.demand_time[i, j][t] * self.beta),
                # )
                actor.info.revenue += actor.pax_action[k] * (actor.price[i, j][t])

            # for acc, the time index would be t+1, but for demand, the time index would be t
            actor.obs = (
                actor.acc,
                self.time,
                actor.dacc,
                actor.demand,
            )

            actor.pax_reward = max(0, actor.pax_reward)

        # if passenger is executed first
        done = False
        ext_done = [done] * self.nregion
        return self.actor_data, done, ext_done

    def reb_step(self) -> T.Tuple[T.List[ActorData], bool]:
        """Take on reb step, Adjusting costs, reward.

        rebAction: the action of rebalancing
        returns: self.obs, self.reward, done, self.info
        """
        t = self.time
        for actor in self.actor_data:
            # reward is calculated from before this to the
            # next rebalancing, we may also have two rewards,
            # one for pax matching and one for rebalancing
            actor.reb_reward = 0
            actor.ext_reward = np.zeros(self.nregion)

        # rebalancing loop
        for actor in self.actor_data:
            for k in range(len(self.edges)):
                i, j = self.edges[k]
                if (i, j) not in self.G.edges:
                    continue
                # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <=
                # self.acc[i][t+1] (in addition to our agent action method)
                # update the number of vehicles
                actor.reb_action[k] = min(actor.acc[i][t + 1], actor.reb_action[k])
                actor.reb_flow[i, j][t + self.reb_time[i, j][t]] = actor.reb_action[k]
                actor.acc[i][t + 1] -= actor.reb_action[k]
                actor.dacc[j][t + self.reb_time[i, j][t]] += actor.reb_flow[i, j][
                    t + self.reb_time[i, j][t]
                ]

                reb_cost = self.reb_time[i, j][t] * self.beta * actor.reb_action[k]

                actor.info.rebalancing_cost += reb_cost
                actor.info.operating_cost += reb_cost
                actor.reb_reward -= reb_cost
                self.ext_reward[i] -= reb_cost

        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed
        # between matching and rebalancing
        for actor in self.actor_data:
            for k in range(len(self.edges)):
                # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me
                # know if you have different opinion
                i, j = self.edges[k]
                if (i, j) in actor.reb_flow and t in actor.reb_flow[i, j]:
                    actor.acc[j][t + 1] += actor.reb_flow[i, j][t]
                if (i, j) in actor.pax_flow and t in actor.pax_flow[i, j]:
                    actor.acc[j][t + 1] += actor.pax_flow[i, j][t]

        self.time += 1

        for actor in self.actor_data:
            actor.obs = (
                actor.acc,
                self.time,
                actor.dacc,
                actor.demand,
            )

        for i, j in self.G.edges:
            self.G.edges[i, j]["time"] = self.reb_time[i, j][self.time]
        done = self.tf == t + 1  # if the episode is completed
        # ext_done = [done] * self.nregion
        return self.actor_data, done

    def reset(self) -> T.List[ActorData]:
        """Reset the episode."""
        for actor in self.actor_data:
            actor.info = PaxStepInfo()
            actor.reb_reward = 0
            actor.pax_reward = 0
            actor.acc = defaultdict(dict)
            actor.dacc = defaultdict(dict)
            actor.reb_flow = defaultdict(dict)
            actor.pax_flow = defaultdict(dict)
            actor.demand = defaultdict(dict)  # demand
            for i, j in self.demand:
                actor.served_demand[i, j] = defaultdict(float)

            for i, j in self.G.edges:
                actor.reb_flow[i, j] = defaultdict(float)
                actor.pax_flow[i, j] = defaultdict(float)

            for n in self.G:
                actor.acc[n][0] = self.G.nodes[n][f"acc_init_{actor.name}"]
                actor.dacc[n] = defaultdict(float)

            tripAttr = self.scenario.get_random_demand(reset=True)
            # trip attribute (origin, destination, time of request, demand, price)
            for i, j, t, d, p in tripAttr:
                self.demand[i, j][t] = d
                actor.price[i, j][0] = p

            actor.demand = defaultdict(dict)

        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))

        self.time = 0
