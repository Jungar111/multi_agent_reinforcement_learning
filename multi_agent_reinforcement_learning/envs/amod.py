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
from multi_agent_reinforcement_learning.data_models.config import BaseConfig
from multi_agent_reinforcement_learning.data_models.model_data_pair import ModelDataPair
from multi_agent_reinforcement_learning.envs.scenario import Scenario
from multi_agent_reinforcement_learning.utils.minor_utils import mat2str
from multi_agent_reinforcement_learning.utils.price_utils import hill_equation


class AMoD:
    """Class for the Autonomous Mobility On Demand."""

    # initialization
    def __init__(
        self,
        actor_data: T.List[ActorData],
        scenario: Scenario,
        config: BaseConfig,
        beta: float = 0.2,
    ):
        """Initialise env.

        scenario: The current scenario
        beta: cost of rebalancing
        """
        # updated to take scenario and beta (cost for rebalancing) as input
        self.config = config
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
        self.nedge = [len(self.G.out_edges(n)) + 1 for n in self.region]  # number of edges leaving each region
        for i, j in self.G.edges:
            self.G.edges[i, j]["time"] = self.reb_time[i, j][
                self.time
            ]  # Append the time it takes to rebalance to all edges
        for actor in actor_data:
            for n in self.region:
                actor.graph_state.acc[n][0] = self.G.nodes[n][f"acc_init_{actor.name}"]
                actor.graph_state.dacc[n] = defaultdict(float)
        self.beta = beta * scenario.tstep  # Rebalancing cost
        t = self.time

        self.N = len(self.region)  # total number of cells

    def matching(
        self,
        actor_data: ActorData,
        CPLEXPATH: T.Optional[str] = None,
        PATH: str = "",
        platform: str = "linux",
    ) -> T.List[int]:
        """Match in the class Matches passengers with vehicles.

        return: paxAction
        """
        t = self.time
        price, demand, acc, name = (
            actor_data.graph_state.price,
            actor_data.graph_state.demand,
            actor_data.graph_state.acc,
            actor_data.name,
        )

        if not self.config.include_price:
            price = self.price

        demandAttr = [
            (i, j, demand[i, j][t], price[i, j][t]) for i, j in demand if t in demand[i, j] and demand[i, j][t] > 1e-3
        ]  # Setup demand and price at time t.
        accTuple = [(n, acc[n][t + 1]) for n in acc]
        modPath = os.getcwd().replace("\\", "/") + "/multi_agent_reinforcement_learning/cplex_mod/"
        matchingPath = os.getcwd().replace("\\", "/") + "/saved_files/cplex_logs/matching/" + PATH + "/"
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
            subprocess.check_call([CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env)
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

    def distribute_based_on_price(
        self,
        model_data_pairs: T.List[ModelDataPair],
        no_customers: int,
        origin: int,
        dest: int,
        t: int,
        cars_in_area_for_each_company: T.List[int],
    ):
        """Distribute cars based on price."""
        if self.config.include_price:
            vot = []
            for model_data_pair in model_data_pairs:
                price = model_data_pair.actor_data.graph_state.price[origin, dest][t]
                travel_time = model_data_pair.actor_data.flow.travel_time[origin, dest][t]
                vot.append(max(price / travel_time, 1))
        else:
            vot = []
            for model_data_pair in model_data_pairs:
                vot.append(
                    max(
                        self.price[origin, dest][t] / model_data_pair.actor_data.flow.travel_time[origin, dest][t],
                        1,
                    )
                )

        rand = np.random.dirichlet(1 / (np.array(vot) + 1e-2), size=no_customers)
        values, counts = np.unique(np.argmax(rand, axis=1), return_counts=True)
        chosen_company = {val: co for val, co in zip(values, counts)}
        for actor_idx in range(self.config.no_actors):
            no_customers_for_company = chosen_company.get(actor_idx, 0)

            if self.config.cancellation:
                probability_of_cancellation = 1 - hill_equation(x=vot[actor_idx], k=self.scenario.vot)
                customers_after_cancellation = np.random.binomial(
                    no_customers_for_company, p=probability_of_cancellation
                )
            else:
                customers_after_cancellation = no_customers_for_company

            no_cars = min(
                cars_in_area_for_each_company[actor_idx],
                customers_after_cancellation,
            )

            model_data_pairs[actor_idx].actor_data.graph_state.demand[origin, dest][t] = no_cars

            excess_cars = chosen_company.get(actor_idx, 0) - cars_in_area_for_each_company[actor_idx]

            overflow_lost_cars = chosen_company.get(actor_idx, 0) - cars_in_area_for_each_company[actor_idx]
            bus_lost_cars = no_customers_for_company - customers_after_cancellation

            model_data_pairs[actor_idx].actor_data.model_log.bus_unmet_demand += bus_lost_cars

            if excess_cars > 0:
                model_data_pairs[actor_idx].actor_data.unmet_demand[origin, dest][t] = (
                    overflow_lost_cars + bus_lost_cars
                )
                model_data_pairs[actor_idx].actor_data.model_log.overflow_unmet_demand += overflow_lost_cars
            else:
                model_data_pairs[actor_idx].actor_data.unmet_demand[origin, dest][t] = bus_lost_cars

            model_data_pairs[actor_idx].actor_data.model_log.total_unmet_demand += model_data_pairs[
                actor_idx
            ].actor_data.unmet_demand[origin, dest][t]

    # pax step
    def pax_step(
        self,
        model_data_pairs: T.List[ModelDataPair],
        pax_action: T.Optional[list] = None,
        cplex_path: T.Optional[str] = None,
        path: str = "",
        platform: str = "linux",
    ) -> bool:
        """Take one pax step.

        paxAction: Passenger actions for timestep
        returns: self.obs, max(0, self.reward), done, self.info
        """
        t = self.time

        # total_market_share = (
        #     np.sum(
        #         [
        #             model_data_pair.actor_data.actions.pax_action
        #             for model_data_pair in model_data_pairs
        #         ]
        #     )
        #     if t > 0
        #     else None
        # )

        for model_data_pair in model_data_pairs:
            model_data_pair.actor_data.info = PaxStepInfo()
            model_data_pair.actor_data.rewards.pax_reward = 0
            # prev_served_demand = np.sum(model_data_pair.actor_data.actions.pax_action)
            # if t > 0:
            #     model_data_pair.actor_data.flow.market_share = (
            #         prev_served_demand / total_market_share
            #     )

        # Distributing customers stochastic given presence in area.
        for (origin, dest), area_demand in self.demand.items():
            no_customers = area_demand[t]
            cars_in_area_for_each_company = [
                int(model_data_pair.actor_data.graph_state.acc[origin][t]) for model_data_pair in model_data_pairs
            ]

            self.distribute_based_on_price(
                model_data_pairs=model_data_pairs,
                no_customers=no_customers,
                cars_in_area_for_each_company=cars_in_area_for_each_company,
                origin=origin,
                dest=dest,
                t=t,
            )
        self.ext_reward = np.zeros(self.nregion)
        for i in self.region:
            for model_data_pair in model_data_pairs:
                model_data_pair.actor_data.graph_state.acc[i][t + 1] = model_data_pair.actor_data.graph_state.acc[i][t]

        if pax_action is None:
            # default matching algorithm used if isMatching is True, matching method will need the
            # information of self.acc[t+1], therefore this part cannot be put forward
            for model_data_pair in model_data_pairs:
                model_data_pair.actor_data.actions.pax_action = self.matching(
                    CPLEXPATH=cplex_path,
                    PATH=path,
                    platform=platform,
                    actor_data=model_data_pair.actor_data,
                )

        for model_data_pair in model_data_pairs:
            # serving passengers, if vehicle is in same section
            for k in range(len(self.edges)):
                i, j = self.edges[k]
                if (
                    (i, j) not in model_data_pair.actor_data.graph_state.demand
                    or t not in model_data_pair.actor_data.graph_state.demand[i, j]
                    or model_data_pair.actor_data.actions.pax_action[k] < 1e-3
                ):
                    continue
                # I moved the min operator above, since we want paxFlow to be consistent with paxAction
                model_data_pair.actor_data.actions.pax_action[k] = min(
                    model_data_pair.actor_data.graph_state.acc[i][t + 1],
                    model_data_pair.actor_data.actions.pax_action[k],
                )
                assert (
                    model_data_pair.actor_data.actions.pax_action[k]
                    < model_data_pair.actor_data.graph_state.acc[i][t + 1] + 1e-3
                )
                # define servedDemand as the current passenger action
                model_data_pair.actor_data.flow.served_demand[i, j][t] = model_data_pair.actor_data.actions.pax_action[
                    k
                ]
                model_data_pair.actor_data.flow.pax_flow[i, j][
                    t + self.demand_time[i, j][t]
                ] = model_data_pair.actor_data.actions.pax_action[k]
                model_data_pair.actor_data.info.operating_cost += (
                    self.demand_time[i, j][t] * self.beta * model_data_pair.actor_data.actions.pax_action[k]
                )
                # define the cost of picking up the current passenger
                model_data_pair.actor_data.graph_state.acc[i][t + 1] -= model_data_pair.actor_data.actions.pax_action[k]
                # Add to served_demand
                model_data_pair.actor_data.info.served_demand += model_data_pair.actor_data.flow.served_demand[i, j][t]
                model_data_pair.actor_data.graph_state.dacc[j][
                    t + self.demand_time[i, j][t]
                ] += model_data_pair.actor_data.flow.pax_flow[i, j][t + self.demand_time[i, j][t]]
                # add to reward
                if self.config.include_price:
                    model_data_pair.actor_data.rewards.pax_reward += model_data_pair.actor_data.actions.pax_action[
                        k
                    ] * (model_data_pair.actor_data.graph_state.price[i, j][t] - self.demand_time[i, j][t] * self.beta)
                    model_data_pair.actor_data.info.revenue += model_data_pair.actor_data.actions.pax_action[k] * (
                        model_data_pair.actor_data.graph_state.price[i, j][t]
                    )
                else:
                    model_data_pair.actor_data.rewards.pax_reward += model_data_pair.actor_data.actions.pax_action[
                        k
                    ] * (self.price[i, j][t] - self.demand_time[i, j][t] * self.beta)
                    model_data_pair.actor_data.info.revenue += model_data_pair.actor_data.actions.pax_action[k] * (
                        self.price[i, j][t]
                    )

            # Update time
            model_data_pair.actor_data.graph_state.time = self.time

        # if passenger is executed first
        done = False
        return done

    def reb_step(self, model_data_pairs: T.List[ModelDataPair]) -> bool:
        """Take on reb step, Adjusting costs, reward.

        rebAction: the action of rebalancing
        returns: self.obs, self.reward, done, self.info
        """
        t = self.time
        for model_data_pair in model_data_pairs:
            # reward is calculated from before this to the
            # next rebalancing, we may also have two rewards,
            # one for pax matching and one for rebalancing
            model_data_pair.actor_data.rewards.reb_reward = 0

        # rebalancing loop
        for model_data_pair in model_data_pairs:
            for k in range(len(self.edges)):
                i, j = self.edges[k]
                if (i, j) not in self.G.edges:
                    continue
                # TODO: add check for actions respecting constraints? e.g. sum of all action[k] starting in "i" <=
                # self.acc[i][t+1] (in addition to our agent action method)
                # update the number of vehicles
                model_data_pair.actor_data.actions.reb_action[k] = min(
                    model_data_pair.actor_data.graph_state.acc[i][t + 1],
                    model_data_pair.actor_data.actions.reb_action[k],
                )
                model_data_pair.actor_data.flow.reb_flow[i, j][
                    t + self.reb_time[i, j][t]
                ] = model_data_pair.actor_data.actions.reb_action[k]
                model_data_pair.actor_data.graph_state.acc[i][t + 1] -= model_data_pair.actor_data.actions.reb_action[k]
                model_data_pair.actor_data.graph_state.dacc[j][
                    t + self.reb_time[i, j][t]
                ] += model_data_pair.actor_data.flow.reb_flow[i, j][t + self.reb_time[i, j][t]]

                reb_cost = self.reb_time[i, j][t] * self.beta * model_data_pair.actor_data.actions.reb_action[k]

                model_data_pair.actor_data.info.rebalancing_cost += reb_cost
                model_data_pair.actor_data.info.operating_cost += reb_cost
                model_data_pair.actor_data.rewards.reb_reward -= reb_cost
                self.ext_reward[i] -= reb_cost

        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed
        # between matching and rebalancing
        for model_data_pair in model_data_pairs:
            for k in range(len(self.edges)):
                # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me
                # know if you have different opinion
                i, j = self.edges[k]
                if (i, j) in model_data_pair.actor_data.flow.reb_flow and t in model_data_pair.actor_data.flow.reb_flow[
                    i, j
                ]:
                    model_data_pair.actor_data.graph_state.acc[j][t + 1] += model_data_pair.actor_data.flow.reb_flow[
                        i, j
                    ][t]
                if (i, j) in model_data_pair.actor_data.flow.pax_flow and t in model_data_pair.actor_data.flow.pax_flow[
                    i, j
                ]:
                    model_data_pair.actor_data.graph_state.acc[j][t + 1] += model_data_pair.actor_data.flow.pax_flow[
                        i, j
                    ][t]

        self.time += 1

        for model_data_pair in model_data_pairs:
            model_data_pair.actor_data.obs = (
                model_data_pair.actor_data.graph_state.acc,
                self.time,
                model_data_pair.actor_data.graph_state.dacc,
                model_data_pair.actor_data.graph_state.demand,
            )

        for i, j in self.G.edges:
            self.G.edges[i, j]["time"] = self.reb_time[i, j][self.time]
        done = self.tf == t + 1  # if the episode is completed
        # ext_done = [done] * self.nregion
        return done

    def reset(self, model_data_pairs: T.List[ModelDataPair]):
        """Reset the episode."""
        for model_data_pair in model_data_pairs:
            model_data_pair.actor_data.info = PaxStepInfo()
            model_data_pair.actor_data.rewards.reb_reward = 0
            model_data_pair.actor_data.rewards.pax_reward = 0
            model_data_pair.actor_data.graph_state.acc = defaultdict(dict)
            model_data_pair.actor_data.graph_state.dacc = defaultdict(dict)
            model_data_pair.actor_data.graph_state.demand = defaultdict(dict)
            model_data_pair.actor_data.flow.reb_flow = defaultdict(dict)
            model_data_pair.actor_data.flow.pax_flow = defaultdict(dict)
            for i, j in self.demand:
                model_data_pair.actor_data.flow.served_demand[i, j] = defaultdict(float)

            for i, j in self.G.edges:
                model_data_pair.actor_data.flow.reb_flow[i, j] = defaultdict(float)
                model_data_pair.actor_data.flow.pax_flow[i, j] = defaultdict(float)

            for n in self.G:
                model_data_pair.actor_data.graph_state.acc[n][0] = self.G.nodes[n][
                    f"acc_init_{model_data_pair.actor_data.name}"
                ]
                model_data_pair.actor_data.graph_state.dacc[n] = defaultdict(float)

            tripAttr = self.scenario.get_random_demand(reset=True)
            # trip attribute (origin, destination, time of request, demand, price)
            self.price = defaultdict(dict)  # price
            for i, j, t, d, p in tripAttr:
                # trip attribute (origin, destination, time of request, demand, price)
                self.demand[i, j][t] = d
                self.price[i, j][t] = p
                self.depDemand[i][t] += d
                self.arrDemand[i][t + self.demand_time[i, j][t]] += d

            for i, j, t, d, p in tripAttr:
                self.demand[i, j][t] = d
                model_data_pair.actor_data.graph_state.price[i, j][0] = p
                model_data_pair.actor_data.flow.value_of_time[i, j][0] = p
                model_data_pair.actor_data.flow.travel_time[i, j][0] = t

            model_data_pair.actor_data.graph_state.demand = defaultdict(dict)

        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))

        self.time = 0
