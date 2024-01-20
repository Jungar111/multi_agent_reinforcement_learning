"""This is the environment for the scenario."""

import json
import typing as T
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd

from multi_agent_reinforcement_learning.data_models.actor_data import ActorData
from multi_agent_reinforcement_learning.data_models.config import BaseConfig
from multi_agent_reinforcement_learning.utils.price_utils import value_of_time


class Scenario:
    """Class for defining a scenario."""

    def __init__(
        self,
        config: BaseConfig,
        actor_data: T.List[ActorData],
        sd=None,
        tripAttr=None,
        demand_input=None,
        demand_ratio: T.Optional[T.Union[float, dict]] = None,
        trip_length_preference: float = 0.25,
        grid_travel_time: int = 1,
        fix_price: bool = True,
        alpha: float = 0.2,
        json_file: T.Optional[str] = None,
        json_hr: int = 19,
        json_tstep: int = 3,
        varying_time: bool = False,
        json_regions=None,
    ):
        """Init method for a scenario.

        trip_length_preference: positive - more shorter trips, negative - more longer trips
        grid_travel_time: travel time between grids
        demand_input： list - total demand out of each region,
                 float/int - total demand out of each region satisfies uniform distribution on [0, demand_input]
                 dict/defaultdict - total demand between pairs of regions
        demand_input will be converted to a variable static_demand to represent the demand between each pair of nodes
        static_demand will then be sampled according to a Poisson distribution
        alpha: parameter for uniform distribution of demand levels - [1-alpha, 1+alpha] * demand_input
        """
        self.sd = sd
        if sd != None:
            np.random.seed(self.sd)
        # simulate enviorienment when json is none.
        if json_file == None:
            self.vot = 0.8
            self.varying_time = varying_time
            self.is_json = False
            self.alpha = alpha
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.demand_input = demand_input
            self.fix_price = fix_price
            self.N1 = config.grid_size_x
            self.N2 = config.grid_size_y
            self.G = nx.complete_graph(self.N1 * self.N2)
            self.G = self.G.to_directed()
            self.demand_time = defaultdict(dict)
            self.reb_time = defaultdict(dict)
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
            self.tstep = json_tstep
            self.tf = config.tf
            self.demand_ratio = defaultdict(list)
            for i, j in self.edges:
                for t in range(0, self.tf * 2):
                    self.demand_time[i, j][t] = (
                        abs(i // self.N1 - j // self.N1) + abs(i % self.N1 - j % self.N1)
                    ) * grid_travel_time

                    self.reb_time[i, j][t] = (
                        abs(i // self.N1 - j // self.N1) + abs(i % self.N1 - j % self.N1)
                    ) * grid_travel_time

            for actor in actor_data:
                for n in self.G.nodes:
                    self.G.nodes[n][f"acc_init_{actor.name}"] = int(actor.no_cars // len(self.G.nodes))

                self.tf = config.tf
                self.demand_ratio = defaultdict(list)

            if demand_ratio == None or isinstance(demand_ratio, list) or isinstance(demand_ratio, dict):
                for i, j in self.edges:
                    if isinstance(demand_ratio, list):
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, self.tf),
                                    np.arange(
                                        0,
                                        self.tf + 1,
                                        self.tf / (len(demand_ratio) - 1),
                                    ),
                                    demand_ratio,
                                )
                            )
                            + [demand_ratio[-1]] * self.tf
                        )
                    if isinstance(demand_ratio, dict):
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, self.tf),
                                    np.arange(
                                        0,
                                        self.tf + 1,
                                        self.tf / (len(demand_ratio[i]) - 1),
                                    ),
                                    demand_ratio[i],
                                )
                            )
                            + [demand_ratio[i][-1]] * self.tf
                        )
                    else:
                        self.demand_ratio[i, j] = [1] * (self.tf + self.tf)
            else:
                for i, j in self.edges:
                    if (i, j) in demand_ratio:
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, self.tf),
                                    np.arange(
                                        0,
                                        self.tf + 1,
                                        self.tf / (len(demand_ratio[i, j]) - 1),
                                    ),
                                    demand_ratio[i, j],
                                )
                            )
                            + [1] * self.tf
                        )
                    else:
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, self.tf),
                                    np.arange(
                                        0,
                                        self.tf + 1,
                                        self.tf / (len(demand_ratio["default"]) - 1),
                                    ),
                                    demand_ratio["default"],
                                )
                            )
                            + [1] * self.tf
                        )
            if self.fix_price:  # fix price
                self.p = defaultdict(dict)
                for i, j in self.edges:
                    self.p[i, j] = (np.random.rand() * 2 + 1) * (self.demand_time[i, j][0] + 1)
            if tripAttr != None:  # given demand as a defaultdict(dict)
                self.tripAttr = deepcopy(tripAttr)
            else:
                self.tripAttr = self.get_random_demand()  # randomly generated demand

        else:  # if json is true
            self.varying_time = varying_time
            self.is_json = True
            with open(json_file, "r") as file:
                data = json.load(file)

            self.tstep = json_tstep
            self.N1 = data["nlat"]
            self.N2 = data["nlon"]
            self.demand_input = defaultdict(dict)
            self.json_regions = json_regions

            if json_regions != None:
                self.G = nx.complete_graph(json_regions)
            elif "region" in data:
                self.G = nx.complete_graph(data["region"])
            else:
                self.G = nx.complete_graph(self.N1 * self.N2)
            self.G = self.G.to_directed()
            self.p = defaultdict(dict)
            self.alpha = 0
            self.demand_time = defaultdict(dict)
            self.reb_time = defaultdict(dict)
            self.json_start = json_hr * 60
            self.config = config
            self.tf = config.tf
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]

            for i, j in self.demand_input:
                self.demand_time[i, j] = defaultdict(int)
                self.reb_time[i, j] = 1

            for item in data["demand"]:
                t, o, d, v, tt, p = (
                    item["time_stamp"],
                    item["origin"],
                    item["destination"],
                    item["demand"],
                    item["travel_time"],
                    item["price"],
                )
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue
                if (o, d) not in self.demand_input:
                    self.demand_input[o, d], self.p[o, d], self.demand_time[o, d] = (
                        defaultdict(float),
                        defaultdict(float),
                        defaultdict(float),
                    )

                self.demand_input[o, d][(t - self.json_start) // json_tstep] += v * demand_ratio
                self.p[o, d][(t - self.json_start) // json_tstep] += p * v * demand_ratio
                self.demand_time[o, d][(t - self.json_start) // json_tstep] += tt * v * demand_ratio / json_tstep

            for o, d in self.edges:
                for t in range(0, self.tf * 2):
                    if t in self.demand_input[o, d]:
                        self.p[o, d][t] /= self.demand_input[o, d][t]
                        self.demand_time[o, d][t] /= self.demand_input[o, d][t]
                        self.demand_time[o, d][t] = max(int(round(self.demand_time[o, d][t])), 1)
                    else:
                        self.demand_input[o, d][t] = 0
                        self.p[o, d][t] = 0
                        self.demand_time[o, d][t] = 0

            for item in data["rebTime"]:
                hr, o, d, rt = (
                    item["time_stamp"],
                    item["origin"],
                    item["destination"],
                    item["reb_time"],
                )
                if json_regions != None and (o not in json_regions or d not in json_regions):
                    continue
                if varying_time:
                    t0 = int((hr * 60 - self.json_start) // json_tstep)
                    t1 = int((hr * 60 + 60 - self.json_start) // json_tstep)
                    for t in range(t0, t1):
                        self.reb_time[o, d][t] = max(int(round(rt / json_tstep)), 1)
                else:
                    if hr == json_hr:
                        for t in range(0, self.tf + 1):
                            self.reb_time[o, d][t] = max(int(round(rt / json_tstep)), 1)

            for actor in actor_data:
                for item in data["totalAcc"]:
                    hr = item["hour"]
                    if hr == json_hr + int(round(json_tstep / 2 * self.tf / 60)):
                        for n in self.G.nodes:
                            self.G.nodes[n][f"acc_init_{actor.name}"] = int(actor.no_cars // len(self.G))
                self.tripAttr = self.get_random_demand()

        # Set value of time form the data.
        df = pd.DataFrame(data["demand"])

        self.vot = value_of_time(
            price=df.price,
            duration=df.travel_time,
            demand_ratio=self.config.demand_ratio[self.config.city],
        )

    def get_random_demand(self, reset: bool = False):
        """Generate demand and price.

        reset = True means that the function is called in the reset() method of AMoD enviroment,
        assuming static demand is already generated
        reset = False means that the function is called when initializing the demand
        return: tripAttr
        """
        demand = defaultdict(dict)
        price = defaultdict(dict)
        tripAttr = []

        # converting demand_input to static_demand
        # skip this when resetting the demand
        # if not reset:
        if self.is_json:
            for t in range(0, self.tf * 2):
                for i, j in self.edges:
                    if (i, j) in self.demand_input and t in self.demand_input[i, j]:
                        demand[i, j][t] = np.random.poisson(self.demand_input[i, j][t])
                        price[i, j][t] = self.p[i, j][t]
                    else:
                        demand[i, j][t] = 0
                        price[i, j][t] = 0
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))
        else:  # Generate random demand
            self.static_demand = dict()
            region_rand = np.random.rand(len(self.G)) * self.alpha * 2 + 1 - self.alpha
            if type(self.demand_input) in [float, int, list, np.array]:
                if type(self.demand_input) in [float, int]:
                    self.region_demand = region_rand * self.demand_input
                else:
                    self.region_demand = region_rand * np.array(self.demand_input)
                for i in self.G.nodes:
                    J = [j for _, j in self.G.out_edges(i)]
                    J.append(i)
                    prob = np.array([np.exp(-self.reb_time[i, j][0] * self.trip_length_preference) for j in J])
                    prob = prob / sum(prob)
                    for idx in range(len(J)):
                        self.static_demand[i, J[idx]] = self.region_demand[i] * prob[idx]
            elif type(self.demand_input) in [dict, defaultdict]:
                for i, j in self.edges:
                    self.static_demand[i, j] = (
                        self.demand_input[i, j] if (i, j) in self.demand_input else self.demand_input["default"]
                    )

                    self.static_demand[i, j] *= region_rand[i]
            else:
                raise Exception("demand_input should be number, array-like, or dictionary-like values")

            # generating demand and prices
            if self.fix_price:
                p = self.p
            self.demand_input2 = defaultdict(dict)
            for t in range(0, self.tf * 2):
                for i, j in self.edges:
                    demand[i, j][t] = np.random.poisson(self.static_demand[i, j] * self.demand_ratio[i, j][t])
                    self.demand_input2[i, j][t] = self.static_demand[i, j] * self.demand_ratio[i, j][t]
                    if self.fix_price:
                        price[i, j][t] = p[i, j]
                    else:
                        price[i, j][t] = min(3, np.random.exponential(2) + 1) * self.demand_time[i, j][t]
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return tripAttr
