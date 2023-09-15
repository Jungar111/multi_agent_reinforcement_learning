"""This is the environment for the scenario."""

from collections import defaultdict
import numpy as np
import networkx as nx
from copy import deepcopy
import json
import typing as T


class Scenario:
    """Class for defining a scenario."""

    def __init__(
        self,
        N1: int = 2,  # grid size
        N2: int = 3,  # grid size
        tf: int = 60,  # Timeframe
        sd=None,
        ninit: int = 5,  # Initial number of vehicles in each region
        tripAttr=None,
        demand_input=None,
        demand_ratio: T.Optional[T.Union[float, dict]] = None,
        trip_length_preference: float = 0.25,
        grid_travel_time: int = 1,
        fix_price: bool = True,
        alpha: float = 0.2,
        json_file: T.Optional[str] = None,
        json_hr: int = 7,
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
        if json_file == None:  # simulate enviorienment when json is none.
            self.varying_time = varying_time
            self.is_json = False
            self.alpha = alpha
            self.trip_length_preference = trip_length_preference
            self.grid_travel_time = grid_travel_time
            self.demand_input = demand_input
            self.fix_price = fix_price
            self.N1 = N1
            self.N2 = N2
            self.G = nx.complete_graph(N1 * N2)
            self.G = self.G.to_directed()
            self.demandTime = defaultdict(dict)
            self.rebTime = defaultdict(dict)
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]
            self.tstep = json_tstep
            for i, j in self.edges:
                for t in range(0, tf * 2):
                    self.demandTime[i, j][t] = (
                        abs(i // N1 - j // N1) + abs(i % N1 - j % N1)
                    ) * grid_travel_time

                    self.rebTime[i, j][t] = (
                        abs(i // N1 - j // N1) + abs(i % N1 - j % N1)
                    ) * grid_travel_time

            for n in self.G.nodes:
                self.G.nodes[n]["accInit"] = int(ninit)
            self.tf = tf
            self.demand_ratio = defaultdict(list)

            if (
                demand_ratio == None
                or isinstance(demand_ratio, list)
                or isinstance(demand_ratio, dict)
            ):
                for i, j in self.edges:
                    if isinstance(demand_ratio, list):
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, tf),
                                    np.arange(0, tf + 1, tf / (len(demand_ratio) - 1)),
                                    demand_ratio,
                                )
                            )
                            + [demand_ratio[-1]] * tf
                        )
                    if isinstance(demand_ratio, dict):
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, tf),
                                    np.arange(
                                        0, tf + 1, tf / (len(demand_ratio[i]) - 1)
                                    ),
                                    demand_ratio[i],
                                )
                            )
                            + [demand_ratio[i][-1]] * tf
                        )
                    else:
                        self.demand_ratio[i, j] = [1] * (tf + tf)
            else:
                for i, j in self.edges:
                    if (i, j) in demand_ratio:
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, tf),
                                    np.arange(
                                        0, tf + 1, tf / (len(demand_ratio[i, j]) - 1)
                                    ),
                                    demand_ratio[i, j],
                                )
                            )
                            + [1] * tf
                        )
                    else:
                        self.demand_ratio[i, j] = (
                            list(
                                np.interp(
                                    range(0, tf),
                                    np.arange(
                                        0,
                                        tf + 1,
                                        tf / (len(demand_ratio["default"]) - 1),
                                    ),
                                    demand_ratio["default"],
                                )
                            )
                            + [1] * tf
                        )
            if self.fix_price:  # fix price
                self.p = defaultdict(dict)
                for i, j in self.edges:
                    self.p[i, j] = (np.random.rand() * 2 + 1) * (
                        self.demandTime[i, j][0] + 1
                    )
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
            self.demandTime = defaultdict(dict)
            self.rebTime = defaultdict(dict)
            self.json_start = json_hr * 60
            self.tf = tf
            self.edges = list(self.G.edges) + [(i, i) for i in self.G.nodes]

            for i, j in self.demand_input:
                self.demandTime[i, j] = defaultdict(int)
                self.rebTime[i, j] = 1

            for item in data["demand"]:
                t, o, d, v, tt, p = (
                    item["time_stamp"],
                    item["origin"],
                    item["destination"],
                    item["demand"],
                    item["travel_time"],
                    item["price"],
                )
                if json_regions != None and (
                    o not in json_regions or d not in json_regions
                ):
                    continue
                if (o, d) not in self.demand_input:
                    self.demand_input[o, d], self.p[o, d], self.demandTime[o, d] = (
                        defaultdict(float),
                        defaultdict(float),
                        defaultdict(float),
                    )

                self.demand_input[o, d][(t - self.json_start) // json_tstep] += (
                    v * demand_ratio
                )
                self.p[o, d][(t - self.json_start) // json_tstep] += (
                    p * v * demand_ratio
                )
                self.demandTime[o, d][(t - self.json_start) // json_tstep] += (
                    tt * v * demand_ratio / json_tstep
                )

            for o, d in self.edges:
                for t in range(0, tf * 2):
                    if t in self.demand_input[o, d]:
                        self.p[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] /= self.demand_input[o, d][t]
                        self.demandTime[o, d][t] = max(
                            int(round(self.demandTime[o, d][t])), 1
                        )
                    else:
                        self.demand_input[o, d][t] = 0
                        self.p[o, d][t] = 0
                        self.demandTime[o, d][t] = 0

            for item in data["rebTime"]:
                hr, o, d, rt = (
                    item["time_stamp"],
                    item["origin"],
                    item["destination"],
                    item["reb_time"],
                )
                if json_regions != None and (
                    o not in json_regions or d not in json_regions
                ):
                    continue
                if varying_time:
                    t0 = int((hr * 60 - self.json_start) // json_tstep)
                    t1 = int((hr * 60 + 60 - self.json_start) // json_tstep)
                    for t in range(t0, t1):
                        self.rebTime[o, d][t] = max(int(round(rt / json_tstep)), 1)
                else:
                    if hr == json_hr:
                        for t in range(0, tf + 1):
                            self.rebTime[o, d][t] = max(int(round(rt / json_tstep)), 1)

            for item in data["totalAcc"]:
                hr, acc = item["hour"], item["acc"]
                if hr == json_hr + int(round(json_tstep / 2 * tf / 60)):
                    for n in self.G.nodes:
                        self.G.nodes[n]["accInit"] = int(acc / len(self.G))
            self.tripAttr = self.get_random_demand()

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
                    prob = np.array(
                        [
                            np.exp(-self.rebTime[i, j][0] * self.trip_length_preference)
                            for j in J
                        ]
                    )
                    prob = prob / sum(prob)
                    for idx in range(len(J)):
                        self.static_demand[i, J[idx]] = (
                            self.region_demand[i] * prob[idx]
                        )
            elif type(self.demand_input) in [dict, defaultdict]:
                for i, j in self.edges:
                    self.static_demand[i, j] = (
                        self.demand_input[i, j]
                        if (i, j) in self.demand_input
                        else self.demand_input["default"]
                    )

                    self.static_demand[i, j] *= region_rand[i]
            else:
                raise Exception(
                    "demand_input should be number, array-like, or dictionary-like values"
                )

            # generating demand and prices
            if self.fix_price:
                p = self.p
            self.demand_input2 = defaultdict(dict)
            for t in range(0, self.tf * 2):
                for i, j in self.edges:
                    demand[i, j][t] = np.random.poisson(
                        self.static_demand[i, j] * self.demand_ratio[i, j][t]
                    )
                    self.demand_input2[i, j][t] = (
                        self.static_demand[i, j] * self.demand_ratio[i, j][t]
                    )
                    if self.fix_price:
                        price[i, j][t] = p[i, j]
                    else:
                        price[i, j][t] = (
                            min(3, np.random.exponential(2) + 1)
                            * self.demandTime[i, j][t]
                        )
                    tripAttr.append((i, j, t, demand[i, j][t], price[i, j][t]))

        return tripAttr
