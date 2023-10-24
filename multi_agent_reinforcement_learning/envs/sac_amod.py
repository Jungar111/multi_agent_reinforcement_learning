"""
Autonomous Mobility-on-Demand Environment
-----------------------------------------
This file contains the specifications for the AMoD system simulator for the SAC.
"""
from collections import defaultdict
import numpy as np
import subprocess
import os
from multi_agent_reinforcement_learning.utils.minor_utils import mat2str
from copy import deepcopy


class AMoD:
    # initialization
    def __init__(
        self, scenario, beta=0.2
    ):  # updated to take scenario and beta (cost for rebalancing) as input
        self.scenario = deepcopy(
            scenario
        )  # I changed it to deep copy so that the scenario input is not modified by env
        self.G = (
            scenario.G
        )  # Road Graph: node - region, edge - connection of regions, node attr: 'accInit', edge attr: 'time'
        self.demandTime = self.scenario.demandTime
        self.rebTime = self.scenario.rebTime
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
        for (
            i,
            j,
            t,
            d,
            p,
        ) in (
            scenario.tripAttr
        ):  # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            self.depDemand[i][t] += d
            self.arrDemand[i][t + self.demandTime[i, j][t]] += d
        self.acc = defaultdict(
            dict
        )  # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(
            dict
        )  # number of vehicles arriving at each region, key: i - region, t - time
        self.rebFlow = defaultdict(
            dict
        )  # number of rebalancing vehicles, key: (i,j) - (origin, destination), t - time
        self.paxFlow = defaultdict(
            dict
        )  # number of vehicles with passengers, key: (i,j) - (origin, destination), t - time
        self.edges = []  # set of rebalancing edges
        self.nregion = len(scenario.G)  # number of regions
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.nedge = [
            len(self.G.out_edges(n)) + 1 for n in self.region
        ]  # number of edges leaving each region
        for i, j in self.G.edges:
            self.G.edges[i, j]["time"] = self.rebTime[i, j][self.time]
            self.rebFlow[i, j] = defaultdict(float)
        for i, j in self.demand:
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.region:
            self.acc[n][0] = self.G.nodes[n]["accInit"]
            self.dacc[n] = defaultdict(float)
        self.beta = beta * scenario.tstep
        t = self.time
        self.servedDemand = defaultdict(dict)
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)

        self.N = len(self.region)  # total number of cells

        # add the initialization of info here
        self.info = dict.fromkeys(
            ["revenue", "served_demand", "rebalancing_cost", "operating_cost"], 0
        )
        self.reward = 0
        # observation: current vehicle distribution, time, future arrivals, demand
        self.obs = (self.acc, self.time, self.dacc, self.demand)

    def matching(
        self, CPLEXPATH=None, PATH="", directory="saved_files", platform="linux"
    ):
        t = self.time
        demandAttr = [
            (i, j, self.demand[i, j][t], self.price[i, j][t])
            for i, j in self.demand
            if t in self.demand[i, j] and self.demand[i, j][t] > 1e-3
        ]
        accTuple = [(n, self.acc[n][t + 1]) for n in self.acc]
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
        datafile = matchingPath + "data_{}.dat".format(t)
        resfile = matchingPath + "res_{}.dat".format(t)
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
        out_file = matchingPath + "out_{}.dat".format(t)
        # print(CPLEXPATH)
        # print(modfile)
        # print(datafile)
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

    # pax step
    def pax_step(
        self,
        paxAction=None,
        CPLEXPATH=None,
        directory="saved_files",
        PATH="",
        platform="linux",
    ):
        t = self.time
        self.reward = 0
        self.ext_reward = np.zeros(self.nregion)
        for i in self.region:
            self.acc[i][t + 1] = self.acc[i][t]
        self.info["served_demand"] = 0  # initialize served demand
        self.info["operating_cost"] = 0  # initialize operating cost
        self.info["revenue"] = 0
        self.info["rebalancing_cost"] = 0
        if (
            paxAction is None
        ):  # default matching algorithm used if isMatching is True, matching method
            # will need the information of self.acc[t+1], therefore this part cannot be put forward
            paxAction = self.matching(
                CPLEXPATH=CPLEXPATH, directory=directory, PATH=PATH, platform=platform
            )
        self.paxAction = paxAction
        # serving passengers

        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (
                (i, j) not in self.demand
                or t not in self.demand[i, j]
                or self.paxAction[k] < 1e-3
            ):
                continue
            # I moved the min operator above, since we want paxFlow to be consistent with paxAction
            self.paxAction[k] = min(self.acc[i][t + 1], paxAction[k])
            assert paxAction[k] < self.acc[i][t + 1] + 1e-3
            self.servedDemand[i, j][t] = self.paxAction[k]
            self.paxFlow[i, j][t + self.demandTime[i, j][t]] = self.paxAction[k]
            self.info["operating_cost"] += (
                self.demandTime[i, j][t] * self.beta * self.paxAction[k]
            )
            self.acc[i][t + 1] -= self.paxAction[k]
            self.info["served_demand"] += self.servedDemand[i, j][t]
            self.dacc[j][t + self.demandTime[i, j][t]] += self.paxFlow[i, j][
                t + self.demandTime[i, j][t]
            ]
            self.reward += self.paxAction[k] * (
                self.price[i, j][t] - self.demandTime[i, j][t] * self.beta
            )
            self.ext_reward[i] += max(
                0,
                self.paxAction[k]
                * (self.price[i, j][t] - self.demandTime[i, j][t] * self.beta),
            )
            self.info["revenue"] += self.paxAction[k] * (self.price[i, j][t])

        self.obs = (
            self.acc,
            self.time,
            self.dacc,
            self.demand,
        )  # for acc, the time index would be t+1, but for demand, the time index would be t
        done = False  # if passenger matching is executed first
        ext_done = [done] * self.nregion
        return self.obs, max(0, self.reward), done, self.info, self.ext_reward, ext_done

    # reb step
    def reb_step(self, rebAction):
        t = self.time
        self.reward = (
            0  # reward is calculated from before this to the next rebalancing,
        )
        # we may also have two rewards, one for pax matching and one for rebalancing
        self.ext_reward = np.zeros(self.nregion)
        self.rebAction = rebAction
        # rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) not in self.G.edges:
                continue
            # TODO: add check for actions respecting constraints? e.g. sum of all action[k]
            # starting in "i" <= self.acc[i][t+1] (in addition to our agent action method)
            # update the number of vehicles
            self.rebAction[k] = min(self.acc[i][t + 1], rebAction[k])
            self.rebFlow[i, j][t + self.rebTime[i, j][t]] = self.rebAction[k]
            self.acc[i][t + 1] -= self.rebAction[k]
            self.dacc[j][t + self.rebTime[i, j][t]] += self.rebFlow[i, j][
                t + self.rebTime[i, j][t]
            ]
            self.info["rebalancing_cost"] += (
                self.rebTime[i, j][t] * self.beta * self.rebAction[k]
            )
            self.info["operating_cost"] += (
                self.rebTime[i, j][t] * self.beta * self.rebAction[k]
            )
            self.reward -= self.rebTime[i, j][t] * self.beta * self.rebAction[k]
            self.ext_reward[i] -= self.rebTime[i, j][t] * self.beta * self.rebAction[k]
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the
        # following codes are executed between matching and rebalancing
        for k in range(len(self.edges)):
            i, j = self.edges[k]
            if (i, j) in self.rebFlow and t in self.rebFlow[i, j]:
                self.acc[j][t + 1] += self.rebFlow[i, j][t]
            if (i, j) in self.paxFlow and t in self.paxFlow[i, j]:
                self.acc[j][t + 1] += self.paxFlow[i, j][
                    t
                ]  # this means that after pax arrived, vehicles can only be rebalanced in
                # the next time step, let me know if you have different opinion

        self.time += 1
        self.obs = (
            self.acc,
            self.time,
            self.dacc,
            self.demand,
        )  # use self.time to index the next time step
        for i, j in self.G.edges:
            self.G.edges[i, j]["time"] = self.rebTime[i, j][self.time]
        done = self.tf == t + 1  # if the episode is completed
        ext_done = [done] * self.nregion
        return self.obs, self.reward, done, self.info, self.ext_reward, ext_done

    def reset(self):
        # reset the episode
        self.acc = defaultdict(dict)
        self.dacc = defaultdict(dict)
        self.rebFlow = defaultdict(dict)
        self.paxFlow = defaultdict(dict)
        self.edges = []
        for i in self.G:
            self.edges.append((i, i))
            for e in self.G.out_edges(i):
                self.edges.append(e)
        self.edges = list(set(self.edges))
        self.demand = defaultdict(dict)  # demand
        self.price = defaultdict(dict)  # price
        tripAttr = self.scenario.get_random_demand(reset=True)
        self.regionDemand = defaultdict(dict)
        for (
            i,
            j,
            t,
            d,
            p,
        ) in (
            tripAttr
        ):  # trip attribute (origin, destination, time of request, demand, price)
            self.demand[i, j][t] = d
            self.price[i, j][t] = p
            if t not in self.regionDemand[i]:
                self.regionDemand[i][t] = 0
            else:
                self.regionDemand[i][t] += d

        self.time = 0
        for i, j in self.G.edges:
            self.rebFlow[i, j] = defaultdict(float)
            self.paxFlow[i, j] = defaultdict(float)
        for n in self.G:
            self.acc[n][0] = self.G.nodes[n]["accInit"]
            self.dacc[n] = defaultdict(float)
        t = self.time
        for i, j in self.demand:
            self.servedDemand[i, j] = defaultdict(float)
        # TODO: define states here
        self.obs = (self.acc, self.time, self.dacc, self.demand)
        self.reward = 0
        return self.obs