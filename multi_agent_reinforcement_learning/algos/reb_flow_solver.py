"""RebFlow Solver."""
import os
import subprocess
from collections import defaultdict
from multi_agent_reinforcement_learning.envs.amod import AMoD
from multi_agent_reinforcement_learning.utils.minor_utils import mat2str


def solveRebFlow(env: AMoD, res_path: str, CPLEXPATH: str):
    """Solves the Reb Flow."""
    t = env.time

    edgeAttr = [(i, j, env.G.edges[i, j]["time"]) for i, j in env.G.edges]
    modPath = (
        os.getcwd().replace("\\", "/")
        + "/multi_agent_reinforcement_learning/cplex_mod/"
    )
    OPTPath = (
        os.getcwd().replace("\\", "/")
        + "/saved_files/cplex_logs/rebalancing/"
        + res_path
        + "/"
    )

    for data in env.actor_data:
        data.cplex_data.acc_actor_tuple = [
            (n, int(round(data.flow.desired_acc[n]))) for n in data.flow.desired_acc
        ]
        data.cplex_data.acc_init_tuple = [
            (n, int(data.graph_state.acc[n][t + 1])) for n in data.graph_state.acc
        ]

        if not os.path.exists(OPTPath):
            os.makedirs(OPTPath)
        datafile = OPTPath + f"data_{data.name}_{t}.dat"
        resfile = OPTPath + f"res_{data.name}_{t}.dat"
        with open(datafile, "w") as file:
            file.write('path="' + resfile + '";\r\n')
            file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
            file.write(
                "acc_init_tuple=" + mat2str(data.cplex_data.acc_init_tuple) + ";\r\n"
            )
            file.write(
                "acc_actor_tuple=" + mat2str(data.cplex_data.acc_actor_tuple) + ";\r\n"
            )
        modfile = modPath + "minRebDistRebOnly.mod"
        if CPLEXPATH is None:
            CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file = OPTPath + f"out_{data.name}_{t}.dat"
        with open(out_file, "w") as output_f:
            subprocess.check_call(
                [CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env
            )
        output_f.close()

        # 3. collect results from file
        flow = defaultdict(float)
        with open(resfile, "r", encoding="utf8") as file:
            for row in file:
                item = row.strip().strip(";").split("=")
                if item[0] == "flow":
                    values = item[1].strip(")]").strip("[(").split(")(")
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f = v.split(",")
                        flow[int(i), int(j)] = float(f)
        data.actions.reb_action = [flow[i, j] for i, j in env.edges]
