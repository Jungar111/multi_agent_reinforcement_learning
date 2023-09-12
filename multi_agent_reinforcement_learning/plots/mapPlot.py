"""File for making mapplots."""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def makeMapPlot(G: nx.Graph, obs: dict, rebAction, t: int):
    """Make a mapplot to visualize the distribution of cars over time.

    G: Graph
    obs: All observations until time t
    rebAction: The rebalancing actions at time t
    t: time t
    """
    t = t + 1
    nrVehicleAtTimeT = {x: i[t] for x, i in obs[0].items()}
    rebMatrix = np.array(rebAction).reshape(len(obs[0]), len(obs[0]))
    # edgeList = {idx: val for idx, val in enumerate(test_list, start = 0)}
    # nx.draw_networkx_edges(G, )
    edgeList = []
    edgeLabels = {}
    # inVehicles = obs[2][0][t]
    for i in range(rebMatrix.shape[0]):
        for j in range(rebMatrix.shape[0]):
            if rebMatrix[i, j] > 0:
                edgeList.append((i, j))
                edgeLabels[i, j] = rebMatrix[i, j]
    # pos = {0: [-1, 1], 1: [0, 1], 2: [1, 1], 3: [-1, -1],
    # 4: [0, -1], 5: [1, -1], 6: [-2,2], 7: [0,2], 8: [2,2], 9: [-2,-2],
    # 10: [0,-2], 11: [2,-2], 12: [-3,3], 13: [0,3], 14: [3,3], 15: [-3,-3]}
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        labels=nrVehicleAtTimeT,
        edgelist=edgeList,
        with_labels=True,
        font_weight="bold",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels)
    plt.show()


if __name__ == "__main__":
    makeMapPlot()
