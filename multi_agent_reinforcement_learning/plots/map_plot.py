"""File for making mapplots."""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image
import os
from glob2 import glob
import wandb


from multi_agent_reinforcement_learning.envs.amod import AMoD


def make_map_plot(G: nx.Graph, obs: dict, t: int, timeEnd: int, env: AMoD):
    """Make a mapplot to visualize the distribution of cars over time.

    G: Graph
    obs: All observations until time t
    rebAction: The rebalancing actions at time t
    t: time t
    """
    t = t + 1
    nrVehicleAtTimeT = {x: int(i[t]) for x, i in obs[0].items()}

    edgeList = []
    edgeLabels = {}
    # inVehicles = obs[2][0][t]
    for i in range(15):
        for j in range(15):
            if (i != j) and (env.rebFlow[i, j][t] > 0):
                edgeList.append((i, j))
                edgeLabels[i, j] = env.rebFlow[i, j][t]
    pos = {
        0: [-1, 1],
        1: [0, 1],
        2: [1, 1],
        3: [-1, -1],
        4: [0, -1],
        5: [1, -1],
        6: [-2, 2],
        7: [0, 2],
        8: [2, 2],
        9: [-2, -2],
        10: [0, -2],
        11: [2, -2],
        12: [-3, 3],
        13: [0, 3],
        14: [3, 3],
        15: [-3, -3],
    }
    # pos = nx.spiral_layout(G)
    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        labels=nrVehicleAtTimeT,
        cmap=cm.get_cmap("viridis"),
        node_color=list(pos.keys()),
        node_size=np.array(list(nrVehicleAtTimeT.values())) * 10,
        edgelist=edgeList,
        width=np.array(list(edgeLabels.values())) / 10,
        with_labels=True,
        font_weight="bold",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels)
    plt.savefig(f"multi_agent_reinforcement_learning/plots/temp_plots/{t}.jpg")
    plt.clf()


def delete_temp_plots():
    """Delete temp plots."""
    files = glob("multi_agent_reinforcement_learning/plots/temp_plots/*")
    for file in files:
        os.remove(file)


def images_to_gif():
    """Make images to gif."""
    image_fnames = glob("multi_agent_reinforcement_learning/plots/temp_plots/*.jpg")
    image_fnames.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))  # sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(
        "multi_agent_reinforcement_learning/plots/temp_plots/map_gif.gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=len(image_fnames),
        loop=0,
    )
    # gif = Image.open("multi_agent_reinforcement_learning/plots/temp_plots/map_gif.gif")
    wandb.log(
        {
            "Map Plot Animation": wandb.Video(
                "multi_agent_reinforcement_learning/plots/temp_plots/map_gif.gif",
                fps=10,
            )
        }
    )
    delete_temp_plots()


if __name__ == "__main__":
    make_map_plot()
