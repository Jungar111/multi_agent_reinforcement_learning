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
from multi_agent_reinforcement_learning.data_models.config import Config


def make_map_plot(
    G: nx.Graph, obs: dict, t: int, timeEnd: int, env: AMoD, config: Config
):
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
    n_nodes = config.grid_size_x * config.grid_size_y
    for i in range(n_nodes):
        for j in range(n_nodes):
            if (i != j) and (env.rebFlow[i, j][t] > 0):
                edgeList.append((i, j))
                edgeLabels[i, j] = env.rebFlow[i, j][t]
    pos = {}
    for i in range(n_nodes):
        pos[i] = [-(i // 4 + 1), i % 4 + 1]
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
