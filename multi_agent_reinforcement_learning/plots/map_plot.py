"""File for making mapplots."""
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from glob2 import glob
from PIL import Image

import multi_agent_reinforcement_learning  # noqa: F401
import wandb
from multi_agent_reinforcement_learning.data_models.actor_data import ActorData
from multi_agent_reinforcement_learning.data_models.config import A2CConfig


def make_map_plot(G: nx.Graph, actor_data: ActorData, t: int, timeEnd: int, config: A2CConfig):
    """Make a mapplot to visualize the distribution of cars over time.

    G: Graph
    obs: All observations until time t
    rebAction: The rebalancing actions at time t
    t: time t
    """
    node_color = list(plt.rcParams["axes.prop_cycle"])[0]["color"]
    edge_color = list(plt.rcParams["axes.prop_cycle"])[1]["color"]

    nrVehicleAtTimeT = {x: int(i[t]) for x, i in actor_data.obs[0].items()}

    edgeList = []
    edgeLabels = {}

    n_nodes = config.grid_size_x * config.grid_size_y
    for i in range(n_nodes):
        for j in range(n_nodes):
            if (i != j) and (actor_data.reb_flow[i, j][t] > 0):
                edgeList.append((i, j))
                edgeLabels[i, j] = actor_data.reb_flow[i, j][t]
    pos = {}
    for i in range(n_nodes):
        pos[i] = [-(i // 4 + 1), i % 4 + 1]

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        labels=nrVehicleAtTimeT,
        node_color=node_color,
        node_size=np.array(list(nrVehicleAtTimeT.values())) * 10,
        edgelist=edgeList,
        edge_color=edge_color,
        width=np.array(list(edgeLabels.values())) / 10,
        with_labels=True,
        font_weight="bold",
        font_color="whitesmoke",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edgeLabels)
    plt.savefig(f"multi_agent_reinforcement_learning/plots/temp_plots/{t}.jpg")
    plt.close()


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
    # Duration is in miliseconds.
    frame_one.save(
        "multi_agent_reinforcement_learning/plots/temp_plots/map_gif.gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=2000,
        loop=0,
    )

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
    pass
