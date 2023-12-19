"""Module for testing and evaluating actor performance."""
import typing as T

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.io import show

from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
from multi_agent_reinforcement_learning.data_models.config import SACConfig
from multi_agent_reinforcement_learning.data_models.model_data_pair import ModelDataPair


def plot_distribution_at_time_step_t(
    actions: np.ndarray,
    models: T.List[ActorCritic],
):
    """Plot average distribution for the actors."""
    no_time_steps = 6
    fig, ax = plt.subplots(no_time_steps, len(actions))
    for idx, model in enumerate(models):
        for t in range(no_time_steps):
            chosen_time_step = t * 11
            actor_actions = actions[idx, chosen_time_step, :].reshape(4, 4)
            for i in range(4):
                for j in range(4):
                    ax[t, idx].text(
                        i,
                        j,
                        # int(actor_actions[j, i]),
                        model.actor_data.demand[j, i][chosen_time_step + 1],
                        color="red",
                    )
            pos = ax[t, idx].matshow(actor_actions)
            fig.colorbar(pos, ax=ax[t, idx])
        ax[0, idx].set_title(f"Actor: {model.actor_data.name}")

    plt.show()


def plot_average_distribution(
    config: SACConfig,
    actions: np.ndarray,
    T: int,
    model_data_pairs: T.List[ModelDataPair],
):
    """Plot average distribution for the actors."""
    max_values_for_cbar = np.array(
        (
            actions[0, :, :].mean(axis=0).max().astype(int),
            actions[1, :, :].mean(axis=0).max().astype(int),
        )
    )
    norm = plt.cm.colors.Normalize(vmin=0, vmax=np.max(max_values_for_cbar))
    sc = plt.cm.ScalarMappable(norm=norm, cmap="bone")
    fig, ax = plt.subplots(1, len(actions))
    no_regions = actions[0, :, :].shape[1]
    for idx, model in enumerate(model_data_pairs):
        if 0 not in config.n_regions[config.city] % np.array(list(range(2, 10))):
            actor_actions = np.pad(
                actions[idx, :, :],
                pad_width=((0, 0), (0, config.n_regions[config.city] + 1 - no_regions)),
            )
            actor_actions = actor_actions.reshape(
                T, config.grid_size_x[config.city], config.grid_size_y[config.city]
            )
        else:
            actor_actions = actions[idx].reshape(
                T, config.grid_size_x[config.city], config.grid_size_y[config.city]
            )
        for i in range(config.grid_size_x[config.city]):
            for j in range(config.grid_size_y[config.city]):
                # Computes demand from i to all other grids
                demand_from_grid = np.array(
                    [
                        np.array(
                            list(
                                model.actor_data.graph_state.demand[
                                    i * config.grid_size_y[config.city] + j, k
                                ].values()
                            )
                        )
                        for k in range(no_regions)
                    ]
                ).sum()
                unmet_demand_from_grid = np.array(
                    [
                        np.array(
                            list(
                                model.actor_data.unmet_demand[
                                    i * config.grid_size_y[config.city] + j, k
                                ].values()
                            )
                        )
                        for k in range(no_regions)
                    ]
                ).sum()
                ax[idx].text(
                    j - 0.3,
                    i,
                    f"{demand_from_grid.round(2)}/{unmet_demand_from_grid.round(2)}",
                    color="red",
                    fontsize=8,
                )
        ax[idx].matshow(actor_actions.mean(axis=0), norm=norm, cmap="bone")
        ax[idx].set_title(f"Actor: {model.actor_data.name}")
    fig.subplots_adjust(right=0.8)
    fig.supxlabel("Actor specific demand / actor specific unmet demand", fontsize=12)
    fig.colorbar(
        sc, ax=ax.ravel().tolist(), label="Mean # of cars departing from", cmap="bone"
    )

    plt.show()


def _get_price_matrix(price_dicts: T.List, tf=20, n_actors=2) -> np.ndarray:
    """Convert a price_dict to a np array."""
    n_epochs = len(price_dicts)
    prices = np.zeros((n_actors, n_epochs, tf))
    for idx, price_dict in enumerate(price_dicts):
        for actor_idx in range(n_actors):
            prices[actor_idx, idx, :] = price_dict[actor_idx]

    return prices


def plot_price_diff_over_time(price_dicts: T.List, tf=20, n_actors=2) -> None:
    """Price vs time plot."""
    x = [i for i in range(tf)]
    prices = _get_price_matrix(price_dicts, tf, n_actors)
    n_epochs = len(price_dicts)

    fig, ax = plt.subplots(n_actors, 1)
    for actor_idx in range(n_actors):
        mean_price = prices[actor_idx, ...].mean(axis=0)
        ax[actor_idx].plot(mean_price, label="Mean price")
        if n_epochs > 1:
            std_price = prices[actor_idx, ...].std(axis=0)
            upper_bound = mean_price + 1.96 * std_price
            lower_bound = mean_price - 1.96 * std_price
            ax[actor_idx].fill_between(
                x, lower_bound, upper_bound, alpha=0.6, label="95% CI", color="#2F4550"
            )
            ax[actor_idx].plot(
                x, lower_bound, alpha=0.8, linestyle="dashed", color="#2F4550"
            )
            ax[actor_idx].plot(
                x, upper_bound, alpha=0.8, linestyle="dashed", color="#2F4550"
            )

        ax[actor_idx].legend()
        ax[actor_idx].set_title(f"Price difference over time - Actor {actor_idx + 1}")

    plt.xlabel("Time")
    plt.ylabel("Delta price")
    plt.show()


def plot_actions_as_function_of_time(
    actions: np.ndarray, chosen_areas: T.List[int], colors: T.List[str]
):
    """Plot boxplot."""
    fig, ax = plt.subplots(actions.shape[0], 1, sharey=True)
    for actor_idx in range(actions.shape[0]):
        box_plots = []
        actual_areas = [x - 1 for x in chosen_areas]
        for idx, area in enumerate(actual_areas):
            box_plot = ax[actor_idx].boxplot(
                actions[actor_idx, area, ...],
                capprops=dict(color=colors[idx]),
                whiskerprops=dict(color=colors[idx]),
                flierprops=dict(color=colors[idx], markeredgecolor=colors[idx]),
                medianprops=dict(color="magenta"),
                patch_artist=True,
            )
            box_plots.append(box_plot)

        for bplot, color in zip(box_plots, colors):
            for patch in bplot["boxes"]:
                patch.set_facecolor(color)
        ax[actor_idx].legend(
            [bp["boxes"][0] for bp in box_plots],
            [f"Area: {area}" for area in chosen_areas],
            loc="upper right",
        )
        ax[actor_idx].set_title(f"Actor {actor_idx+1}", fontsize=16)
        ax[actor_idx].set_ylabel("Concentration")
    plt.xlabel("Time")
    plt.show()


def plot_price_distribution(
    model_data_pairs: T.List, data: pd.DataFrame, tf=20, n_actors=2, n_epochs=10
):
    """Price distribution plot."""
    fig, ax = plt.subplots(n_actors, 1)
    for actor_idx, model_data_pair in enumerate(model_data_pairs):
        price = model_data_pair.actor_data.graph_state.price
        prices = [
            value
            for inner_dict in price.values()
            for value in inner_dict.values()
            if value != 0
        ]

        price_actor = prices
        sns.kdeplot(x=price_actor, ax=ax[actor_idx], label="Estimated price")
        sns.kdeplot(x="price", data=data, ax=ax[actor_idx], label="Price in data")
        # sns.histplot(x=price_actor, stat="density", alpha=0.4, ax=ax[actor_idx])
        ax[actor_idx].set_title(f"Price distribution - Actor {actor_idx + 1}")
        ax[actor_idx].legend()

    plt.show()


def flatten_data(data: T.List, column_name: str):
    """Flatten price the specific defaultdict structure we use."""
    return [
        {
            "epoch": epoch,
            "time_step": point,
            "actor": time_step,
            column_name: float(value),
        }
        for epoch, time_steps in enumerate(data)
        for time_step, actors in time_steps.items()
        for point, value in enumerate(actors)
    ]


def plot_price_vs_other_attribute(
    price_dicts: T.List,
    other_attribute: T.List,
    name_for_other: str,
):
    """Price vs. other attribute plot."""
    df_price = pd.DataFrame(flatten_data(price_dicts, "price"))
    df_demand = pd.DataFrame(flatten_data(other_attribute, name_for_other))
    df = df_price.merge(df_demand, on=["epoch", "time_step", "actor"]).sort_values(
        "price"
    )
    print(df.head())

    df_actor_1 = df[df["actor"] == 0]
    df_actor_2 = df[df["actor"] == 1]
    plt.scatter(df_actor_1["price"], df_actor_1[name_for_other])
    plt.scatter(df_actor_2["price"], df_actor_2[name_for_other])
    plt.xlabel("Price")
    plt.ylabel(name_for_other.replace("_", " ").capitalize())
    plt.title(f"Price vs. {name_for_other.replace('_', ' ').capitalize()}")
    plt.show()


def chord_chart_of_trips_in_data(data: pd.DataFrame):
    """Chord chart of org, dest pairs."""
    hv.extension("bokeh")
    hv.output(size=500)
    df = data.groupby(["origin", "destination"]).price.count().reset_index()
    df.columns = ["source", "target", "value"]
    chord = hv.Chord(df)
    show(hv.render(chord))
