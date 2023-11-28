"""Module for testing and evaluating actor performance."""
import multi_agent_reinforcement_learning  # noqa: F401
import matplotlib.pyplot as plt
import seaborn as sns
import typing as T
import numpy as np
import pandas as pd
from bokeh.io import show
import holoviews as hv

from multi_agent_reinforcement_learning.algos.actor_critic_gnn import ActorCritic
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
    sc = plt.cm.ScalarMappable(norm=norm)
    fig, ax = plt.subplots(1, len(actions))
    no_grids = actions[0, :, :].shape[1]
    for idx, model in enumerate(model_data_pairs):
        if no_grids < 16:
            actor_actons = np.pad(
                actions[idx, :, :],
                pad_width=((0, 0), (0, 16 - no_grids)),
            )
        actor_actions = actor_actons.reshape(T, 4, 4)
        for i in range(4):
            for j in range(4):
                # Computes demand from i to all other grids
                demand_from_grid = np.array(
                    [
                        np.array(
                            list(
                                model.actor_data.graph_state.demand[
                                    i * 4 + j, k
                                ].values()
                            )
                        )
                        for k in range(no_grids)
                    ]
                ).sum()
                unmet_demand_from_grid = np.array(
                    [
                        np.array(
                            list(model.actor_data.unmet_demand[i * 4 + j, k].values())
                        )
                        for k in range(no_grids)
                    ]
                ).sum()
                ax[idx].text(
                    j - 0.3,
                    i,
                    f"{demand_from_grid.round(2)}/{unmet_demand_from_grid.round(2)}",
                    color="White",
                )
        ax[idx].matshow(actor_actions.mean(axis=0), norm=norm)
        ax[idx].set_title(f"Actor: {model.actor_data.name}")
    fig.subplots_adjust(right=0.8)
    fig.supxlabel("Total demand / total unmet demand", fontsize=12)
    fig.colorbar(sc, ax=ax.ravel().tolist(), label="Mean # of cars departing from")

    plt.show()


def _get_price_matrix(price_dicts: T.List, tf=20, n_actors=2) -> np.ndarray:
    n_epochs = len(price_dicts)
    prices = np.zeros((n_actors, n_epochs, tf))
    for idx, price_dict in enumerate(price_dicts):
        for actor_idx in range(n_actors):
            prices[actor_idx, idx, :] = price_dict[actor_idx]

    return prices


def plot_price_diff_over_time(price_dicts: T.List, tf=20, n_actors=2) -> None:
    x = [i for i in range(tf)]
    prices = _get_price_matrix(price_dicts, tf, n_actors)
    n_epochs = len(price_dicts)

    fig, ax = plt.subplots(n_actors, 1)
    for actor_idx in range(n_actors):
        mean_price = prices[1, ...].mean(axis=0)
        ax[actor_idx].plot(mean_price, label="Mean price")
        if n_epochs > 1:
            std_price = prices[1, ...].std(axis=0)
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


def plot_price_distribution(price_dicts: T.List, data: pd.DataFrame, tf=20, n_actors=2):
    mean_prices = data.groupby(["origin", "destination"])["price"].transform("mean")
    data["price"] -= mean_prices
    n_epochs = len(price_dicts)
    prices = _get_price_matrix(price_dicts, tf, n_actors)

    fig, ax = plt.subplots(n_actors, 1)
    for actor_idx in range(n_actors):
        price_actor = prices[actor_idx, ...].reshape(tf * n_epochs)
        sns.kdeplot(x=price_actor, ax=ax[actor_idx], label="Estimated price difference")
        sns.kdeplot(
            x="price", data=data, ax=ax[actor_idx], label="Price difference in data"
        )
        # sns.histplot(x=price_actor, stat="density", alpha=0.4, ax=ax[actor_idx])
        ax[actor_idx].set_title(f"Price distribution - Actor {actor_idx + 1}")
        ax[actor_idx].legend()

    plt.show()


def flatten_data(data: T.List, column_name: str):
    return [
        {
            "epoch": epoch,
            "time_step": time_step,
            "actor": point,
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
    df_price = pd.DataFrame(flatten_data(price_dicts, "price"))
    df_demand = pd.DataFrame(flatten_data(other_attribute, name_for_other))
    df = df_price.merge(df_demand, on=["epoch", "time_step", "actor"]).sort_values(
        "price"
    )

    plt.scatter(df["price"], df[name_for_other])
    plt.xlabel("Price")
    plt.ylabel(name_for_other.replace("_", " ").capitalize())
    plt.title(f"Price vs. {name_for_other.replace('_', ' ').capitalize()}")
    plt.show()


def chord_chart_of_trips_in_data(data: pd.DataFrame):
    hv.extension("bokeh")
    hv.output(size=500)
    df = data.groupby(["origin", "destination"]).price.count().reset_index()
    df.columns = ["source", "target", "value"]
    chord = hv.Chord(df)
    show(hv.render(chord))
