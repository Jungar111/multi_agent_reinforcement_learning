"""A module for different utils regarding price."""
import typing as T

import numpy as np


def map_to_price(x, lower: float, upper: float):
    """Map from tanh to linear price."""
    return (upper - lower) / 2 * x + (lower + upper) / 2


def value_of_time(price: np.ndarray, duration: np.ndarray, demand_ratio: float) -> np.float64:
    """Compute value of time (vot)."""
    return np.max(price / duration)  # * demand_ratio


def hill_equation(x: T.Union[float, np.ndarray], k: float, alpha: float = 4) -> float:
    """Compute shifted sigmoid."""
    return x**alpha / (k**alpha + x**alpha)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    import multi_agent_reinforcement_learning  # noqa: F401

    colors = ["#8C1C13", "#2F4550", "#A3BBAD", "#C49991", "#F7A072"]
    vots: list[float] = [1.0, 2.0, 3.0][::-1]
    for idx, vot in enumerate(vots):
        x = np.linspace(0, 10, 1000)
        plt.plot(x, hill_equation(x, vot, 4), label=f"k={vot}", color=colors[idx])
        plt.vlines(vot, 0, hill_equation(vot, vot, 4), linestyles="dashed", color=colors[idx])
        plt.hlines(hill_equation(vot, vot, 4), vot, 0, linestyles="dashed", color=colors[idx])

    plt.title(r"Hill function $\alpha=4$")
    plt.xlabel("Value of ride (VOR)")
    plt.ylabel("Probability of cancellation")
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Create a custom legend handle
    custom_line = Line2D([0], [0], color="grey", linestyle="dashed", lw=2)
    # Get the existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # Add the custom handle
    handles.append(custom_line)
    labels.append("Function value at k")
    # Create the legend with the custom handle
    plt.legend(handles=handles, labels=labels)
    plt.show()
