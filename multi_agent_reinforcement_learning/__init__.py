"""Init method for the package."""

import matplotlib.pyplot as plt
from pathlib import Path

style_path = Path(
    "multi_agent_reinforcement_learning", "utils", "master_2023.mplstyle"
).absolute()

plt.style.use(str(style_path))


if __name__ == "__main__":
    """For viewing test plot."""
    import numpy as np

    x = np.linspace(0, 3, 1000)

    noise_mean = 0
    noise_sd = 0.2
    y = x**2 + np.random.normal(noise_mean, noise_sd, 1000)
    y2 = x + np.random.normal(noise_mean, noise_sd, 1000)
    y3 = np.log(x) + np.random.normal(noise_mean, noise_sd, 1000)
    y4 = np.exp(x) + np.random.normal(noise_mean, noise_sd, 1000)
    y5 = 2 * x + np.random.normal(noise_mean, noise_sd, 1000)
    plt.scatter(x, y, label="y=x^2")
    plt.scatter(x, y2, label="y=x")
    plt.scatter(x, y3, label="y=log(x)")
    plt.scatter(x, y4, label="y=exp(x)")
    plt.scatter(x, y5, label="y=2x")
    plt.legend()
    plt.title("Test title")
    plt.show()
