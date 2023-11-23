import numpy as np


def value_of_time(price: np.ndarray, duration: np.ndarray) -> np.float64:
    """Compute value of time (vot)."""
    return np.max(price / duration)


def hill_equation(x: float, k: float, alpha: float = 4) -> float:
    """Compute shifted sigmoid"""
    return x**alpha / (k**alpha + x**alpha)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import multi_agent_reinforcement_learning  # noqa: F401

    vot = 0.6
    x = np.linspace(0, 1.2, 1000)
    plt.plot(x, hill_equation(x, vot, 4))
    plt.title(f"Hill function with k={vot} and alpha={4}")
    plt.xlabel("Value of time")
    plt.ylabel("Probability of choosing bus")
    plt.show()
