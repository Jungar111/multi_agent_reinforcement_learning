"""Utils for the project."""
import numpy as np


def mat2str(mat):
    """Mat2Str method."""
    return str(mat).replace("'", '"').replace("(", "<").replace(")", ">").replace("[", "{").replace("]", "}")


def dictsum(dic, t):
    """Sum all keys in dict."""
    return sum([dic[key][t] for key in dic if t in dic[key]])


def moving_average(a, n=3):
    """Compute a moving average used for reward trace smoothing."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
