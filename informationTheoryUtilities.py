import numpy as np


def entropy(px):
    return sum([-x*np.log2(x) if 0 < x < 1 else 0 for x in px.flat])


def entropy_masked(px):
    return sum([-x * np.log2(x) if 0 < x < 1 else 0 if x == 0 or x == 1 else np.ma.masked for x in px.flat])


def kl_distance(px, py):
    return sum(px*np.log2(px/py))

