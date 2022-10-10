import numpy as np


def entropy(px):
    return sum([-x*np.log2(x) if 0 < x < 1 else 0 for x in px.flat])


def entropy_masked(px):
    return sum([-x * np.log2(x) if 0 < x < 1 else 0 if x == 0 or x == 1 else np.ma.masked for x in px.flat])


def kl_distance(a, b):
    return sum(a*np.log2(a/b))


def mutual_information(pxy):
    return entropy(pxy.sum(axis=0)) + entropy(pxy.sum(axis=1)) - entropy(pxy)

