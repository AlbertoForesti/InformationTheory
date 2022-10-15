import numpy as np
import matplotlib.pyplot as plt


def entropy(px):
    return sum([-x*np.log2(x) if 0 < x < 1 else 0 for x in px.flat])


def entropy_masked(px):
    return sum([-x * np.log2(x) if 0 < x < 1 else 0 if x == 0 or x == 1 else np.ma.masked for x in px.flat])


def kl_distance(a, b):
    return sum(a*np.log2(a/b))


def mutual_information(pxy):
    return entropy(pxy.sum(axis=0)) + entropy(pxy.sum(axis=1)) - entropy(pxy)


def pmf(data):
    return data/sum(data.flat)


def save_bar(x, px, path, xlabel='x' ,ylabel='probability', title='Probability Mass Function'):
    fig, ax = plt.subplots( )
    ax.set_xlabel( xlabel )
    ax.set_ylabel( ylabel )
    ax.set_title( title )
    plt.bar( x, px, align='center' )
    plt.savefig( path )
    plt.close( )