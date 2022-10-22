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


def save_plot(y, path, x=None,xlabel='' ,ylabel='', title='', ymin=None, ymax=None, xmin=None, xmax=None):
    if x is None:
        x = np.arange(0, len(y), 1)
    fig, ax = plt.subplots( )
    ax.plot( x, y )
    plt.title( title )
    if ymin is None:
        ymin = np.min(y)
    if ymax is None:
        ymax = np.max(y)
    if xmax is None:
        xmax = np.max(x)
    if xmin is None:
        xmin = np.min(x)
    ax.set_ylim( ymin=ymin, ymax=ymax )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig( path )
    plt.close( )


def renyi_entropy(px, alpha):
    if alpha == 1:
        return entropy_masked(px)
    px = px.flat
    if sum(px) >= 1.01 or np.any(px[px < 0]) or np.any(px[px > 1]):
        return np.ma.masked
    return 1/(1-alpha)*np.log2(sum(np.power(px, alpha)))


def permutation_entropy(time_series, nr, sliding_window = None):
    time_series = time_series.flat  # avoid bug with multidimensionial arrays
    if sliding_window is None:
        sliding_window = len(time_series)
    permutation_matrix = np.array([np.argsort(time_series[i:i+nr]) for i in range(0, sliding_window)]) #starting permutation matrix
    distribution = {} #dictionaries of patterns
    hash_function = lambda x: sum([(10**i)*x[i] for i in range(0, x.size)])
    for element in permutation_matrix:
        if hash_function(element) in distribution:
            distribution[ hash_function( element ) ] += 1
        else:
            distribution[ hash_function( element ) ] = 1
    print(pmf( np.array( [v for v in distribution.values( )] ) ))
    p_entropy = [entropy( pmf( np.array( [v for v in distribution.values( )] ) ) )]
    for i in range(1, len(time_series)-nr-sliding_window):
        exiting = np.argsort(time_series[i-1:i-1+nr])
        entering = np.argsort(time_series[i+sliding_window:i+sliding_window+nr])
        if hash_function(entering) in distribution:
            distribution[hash_function(entering)] += 1
        else:
            distribution[hash_function(entering)] = 1
        distribution[ hash_function( exiting ) ] -= 1
        p_entropy.append(entropy( pmf( np.array( [v for v in distribution.values( )] ) ) ))
    return p_entropy
