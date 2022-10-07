import random
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import fsolve
import seaborn as sns


def f(n):
    if n <= 0 or n >= 1:
        return 0
    return n * math.log2(1/n)


def fc(a, b):
    if a + b > 1:
        return np.ma.masked
    if a + b == 1:
        return 0
    return f(a) + f(b) + f(1 - a - b)


def h(p):
    if p == 0 or p == 1:
        return 0
    return p * np.log2( 1 / p ) + (1 - p) * np.log2( 1 / (1 - p) )


def hc(p_fixed, p_array):
    return np.array([fc(p, p_fixed) for p in p_array])


def entropy_plotter_2d():
    prob = np.linspace(0, 1, 100)
    H = np.array([h(p) for p in prob])
    fig, ax = plt.subplots( )
    ax.plot( prob, H )
    plt.show( )


def max_entropy_vector_plot():
    H = entropy_plot()
    max_vector = np.unravel_index( np.nanargmax( H, axis=None ), H.shape )  # returns a tuple
    plt.scatter(max_vector[0], max_vector[1], color="white", edgecolors="black")
    plt.title(f"Hmax = {round(H[ max_vector[ 0 ] ][ max_vector[ 1 ] ], 2)} p1 = {max_vector[ 0 ]}% p2 = {max_vector[ 1 ]}%")
    plt.show( )


def entropy_plot():
    p1 = np.linspace( 0, 1, 100 )
    H = np.array( [ hc( p, np.arange( 0, 1, 0.01 ) ) for p in p1 ] )
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.xticks(np.arange(0, 101, 10), np.round(np.linspace(0, 1, 11), 2))  # Set text labels and properties.
    plt.yticks( np.arange( 0, 101, 10 ), np.round( np.linspace( 0, 1, 11 ), 2 ) )  # Set text labels and properties.
    plt.imshow( H, cmap='rainbow', interpolation='bilinear' )
    # sns.heatmap(H)
    # sns.heatmap(H, xticklabels=np.arange(0, 1, 0.01).round(2), yticklabels=np.arange(0, 1, 0.01).round(2))
    return H


def average_vector_entropy_plot():
    entropy_plot()
    p1 = round( random.random( ), 2 )
    p2 = round( random.uniform( 0, 1 - p1 ), 2 )
    plt.scatter( p1 * 100, p2 * 100, color="hotpink", edgecolors="black" )
    p = (p1 + p2) / 2
    plt.scatter( p * 100, p * 100, color="red", edgecolors="black" )
    plt.title(
        f"p1 = {round(p1, 2)}, p2 = {round(p2, 2)}, p3 = {round(1 - p1 - p2, 2)}, H = {round( fc( p1, p2 ), 2 )}\n"
        f"p1 = {round(p, 2)}, p2 = {round(p, 2)}, p3 = {round(1 - p1 - p2, 2)}, H = {round( fc( p, p ), 2 )}" )
    plt.show( )


def meal_pmf(costs, average):
    print(costs)
    print(np.average(costs))
    fun = lambda x : sum((costs - average)*(x**costs))/(x**costs[0]) #division by costs[0] to discard trivial solutions
    beta = fsolve(fun, 2/costs.shape[0])
    alpha = 1/(sum(beta**costs))
    p = np.asarray([alpha*(beta**c) for c in costs])
    fig, ax = plt.subplots()
    ax.set_xlabel( 'cost' )
    ax.set_ylabel( 'probability' )
    ax.set_title( 'Probability Mass Function' )
    plt.ylim([0, 1])
    plt.scatter(costs, p, c="white", edgecolors="black")
    #sns.scatterplot(x=costs, y=p)
    plt.grid()
    plt.show()