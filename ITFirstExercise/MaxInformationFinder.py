import matplotlib.pyplot as plt
import numpy as np
import math


def f(n):
    if n <= 0 or n >= 1:
        return 0
    return n * math.log2(1/n)


def fc(a, b):
    if a + b >= 1:
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


def entropy_plotter_3d():
    p1 = np.linspace(0, 1, 100)
    H = np.array([hc(p, np.arange(0, 1, 0.01)) for p in p1])
    print(H)
    plt.imshow( H, cmap='rainbow', interpolation='bilinear' )
    plt.title( "Heat Map" )
    plt.show( )