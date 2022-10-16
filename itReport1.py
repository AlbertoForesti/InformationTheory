import random
import matplotlib.pyplot as plt
import numpy as np
import math
import informationTheoryUtilities as it
from scipy.optimize import fsolve
from scipy.stats import binom
import graphviz


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
    distributions = [np.array([p, 1-p]) for p in prob]
    H = np.array([it.entropy(px) for px in distributions])
    fig, ax = plt.subplots( )
    ax.plot( prob, H )
    plt.show( )
    plt.close( )


def max_entropy_vector_plot():
    path = "C:\\Users\\Gian Luca Foresti\\Desktop\\Materiale Uni\\4 - anno\\IT"
    H = entropy_plot()
    max_vector = np.unravel_index( np.nanargmax( H, axis=None ), H.shape )  # returns a tuple
    plt.scatter(max_vector[0], max_vector[1], color="white", edgecolors="black")
    plt.title(f"Hmax = {round(H[ max_vector[ 0 ] ][ max_vector[ 1 ] ], 2)}, p1 = {round(max_vector[ 0 ]/100., 2)}, p2 = {round(max_vector[ 1 ]/100., 2)}")
    plt.savefig(path+"\\Ex1A.png")
    plt.close( )


def entropy_plot():
    p1 = np.linspace( 0, 1, 100 )
    H = np.array( [ hc( p, np.arange( 0, 1, 0.01 ) ) for p in p1 ] )
    plt.xlabel( "p1" )
    plt.ylabel( "p2" )
    plt.xticks( np.arange( 0, 101, 10 ), np.round( np.linspace( 0, 1, 11 ), 2 ) )  # Set text labels and properties.
    plt.yticks( np.arange( 0, 101, 10 ), np.round( np.linspace( 0, 1, 11 ), 2 ) )  # Set text labels and properties.
    plt.imshow( H, cmap='rainbow', interpolation='bilinear' )
    return H


def entropy_plot_3d():
    p1 = np.arange(0, 1.01, 0.01)
    p2 = np.arange(0, 1.01, 0.01)
    distributions = np.array( [ [round(px, 4), round(py, 4), round(1 - px - py, 4)] for px in p1 for py in p2 if 0 <= px + py <= 1 ] )
    print(distributions)
    H = np.array( [ it.renyi_entropy( px , 200) for px in distributions ] )
    print(H)
    x = distributions[:, 0]
    y = distributions[:, 1]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, H, c = H, cmap = 'turbo')
    plt.show()
    return H

def average_vector_entropy_plot():
    path = "C:\\Users\\Gian Luca Foresti\\Desktop\\Materiale Uni\\4 - anno\\IT"
    entropy_plot()
    p1 = round( random.random( ), 2 )
    p2 = round( random.uniform( 0, 1 - p1 ), 2 )
    p = (p1 + p2) / 2
    plt.scatter( p1 * 100, p2 * 100, color="green", edgecolors="black", label="Original vector" )
    plt.scatter( p * 100, p * 100, color="red", edgecolors="black", label="Averaged vector" )
    plt.title(
        f"p1 = {round(p1, 2)}, p2 = {round(p2, 2)}, p3 = {round(1 - p1 - p2, 2)}, H = {round( fc( p1, p2 ), 2 )}\n"
        f"p = {round(p, 2)}, p = {round(p, 2)}, p3 = {round(1 - p1 - p2, 2)}, H = {round( fc( p, p ), 2 )}" )
    plt.legend()
    plt.savefig(path+"\\Ex1B.png")
    plt.close()


def meal_pmf(costs, average, path):
    fun = lambda x: sum((costs - average)*(x**costs))/(x**costs[0]) #division by costs[0] to discard trivial solutions
    beta = fsolve(fun, 1)
    alpha = 1/(sum(beta**costs))
    p = np.asarray([alpha*(beta**c) for c in costs])
    fig, ax = plt.subplots()
    ax.set_xlabel( 'cost' )
    ax.set_ylabel( 'probability' )
    ax.set_title( 'Probability Mass Function' )
    plt.ylim([0, 1])
    plt.scatter(costs, p, c="white", edgecolors="black")
    plt.grid()
    plt.savefig(path)
    plt.close( )


def joint_distribution_stats(pxy):
    px = pxy.sum(axis=0)
    py = pxy.sum(axis=1)
    hx = it.entropy(px)
    hy = it.entropy(py)
    hxy = it.entropy(pxy)
    hxcondy = hxy - hx
    hycondx = hxy - hy
    mutual_information = it.mutual_information(pxy)
    print(f"px = {px}")
    print(f"py = {py}")
    print(f"Hx = {hx}")
    print(f"Hy = {hy}")
    print(f"Hxy = {hxy}")
    print(f"Mutual information = {mutual_information}")
    print(f"H(x|y) = {hxcondy}")
    print(f"H(y|x) = {hycondx}")


class C4dot5classifier:
    def __init__(self, training_set):
        used_thresholds = [ set( ) for x in range( training_set.shape[ 1 ] - 1 ) ]
        self.root = C4dot5node( training_set, None, used_thresholds )

    def classify(self, vector):
        return self.root.classify(vector)

    def print_tree(self):
        tree = graphviz.Digraph("Decision tree")
        self.root.build_graphviz_tree(tree, 1)
        tree.render(directory='doctest-output', view=True)


class C4dot5node:
    def __init__(self, training_set, parent_node, used_thresholds):
        self.left_node = None
        self.right_node = None
        self.parent_node = parent_node
        self.training_set = training_set
        # Training set is a matrix with rows as vectors, all columns but the last as feature and the last column as class label
        if training_set.size == 0: #if subset is empty it's child node, roll back
            self.is_leaf_node = True
            self.node_label = parent_node.node_label
            return
        self.node_label = np.bincount(training_set[:][-1]).argmax()
        best_fid = 0
        maxigr_feature = 0
        best_pxy = np.zeros( [ 2, 2 ] )
        best_pxy_threshold = best_pxy
        super_best_threshold = 0
        all_feature_used = True
        for fid in range(training_set.shape[1] - 1): #for each feature
            maxigr_threshold = 0
            best_threshold = 0
            # print(training_set[:, fid])
            available_thresholds = set(training_set[:, fid])-used_thresholds[fid]
            #print(fid)
            for t in available_thresholds: #for each threshold that hasn't been considered compute igr
                # compute distribution for binary representation
                all_feature_used = False
                pxy = np.zeros([2, 2]) #row is binary representation of threshold, column is for class
                for vector in training_set[:, [fid, -1]]: #check slicing
                    if vector[0] >= t:
                        pxy[1, vector[1]] += 1 #larger than threshold
                    else:
                        pxy[0, vector[1]] += 1 #smaller than threshold
                pxy /= sum(sum(pxy)) #normalization
                if it.entropy(pxy.sum(axis=1)) == 0: #probably means that of that feature only one instance survived
                    # print(pxy)
                    # print(pxy.sum(axis=1))
                    igr = 0
                else:
                    igr = it.mutual_information( pxy ) / it.entropy( pxy.sum( axis=1 ) )  # information gain ratio
                # print(f"igr = {igr} = {it.mutual_information(pxy)}/{it.entropy(pxy.sum(axis=1))}")
                if igr > maxigr_threshold:
                    maxigr_threshold = igr
                    best_threshold = t
                    best_pxy_threshold = pxy
            if maxigr_threshold > maxigr_feature:
                #print(fid)
                maxigr_feature = maxigr_threshold
                super_best_threshold = best_threshold
                best_fid = fid
                best_pxy = best_pxy_threshold
        """print( f"Probability distribution of best is {best_pxy}" )
        print( f"Best threshold is {best_threshold}")
        print(f"Best information gain ratio is: {maxigr_feature}")"""
        #check stopping conditions
        if all_feature_used:
            #print("used all the features")
            self.is_leaf_node = True
            self.node_label = parent_node.node_label
            return
        used_thresholds[best_fid].add(super_best_threshold)
        #print("Added threshold ", super_best_threshold, " of feature ", best_fid, "to used thresholds")
        if best_pxy.sum(axis=1)[0] == 0 or best_pxy.sum(axis=1)[0] == 1:
            self.is_leaf_node = True
            #print("Returned")
            return
        self.is_leaf_node = False
        # now compute left and right training set
        right_training_set = training_set[training_set[:, best_fid] >= super_best_threshold]
        left_training_set = training_set[training_set[:, best_fid] < super_best_threshold]

        self.right_node = C4dot5node(right_training_set, self, used_thresholds)
        self.left_node = C4dot5node(left_training_set, self, used_thresholds)
        self.node_feature = best_fid
        self.node_threshold = super_best_threshold

    def classify(self, vector):
        if self.is_leaf_node:
            #print(f"{vector} classified {self.node_label}")
            return self.node_label
        else:
            if vector[self.node_feature] >= self.node_threshold:
                #print(f"{vector[ self.node_feature ]} >= {self.node_threshold}")
                return self.right_node.classify(vector)
            else:
                #print( f"{vector[ self.node_feature ]} < {self.node_threshold}" )
                return self.left_node.classify(vector)

    def build_graphviz_tree(self, tree, node_id):
        if self.is_leaf_node:
            tree.node( name=str(node_id), label=str(self.node_label) )
        else:
            tree.node( name=str(node_id), label="" )
            if self.left_node is not None:
                next_id = 2*node_id
                self.left_node.build_graphviz_tree( tree, next_id )
                tree.edge( str(node_id), str(next_id), label=f"X{self.node_feature + 1} < {self.node_threshold}" )
            if self.right_node is not None:
                next_id = 2*node_id + 1
                self.right_node.build_graphviz_tree( tree, next_id )
                tree.edge( str(node_id), str(next_id), label=f"X{self.node_feature + 1} >= {self.node_threshold}" )
        return


def ex1a():
    max_entropy_vector_plot( )


def ex1b():
    average_vector_entropy_plot()


def ex2a():
    path = "C:\\Users\\Gian Luca Foresti\\Desktop\\Materiale Uni\\4 - anno\\IT"
    costs = np.asarray( [ 4, 7, 10, 20, 15 ] )
    meal_pmf( costs, 9, path + "\\Ex2A.png" )


def ex2b():
    costs = np.asarray( [ 4, 7, 10, 20, 15 ] )
    path = "C:\\Users\\Gian Luca Foresti\\Desktop\\Materiale Uni\\4 - anno\\IT"
    meal_pmf( costs, np.average( costs ), path + "\\Ex2B.png" )
    meal_pmf( costs, 13, path + "\\Ex2B13.png" )
    meal_pmf( costs, 5, path + "\\Ex2B5.png" )


def ex3():
    sassuolo_data = np.array([[7, 0, 0], [6, 11, 14]])
    bvb_data = np.array([[17, 2, 2], [4, 9, 4]])
    #Rows: Sassuolo scoring 3 or more goals in a match in 21/22, Columns: Sassuolo W/D/L in 21/22
    distribution = it.pmf(sassuolo_data)
    print(distribution)
    joint_distribution_stats(distribution)


def ex4():
    x = np.arange(0, 11, 1)
    num = np.asarray([1, 3, 2, 8, 22, 45, 44, 42, 24, 8, 3])
    px = it.pmf(num)
    print(f"px = {px}")
    path = "C:\\Users\\Gian Luca Foresti\\Desktop\\Materiale Uni\\4 - anno\\IT"
    it.save_bar(x, px, path + "\\Ex4empiricalpmf.png")
    uniform = 1/11*np.ones(11)
    it.save_bar(x, uniform, path + "\\Ex4uniformpmf.png", title=f'KL-distance = {round(it.kl_distance(uniform, px), 2)}')
    dmin = 100
    pmin = 0
    for p in np.arange(0.01, 1, 0.01):
        if it.kl_distance(np.asarray([binom.pmf(r, 10, p) for r in np.arange(0, 11, 1)]), px) < dmin:
            dmin = it.kl_distance(np.asarray([binom.pmf(r, 10, p) for r in np.arange(0, 11, 1)]), px)
            pmin = p
    print(f"The parameter p that allows minimum kl-distance for a binomial is: {pmin}")
    print(f"The minimum distance corresponding to p = {pmin} is {dmin}")
    print(f"The kl-distance between the uniform distribution and px is {it.kl_distance(uniform, px)}")
    # plt.scatter(x, np.asarray([binom.pmf(r, 10, pmin) for r in np.arange(0, 11, 1)]))
    it.save_bar(x, np.asarray([binom.pmf(r, 10, pmin) for r in np.arange(0, 11, 1)]), path + "\\Ex4binomialpmf.png", title=f"p = {round(pmin,2)}, kl-distance = {round(dmin,2)}")
    # plt.hist(np.asarray([binom.pmf(r, 10, pmin) for r in np.arange(0, 11, 1)]))


def ex5():
    training_set = np.asarray(
        [ [ 30, 0, 10, 0 ], [ 30, 0, 70, 0 ], [ 30, 1, 20, 0 ], [ 30, 1, 80, 1 ], [ 60, 0, 40, 0 ], [ 60, 0, 60, 1 ],
          [ 60, 1, 50, 0 ], [ 60, 1, 60, 1 ] ] )
    classifier = C4dot5classifier( training_set )
    for vector in training_set:
        res = classifier.classify( vector[ 0:3 ] )
        if res != vector[ 3 ]:
            print( "Failure" )
    classifier.print_tree( )


def ex6():
    time_series = np.random.rand(10000)
    pattern = np.array([np.random.normal(i, 100) if i % 2 == 0 else i for i in range(0, 1000)])
    print(pattern)
    time_series[1000:2000] = pattern #pattern example
    sliding_window = 500
    nr = 3
    y = it.permutation_entropy(time_series, nr, sliding_window)
    fig, ax = plt.subplots( )
    ax.plot( np.arange(0, len(y), 1), y)
    plt.title( f"Average entropy: {np.average( y )}" )
    ax.set_ylim( ymin=0, ymax=np.max(y)+1 )
    plt.show( )
    plt.close( )

#ordinal
#igr = 0
#how to plot?
#enough training set verification?