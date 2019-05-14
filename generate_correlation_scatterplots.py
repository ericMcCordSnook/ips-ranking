import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from itertools import permutations
from rankobjects.Weight import Unweighted, Arithmetic, Geometric, Harmonic
from rankobjects.Ranking import Ranking

def read_cmd_args():
    usage = "Usage: python generate_correlation_scatterplots.py <n> <weight_type> <b> <cutoff_or_-1>"
    try:
        n = int(sys.argv[1])
        weight_type = sys.argv[2]
        b = float(sys.argv[3])
        cutoff = int(sys.argv[4]) if int(sys.argv[4]) != -1 else None
    except:
        print(usage)
        sys.exit(0)
    return n, weight_type, b, cutoff

def create_weight_objs(weight_type, b, cutoff):
    weighted_obj = None
    unweighted_obj = Unweighted(a=1.0, b=b, cutoff=cutoff)
    if weight_type == "Unweighted":
        weighted_obj = Unweighted(a=1.0, b=b, cutoff=cutoff)
    elif weight_type == "Arithmetic":
        weighted_obj = Arithmetic(a=1.0, b=b, cutoff=cutoff)
    elif weight_type == "Geometric":
        weighted_obj = Geometric(a=1.0, b=b, cutoff=cutoff)
    elif weight_type == "Harmonic":
        weighted_obj = Harmonic(a=1.0, b=b, cutoff=cutoff)
    return weighted_obj, unweighted_obj

def compute_distances(n, weighted_obj, unweighted_obj):
    perms = list(permutations([i for i in range(1, n+1)]))
    weighted_dists = np.empty(len(perms))
    weighted_ground_truth = Ranking(n, weighted_obj, rank=np.array([i for i in range(1, n+1)]))
    weighted_ranking = Ranking(n, weighted_obj, rank=np.array([i for i in range(1, n+1)]))
    unweighted_dists = np.empty(len(perms))
    unweighted_ground_truth = Ranking(n, unweighted_obj, rank=np.array([i for i in range(1, n+1)]))
    unweighted_ranking = Ranking(n, unweighted_obj, rank=np.array([i for i in range(1, n+1)]))
    for i in range(len(perms)):
        weighted_ranking.set_rank(np.array(perms[i]))
        unweighted_ranking.set_rank(np.array(perms[i]))
        weighted_dists[i] = weighted_ranking.calc_dist(weighted_ground_truth)
        unweighted_dists[i] = unweighted_ranking.calc_dist(unweighted_ground_truth)
    return weighted_dists, unweighted_dists

def make_scatterplot(x, y, n, weight_type, b, cutoff):
    title = weight_type + ", n = " + str(n) + ", b = " + str(b) + ", "
    if cutoff is None:
        title = title + "no cutoff"
    else:
        title = title + "cutoff = " + str(cutoff)
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel("Weighted Distances")
    plt.ylabel("Unweighted Distances")
    plt.show()

def main():
    n, weight_type, b, cutoff = read_cmd_args()
    weighted_obj, unweighted_obj = create_weight_objs(weight_type, b, cutoff)
    weighted_dists, unweighted_dists = compute_distances(n, weighted_obj, unweighted_obj)
    make_scatterplot(weighted_dists, unweighted_dists, n, weight_type, b, cutoff)

if __name__ == '__main__':
    main()
