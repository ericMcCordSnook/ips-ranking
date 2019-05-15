import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from itertools import permutations
from rankobjects.Weight import Unweighted, Arithmetic, Geometric, Harmonic
from rankobjects.Ranking import Ranking

def harm(n):
    a = [1.0/i for i in range(1, n+1)]
    return np.sum(np.array(a))

def read_cmd_args():
    usage = "Usage: python generate_distance_correlation_scatterplots.py <n> <weight_type> <b> <cutoff_or_-1>"
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

def compute_max_dist(n, weight_obj):
    rank_obj_1 = Ranking(n, weight_obj, rank=np.array([i for i in range(1, n+1)]))
    rank_obj_2 = Ranking(n, weight_obj, rank=np.array([i for i in range(n, 0, -1)]))
    max_dist = round(rank_obj_1.calc_dist(rank_obj_2), 3)
    return max_dist

def compute_expected_dist(n, weight_obj):
    w_vec = weight_obj.generate_weights_vect(n)
    avg_dist = 0
    for s in range(n-1):
        avg_dist += w_vec[s]*(n-s)*(harm(n)-harm(n-s))
    return avg_dist

def simulate_expected_dist(n, weight_obj):
    ground_truth_ranking = Ranking(n, weight_obj, rank=np.array([i for i in range(1, n+1)]))
    other_ranking = Ranking(n, weight_obj)
    num_trials = 1000
    dists = np.empty(num_trials)
    for i in range(num_trials):
        other_ranking.shuffle_ranking()
        dists[i] = other_ranking.calc_dist(ground_truth_ranking)
    return np.average(dists)

def compute_correlation_from_dist(dist, avg, max_dist):
    if dist <= avg:
        return 1 - dist/avg
    else:
        return -1*(dist-avg)/(max_dist-avg)

def compute_correlations(n, weighted_dists, weighted_obj, unweighted_dists, unweighted_obj):
    num_perms = len(weighted_dists)
    max_dist_weighted = compute_max_dist(n, weighted_obj)
    # expected_dist_weighted = compute_expected_dist(n, weighted_obj)
    expected_dist_weighted = simulate_expected_dist(n, weighted_obj)
    max_dist_unweighted = compute_max_dist(n, unweighted_obj)
    # expected_dist_unweighted = compute_expected_dist(n, unweighted_obj)
    expected_dist_unweighted = simulate_expected_dist(n, unweighted_obj)
    weighted_corrs = np.empty(num_perms)
    unweighted_corrs = np.empty(num_perms)
    for i in range(num_perms):
        weighted_corrs[i] = compute_correlation_from_dist(weighted_dists[i], expected_dist_weighted, max_dist_weighted)
        unweighted_corrs[i] = compute_correlation_from_dist(unweighted_dists[i], expected_dist_unweighted, max_dist_unweighted)
    return weighted_corrs, unweighted_corrs

def make_scatterplot(x, xlabel, y, ylabel, n, weight_type, b, cutoff):
    title = weight_type + ", n = " + str(n) + ", b = " + str(b) + ", "
    if cutoff is None:
        title = title + "no cutoff"
    else:
        title = title + "cutoff = " + str(cutoff)
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def main():
    n, weight_type, b, cutoff = read_cmd_args()
    weighted_obj, unweighted_obj = create_weight_objs(weight_type, b, cutoff)
    weighted_dists, unweighted_dists = compute_distances(n, weighted_obj, unweighted_obj)
    weighted_corrs, unweighted_corrs = compute_correlations(n, weighted_dists, weighted_obj, unweighted_dists, unweighted_obj)
    make_scatterplot(weighted_dists, "Weighted Distances", unweighted_dists, "Unweighted Distances", n, weight_type, b, cutoff)
    make_scatterplot(weighted_corrs, "Weighted Correlations", unweighted_corrs, "Unweighted Correlations", n, weight_type, b, cutoff)

if __name__ == '__main__':
    main()
