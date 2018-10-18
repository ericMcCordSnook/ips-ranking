from heuristics.Generic_Heuristic import Generic_Heuristic
from math import ceil, factorial
import sys
from copy import deepcopy
import numpy as np
import logging

class Single_Start_Greedy(Generic_Heuristic):
    def __init__(self):
        super().__init__()
        self.start = None # will be the starting node

    def set_params(self, params):
        self.start = params["start"]

    def get_neighbors(self, perm):
        neighbors = np.empty((len(perm)-1, len(perm)))
        for i in range(len(perm)-1):
            perm_cpy = deepcopy(perm)
            perm_cpy[i], perm_cpy[i+1] = perm_cpy[i+1], perm_cpy[i]
            neighbors[i] = perm_cpy
        return neighbors

    def run_heuristic(self, params):
        ground_truth = np.array(list(self.start), dtype=float)
        single_path_results = [] # path from start -> optimal ground truth node based on greedy evaluation of maximum_log_likelihood
        newly_visited = set() # set of ground truths already visited on this single start greedy
        # if this is a single start worker of a multi-start greedy then store the precomputed log-likelihoods
        precomputed = {}
        if 'precomputed' in params:
            precomputed = params['precomputed']
        self.optimization_params["ground_truth"] = ground_truth
        self.optimization.set_params(self.optimization_params)
        result = self.optimization.optimize() # returns best_b, best_phi, max_log_like
        single_path_results.append((np.array_str(ground_truth), result))
        max_log_like = result[2]
        newly_visited.add(np.array_str(ground_truth))
        precomputed[np.array_str(ground_truth)] = result
        while True:
            neighbors = self.get_neighbors(ground_truth)
            neighbor_results = [] # list of results for each neighbor
            new_neighbors = 0
            for i, neighbor in enumerate(neighbors):
                if np.array_str(neighbor) not in newly_visited:
                    new_neighbors += 1
                    newly_visited.add(np.array_str(neighbor))
                    if np.array_str(neighbor) in precomputed:
                        neighbor_results.append((neighbor, i, precomputed[np.array_str(neighbor)]))
                    else:
                        self.optimization_params["ground_truth"] = neighbor
                        self.optimization.set_params(self.optimization_params)
                        result = self.optimization.optimize()
                        neighbor_results.append((neighbor, i, result))
                        precomputed[np.array_str(neighbor)] = result
            if new_neighbors == 0:
                break
            # sort by ascending log-likelihood, descending index of swap
            neighbor_results.sort(key= lambda x: (x[2][2], -1*x[1]))
            # last element was the best choice
            best_neighbor_log_like = neighbor_results[-1][2][2]
            if best_neighbor_log_like >= max_log_like:
                max_log_like = best_neighbor_log_like
                single_path_results.append((np.array_str(neighbor_results[-1][0]), neighbor_results[-1][2]))
                ground_truth = neighbor_results[-1][0]
            else:
                break
        logging.info("Attempted %d possible ground truths" % len(newly_visited))
        # print("Attempted %d possible ground truths" % len(newly_visited))
        return single_path_results, precomputed
