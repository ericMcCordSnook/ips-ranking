from utils.miscutils import generate_weighted_permutation_graph
from math import ceil, factorial
import sys
from copy import deepcopy
import numpy as np

class Unimode_Greedy:
    def __init__(self):
        super().__init__()
        self.mode = None # will be the starting node

    def set_params(self, params):
        self.mode = params["mode"]

    def get_neighbors(self, perm):
        neighbors = np.empty((len(perm)-1, len(perm)))
        for i in range(len(perm)-1):
            perm_cpy = deepcopy(perm)
            perm_cpy[i], perm_cpy[i+1] = perm_cpy[i+1], perm_cpy[i]
            neighbors[i] = perm_cpy
        return neighbors

    def run_heuristic(self):
        ground_truth = np.array(list(self.mode), dtype=int)
        results = []
        visited = set()
        self.optimization_params["ground_truth"] = ground_truth
        self.optimization.set_params(self.optimization_params)
        result = self.optimization.optimize() # returns best_b, best_phi, max_log_like
        results.append((ground_truth, result))
        max_log_like = result[2]
        visited.add(np.array_str(ground_truth))
        while True:
            neighbors = self.get_neighbors(ground_truth)
            cur_results = [] # list of results for each neighbor
            new_neighbors = 0
            for i, neighbor in enumerate(neighbors):
                if np.array_str(neighbor) not in visited:
                    new_neighbors += 1
                    visited.add(np.array_str(neighbor))
                    self.optimization_params["ground_truth"] = neighbor
                    self.optimization.set_params(self.optimization_params)
                    cur_results.append((neighbor, self.optimization.optimize()))
            if new_neighbors == 0:
                break
            cur_results.sort(key= lambda x: x[1][2]) # sorts by ascending log-likelihood
            best_neighbor_log_like = cur_results[-1][1][2]
            if best_neighbor_log_like > max_log_like:
                max_log_like = best_neighbor_log_like
                results.append(cur_results[-1])
                ground_truth = cur_results[-1][0]
            else:
                break
        return results






    # THE METHODS BELOW ARE FLAWED AS THEY ASSUMED THAT THE B AND PHI WERE ALWAYS THE SAME
    ## DEPRECATED
    def get_greedy_traversal_order(self, graph, start_node):
        order = [start_node]
        visited = set()
        visited.add(start_node)
        current_nodes = deepcopy(graph[start_node])
        while bool(current_nodes): # tests if empty
            min_dist_node = None
            min_dist = sys.maxsize
            for node, dist in current_nodes.items():
                if dist < min_dist:
                    min_dist_node = node
                    min_dist = dist
            visited.add(min_dist_node)
            order.append(min_dist_node)
            for node, dist in graph[min_dist_node].items():
                if node not in visited:
                    if (node not in current_nodes) or (min_dist + dist < current_nodes[node]):
                        current_nodes[node] = min_dist + min_dist
            del current_nodes[min_dist_node]
        return order

    # All heuristics will have a method of the same name
    def get_perms_to_explore(self, graph, mode_ranking, pct=0.2, num=None):
        num_to_return = 0
        if num is not None:
            if not isinstance(num, int):
                raise ValueError("num argument must be an integer")
            else:
                num_to_return = num
        else:
            num_to_return = int(ceil(pct * factorial(len(mode_ranking))))
        return self.get_greedy_traversal_order(graph, mode_ranking)[:num_to_return]
