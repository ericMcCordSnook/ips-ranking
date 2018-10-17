from heuristics.Generic_Heuristic import Generic_Heuristic
from heuristics.Single_Start_Greedy import Single_Start_Greedy
from math import ceil, factorial
import sys
from copy import deepcopy
import numpy as np

class Multi_Start_Greedy(Generic_Heuristic):
    def __init__(self):
        super().__init__()
        self.starts = None # list of starting nodes
        self.greedy_worker = Single_Start_Greedy()

    def set_params(self, params):
        self.starts = params["starts"]

    def run_heuristic(self, params):
        precomputed = {} # a map from ground_truth -> best_phi, best_b, max_log_like for all we've ever seen
        multi_start_results = {}
        self.greedy_worker.num_elements = self.num_elements
        self.greedy_worker.optimization = self.optimization
        self.greedy_worker.optimization_params = self.optimization_params
        self.greedy_worker.weight = self.weight
        for start in self.starts:
            self.greedy_worker.start = start
            start_str = np.array_str(np.array(list(start), dtype=float))
            multi_start_results[start_str], precomputed = self.greedy_worker.run_heuristic({"precomputed":precomputed})
            print("Start = " , start_str, "attempted %d possible ground truths" % len(multi_start_results[start_str]))
        return multi_start_results
