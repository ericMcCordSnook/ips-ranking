from heuristics.Generic_Heuristic import Generic_Heuristic
from itertools import permutations
import numpy as np

class No_Heuristic(Generic_Heuristic):

    def __init__(self):
        super().__init__()

    def set_params(self, params):
        pass # there are no params to set

    def run_heuristic(self):
        ground_truths = list(permutations([i for i in range(1, self.num_elements+1)]))
        # array containing best_phi, best_b, max_log_like for each ground_truth
        results = np.empty((len(ground_truths), 3))
        for i in range(len(ground_truths)):
            self.optimization_params["ground_truth"] = np.array(ground_truths[i])
            self.optimization.set_params(self.optimization_params)
            results[i] = self.optimization.optimize()
        return results
