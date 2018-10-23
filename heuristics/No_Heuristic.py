from heuristics.Generic_Heuristic import Generic_Heuristic
from itertools import permutations
import numpy as np

class No_Heuristic(Generic_Heuristic):

    def __init__(self):
        super().__init__()

    def set_params(self, params):
        pass # there are no params to set

    def run_heuristic(self, params):
        ground_truths = list(permutations([i for i in range(1, self.num_elements+1)]))
        # array containing best_phi, best_b, max_log_like for each ground_truth
        best_result = None
        for i, ground_truth in enumerate(ground_truths):
            self.optimization_params["ground_truth"] = np.array(ground_truth)
            self.optimization.set_params(self.optimization_params)
            cur_result = self.optimization.optimize()
            if best_result is None or cur_result[2] > best_result[1][2]:
                best_result = (ground_truth, cur_result)
        return [best_result], {} # {} takes the place of precomputed results, we don't use this here
