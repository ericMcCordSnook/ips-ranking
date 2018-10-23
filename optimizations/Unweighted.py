from optimizations.Generic_Optimization import Generic_Optimization
from rankobjects.MultiRanking import MultiRanking
import numpy as np
import logging

class Unweighted(Generic_Optimization):
    def __init__(self):
        super().__init__()

    def set_params(self, params):
        self.ground_truth = params["ground_truth"]

    # assuming weights are uniform, find the best ground-truth
    def optimize(self):
        self.weight.b = 0.0
        multi_rank_obj = MultiRanking(self.data, self.weight, self.ground_truth, phi=1.0)
        max_log_like = multi_rank_obj.log_likelihood()
        logging.info("Unweighted max log likelihood for %s is %f" % (self.ground_truth, max_log_like))
        return (1.0, 0.0, max_log_like)
