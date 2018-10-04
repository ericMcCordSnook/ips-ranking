from optimizations.Generic_Optimization import Generic_Optimization
from rankobjects.MultiRanking import MultiRanking
import numpy as np

class Exhaustive_Search(Generic_Optimization):
    def __init__(self):
        super().__init__()
        self.granularity = None

    def set_params(self, params):
        self.ground_truth = params["ground_truth"]
        self.granularity = params["granularity"]

    # Finds b and phi that maximizes log-likelihood using exhaustive search
    # Returns this b and its corresponding log-like
    # Assumes arithmetic weights with a = 1.
    def optimize(self):
        print("Optimizing parameters for ground_truth: ", self.ground_truth)
        num_elements = len(self.ground_truth)

        WEIGHT_A = 1.0
        b_min = 0.
        b_max = 1.0 / (num_elements-2)
        phi_min = 0.
        phi_max = 2.0

        b_vect = np.linspace(b_min, b_max, self.granularity)
        phi_vect = np.linspace(phi_min, phi_max, self.granularity)

        log_likes = np.zeros((self.granularity, self.granularity))

        for i, phi in enumerate(phi_vect):
            for j, b in enumerate(b_vect):
                multi_rank_obj = MultiRanking(self.data, self.weight, self.ground_truth, phi=phi)
                log_likes[i,j] = multi_rank_obj.log_likelihood()

        max_index_phi, max_index_b = np.unravel_index(np.argmax(log_likes), log_likes.shape)
        best_phi = phi_vect[max_index_phi]
        best_b = b_vect[max_index_b]
        max_log_like = log_likes[max_index_phi][max_index_b]

        print("Optimal (phi, b, log_like) = (%f, %f, %f) \n" % (best_phi, best_b, max_log_like))

        return (best_phi, best_b, max_log_like)
