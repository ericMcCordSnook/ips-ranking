from rankobjects.Ranking import Ranking
from rankobjects.Weight import Weight
import numpy as np
from math import exp, log

# MultiRanking holds multiple Ranking objects and can calculate log likelihood based on these samples
class MultiRanking:
    def __init__(self, rankings_arr, weights, ground_truth=None, phi=1.0):
        # if not isinstance(ground_truth, np.ndarray):
        #     raise TypeError("ground_truth is not an array")
        if not isinstance(rankings_arr, np.ndarray):
            raise TypeError("rankings_arr is not an array")
        if len(rankings_arr.shape) != 2:
            raise ValueError("Incorrect number of dimensions of rankings array. Must be 2D")
        if not isinstance(weights, Weight):
            raise TypeError("weight is not an instance of the Weight class")

        # number of samples of rankings
        self.num_samples = rankings_arr.shape[0]
        # number of elements in each rank
        self.num_elements = rankings_arr.shape[1]

        self.rankings_arr = rankings_arr

        self.weights = weights

        # delta matrix contains deltas for all the rankings in our dataset.
        # Necessary to calculate for gradient of log-likes w/ respect to b
        self.delta_matrix = np.zeros((self.num_samples, self.num_elements))

        self.phi = phi

        # list of Ranking objects created from data in rankings_arr
        self.ranking_obj_list = []

        # pi_rank_i = None
        for i in range(self.num_samples):

            # populate ranking_obj_list with data
            pi_rank_i = Ranking(self.num_elements, self.weights,
                rank=rankings_arr[i], ground_truth=ground_truth, phi=self.phi)
            self.ranking_obj_list.append(pi_rank_i)

            # store delta_vect of pi_rank_i into delta_matrix
            self.delta_matrix[i] = pi_rank_i.deltas

            self.Z_j = pi_rank_i.Z_j_vect

    # calculates log likelihood with num_samples of ranking data.
    # Note that this is NOT optimized yet (fewer calculations can be done)
    def log_likelihood(self):
        ret_sum = 0
        for i in range(self.num_samples):
            pi_rank_i = self.ranking_obj_list[i]
            ret_sum += pi_rank_i.calc_log_pi_prob()
        return ret_sum

    # returns vectors over j containing s_j and v_j
    def calc_delta_j_means(self):
        s_j = np.sum(self.delta_matrix, axis=0) / self.num_samples
        v_j = np.sum(self.delta_matrix*self.delta_matrix, axis=0) / self.num_samples

        return s_j, v_j

    # Calculates gradient at a given b
    # Assumes arithmetic weights
    def calc_gradients(self, b, phi):
        s_j, v_j = self.calc_delta_j_means()
        # print(s_j, v_j)
        j_vect = np.arange(self.num_elements)
        # print(j_vect)

        phi_first_term = -self.num_samples*(np.sum(s_j) - b * (np.dot(j_vect, s_j) + 0.5*(np.sum(v_j - s_j))))

        b_first_term = phi*self.num_samples*(np.dot(j_vect, s_j) + 0.5*(np.sum(v_j - s_j)))

        b_second_term = 0
        phi_second_term = 0

        # calculating the second term
        for j in range(self.num_elements):
            i_sum_b = 0
            i_sum_phi = 0

            for i in range(self.num_elements - j):
                ij_coef = (0.5*(i-1)*i)+i*j
                i_sum_b += ij_coef * phi * np.exp(phi * (ij_coef * b - i))
                i_sum_phi += (ij_coef*b - i) * np.exp(phi * (ij_coef * b - i))

            b_second_term += (i_sum_b / self.Z_j[j])
            phi_second_term += (i_sum_phi / self.Z_j[j])

        b_grad = b_first_term - self.num_samples * b_second_term
        phi_grad = phi_first_term - self.num_samples * phi_second_term

        return b_grad, phi_grad
