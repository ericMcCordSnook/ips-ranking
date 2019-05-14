from rankobjects.Weight import Weight
import numpy as np
from math import exp, log

# Each ranking class holds one Pi permutation with a specific weighting structure
class Ranking:
    def __init__(self, num_elements, weights, rank=None, ground_truth=None, phi=1.0):
        if rank is not None:
            if not isinstance(rank, np.ndarray):
                raise TypeError("rank is not a numpy array.")
            if len(rank) != num_elements:
                raise ValueError("num_elements does not match the number of elements in given rank")
        if ground_truth is not None:
            if not isinstance(ground_truth, np.ndarray):
                raise TypeError("ground_truth is not a numpy array.")
            if len(ground_truth) != num_elements:
                raise ValueError("ground_truth does not match the number of elements in given rank")
        if not isinstance(num_elements, int):
            raise TypeError("num_elements is not an integer.")
        if not isinstance(weights, Weight):
            raise TypeError("weight is not an instance of the Weight class")
        if num_elements < 1:
            raise ValueError("num_elements is not a positive number")

        self.num_elements = num_elements

        self.identity = np.arange(1, num_elements+1)

        # weight object holds information from weight
        self.weight = weights

        self.phi = phi

        # create a weights vector from weight class
        self.weights_vect = weights.generate_weights_vect(num_elements)

        if rank is None:
            # create a Pi permutation from identity vector
            self.rank = np.random.permutation(self.identity).astype(int)
        else:
            self.rank = rank.astype(int)

        self.ground_truth = ground_truth

        self.deltas = self.calculate_deltas()

        self.Z_j_vect = self.calc_marginal_weight_sum()


    # transform ground truth to identity and orig ranking
    #   to new ranking w/ same distance
    def ground_truth_transform(self):
        if self.ground_truth is None:
            return self.rank

        # 1. Calculate inv ground truth
        ground_truth_inv = np.argsort(self.ground_truth)

        # 2. Left multiply orig_ranking by 1.
        new_rank = ground_truth_inv[self.rank-1]

        # 3. return 2. as our transformed ranking (with id as grnd truth)
        return new_rank+1

    # assumes ground truth is the identity
    # transformation from ground truth identity is required
    # deltas are calculated in Big O(num_elements squared)
    def calculate_deltas(self):
        # delta vect temporarily holds zeros
        delta_vect = np.zeros(self.num_elements)

        transformed_rank = self.ground_truth_transform()

        for j in range(1, self.num_elements+1):
            j_sum = 0
            for i in range(1, self.num_elements+1):
                if transformed_rank[i-1] > j:
                    j_sum += 1
                elif transformed_rank[i-1] == j:
                    break
            delta_vect[j-1] = j_sum

        return delta_vect.astype(int)

    def calc_weight_sum(self):
        weight_sum_vect = np.zeros(self.num_elements)

        for j in range(self.num_elements):
            weight_sum_vect[j] = self.weight.calc_weight_sum(j, self.deltas[j])

        return weight_sum_vect

    # used to calculate marginal (Z_j) for all j's in the vector, marg_weight_sum_vect
    def calc_marginal_weight_sum(self):

        marg_weight_sum_vect = np.zeros(self.num_elements)
        a = self.weight.a
        b = self.weight.b
        for j in range(self.num_elements): # loop through all j's
            z_j = 0.0

            for i in range(self.num_elements - j):
                weight_sum = self.weight.calc_weight_sum(j, i)

                z_j += exp(-self.phi * weight_sum)

            marg_weight_sum_vect[j] = z_j

        return marg_weight_sum_vect

    # calculate probability of delta_j exactly for all j's
    def calc_delt_prob(self):
        weight_sum = self.calc_weight_sum()
        weight_sum_exp = np.exp(-self.phi*weight_sum)
        marg_weight_sum = self.calc_marginal_weight_sum()
        return weight_sum_exp / marg_weight_sum

    # calculate probability of ranking based on delta_probs
    def calc_pi_prob(self):
        return np.prod(self.calc_delt_prob())

    # calculate log of probability of rankings (uses sums not products)
    def calc_log_pi_prob(self):
        return np.log(self.calc_pi_prob())

    def calc_dist(self, other_rank):
        self.ground_truth = other_rank.rank
        self.rank = self.ground_truth_transform()
        weight_sum = 0

        # bubble sort to compute (weighted) kemeny distance
        for i in range(self.num_elements):
            for j in range(0, self.num_elements-i-1):
                if self.rank[j] > self.rank[j+1]:
                    self.rank[j], self.rank[j+1] = self.rank[j+1], self.rank[j]
                    weight_sum += self.weights_vect[j]

        return weight_sum

    def shuffle_ranking(self):
        self.rank = np.random.permutation(self.identity).astype(int)

    def get_num_elements(self):
        return self.num_elements

    def get_weights_vect(self):
        return self.weights_vect

    def get_identity(self):
        return self.identity

    def get_rank(self):
        return self.rank

    def set_rank(self, new_rank):
        self.rank = new_rank.astype(int)

    def __str__(self):
        return "\nRanking:\t" + str(self.rank) + "\nWeights:\t" + str(self.weights_vect) + \
            "\nDeltas:\t\t" + str(self.deltas)
