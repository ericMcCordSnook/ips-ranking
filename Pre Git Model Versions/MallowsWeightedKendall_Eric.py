"""
  Nikhil Bhaip and Eric McCord-Snook
  MallowsWeightedKendall.py

  This program is used to calculate probabilities of rankings with different weights.
  Weights are abstract classes that contain information on weighting structure based on two parameters.
  Currently Arithmetic class is the only child class of Weight, but more (like Geometric) can be easily extended.


  A Ranking is a class that holds information on a particular ranking and its properties. All rankings will have a
  Weight class to calculate operations related to weights like getting the sum of weights.

  A MultiRanking is a class that holds multiple Ranking objects. We can use it calculate the log-likelihood.
"""

import numpy as np
from math import exp, log
from abc import abstractmethod
from itertools import permutations
import matplotlib.pyplot as plt


# Weights hold information of a two-parameter (a, b) weighting mechanism for rankings
class Weight:
    def __init__(self, a=1., b=1.):
        if not isinstance(a, float):
            raise TypeError("First argument, 'a', is not a float.")
        if not isinstance(b, float):
            raise TypeError("Second argument, 'b', is not a float.")

        self.a = a
        self.b = b
        self.sequence = None

    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_sequence(self):
        return self.sequence

    @abstractmethod
    def generate_weights_vect(self, num_elements):
        pass

    @abstractmethod
    def calc_weight_sum(self, j, delt):
        pass


# Arithmetic weights are a Weight in which weights are determined arithmetically
# of the form: a-l*b, where l is the position of the identity vector of size num_elements
class Arithmetic(Weight):
    def __init__(self, a=1.0, b=1.0):
        Weight.__init__(self, a, b)
        self.sequence = "arithmetic"

    def generate_weights_vect(self, num_elements):
        # weights are 1 less than num_elements
        indices = np.arange(1, num_elements)
        return self.a-indices*self.b

    def calc_weight_sum(self, j, delt):
        # weight_sum = 0
        # for l in range(j, j+delt): # loops from l=j to l=j+delt-1
        #     weight_sum += (self.a - l*self.b)
        #return weight_sum
        return self.a*delt - self.b*delt*(j-0.5) - 0.5*self.b*delt*delt


# Geometric weights are a Weight in which weights are determined geometrically
# of the form: a(b**l), where l is the position of the identity vector of size num_elements
class Geometric(Weight):
    def __init__(self, a=1.0, b=1.0):
        Weight.__init__(self, a, b)
        self.sequence = "geometric"

    def generate_weights_vect(self, num_elements):
        indices = np.arange(1, num_elements)
        return self.a*(self.b ** indices)

    def calc_weight_sum(self, j, delt):
        return (self.a*(self.b ** j)*(1 - (self.b ** delt))) / (1-self.b)


# Each ranking class holds one Pi permutation with a specific weighting structure
class Ranking:
    def __init__(self, num_elements, weights, rank=None):
        if rank is not None:
            if not isinstance(rank, np.ndarray):
                raise TypeError("rank is not a numpy array.")
            if len(rank) != num_elements:
                raise ValueError("num_elements does not match the number of elements in given rank")
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

        # create a weights vector from weight class
        self.weights_vect = weights.generate_weights_vect(num_elements)

        if rank is None:
            # create a Pi permutation from identity vector
            self.rank = np.random.permutation(self.identity)
        else:
            self.rank = rank

        self.deltas = self.calculate_deltas()

    # deltas are calculated in Big O(num_elements squared)
    def calculate_deltas(self):
        # delta vect temporarily holds zeros
        delta_vect = np.zeros(self.num_elements)

        for j in range(1, self.num_elements+1):
            j_sum = 0
            for i in range(1, self.num_elements+1):
                if self.rank[i-1] > j:
                    j_sum += 1
                elif self.rank[i-1] == j:
                    break
            delta_vect[j-1] = j_sum

        return delta_vect.astype(int)

    def calc_weight_sum(self):
        """
        weight_sum_vect = np.zeros(self.num_elements)

        for j in range(self.num_elements):
            #### CHANGED j+1 to j
            weight_sum_vect[j] = self.weight.calc_weight_sum(j, self.deltas[j])

        return weight_sum_vect
        """
        weight_sum_vect = np.zeros(self.num_elements)

        for j in range(self.num_elements):
            #print("j is: ", j)
            #print("delta_j is: ", self.deltas[j])
            weight_sum_vect[j] = self.weight.calc_weight_sum(j, self.deltas[j])

        return weight_sum_vect

    # used to calculate marginal (Zj) for all j's in the vector, marg_weight_sum_vect
    def calc_marginal_weight_sum(self):

        marg_weight_sum_vect = np.zeros(self.num_elements)
        a = self.weight.get_a()
        b = self.weight.get_b()
        for j in range(self.num_elements): # loop through all j's
            z_j = 0.0

            for i in range(self.num_elements - j):
                weight_sum = self.weight.calc_weight_sum(j, i)
                z_j += exp(-weight_sum)

            marg_weight_sum_vect[j] = z_j

        return marg_weight_sum_vect

    # calculate probability of delta_j exactly for all j's
    def calc_delt_prob(self):
        weight_sum = self.calc_weight_sum()
        weight_sum_exp = np.exp(-1.0*weight_sum)
        marg_weight_sum = self.calc_marginal_weight_sum()
        return weight_sum_exp / marg_weight_sum

    # calculate probability of ranking based on delta_probs
    def calc_pi_prob(self):
        return np.prod(self.calc_delt_prob())

    # calculate log of probability of rankings (uses sums not products)
    def calc_log_pi_prob(self):
        weight_sum = self.calc_weight_sum()
        log_marg_weight_sum = np.log(self.calc_marginal_weight_sum())

        log_vect = -1*weight_sum - log_marg_weight_sum

        return np.sum(log_vect)

    def get_num_elements(self):
        return self.num_elements

    def get_weights_vect(self):
        return self.weights_vect

    def get_identity(self):
        return self.identity

    def get_rank(self):
        return self.rank

    def __str__(self):
        return "\nRanking:\t" + str(self.rank) + "\nWeights:\t" + str(self.weights_vect) + \
            "\nDeltas:\t\t" + str(self.deltas)


# MultiRanking holds multiple Ranking objects and can calculate log likelihood based on these samples
class MultiRanking:
    def __init__(self, rankings_arr, weights):
        if not isinstance(rankings_arr, np.ndarray):
            raise TypeError("rankings_lists is not an array")
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

        # list of Ranking objects created from data in rankings_arr
        self.ranking_obj_list = []

        # populate ranking_obj_list with data
        for i in range(self.num_samples):
            pi_rank_i = Ranking(self.num_elements, self.weights, rankings_arr[i])
            self.ranking_obj_list.append(pi_rank_i)

    # calculates log likelihood with num_samples of ranking data.
    # Note that this is NOT optimized yet (fewer calculations can be done)
    def log_likelihood(self):
        ret_sum = 0
        for i in range(self.num_samples):
            pi_rank_i = self.ranking_obj_list[i]
            ret_sum += pi_rank_i.calc_log_pi_prob()
        return ret_sum

def plot_log_like(b_arr, log_like):
    plt.plot(b_arr, log_like)
    plt.xlabel("b")
    plt.ylabel("log-likelihood")
    plt.show()

def generate_simulated_ranks(BASE_LIST, NUM_RANKING_ELEMENTS, WEIGHT_A, WEIGHT_B, NUM_SIM_RANKINGS):
    geom_weights = Geometric(WEIGHT_A, WEIGHT_B)
    all_rankings_data = np.array([])
    perms = list(permutations(BASE_LIST))
    perm_probs = np.zeros(len(perms))

    # calculate the probabilities of each permutation
    for i in range(len(perms)):
        rank_i = perms[i]
        rank_as_arr = np.array(rank_i)
        pi_rank = Ranking(len(BASE_LIST), geom_weights, rank_as_arr)
        perm_probs[i] = pi_rank.calc_pi_prob()

    # create list of simulated rankings based on probabilities
    for i in range(NUM_SIM_RANKINGS):
        all_rankings_data = np.append(all_rankings_data, perms[np.random.choice(range(len(perms)), p=perm_probs)])

    # Reshape 1D array to different dimenstions (Num_rankings by Number of elements in each ranking) and return
    return all_rankings_data.reshape(int(np.size(all_rankings_data)/NUM_RANKING_ELEMENTS), NUM_RANKING_ELEMENTS)

def main():
    WEIGHT_A = float(input("Enter a: "))
    WEIGHT_B = float(input("Enter b: "))
    NUM_RANKING_ELEMENTS = int(input("Enter number of elements: "))
    NUM_SIM_RANKINGS = int(input("Enter number of simulated rankings: "))
    BASE_LIST = [i for i in range(1, NUM_RANKING_ELEMENTS+1)]
    # MULTI_RANK_DATA = np.array([[1, 4, 3, 2], [1, 4, 3, 2], [1, 3, 2, 4], [2, 3, 4, 1], [1, 3, 4, 2]])

    # avoid scientific notation when printing numbers
    np.set_printoptions(suppress=True)

    all_rankings_data = generate_simulated_ranks(BASE_LIST, NUM_RANKING_ELEMENTS, WEIGHT_A, WEIGHT_B, NUM_SIM_RANKINGS)
    print("Simulated rankings: \n", all_rankings_data)

    # Create Multiranking object from simulated rankings array
    log_likes = np.zeros((99,1))
    for i in range(1, 100):
        b = 0.01 * i
        geom_weights_adj = Geometric(WEIGHT_A, b)
        multi_rank_obj = MultiRanking(all_rankings_data, geom_weights_adj)
        log_like = multi_rank_obj.log_likelihood()
        log_likes[i - 1] = log_like

    plot_log_like(np.arange(0.01, 1, 0.01), log_likes)


if __name__ == "__main__":
    main()
