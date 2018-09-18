"""
  Nikhil Bhaip & Eric McCord-Snook
  MallowsWeightedKendall.py

  This program is used to calculate probabilities of rankings with different weights.
  Weights are abstract classes that contain information on weighting structure based on two parameters. 
  
  
  A Ranking is a class that holds information on a particular ranking and its properties. All rankings will have a 
  Weight class to calculate operations related to weights like getting the sum of weights.  
  
  A MultiRanking is a class that holds multiple Ranking objects. We can use it calculate the log-likelihood.
"""


"""
Changelog 4/3
Added negative sign to line 326




"""

import numpy as np
from math import exp, log
from abc import abstractmethod
from itertools import permutations
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


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
        indices = np.arange(0, num_elements-1)
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

    # used to calculate marginal (Z_j) for all j's in the vector, marg_weight_sum_vect
    def calc_marginal_weight_sum(self):

        marg_weight_sum_vect = np.zeros(self.num_elements)
        a = self.weight.get_a()
        b = self.weight.get_b()
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

    """
    def calc_log_pi_prob_v2(self):
        weight_sum = self.calc_weight_sum()
        log_marg_weight_sum = np.log(self.calc_marginal_weight_sum())

        log_vect = -self.phi*weight_sum - log_marg_weight_sum

        return np.sum(log_vect)
    """

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

# Read in the data from given file
def read_csv(file, has_header=True):
    data = np.loadtxt(file, dtype=float, delimiter=',', skiprows=has_header)
    return data

# Plot graph of possible b's and their respective log-likelihood (a is fixed)
def plot_graph(x, y, x_label, y_label):
    plt.plot(x, y)
    # plt.plot(b_arr, grad)
    plt.xlabel(x_label)
    # plt.xlim(0, 0.3333)
    # plt.ylim(-380,-350)
    # plt.title("Log-likes")
    plt.ylabel(y_label)
    plt.show()

# Plot contour likelihood graph of possible phi's and b's 
def plot_2D_log_like(phi_arr, b_arr, log_like, dataset, \
    num_contours=20, viz_folder="Plots", ground_truth=None):
    plt.figure(figsize=(20,10))
    plt.contourf(b_arr,phi_arr,log_like,num_contours)
    cb = plt.colorbar()
    plt.xlabel("b")
    plt.ylabel("phi")
    # plt.ylim(ymax=b_arr[-1])
    if ground_truth is None:
        plt.title(dataset)
        plt.savefig(viz_folder + "/" + dataset)
    else:
        gt_str = "".join(ground_truth.astype(str))
        plt.title(dataset + " " + " GT " + gt_str)
        plt.savefig(viz_folder + "/" + dataset 
            + " GT " + gt_str)
    cb.remove()

def calc_2D_log_like(data, a_arr, b_arr, weight_type, ground_truth=None):
    # Get length of a and b arrays
    a_num = a_arr.shape[0]
    b_num = b_arr.shape[0]

    weight_type = weight_type.lower()
    weight = None 

    # Create weight object with defaults
    if weight_type == "arithmetic":
        weight = Arithmetic()
    elif weight_type == "geometric":
        weight = Geometric()
    else:
        raise Exception("Invalid Weight Type")


    log_likes = np.zeros((a_num, b_num))
    for i in range(a_num):
        for j in range(b_num):
            # Set weight object with new values
            weight.a = a_arr[i]
            weight.b = b_arr[j]

            multi_rank_obj = MultiRanking(data, weight, ground_truth=ground_truth)
            log_likes[i][j] = multi_rank_obj.log_likelihood()

    return log_likes

# Convert list of ranks to concatenated string versions
# (e.g. [[1,2,4,3],[2,4,3,1]] --> ["1243", "2431"])
def ranks_to_str(all_ranks):
    num_rows = all_ranks.shape[0]
    ranks_str = np.empty(num_rows).astype(str)

    for i in range(num_rows):
        ranks_str[i] = "".join(all_ranks[i].astype(str))

    return ranks_str


# Plot bar chart of ground truths and respective max log-likelihoods
def plot_gt_bars(all_ground_truths, max_log_likes, title, viz_folder="Plots", 
    xlab="Max Log like", ylab="Ground Truths"):
    all_gt_str = ranks_to_str(all_ground_truths)

    plt.barh(all_gt_str, max_log_likes, align='center')
    plt.ylabel(ylab)
    # plt.xticks(rotation=90)
    plt.xlabel(xlab)
    plt.savefig(viz_folder + "/" + title)
    plt.clf()
    # plt.show()

# Gets data from CSV file and converts it appropriately
def get_data(dataset, folder="RankingDatasets/"):
    data_file = folder+dataset+".csv"
    return read_csv(data_file).astype(int)

# Finds b and phi that maximizes log-likelihood using exhaustive search
# Returns this b and its corresponding log-like
# Assumes arithmetic weights with a = 1.
def exhaustive_search(data, granularity=21, ground_truth=None, plot_log_likes=True):
    num_elements = data[0].shape[0]
    
    WEIGHT_A = 1.0
    b_min = 0.
    b_max = 1.0 / (num_elements-2)
    phi_min = 0.
    phi_max = 2.0

    b_vect = np.linspace(b_min, b_max, granularity)
    phi_vect = np.linspace(phi_min, phi_max, granularity)
    
    log_likes = np.zeros((granularity, granularity))

    for i, phi in tqdm(enumerate(phi_vect)):
        for j, b in tqdm(enumerate(b_vect)):    
            multi_rank_obj = MultiRanking(data, Arithmetic(WEIGHT_A, b), ground_truth, phi=phi)
            log_likes[i,j] = multi_rank_obj.log_likelihood()


    max_index_phi, max_index_b = np.unravel_index(np.argmax(log_likes), log_likes.shape)
    best_phi = phi_vect[max_index_phi]
    best_b = b_vect[max_index_b]
    max_log_like = log_likes[max_index_phi][max_index_b]

    print("best phi:", best_phi)
    print("best b:", best_b)
    print("best log-like:", max_log_like)

    if plot_log_likes:
        plot_2D_log_like(phi_vect, b_vect, log_likes, "Simulated")

    return best_phi, best_b, max_log_like

# assuming weights are uniform, find the best ground-truth
def conventional(data, ground_truth=None):
    
    WEIGHT_A = 1.0
    WEIGHT_B = 0.0
    
    multi_rank_obj = MultiRanking(data, Arithmetic(WEIGHT_A, WEIGHT_B), ground_truth=ground_truth)
    max_likelihood = multi_rank_obj.log_likelihood()

    return max_likelihood

# Finds b that maximizes log-likelihood using gradient ascent
# Returns this b and its corresponding log-like
# Assumes arithmetic weights with a = 1.

def grad_ascent(data, num_iter=100, alpha=0.001, tolerance=0.00001, 
    ground_truth=None, plot_grads=True, bounds=True, max_step_ratio=0.5):
    num_elements = data[0].shape[0]
    
    WEIGHT_A = 1.0

    # starting b 
    b = 1.0 / (2*(num_elements-2))

    # bounds avoids values of b, which would cause negative weights
    b_min = 0.
    b_max = 1.0 / (num_elements-2)
    phi_min = 0.
    phi = 1.0

    grad_b_min, _ = MultiRanking(data, Arithmetic(WEIGHT_A, 
        b_min), ground_truth=ground_truth).calc_gradients(b_min, phi)

    grad_b_max, _ = MultiRanking(data, Arithmetic(WEIGHT_A, 
       b_max), ground_truth=ground_truth).calc_gradients(b_max, phi)

    max_grad = max(abs(grad_b_min), abs(grad_b_max))

    alpha =  (b_max * max_step_ratio) / max_grad

    log_likes = np.zeros(num_iter)
    grad_vect = np.zeros((num_iter, 2))
    b_vect = np.zeros(num_iter)
    b_vect[0] = b

    phi_vect = np.zeros(num_iter)
    phi_vect[0] = phi



    # assume that we don't converge
    converge_iter = num_iter-1

    for i in tqdm(range(num_iter)):
        # print(phi)

        # test for convergence
        if (i>0 and abs(b_vect[i]-b_vect[i-1]) < tolerance and abs(phi_vect[i]-phi_vect[i-1]) < tolerance):
            converge_iter = i
            break

        multi_rank_obj = MultiRanking(data, Arithmetic(WEIGHT_A, b), ground_truth=ground_truth, phi=phi)

        log_likes[i] = multi_rank_obj.log_likelihood()
        grad_vect[i] = np.array(multi_rank_obj.calc_gradients(b, phi))

        # Enforcing boundary conditions
        if bounds:
            b_new = b + alpha*grad_vect[i, 0]
            b = max(b_min, b_new)
            b = min(b, b_max)
            phi = max(phi_min, phi + alpha*grad_vect[i, 1])
        else:
            b = b + alpha*grad_vect[i, 0]
            phi = phi + alpha*grad_vect[i, 1]

        if i< num_iter-1:
            b_vect[i+1] = b
            phi_vect[i+1] = phi


    best_b = b_vect[converge_iter]
    best_phi = phi_vect[converge_iter]
    max_likelihood = MultiRanking(data, Arithmetic(WEIGHT_A, best_b), 
        ground_truth=ground_truth, phi=best_phi).log_likelihood()

    if plot_grads:
        plot_graph(b_vect[:(converge_iter)], grad_vect[:(converge_iter), 0], "b-vect", "grad_b")
        plot_graph(b_vect[:(converge_iter)], log_likes[:(converge_iter)], "b-vect", "log-likes")
        plot_graph(phi_vect[:(converge_iter)], grad_vect[:(converge_iter), 1], "phis", "grad_phi")
        plot_graph(phi_vect[:(converge_iter)], log_likes[:(converge_iter)], "phis", "log-likes")

    return best_b, best_phi, max_likelihood

# Utilizes alternating optimization algorithm to find max
def alt_grad_ascent(data, num_iter=100, alpha=0.001, tolerance=0.00001, 
    ground_truth=None, plot_grads=True, bounds=True, max_step_ratio=0.5):
    num_elements = data[0].shape[0]
    
    WEIGHT_A = 1.0

    # starting b 
    b = 1.0 / (2*(num_elements-2))

    # bounds avoids values of b, which would cause negative weights
    b_min = 0.
    b_max = 1.0 / (num_elements-2)
    phi_min = 0.
    phi = 1.0

    grad_b_min, _ = MultiRanking(data, Arithmetic(WEIGHT_A, 
        b_min), ground_truth=ground_truth).calc_gradients(b_min, phi)

    grad_b_max, _ = MultiRanking(data, Arithmetic(WEIGHT_A, 
       b_max), ground_truth=ground_truth).calc_gradients(b_max, phi)

    max_grad = max(abs(grad_b_min), abs(grad_b_max))

    alpha =  (b_max * max_step_ratio) / max_grad

    # log_likes = np.zeros(num_iter)
    grad_vect = np.zeros((num_iter**2, 2))
    b_vect = np.zeros(num_iter**2)
    b_vect[0] = b

    phi_vect = np.zeros(num_iter**2)
    phi_vect[0] = phi

    # count of alternations between optimizing b and optimizing phi
    outer_it = 0 

    # overall number of steps in gradient ascent
    overall_it = 0
    while outer_it < num_iter:
        print("outer_it", outer_it)
        # loop optimizing for b
        inner_it = 0 # number of iterations for optimizing b on this alternation
        while inner_it < num_iter-1:
            
            multi_rank_obj = MultiRanking(data, Arithmetic(WEIGHT_A, b), ground_truth=ground_truth, phi=phi)

            # log_likes[i] = multi_rank_obj.log_likelihood()
            grad_vect[overall_it] = np.array(multi_rank_obj.calc_gradients(b, phi))
            if bounds:
                b_new = b + alpha*grad_vect[overall_it, 0]
                b = max(b_min, b_new)
                b = min(b, b_max)
            else:
                b = b + alpha*grad_vect[overall_it, 0]

            if abs(b-b_vect[overall_it-1]) < tolerance:
                break

            overall_it += 1
            b_vect[overall_it] = b
            phi_vect[overall_it] = phi
            
            inner_it += 1

        outer_it += 1
        # print("inner_it", inner_it)
        # print("opt_b = ", b)

        # testing to see if we didn't update b at all (we have reached a global max)
        if outer_it > 2 and inner_it == 0:
            break

        # loop optimizing for phi
        inner_it = 0 # number of iterations for optimizing phi on this alternation
        while inner_it < num_iter-1:
            
            multi_rank_obj = MultiRanking(data, Arithmetic(WEIGHT_A, b), ground_truth=ground_truth, phi=phi)

            # log_likes[i] = multi_rank_obj.log_likelihood()
            grad_vect[overall_it] = np.array(multi_rank_obj.calc_gradients(b, phi))
            if bounds:
                phi = max(phi_min, phi + alpha*grad_vect[overall_it, 1])
            else:
                phi = phi + alpha*grad_vect[overall_it, 1]

            if abs(phi-phi_vect[overall_it-1]) < tolerance:
                break

            overall_it += 1
            b_vect[overall_it] = b
            phi_vect[overall_it] = phi
            
            inner_it += 1

        outer_it += 1
        # print("inner_it", inner_it)
        # print("opt_phi = ", phi)

        # testing to see if we didn't update phi at all (we have reached a global max)
        if inner_it == 0:
            break


    best_b = b_vect[overall_it]
    best_phi = phi_vect[overall_it]
    max_likelihood = MultiRanking(data, Arithmetic(WEIGHT_A, best_b), 
        ground_truth=ground_truth, phi=best_phi).log_likelihood()

    if plot_grads:
        plot_graph(b_vect[:(overall_it)], grad_vect[:(overall_it), 0], "b-vect", "grad_b")
        # plot_graph(b_vect[:(overall_it)], log_likes[:(overall_it)], "b-vect", "log-likes")
        plot_graph(phi_vect[:(overall_it)], grad_vect[:(overall_it), 1], "phis", "grad_phi")
        # plot_graph(phi_vect[:(overall_it)], log_likes[:(overall_it)], "phis", "log-likes")

    return best_b, best_phi , max_likelihood

# Utilizes alternating optimization algorithm to find max
def alt_grad_ascent_v2(data, num_iter=100, alpha=0.001, tolerance=0.00001, 
    ground_truth=None, plot_grads=True, bounds=True, max_step_ratio=0.5):
    num_elements = data[0].shape[0]
    
    WEIGHT_A = 1.0

    # starting b 
    b = 1.0 / (2*(num_elements-2))

    # bounds avoids values of b, which would cause negative weights
    b_min = 0.
    b_max = 1.0 / (num_elements-2)
    phi_min = 0.
    phi = 1.0

    grad_b_min, _ = MultiRanking(data, Arithmetic(WEIGHT_A, 
        b_min), ground_truth=ground_truth).calc_gradients(b_min, phi)

    grad_b_max, _ = MultiRanking(data, Arithmetic(WEIGHT_A, 
       b_max), ground_truth=ground_truth).calc_gradients(b_max, phi)

    max_grad = max(abs(grad_b_min), abs(grad_b_max))

    alpha =  (b_max * max_step_ratio) / max_grad

    # log_likes = np.zeros(num_iter)
    grad_vect = np.zeros((num_iter**2, 2))
    b_vect = np.zeros(num_iter**2)
    b_vect[0] = b

    phi_vect = np.zeros(num_iter**2)
    phi_vect[0] = phi

    # count of alternations between optimizing b and optimizing phi
    outer_it = 0 

    # overall number of steps in gradient ascent
    overall_it = 0
    while outer_it < num_iter:
        print("outer_it", outer_it)
        # loop optimizing for phi
        inner_it = 0 # number of iterations for optimizing phi on this alternation
        while inner_it < num_iter-1:
            
            multi_rank_obj = MultiRanking(data, Arithmetic(WEIGHT_A, b), ground_truth=ground_truth, phi=phi)

            # log_likes[i] = multi_rank_obj.log_likelihood()
            grad_vect[overall_it] = np.array(multi_rank_obj.calc_gradients(b, phi))
            if bounds:
                phi = max(phi_min, phi + alpha*grad_vect[overall_it, 1])
            else:
                phi = phi + alpha*grad_vect[overall_it, 1]

            if abs(phi-phi_vect[overall_it-1]) < tolerance:
                break

            overall_it += 1
            b_vect[overall_it] = b
            phi_vect[overall_it] = phi
            
            inner_it += 1

        outer_it += 1
        print("inner_it", inner_it)
        print("opt_phi = ", phi)

        # testing to see if we didn't update phi at all (we have reached a global max)
        if outer_it > 2 and inner_it == 0:
            break

        # loop optimizing for b
        inner_it = 0 # number of iterations for optimizing b on this alternation
        while inner_it < num_iter-1:
            
            multi_rank_obj = MultiRanking(data, Arithmetic(WEIGHT_A, b), ground_truth=ground_truth, phi=phi)

            # log_likes[i] = multi_rank_obj.log_likelihood()
            grad_vect[overall_it] = np.array(multi_rank_obj.calc_gradients(b, phi))
            if bounds:
                b_new = b + alpha*grad_vect[overall_it, 0]
                b = max(b_min, b_new)
                b = min(b, b_max)
            else:
                b = b + alpha*grad_vect[overall_it, 0]

            if abs(b-b_vect[overall_it-1]) < tolerance:
                break

            overall_it += 1
            b_vect[overall_it] = b
            phi_vect[overall_it] = phi
            
            inner_it += 1

        outer_it += 1
        print("inner_it", inner_it)
        print("opt_b = ", b)

        # testing to see if we didn't update b at all (we have reached a global max)
        if inner_it == 0:
            break


    best_b = b_vect[overall_it]
    best_phi = phi_vect[overall_it]
    max_likelihood = MultiRanking(data, Arithmetic(WEIGHT_A, best_b), 
        ground_truth=ground_truth, phi=best_phi).log_likelihood()

    if plot_grads:
        plot_graph(b_vect[:(overall_it)], grad_vect[:(overall_it), 0], "b-vect", "grad_b")
        # plot_graph(b_vect[:(overall_it)], log_likes[:(overall_it)], "b-vect", "log-likes")
        plot_graph(phi_vect[:(overall_it)], grad_vect[:(overall_it), 1], "phis", "grad_phi")
        # plot_graph(phi_vect[:(overall_it)], log_likes[:(overall_it)], "phis", "log-likes")

    return best_b, best_phi #, max_likelihood

def generate_simulated_ranks(BASE_LIST, NUM_RANKING_ELEMENTS, WEIGHT_A, WEIGHT_B, NUM_SIM_RANKINGS, data_summary=False):
    weights = Arithmetic(WEIGHT_A, WEIGHT_B)
    all_rankings_data = np.array([])
    perms = list(permutations(BASE_LIST))
    perm_probs = np.zeros(len(perms))

    np.random.seed(42)

    # calculate the probabilities of each permutation
    for i in range(len(perms)):
        rank_i = perms[i]
        rank_as_arr = np.array(rank_i)
        pi_rank = Ranking(len(BASE_LIST), weights, rank_as_arr)
        perm_probs[i] = pi_rank.calc_pi_prob()
    # print(perm_probs)
    # print(np.sum(perm_probs))
    # create list of simulated rankings based on probabilities
    for i in range(NUM_SIM_RANKINGS):
        all_rankings_data = np.append(all_rankings_data, perms[np.random.choice(range(len(perms)), p=perm_probs)])

    all_rankings_data = all_rankings_data.reshape(int(np.size(all_rankings_data)/NUM_RANKING_ELEMENTS), NUM_RANKING_ELEMENTS)

    if data_summary:
        uniques = np.unique(all_rankings_data, axis=0, return_counts=True)
        unique_ranks = uniques[0].astype(int)
        counts = uniques[1]
        for i in range(len(unique_ranks)):
            unique_rank_str = " ".join(unique_ranks[i].astype(str))
            # print("Permutation: " + unique_rank_str + "\t" + " Count: "  + str(counts[i]))


    # Reshape 1D array to different dimensions (Num_rankings by Number of elements in each ranking) and return
    return all_rankings_data



def main():
    ##############################################
    # Simulated Dataset (Fixed_Ground_Truth)
    # WEIGHT_A = 1.0
    # WEIGHT_B = 0.1

    # NUM_RANKING_ELEMENTS = 4

    # NUM_SIM_RANKINGS = 10000
    # BASE_LIST = [i for i in range(1, NUM_RANKING_ELEMENTS+1)]

    # # avoid scientific notation when printing numbers
    # np.set_printoptions(suppress=True)

    # all_rankings_data = generate_simulated_ranks(BASE_LIST, NUM_RANKING_ELEMENTS, 
    #     WEIGHT_A, WEIGHT_B, NUM_SIM_RANKINGS, data_summary=True)

    # actual_best_phi, actual_best_b, actual_max_log_like = exhaustive_search(all_rankings_data, plot_log_likes=True)
    # pred_best_b, pred_best_phi, pred_max_log_like = alt_grad_ascent(all_rankings_data, plot_grads=True)

    # print("Actual best_b:", actual_best_b)
    # print("Actual best_phi:", actual_best_phi)
    # print("Actual max_log_like:", actual_max_log_like)

    # print("Predicted best_b:", pred_best_b)
    # print("Predicted best_phi:", pred_best_phi)
    # print("Predicted max_log_like:", pred_max_log_like)

    ##############################################
    # Simulated Datasets (Across_All_Ground_Truths)
    # WEIGHT_A = 1.0
    # WEIGHT_B = 0.33

    # NUM_RANKING_ELEMENTS = 4

    # NUM_SIM_RANKINGS = 10000
    # BASE_LIST = [i for i in range(1, NUM_RANKING_ELEMENTS+1)]

    # data = generate_simulated_ranks(BASE_LIST, NUM_RANKING_ELEMENTS, 
    #     WEIGHT_A, WEIGHT_B, NUM_SIM_RANKINGS, data_summary=True)

    # first_ranking = data[0]

    # all_ground_truths = np.array(list(permutations(first_ranking))).astype(int)

    # # best b values for each corresponding ground truth
    # best_b_vect = np.zeros(np.shape(all_ground_truths)[0])

    # max_log_likes = np.zeros(np.shape(all_ground_truths)[0])

    # gt_it = 0
    # for gt in tqdm(all_ground_truths):

    #     best_b, max_log_like = grad_ascent(data, ground_truth=gt, plot_grads=False)
    #     max_log_likes[gt_it] = max_log_like
    #     best_b_vect[gt_it] = best_b
    #     gt_it += 1

    # # plot_gt_bars(all_ground_truths, best_b_vect, title="Simulated_Arith_Best_B", xlab="Best B")
    # plot_gt_bars(all_ground_truths, max_log_likes, title="Simulated_Arith_Log_Like", xlab="Max Log Like")



    ##############################################
    # determining log_likelihood of real dataset
    #   arithmetic weights w/o grad descent
    #   assumes identity as ground truth

    # DATASET = "soccerPL"

    # data = get_data(DATASET)

    # # best_b, max_log_like = exhaustive_search(data, plot_log_likes=False)
    # best_b, max_log_like = grad_ascent(data, plot_grads=False)

    # print(best_b)
    # print(max_log_like)

    ##################################
    # Determine max-log-likes and best b's and phi's across different ground_truths
    
    DATASETS = ["APA"]

    for dataset in DATASETS:
        data = get_data(dataset)

        # convert items to ranks and switch order of importance
        if dataset=="idea":
            data = 6-data
            data= np.argsort(data)+1

        # Print freq dist of dataset
        ######
        # print(dataset)
        # uniques = np.unique(data, axis=0, return_counts=True)
        # unique_ranks = uniques[0].astype(int)
        # counts = uniques[1]
        # for i in range(len(unique_ranks)):
        #     unique_rank_str = " ".join(unique_ranks[i].astype(str))
        #     print("Permutation: " + unique_rank_str + "\t" + " Count: "  + str(counts[i]))
        ######

        first_ranking = data[0]

        all_ground_truths = np.array(list(permutations(first_ranking))).astype(int)

        best_b_vect = np.zeros(np.shape(all_ground_truths)[0])
        best_phi_vect = np.zeros(np.shape(all_ground_truths)[0])
        max_log_likes = np.zeros(np.shape(all_ground_truths)[0])

        gt_it = 0
        for gt in tqdm(all_ground_truths):
            best_b, best_phi, max_log_like = alt_grad_ascent(data, ground_truth=gt, plot_grads=False)
            max_log_likes[gt_it] = max_log_like
            best_b_vect[gt_it] = best_b
            best_phi_vect[gt_it] = best_phi
            gt_it += 1

        # # convert to 1-d array to fit in df
        gt_str = ranks_to_str(all_ground_truths)

        # # # Create a dataframe that summarizes best-b and max log-like for each ground truth
        data = {'Ground Truths':gt_str, 'Max Log-Like':max_log_likes, 
        'Best b':best_b_vect, 'Best phi':best_phi_vect}

        gt_df = pd.DataFrame(data=data)

        # # # Move columns in correct order
        gt_df = gt_df[['Ground Truths','Max Log-Like','Best b', 'Best phi']]

        outfile = 'GT_outputs/'+dataset+"_out_alt_grad.csv"

        gt_df.to_csv(outfile)

        # # Plot ground truths in bar charts 
        # # Please do for data with 4 or fewer ranking elements
        # best_b_title="3_19_Plots/"+dataset+"_Arith_Best_B"
        # log_like_title="3_19_Plots/"+dataset+"Uniform_Weights"
        # plot_gt_bars(all_ground_truths, best_b_vect, title=best_b_title, xlab="Best B")
        # plot_gt_bars(all_ground_truths, max_log_likes, title=log_like_title, xlab="Max Log Like")
    

if __name__ == "__main__":
    main()