from rankobjects import *
import numpy as np
from itertools import permutations
import logging

class Data_Simulator:

    def __init__(self, data):
        self.weight_types = {"Arithmetic": Arithmetic, "Geometric": Geometric}
        self.num_elements = data["num_elements"]
        self.num_rankings = data["num_rankings"]
        self.weights = data["weights"]
        self.ground_truths = data["ground_truths"]
        self.proportions = data["proportions"]
        self.filepath = data["filepath"]

    def simulate(self):
        all_rankings_data = self.generate_simulated_data()
        header_str = 'x,' * (self.num_elements-1) + 'x'
        fmt_str = '%i,' * (self.num_elements-1) + '%i'
        np.savetxt(self.filepath, all_rankings_data.astype(int), header=header_str, fmt=fmt_str, delimiter='\n')

    def generate_simulated_data(self, data_summary=True):
        all_rankings_data = np.array([])
        BASE_LIST = [i for i in range(1, self.num_elements+1)]
        perms = list(permutations(BASE_LIST))
        perm_probs = np.zeros(len(perms))

        for i, ground_truth_str in enumerate(self.ground_truths):
            cur_ground_truth = np.array(list(map(int, list(ground_truth_str))))
            cur_weight = self.weights[i]
            cur_proportion = self.proportions[i]
            cur_a = float(cur_weight[1])
            cur_b = float(cur_weight[2])
            cur_weight_obj = self.weight_types[cur_weight[0]](cur_a, cur_b)

            # calculate the probabilities of each permutation
            for i in range(len(perms)):
                rank_i = perms[i]
                rank_as_arr = np.array(rank_i)
                pi_rank = Ranking(len(BASE_LIST), cur_weight_obj, rank_as_arr, ground_truth=cur_ground_truth)
                perm_probs[i] = pi_rank.calc_pi_prob()

            # create list of simulated rankings based on probabilities
            cur_num_rankings = int(cur_proportion * self.num_rankings)
            for i in range(cur_num_rankings): # TODO maybe change the line below to be more efficient
                all_rankings_data = np.append(all_rankings_data, perms[np.random.choice(range(len(perms)), p=perm_probs)])


        # Reshape 1D array to different dimensions (Num_rankings by Number of elements in each ranking) and return
        actual_num_rankings = int(len(all_rankings_data) / self.num_elements)
        all_rankings_data = all_rankings_data.reshape((actual_num_rankings, self.num_elements))

        if data_summary:
            uniques = np.unique(all_rankings_data, axis=0, return_counts=True)
            unique_ranks = uniques[0].astype(int)
            counts = uniques[1]
            for i in range(len(unique_ranks)):
                unique_rank_str = " ".join(unique_ranks[i].astype(str))
                logging.info("Permutation: %s \t Count: %d" % (unique_rank_str, counts[i]))

        return all_rankings_data
