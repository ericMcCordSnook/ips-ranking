from rankobjects import *
import numpy as np
from itertools import permutations

class Data_Simulator:

    def __init__(self, data):
        weight_types = {"Arithmetic": Arithmetic, "Geometric": Geometric}
        self.num_elements = data["num_elements"]
        self.num_rankings = data["num_rankings"]
        self.weight_a = data["weight_a"]
        self.weight_b = data["weight_b"]
        self.weight = weight_types[data["weight_type"]](self.weight_a, self.weight_b)
        self.ground_truth = np.array(list(map(int, list(data["ground_truth"]))))
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

        # calculate the probabilities of each permutation
        for i in range(len(perms)):
            rank_i = perms[i]
            rank_as_arr = np.array(rank_i)
            pi_rank = Ranking(len(BASE_LIST), self.weight, rank_as_arr, ground_truth=self.ground_truth)
            perm_probs[i] = pi_rank.calc_pi_prob()
        # create list of simulated rankings based on probabilities
        for i in range(self.num_rankings):
            all_rankings_data = np.append(all_rankings_data, perms[np.random.choice(range(len(perms)), p=perm_probs)])

        # Reshape 1D array to different dimensions (Num_rankings by Number of elements in each ranking) and return
        all_rankings_data = all_rankings_data.reshape(self.num_rankings, int(np.size(all_rankings_data)/self.num_rankings))

        if data_summary:
            uniques = np.unique(all_rankings_data, axis=0, return_counts=True)
            unique_ranks = uniques[0].astype(int)
            counts = uniques[1]
            for i in range(len(unique_ranks)):
                unique_rank_str = " ".join(unique_ranks[i].astype(str))
                # self.logger.info("Permutation: " + unique_rank_str + "\t" + " Count: "  + str(counts[i]))

        return all_rankings_data
