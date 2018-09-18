from rankobjects.weight import Arithmetic
from rankobjects.ranking import Ranking
import numpy as np
from itertools import permutations

# Read in the data from given file
def read_csv(file, has_header=True):
    data = np.loadtxt(file, dtype=float, delimiter=',', skiprows=has_header)
    return data

# Gets data from CSV file and converts it appropriately
def get_data(dataset, folder="../datasets/"):
    data_file = folder+dataset+".csv"
    return read_csv(data_file).astype(int)

# Convert list of ranks to concatenated string versions
# (e.g. [[1,2,4,3],[2,4,3,1]] --> ["1243", "2431"])
def ranks_to_str(all_ranks):
    num_rows = all_ranks.shape[0]
    ranks_str = np.empty(num_rows).astype(str)
    for i in range(num_rows):
        ranks_str[i] = "".join(all_ranks[i].astype(str))
    return ranks_str

def generate_simulated_ranks(BASE_LIST, NUM_RANKING_ELEMENTS, WEIGHT_A, WEIGHT_B, NUM_SIM_RANKINGS, data_summary=False, ground_truth=None):
    weights = Arithmetic(WEIGHT_A, WEIGHT_B)
    all_rankings_data = np.array([])
    perms = list(permutations(BASE_LIST))
    perm_probs = np.zeros(len(perms))

    # calculate the probabilities of each permutation
    for i in range(len(perms)):
        rank_i = perms[i]
        rank_as_arr = np.array(rank_i)
        pi_rank = Ranking(len(BASE_LIST), weights, rank_as_arr, ground_truth=ground_truth)
        perm_probs[i] = pi_rank.calc_pi_prob()
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
            print("Permutation: " + unique_rank_str + "\t" + " Count: "  + str(counts[i]))

    # Reshape 1D array to different dimensions (Num_rankings by Number of elements in each ranking) and return
    return all_rankings_data
