from rankobjects.Weight import *
from rankobjects.Ranking import Ranking
import numpy as np
from itertools import permutations

# Read in the data from given file
def read_csv(file, has_header=True):
    data = np.loadtxt(file, dtype=float, delimiter=',', skiprows=has_header)
    return data

# Gets data from CSV file and converts it appropriately
def get_data(data_file):
    return read_csv(data_file).astype(int)

# Convert list of ranks to concatenated string versions
# (e.g. [[1,2,4,3],[2,4,3,1]] --> ["1243", "2431"])
def ranks_to_str(all_ranks):
    num_rows = all_ranks.shape[0]
    ranks_str = np.empty(num_rows).astype(str)
    for i in range(num_rows):
        ranks_str[i] = "".join(all_ranks[i].astype(str))
    return ranks_str

def generate_weighted_permutation_graph(NUM_RANKING_ELEMENTS, weight):
    if not isinstance(weight, Weight):
        raise TypeError("weight argument is not a Weight object")
    IDENTITY_LIST = [i for i in range(1,NUM_RANKING_ELEMENTS+1)]
    weights = weight.generate_weights_vect(NUM_RANKING_ELEMENTS)
    perms = list(permutations(IDENTITY_LIST))
    graph = {}
    for perm in perms:
        graph[perm] = {}
        for i in range(NUM_RANKING_ELEMENTS-1):
            perm_copy = list(perm)
            perm_copy[i], perm_copy[i+1] = perm_copy[i+1],perm_copy[i]
            graph[perm][tuple(perm_copy)] = weights[i]
    return graph
