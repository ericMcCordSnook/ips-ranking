import numpy as np
import math
from itertools import permutations

def create_freq_table(num_rankings=50, num_elements=4):
    base_list = [i for i in range(1, num_elements+1)]
    perms = list(permutations(base_list))
    num_perms = len(perms)
    perm_indices = [i for i in range(num_perms)]
    freq_tbl = {}
    for perm_l in perms:
        freq_tbl[perm_l] = 0
    p_dist = np.random.rand(num_perms)
    p_dist /= np.sum(p_dist)
    for i in range(num_rankings):
        rand_perm_choice = perms[np.random.choice(num_perms, p=p_dist)]
        freq_tbl[rand_perm_choice] += 1
    return freq_tbl

def write_data_to_file(freq_tbl, file_name):
    with open(file_name, 'w') as f:
        f.write("# x,x,x,x\n")
        for perm, freq in freq_tbl.items():
            perm_str = str(perm)[1:-1].replace(' ','')
            for i in range(freq):
                f.write(perm_str + "\n")

def write_config_file(file_name, data_file):
    config_str = """optimization: optimizations.Unweighted
optimization_params: {}
heuristic: heuristics.No_Heuristic
heuristic_params: {}
weight: rankobjects.Arithmetic
data_file: """ + data_file
    with open(file_name, 'w') as f:
        f.write(config_str)

def main():
    num_experiments = 50
    num_rankings = 50
    num_elements = 4
    for exp in range(num_experiments):
        # freq_tbl = create_freq_table(num_rankings=num_rankings, num_elements=num_elements)
        file_name = "config/experiments/random/unweighted/rand_exp_"+str(num_elements)+"_"+str(num_rankings)+"_"+str(exp)+".yml"
        data_file = "data/random/rand_exp_"+str(num_elements)+"_"+str(num_rankings)+"_"+str(exp)+".csv"
        # write_data_to_file(freq_tbl, file_name)
        write_config_file(file_name, data_file)

if __name__ == '__main__':
    main()
