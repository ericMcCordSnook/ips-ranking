import os
import sys
import random
import numpy as np
from scipy.special import comb
from rankobjects.Weight import Unweighted, Arithmetic, Geometric, Harmonic
from rankobjects.Ranking import Ranking

# Read in the data from given file
def read_csv(file_path, has_header=True):
    data = np.loadtxt(file_path, dtype=float, delimiter=',', skiprows=has_header)
    return data.astype(int)

def read_cmd_args():
    try:
        num_items = int(sys.argv[1])
        weight_type = sys.argv[2]
        if weight_type not in ["Arithmetic", "Geometric", "Harmonic"]:
            raise Exception()
        b = float(sys.argv[3])
    except:
        print("Usage: python generate_distance_to_correlation_piecewise.py <num_items> <weight_type> <b>")
        sys.exit(0)
    return num_items, weight_type, b

def compute_avg_and_max(num_items, weight_type, b=0.0, cutoff=None):
    weight_obj = None
    if weight_type == "Unweighted":
        if cutoff is not None:
            weight_obj = Unweighted(a=1.0, b=b, cutoff=cutoff)
        else:
            max_dist = comb(num_items, 2)
            avg = max_dist / 2
            return avg, max_dist
    elif weight_type == "Arithmetic":
        weight_obj = Arithmetic(a=1.0, b=b, cutoff=cutoff)
    elif weight_type == "Geometric":
        weight_obj = Geometric(a=1.0, b=b, cutoff=cutoff)
    elif weight_type == "Harmonic":
        weight_obj = Harmonic(a=1.0, b=b, cutoff=cutoff)
    rank_obj_1 = Ranking(num_items, weight_obj, rank=np.array([i for i in range(1, num_items+1)]))
    rank_obj_2 = Ranking(num_items, weight_obj, rank=np.array([i for i in range(num_items, 0, -1)]))
    max_dist = round(rank_obj_1.calc_dist(rank_obj_2), 3)
    num_trials = 1000
    dists = np.empty((num_trials))
    for i in range(num_trials):
        rank_obj_1.shuffle_ranking()
        rank_obj_2.shuffle_ranking()
        dists[i] = rank_obj_1.calc_dist(rank_obj_2)
    avg = round(np.average(dists), 3)
    return avg, max_dist

def compute_correlation_from_dist(dist, avg, max_dist):
    if dist <= avg:
        return 1 - dist/avg
    else:
        return -1*(dist-avg)/(max_dist-avg)

def main():
    with open("output/preflib/web_search_correlations_harmonic_cutoff_30.log", 'w') as f:
        weight_type = "Harmonic"
        b = 0.0
        cutoff = 30
        f.write('weight_type: ' + weight_type + "\n")
        f.write('b: ' + str(b) + "\n")
        f.write('cutoff: ' + str(cutoff) + "\n\n")
        for file_name in os.listdir("data/preflib/web_search"):
            print(file_name)
            f.write(file_name + "\n")
            data = read_csv("data/preflib/web_search/" + file_name)
            num_rankings = np.shape(data)[0]
            num_items = np.shape(data)[1]
            f.write("num_items: " + str(num_items) + "\n")
            f.write("num_rankings: " + str(num_rankings) + "\n")
            avg_weighted, max_dist_weighted = compute_avg_and_max(num_items, weight_type, b=b, cutoff=(cutoff if cutoff is not None and cutoff<num_items else None))
            avg_unweighted, max_dist_unweighted = compute_avg_and_max(num_items, "Unweighted", b=0.0, cutoff=(cutoff if cutoff is not None and cutoff<num_items else None))
            f.write("avg_weighted, max_dist_weighted: " + str(avg_weighted) + ", " + str(max_dist_weighted) + "\n")
            f.write("avg_unweighted, max_dist_unweighted: " + str(avg_unweighted) + ", " + str(max_dist_unweighted) + "\n")

            # Computing the average correlation between two rankings in dataset
            corrs_weighted = []
            corrs_unweighted = []
            for i in range(num_rankings):
                for j in range(i+1, num_rankings):
                    rank_obj_1 = Ranking(num_items, Harmonic(a=1.0, b=b, cutoff=cutoff), rank=data[i])
                    rank_obj_2 = Ranking(num_items, Harmonic(a=1.0, b=b, cutoff=cutoff), rank=data[j])
                    corrs_weighted.append(compute_correlation_from_dist(rank_obj_1.calc_dist(rank_obj_2), avg_weighted, max_dist_weighted))
                    rank_obj_3 = Ranking(num_items, Unweighted(a=1.0, b=0.0), rank=data[i])
                    rank_obj_4 = Ranking(num_items, Unweighted(a=1.0, b=0.0), rank=data[j])
                    corrs_unweighted.append(compute_correlation_from_dist(rank_obj_3.calc_dist(rank_obj_4), avg_unweighted, max_dist_unweighted))
            avg_corr_weighted = sum(corrs_weighted)/len(corrs_weighted)
            avg_corr_unweighted = sum(corrs_unweighted)/len(corrs_unweighted)
            f.write("avg_corr_weighted: " + str(avg_corr_weighted) + "\n")
            f.write("avg_corr_unweighted: " + str(avg_corr_unweighted) + "\n")
            f.write("\n")

if __name__ == '__main__':
    main()
