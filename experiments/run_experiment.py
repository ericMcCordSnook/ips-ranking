from rankobjects.multiranking import MultiRanking
from rankobjects.weight import Arithmetic
from utils.miscutils import generate_simulated_ranks

import numpy as np
from tqdm import tqdm
import time

def run():
    ground_truth = np.array([1,2,3])
    data = generate_simulated_ranks([1,2,3], 3, 1., 0., 100, data_summary=False, ground_truth=np.array([1,2,3]))
    multi_rank_obj = MultiRanking(data, Arithmetic(1., 0.), ground_truth=ground_truth)
    print("Success... we think")

# def run():
#     ##############################################
#     # Simulated Dataset (Fixed_Ground_Truth)
#     WEIGHT_A = 1.0
#
#     NUM_RANKING_ELEMENTS = 3
#     # np.random.seed(42)
#     NUM_SIM_RANKINGS = 100
#     BASE_LIST = [i for i in range(1, NUM_RANKING_ELEMENTS+1)]
#
#     # avoid scientific notation when printing numbers
#     np.set_printoptions(suppress=True)
#     # all_b_values = [0.0, 0.1,0.2,0.3,0.4,0.5]
#     all_b_values = [0.20]
#     num_rows = len(all_b_values)
#
#     NUM_TRIALS = 3
#
#     all_b_arr = np.array(all_b_values)
#
#     # Each row corresponds to all_b_values
#     # first column is mean across number of trials
#     # second column is SD across number of trials
#     time_summary = np.zeros((num_rows, 2))
#     pred_b_summary = np.zeros((num_rows,2))
#     pred_phi_summary = np.zeros((num_rows,2))
#     log_like_summary = np.zeros((num_rows, 2))
#
#
#     for j, b in enumerate(tqdm(all_b_arr)):
#
#         time_vect = np.zeros(NUM_TRIALS)
#         pred_b_vect = np.zeros(NUM_TRIALS)
#         pred_phi_vect = np.zeros(NUM_TRIALS)
#         max_log_like_vect = np.zeros(NUM_TRIALS)
#
#         for i in tqdm(range(NUM_TRIALS)):
#             t0 = time.time()
#
#             all_rankings_data = generate_simulated_ranks(BASE_LIST, NUM_RANKING_ELEMENTS,
#                 WEIGHT_A, b, NUM_SIM_RANKINGS, data_summary=False, ground_truth=np.array(BASE_LIST))
#
#             pred_best_b, pred_best_phi, pred_max_log_like = backtrack_line(all_rankings_data, plot_grads=False)
#
#             t1 = time.time()
#
#             pred_b_vect[i] = pred_best_b
#             pred_phi_vect[i] = pred_best_phi
#             max_log_like_vect[i] = pred_max_log_like
#             time_vect[i] = t1-t0
#
#         # print(pred_b_vect)
#         # print(pred_phi_vect)
#         # print(max_log_like_vect)
#         pred_phi_summary[j] = np.mean(pred_phi_vect), np.std(pred_phi_vect)
#         pred_b_summary[j] = np.mean(pred_b_vect), np.std(pred_b_vect)
#         log_like_summary[j] = np.mean(max_log_like_vect), np.std(max_log_like_vect)
#         time_summary[j] = np.mean(time_vect), np.std(time_vect)
#
#     all_summary = np.hstack((time_summary,pred_b_summary, pred_phi_summary, log_like_summary))
#     all_b_arr = np.expand_dims(all_b_arr, axis=1)
#     output = np.concatenate((all_b_arr,all_summary), axis=1)
#     head_str = "actual_b_vals,time_avg,time_std,pred_b_avg,pred_b_std,pred_phi_avg,pred_phi_std,log_like_avg,log_like_std"
#
#     np.savetxt("b_experiment.csv", output, delimiter=",", header=head_str, comments='')
#
#     print("Done!")
