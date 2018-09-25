from rankobjects.MultiRanking import MultiRanking
import numpy as np

class Exhaustive_Search:
    def __init__(self):
        print("Exhaustive_Search object created")

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
