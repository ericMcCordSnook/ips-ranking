from rankobjects.MultiRanking import MultiRanking
import numpy as np

class Grad_Ascent:
    def __init__(self):
        print("Grad_Ascent object created")

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
