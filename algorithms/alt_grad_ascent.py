from rankobjects.MultiRanking import MultiRanking
from numpy import np

class Alt_Grad_Ascent:
    def __init__(self):
        print("Alt_Grad_Ascent object created")

    # Utilizes alternating optimization algorithm to find max
    def alt_grad_ascent(data, num_iter=100, alpha=0.001, tolerance=0.00001,
        ground_truth=None, plot_grads=True, bounds=True, max_step_ratio=0.5, b_start=None, phi_start=None):
        num_elements = data[0].shape[0]

        WEIGHT_A = 1.0

        # starting b
        if b_start:
            b = b_start
        else:
            b = 1.0 / (2*(num_elements-2))

        # bounds avoids values of b, which would cause negative weights
        b_min = 0.
        b_max = 1.0 / (num_elements-2)
        phi_min = 0.

        if phi_start:
            phi = phi_start
        else:
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

        # progress bar
        pbar = tqdm(total=num_iter)

        while outer_it < num_iter:
            # print("outer_it", outer_it)
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
            pbar.update(1)


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
            pbar.update(1)

            # testing to see if we didn't update phi at all (we have reached a global max)
            if inner_it == 0:
                break

        pbar.close()
        best_b = b_vect[overall_it]
        best_phi = phi_vect[overall_it]
        max_likelihood = MultiRanking(data, Arithmetic(WEIGHT_A, best_b),
            ground_truth=ground_truth, phi=best_phi).log_likelihood()

        if plot_grads:
            plot_graph(b_vect[:(overall_it)], grad_vect[:(overall_it), 0], "b-vect", "grad_b")
            plot_graph(phi_vect[:(overall_it)], grad_vect[:(overall_it), 1], "phis", "grad_phi")

        return best_b, best_phi, max_likelihood
