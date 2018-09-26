from rankobjects.MultiRanking import MultiRanking
import numpy as np

class Backtrack_Line:
    def __init__(self):
        print("Backtrack_Line object created")

    # TODO: modularize all this code and optimize

    # Extends alternating gradient descent and uses backtracking algorithm to find the best learning rate
    def backtrack_line(data, num_iter=100, tolerance=0.0005, ground_truth=None,
        plot_grads=True, bounds=True, b_start=None, phi_start=None, unweighted=False):
        num_elements = data[0].shape[0]

        WEIGHT_A = 1.0

        if not unweighted:
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

        grad_vect = np.zeros((num_iter**2, 2))
        b_vect = np.zeros(num_iter**2)
        if not unweighted:
            b_vect[0] = b

        phi_vect = np.zeros(num_iter**2)
        phi_vect[0] = phi

        # count of alternations between optimizing b and optimizing phi
        outer_it = 0

        # overall number of steps in gradient ascent
        overall_it = 0

        # progress bar
        # pbar = tqdm(total=num_iter)

        while outer_it < num_iter:
            # print("outer_it", outer_it)
            # loop optimizing for b
            inner_it = 0 # number of iterations for optimizing b on this alternation
            while inner_it < num_iter-1:

                multi_rank_obj = MultiRanking(data, Arithmetic(WEIGHT_A, b), ground_truth=ground_truth, phi=phi)

                # log_likes[i] = multi_rank_obj.log_likelihood()
                grad_vect[overall_it] = np.array(multi_rank_obj.calc_gradients(b, phi))
                if bounds:

                    alpha_continue = True
                    alpha = 0.01
                    while(alpha_continue):
                        b_cur_alpha = b + alpha*grad_vect[overall_it, 0]
                        b_cur_alpha = min(max(b_min, b_cur_alpha), b_max)

                        new_alpha = alpha / 2
                        b_new_alpha = b + new_alpha*grad_vect[overall_it, 0]
                        b_new_alpha = min(max(b_min, b_new_alpha), b_max)

                        multi_rank_obj_b_cur = MultiRanking(data, Arithmetic(WEIGHT_A, b_cur_alpha), ground_truth=ground_truth, phi=phi)
                        multi_rank_obj_b_new = MultiRanking(data, Arithmetic(WEIGHT_A, b_new_alpha), ground_truth=ground_truth, phi=phi)

                        b_cur_log_like = multi_rank_obj_b_cur.log_likelihood()
                        b_new_log_like = multi_rank_obj_b_new.log_likelihood()

                        if b_cur_log_like >= b_new_log_like:
                            # Stop changing alpha
                            b = b_cur_alpha
                            grad_vect[overall_it+1] = np.array(multi_rank_obj_b_cur.calc_gradients(b, phi))
                            b_vect[overall_it+1] = b
                            phi_vect[overall_it+1] = phi

                            alpha_continue = False
                        else:
                            alpha = new_alpha

                if abs(b-b_vect[overall_it]) < tolerance:
                    # print("b", b)
                    # print("b_vect", b_vect[overall_it])
                    # print("Hello")
                    break

                overall_it += 1
                inner_it += 1

            outer_it += 1
            # pbar.update(1)
            # print("inner_it", inner_it)
            # print("opt_b = ", b)

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

                    alpha_continue = True
                    alpha = 0.01
                    while(alpha_continue):
                        phi_cur_alpha = phi + alpha*grad_vect[overall_it, 1]
                        phi_cur_alpha = max(phi_min, phi_cur_alpha)

                        new_alpha = alpha / 2
                        phi_new_alpha = phi + new_alpha*grad_vect[overall_it, 1]
                        phi_new_alpha = max(phi_min, phi_new_alpha)

                        multi_rank_obj_phi_cur = MultiRanking(data, Arithmetic(WEIGHT_A, b), ground_truth=ground_truth, phi=phi_cur_alpha)
                        multi_rank_obj_phi_new = MultiRanking(data, Arithmetic(WEIGHT_A, b), ground_truth=ground_truth, phi=phi_new_alpha)

                        phi_cur_log_like = multi_rank_obj_phi_cur.log_likelihood()
                        phi_new_log_like = multi_rank_obj_phi_new.log_likelihood()

                        if phi_cur_log_like >= phi_new_log_like:
                            # Stop changing alpha
                            phi = phi_cur_alpha
                            grad_vect[overall_it+1] = np.array(multi_rank_obj_phi_cur.calc_gradients(b, phi))
                            b_vect[overall_it+1] = b
                            phi_vect[overall_it+1] = phi

                            alpha_continue = False
                        else:
                            alpha = new_alpha

                if abs(phi-phi_vect[overall_it]) < tolerance:
                    # print("Is it me you're looking for?")
                    break

                overall_it += 1
                inner_it += 1

            outer_it += 1
            # pbar.update(1)
            # print("inner_it", inner_it)
            # print("opt_phi = ", phi)

            # testing to see if we didn't update phi at all (we have reached a global max)
            if inner_it == 0:
                # print("Eric says Hello too.")
                break

        # pbar.close()
        best_b = b_vect[overall_it]
        best_phi = phi_vect[overall_it]
        max_likelihood = MultiRanking(data, Arithmetic(WEIGHT_A, best_b),
            ground_truth=ground_truth, phi=best_phi).log_likelihood()

        if plot_grads:
            plot_graph(b_vect[:(overall_it)], grad_vect[:(overall_it), 0], "b-vect", "grad_b")
            # plot_graph(b_vect[:(overall_it)], log_likes[:(overall_it)], "b-vect", "log-likes")
            plot_graph(phi_vect[:(overall_it)], grad_vect[:(overall_it), 1], "phis", "grad_phi")
            # plot_graph(phi_vect[:(overall_it)], log_likes[:(overall_it)], "phis", "log-likes")

        return best_b, best_phi, max_likelihood
