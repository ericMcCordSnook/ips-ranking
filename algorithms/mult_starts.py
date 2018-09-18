def mult_starts(data, num_iter=100, tolerance=0.0005,
    ground_truth=None, plot_grads=True, bounds=True, max_step_ratio=0.5, num_grid_pts_b=3, num_grid_pts_phi=3):

    num_elements = data[0].shape[0]
    b_min =  0.
    b_max = 1.0 / (num_elements-2)

    # We only want to run multiple starts on the middle parameter values, not the edges
    b_start_vect = np.linspace(b_min, b_max, num_grid_pts_b+2)
    b_start_vect = b_start_vect[1:-1]

    phi_min = 0.
    phi_max = num_grid_pts_phi / 2.0

    phi_start_vect = np.linspace(phi_min, phi_max, num_grid_pts_phi+2)
    phi_start_vect = phi_start_vect[1:-1]

    output = np.zeros((num_grid_pts_b*num_grid_pts_phi, 5))

    i = 0
    for b_start in tqdm(b_start_vect):
        for phi_start in tqdm(phi_start_vect):

            pred_best_b, pred_best_phi, pred_max_log_like = backtrack_line(data, num_iter=num_iter, tolerance=tolerance,
    ground_truth=ground_truth, plot_grads=plot_grads, bounds=bounds, b_start=b_start, phi_start=phi_start)

            output[i] = np.array((b_start, phi_start, pred_best_b, pred_best_phi, pred_max_log_like))

            i+=1

    return output
