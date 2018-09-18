import matplotlib.pyplot as plt

# TODO: clean up these functions

# Plot graph of possible b's and their respective log-likelihood (a is fixed)
def plot_graph(x, y, x_label, y_label, title=None, xlim=None, ylim=None):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if xlim and len(xlim) == 2:
        plt.xlim(xlim[0], xlim[1])
    if ylim and len(ylim) == 2:
        plt.ylim(ylim[0], ylim[1])
    if title:
        plt.title(title)
    plt.show()

# Plot contour likelihood graph of possible phi's and b's
def plot_2D_log_like(phi_arr, b_arr, log_like, dataset, \
    num_contours=20, viz_folder="Plots", ground_truth=None):
    plt.figure(figsize=(20,10))
    plt.contourf(b_arr,phi_arr,log_like,num_contours)
    cb = plt.colorbar()
    plt.xlabel("b")
    plt.ylabel("phi")
    # plt.ylim(ymax=b_arr[-1])
    if ground_truth is None:
        plt.title(dataset)
        plt.savefig(viz_folder + "/" + dataset)
    else:
        gt_str = "".join(ground_truth.astype(str))
        plt.title(dataset + " " + " GT " + gt_str)
        plt.savefig(viz_folder + "/" + dataset
            + " GT " + gt_str)
    cb.remove()

# Plot bar chart of ground truths and respective max log-likelihoods
def plot_gt_bars(all_ground_truths, max_log_likes, title, viz_folder="Plots",
    xlab="Max Log like", ylab="Ground Truths"):
    all_gt_str = ranks_to_str(all_ground_truths)

    plt.barh(all_gt_str, max_log_likes, align='center')
    plt.ylabel(ylab)
    # plt.xticks(rotation=90)
    plt.xlabel(xlab)
    plt.savefig(viz_folder + "/" + title)
    plt.clf()
    # plt.show()
