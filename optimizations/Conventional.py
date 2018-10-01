from rankobjects.MultiRanking import MultiRanking

class Conventional:
    def __init__(self):
        print("Conventional optimization created")

    # assuming weights are uniform, find the best ground-truth
    def conventional(data, ground_truth=None):
        multi_rank_obj = MultiRanking(data, Arithmetic(WEIGHT_A, WEIGHT_B), ground_truth=ground_truth)
        max_likelihood = multi_rank_obj.log_likelihood()
        return max_likelihood
