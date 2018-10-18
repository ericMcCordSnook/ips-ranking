from abc import abstractmethod

# Heuristics are used to find the optimal ground truth
# They iteratively try possible ground_truths and what differentiates them
# is the way they decide where to go next
class Generic_Heuristic:

    def __init__(self):
        self.num_elements = None
        self.optimization = None
        self.optimization_params = None
        self.weight = None

    def set_optimization_params(self, params):
        self.optimization.set_params(params)

    @abstractmethod
    def set_params(self, params):
        pass

    @abstractmethod
    def run_heuristic(self, params):
        pass
