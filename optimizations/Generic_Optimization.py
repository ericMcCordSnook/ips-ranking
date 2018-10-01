from abc import abstractmethod

# Used to find the values of the parameters phi, b (and optionally a)
# which maximize the log-likelihood of the given ranking being the
# ground_truth of the data set
class Generic_Optimization:

    def __init__(self):
        self.data = None
        self.ground_truth = None
        self.weight = None

    @abstractmethod
    def set_params(self):
        pass

    @abstractmethod
    def optimize(self):
        pass
