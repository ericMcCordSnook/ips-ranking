import numpy as np
from abc import abstractmethod

# Weights hold information of a two-parameter (a, b) weighting mechanism for rankings
class Weight:
    def __init__(self, a=1., b=0.2, cutoff=None):
        if not isinstance(a, float):
            raise TypeError("First argument, 'a', is not a float.")
        if not isinstance(b, float):
            raise TypeError("Second argument, 'b', is not a float.")

        self.a = a
        self.b = b
        self.cutoff = cutoff # This is the number of items AFTER WHICH weights are all 0
        self.sequence = None

    @abstractmethod
    def generate_weights_vect(self, num_elements):
        pass

    @abstractmethod
    def calc_weight_sum(self, j, delt):
        pass

# Unweighted weights are a Weight in which weights are all equal, i.e., all weights are 1
class Unweighted(Weight):
    def __init__(self, a=0.0, b=0.0, cutoff=None):
        super().__init__(a=a, b=b, cutoff=cutoff)
        self.sequence = "unweighted"

    def generate_weights_vect(self, num_elements):
        weights = np.ones(num_elements-1)
        if self.cutoff is not None:
            weights[self.cutoff:] = 0.0
        return weights

    def calc_weight_sum(self, j, delt):
        if self.cutoff is not None:
            if j > self.cutoff:
                return 0.0
            elif j + delt - 1 >= self.cutoff:
                delt = self.cutoff - j
        return delt

# Arithmetic weights are a Weight in which weights are determined arithmetically
# of the form: a-l*b, where l is the position of the identity vector of size num_elements
class Arithmetic(Weight):
    def __init__(self, a=1.0, b=0.2, cutoff=None):
        super().__init__(a=a, b=b, cutoff=cutoff)
        self.sequence = "arithmetic"

    def generate_weights_vect(self, num_elements):
        # weights are 1 less than num_elements
        indices = np.arange(0, num_elements-1)
        weights = self.a-indices*self.b
        if self.cutoff is not None:
            weights[self.cutoff:] = 0.0
        return weights

    def calc_weight_sum(self, j, delt):
        if self.cutoff is not None:
            if j >= self.cutoff:
                return 0.0
            elif j + delt - 1 >= self.cutoff:
                delt = self.cutoff - j
        return self.a*delt - self.b*delt*(j-0.5) - 0.5*self.b*delt*delt


# Geometric weights are a Weight in which weights are determined geometrically
# of the form: a(b**l), where l is the position of the identity vector of size num_elements
class Geometric(Weight):
    def __init__(self, a=1.0, b=0.8, cutoff=None):
        super().__init__(a=a, b=b, cutoff=cutoff)
        self.sequence = "geometric"

    def generate_weights_vect(self, num_elements):
        indices = np.arange(1, num_elements)
        weights = self.a*(self.b ** indices)
        if self.cutoff is not None:
            weights[self.cutoff:] = 0.0
        return weights

    def calc_weight_sum(self, j, delt):
        if self.cutoff is not None:
            if j >= self.cutoff:
                return 0.0
            elif j + delt - 1 >= self.cutoff:
                delt = self.cutoff - j
        return (self.a*(self.b ** j)*(1 - (self.b ** delt))) / (1-self.b)


# Harmonic weights are a Weight in which weights are determined harmonically
# of the form: 1/l, where l is the position of the identity vector of size num_elements
class Harmonic(Weight):
    def __init__(self, a=1.0, b=0.0, cutoff=None):
        super().__init__(a=a, b=b, cutoff=cutoff)
        self.sequence  = "harmonic"

    def generate_weights_vect(self, num_elements):
        weights = np.array([1.0/i for i in range(1, num_elements+1)])
        if self.cutoff is not None:
            weights[self.cutoff:] = 0.0
        return weights

    def calc_weight_sum(self, j, delt):
        s = 0
        if self.cutoff is None:
            for l in range(j+1, j + delt + 1):
                s += 1.0/l
        else:
            for l in range(j+1, min(j + delt + 1, self.cutoff)):
                s += 1.0/l
        return s
