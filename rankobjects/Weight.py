import numpy as np
from abc import abstractmethod

# Weights hold information of a two-parameter (a, b) weighting mechanism for rankings
class Weight:
    def __init__(self, a=1., b=0.2):
        if not isinstance(a, float):
            raise TypeError("First argument, 'a', is not a float.")
        if not isinstance(b, float):
            raise TypeError("Second argument, 'b', is not a float.")

        self.a = a
        self.b = b
        self.sequence = None

    @abstractmethod
    def generate_weights_vect(self, num_elements):
        pass

    @abstractmethod
    def calc_weight_sum(self, j, delt):
        pass


# Arithmetic weights are a Weight in which weights are determined arithmetically
# of the form: a-l*b, where l is the position of the identity vector of size num_elements
class Arithmetic(Weight):
    def __init__(self, a=1.0, b=0.2):
        super().__init__(a=a, b=b)
        self.sequence = "arithmetic"

    def generate_weights_vect(self, num_elements):
        # weights are 1 less than num_elements
        indices = np.arange(0, num_elements-1)
        return self.a-indices*self.b

    def calc_weight_sum(self, j, delt):
        return self.a*delt - self.b*delt*(j-0.5) - 0.5*self.b*delt*delt


# Geometric weights are a Weight in which weights are determined geometrically
# of the form: a(b**l), where l is the position of the identity vector of size num_elements
class Geometric(Weight):
    def __init__(self, a=1.0, b=0.8):
        super().__init__(a=a, b=b)
        self.sequence = "geometric"

    def generate_weights_vect(self, num_elements):
        indices = np.arange(1, num_elements)
        return self.a*(self.b ** indices)

    def calc_weight_sum(self, j, delt):
        return (self.a*(self.b ** j)*(1 - (self.b ** delt))) / (1-self.b)
