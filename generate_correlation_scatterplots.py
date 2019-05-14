import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from rankobjects.Weight import Unweighted, Arithmetic, Geometric, Harmonic
from rankobjects.Ranking import Ranking

def main():
    x = np.random.rand(10)
    y = np.random.rand(10)
    plt.scatter(x,y)
    plt.show()

if __name__ == '__main__':
    main()
