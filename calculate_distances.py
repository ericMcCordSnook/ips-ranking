import numpy as np
import math
from rankobjects.Weight import Weight, Arithmetic
from rankobjects.Ranking import Ranking

def main():
    weight_obj = Arithmetic(a=1.0, b=0.2)
    weights = weight_obj.generate_weights_vect(4)
    rank_obj_1 = Ranking(4, weight_obj)
    rank_obj_2 = Ranking(4, weight_obj)
    rank_obj_1.calc_dist(rank_obj_2)

if __name__ == '__main__':
    main()
