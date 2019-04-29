import numpy as np
import math
from rankobjects.Weight import Weight, Arithmetic
from rankobjects.Ranking import Ranking

def main():
    weight_obj = Arithmetic(a=1.0, b=0.08)
    weights = weight_obj.generate_weights_vect(12)
    rank_obj_1 = Ranking(12, weight_obj, rank=np.array([1,2,3,4,5,6,7,8,9,10,11,12]))
    rank_obj_2 = Ranking(12, weight_obj, rank=np.array([12,11,10,9,8,7,6,5,4,3,2,1]))
    print(rank_obj_1.calc_dist(rank_obj_2))

if __name__ == '__main__':
    main()
