from rankobjects.Weight import *
from utils.miscutils import generate_weighted_permutation_graph
# from heuristics.Single_Start_Greedy import get_greedy_traversal_order, get_perms_to_explore
from heuristics import Single_Start_Greedy

class Single_Start_Greedy_Experiment:
    def __init__(self):
        print("New object created: Single_Start_Greedy_Experiment")

    def run(self):
        weight = Arithmetic(a=1.0,b=0.1)
        Single_Start_Greedy = Single_Start_Greedy()
        graph = generate_weighted_permutation_graph(4,weight)
        order = Single_Start_Greedy.get_greedy_traversal_order(graph, (1,2,3,4))
        print(order, "\n")
        order2 = Single_Start_Greedy.get_perms_to_explore(graph, (1,2,3,4), num=10)
        print(order2)
