from experiments.run_experiment import run
from rankobjects.weight import *
from utils.miscutils import generate_weighted_permutation_graph
from heuristics.unimode_greedy import get_greedy_traversal_order, get_perms_to_explore

if __name__ == '__main__':
    weight = Arithmetic(a=1.0,b=0.1)
    graph = generate_weighted_permutation_graph(7,weight)
    order = get_greedy_traversal_order(graph, (1,2,3,4,5,6,7))
    print(order, "\n")
    order2 = get_perms_to_explore(graph, (1,2,3,4,5,6,7), num=10)
    print(order2)

# if __name__ == '__main__':
#     run()
