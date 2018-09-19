from experiments.run_experiment import run
from rankobjects.weight import *
from utils.miscutils import generate_weighted_permutation_graph
from heuristics.unimode_greedy import get_greedy_traversal_order

if __name__ == '__main__':
    weight = Arithmetic(a=1.0,b=0.2)
    graph = generate_weighted_permutation_graph(4,weight)
    order = get_greedy_traversal_order(graph, (1,2,3,4))
    print(order)

# if __name__ == '__main__':
#     run()
