from utils.miscutils import generate_weighted_permutation_graph
from math import ceil, factorial
import sys
from copy import deepcopy

# Try making a class that has all the heuristics objects then create generic
# heuristics object by doing getattr(HeuristicsClass, "Unimode_Greedy")

class Unimode_Greedy:
    def __init__(self):
        print("New object created: Unimode_Greedy")

    def get_greedy_traversal_order(self, graph, start_node):
        order = [start_node]
        visited = set()
        visited.add(start_node)
        current_nodes = deepcopy(graph[start_node])
        while bool(current_nodes): # tests if empty
            min_dist_node = None
            min_dist = sys.maxsize
            for node, dist in current_nodes.items():
                if dist < min_dist:
                    min_dist_node = node
                    min_dist = dist
            visited.add(min_dist_node)
            order.append(min_dist_node)
            for node, dist in graph[min_dist_node].items():
                if node not in visited:
                    if (node not in current_nodes) or (min_dist + dist < current_nodes[node]):
                        current_nodes[node] = min_dist + min_dist
            del current_nodes[min_dist_node]
        return order

    # All heuristics will have a method of the same name
    def get_perms_to_explore(self, graph, mode_ranking, pct=0.2, num=None):
        num_to_return = 0
        if num is not None:
            if not isinstance(num, int):
                raise ValueError("num argument must be an integer")
            else:
                num_to_return = num
        else:
            num_to_return = int(ceil(pct * factorial(len(mode_ranking))))
        return self.get_greedy_traversal_order(graph, mode_ranking)[:num_to_return]
