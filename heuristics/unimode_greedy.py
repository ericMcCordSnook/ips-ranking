def get_greedy_traversal_order(graph, start_node):
    order = [start_node]
    visited = set()
    visited.add(start_node)
    current_nodes = graph[start_node]
    i = 0
    while bool(current_nodes): # tests if empty
        i = i + 1
        print("Step", i, current_nodes)
        min_dist_node = None
        min_dist = 100
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
