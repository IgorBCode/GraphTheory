import pandas as pd
import numpy as np

file_path = 'distances_matrix.csv'
df = pd.read_csv(file_path, index_col=0)

# Convert DataFrame to a NumPy array and get node names as a list
adj_matrix = df.to_numpy()
nodes = df.index.tolist()

def dijkstra(adj_matrix, start_node, end_node):
    n = len(adj_matrix)
    distances = [float('inf')] * n  # Initialize distances with infinity
    previous_nodes = [None] * n     # To store the path
    unvisited = set(range(n))       # Unvisited nodes by index
    start_idx = nodes.index(start_node)
    end_idx = nodes.index(end_node)
    distances[start_idx] = 0        # Distance to start node is 0

    while unvisited:
        # Get the node with the smallest distance
        current_idx = min(unvisited, key=lambda idx: distances[idx])

        if distances[current_idx] == float('inf'):
            break  # Remaining nodes are inaccessible from start_node

        unvisited.remove(current_idx)

        # Update distances for neighbors
        for neighbor_idx in range(n):
            distance = adj_matrix[current_idx][neighbor_idx]
            if distance != -1 and neighbor_idx in unvisited:  # -1 indicates no direct path
                new_distance = distances[current_idx] + distance
                if new_distance < distances[neighbor_idx]:
                    distances[neighbor_idx] = new_distance
                    previous_nodes[neighbor_idx] = current_idx

        # Stop if we reach the end node
        if current_idx == end_idx:
            break

    # Reconstruct the path from start_node to end_node
    path = []
    node_idx = end_idx
    while node_idx is not None:
        path.append(nodes[node_idx])
        node_idx = previous_nodes[node_idx]
    path = path[::-1]  # Reverse path to start from the start_node

    if distances[end_idx] == float('inf'):
        print(f"No path exists from {start_node} to {end_node}.")
    else:
        print(f"Path from {start_node} to {end_node}: {' -> '.join(path)}")
        print(f"Total distance: {distances[end_idx]}")


start = 'Madison, WI'  # Replace with actual starting node name
end = 'Santa Fe, NM'  # Replace with actual ending node name
dijkstra(adj_matrix, start, end)
