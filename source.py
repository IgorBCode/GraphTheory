import pandas as pd

file_path = 'distances_matrix.csv'
df = pd.read_csv(file_path, index_col=0)


def dijkstra(adj_matrix, start_node, end_node):
    n = len(adj_matrix)
    distances = {node: float('inf') for node in adj_matrix.index}  # Initialize distances
    distances[start_node] = 0
    previous_nodes = {node: None for node in adj_matrix.index}  # To store path
    unvisited = set(adj_matrix.index)  # All nodes as unvisited

    while unvisited:
        # Get node with smallest distance
        current_node = min(unvisited, key=lambda node: distances[node])

        if distances[current_node] == float('inf'):
            break  # Remaining nodes are inaccessible from start_node

        unvisited.remove(current_node)

        # Update distances for neighbors
        for neighbor, distance in adj_matrix.loc[current_node].items():
            if distance != -1 and neighbor in unvisited:  # -1 indicates no direct path
                new_distance = distances[current_node] + distance
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node

        # Stop if we reach the end node
        if current_node == end_node:
            break

    # Create path from start to end
    path = []
    node = end_node
    while node is not None:
        path.append(node)
        node = previous_nodes[node]
    # reverse path to show correctly
    path = path[::-1]

    if distances[end_node] == float('inf'):
        print(f"No path exists from {start_node} to {end_node}.")
    else:
        print(f"Path from {start_node} to {end_node}: {' -> '.join(path)}")
        print(f"Total distance: {distances[end_node]}")


start = 'Madison, WI'  # Replace with actual starting node name
end = 'Santa Fe, NM'  # Replace with actual ending node name
dijkstra(df, start, end)
