import pandas as pd
# import numpy as np
import heapq
# from AStar import *

file_path = 'distances_matrix.csv'
df = pd.read_csv(file_path, index_col=0)

# Convert DataFrame to a NumPy array and get node names as a list
adj_matrix = df.to_numpy()
nodes = df.index.tolist()



def dijkstra(adj_matrix, start_city, end_city):
    n = len(adj_matrix)
    distances = [float('inf')] * n  # Initialize distances with infinity
    path_taken = [None] * n  # Track path taken
    start_idx = nodes.index(start_city)
    end_idx = nodes.index(end_city)
    distances[start_idx] = 0  # Distance to start node is 0

    # Priority queue data format: (distance, node index)
    pq = [(0, start_idx)]

    while pq:
        # Extract the node with the smallest distance
        current_distance, current_idx = heapq.heappop(pq)

        # Skip if this node was already processed with a shorter distance
        if current_distance > distances[current_idx]:
            continue

        # Update distances for neighbors
        for neighbor_idx in range(n):
            distance = adj_matrix[current_idx][neighbor_idx]
            if distance != -1:
                new_distance = distances[current_idx] + distance
                if new_distance < distances[neighbor_idx]:
                    distances[neighbor_idx] = new_distance
                    path_taken[neighbor_idx] = current_idx
                    heapq.heappush(pq, (new_distance, neighbor_idx))

    # Build the path for print out
    path = []
    node_idx = end_idx
    while node_idx is not None:
        # look up city name and add to path
        path.append(nodes[node_idx])
        node_idx = path_taken[node_idx]
    path = path[::-1]  # Reverse to get the correct order


    return path, distances[end_idx]


if __name__ == "__main__":
    print("\n*****  Welcome to flight path calculator.  *****")
    # program loop
    while True:
        start = input("Enter starting city, format: City, ST (or x to exit): ").strip()
        if start == 'x':  # check if user wants to exit
            break
        end = input("Enter destination city, format: City, ST (or x to exit): ").strip()
        if end == 'x':  # check if user wants to exit
            break

        if start not in nodes or end not in nodes:
            print("Invalid city entered, please try again.\n\n")
        else:
            path, distance = dijkstra(adj_matrix, start, end)
            print("\n\n")
            print(f"Path from {start} to {end}: {' -> '.join(path)}")
            print(f"Total distance: {distance}")


