import pandas as pd
import numpy as np
import heapq

# import csv
df = pd.read_csv("distances_matrix.csv", index_col=0)

# replace -1 with infinity
df = df.replace(-1, np.inf)

# create graph in form of dictionary of dictionaries
graph = {city: {adj_city: df.loc[city, adj_city] for adj_city in df.columns if df.loc[city, adj_city] != np.inf}
         for city in df.index}

# Dijkstra’s algorithm to find the shortest path and reconstruct the path
def dijkstra(graph, start, end):
    # Initialize the distance dictionary with infinity for all nodes except the start node
    distances = {city: float('inf') for city in graph}
    distances[start] = 0

    # store city path
    previous_cities = {city: None for city in graph}

    # Priority queue to manage visiting nodes in order of smallest distance
    pq = [(0, start)]  # (distance, city)

    while pq:
        current_distance, current_city = heapq.heappop(pq)

        # Skip visited cities
        if current_distance > distances[current_city]:
            continue

        # Check all adjacent cities and update their distances if a shorter path is found
        for neighbor, weight in graph[current_city].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_cities[neighbor] = current_city
                heapq.heappush(pq, (distance, neighbor))

    # If the end city is unreachable, return None
    if distances[end] == float('inf'):
        return None, None

    # Create full path
    path = []
    current_city = end
    while current_city is not None:
        path.append(current_city)
        current_city = previous_cities[current_city]
    path = path[::-1]  # Reverse the path to get it from start to end

    return path, distances[end]


# Function to print the shortest path with distance
def print_shortest_path(start_city, end_city):
    path, total_distance = dijkstra(graph, start_city, end_city)

    if path is not None:
        path_str = " -> ".join(path)
        print(f"Shortest path from {start_city} to {end_city} is:")
        print(f"{path_str}")
        print(f"Total Distance: {total_distance}")
    else:
        print(f"There is no path from {start_city} to {end_city}")

start_city = "Boston, MA"
end_city = "Olympia, WA"

print_shortest_path(start_city, end_city)