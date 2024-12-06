import time
import random
from source import nodes, dijkstra, adj_matrix
from AStar import create_heuristic, a_star, city_coordinates

while True:
    d_times = []
    a_times = []
    test_count = input("Enter how many tests you would like to run (x to exit): ")

    if test_count == 'x':
        break

    for i in range(int(test_count)):
        # choose 2 random cities
        random_cities = random.sample(nodes, 2)
        # Dijkstra's Algorithm
        dijkstra_start_time = time.time()
        d_path, dijkstra_distance = dijkstra(adj_matrix, random_cities[0], random_cities[1])
        dijkstra_execution_time = time.time() - dijkstra_start_time

        # A* Algorithm
        heuristic = create_heuristic(adj_matrix, random_cities[1], city_coordinates)
        a_star_start_time = time.time()
        a_star_path, a_star_distance = a_star(adj_matrix, random_cities[0], random_cities[1], heuristic)
        a_star_execution_time = time.time() - a_star_start_time

        # if algorithms get different results: test failed
        if (dijkstra_distance != a_star_distance) or (d_path != a_star_path):
            print(f"Test {i} failed. Cities: {random_cities[0]}, {random_cities[1]}")
        else:
            d_times.append(dijkstra_execution_time)
            a_times.append(a_star_execution_time)

    avg_dijkstra = sum(d_times) / len(d_times)
    avg_astar = sum(a_times) / len(a_times)

    print(f"Average processing time Dijkstra's: {avg_dijkstra}")
    print(f"Average processing time A*:         {avg_astar}")
