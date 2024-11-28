import pandas as pd
import heapq
import math
import os

# Get the current working directory dynamically
current_dir = os.getcwd()

# file path using the current working directory
file_path = os.path.join(current_dir, 'distances_matrix.csv')

# Try to load the CSV file
try:
    df = pd.read_csv(file_path, index_col=0)
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the path.")
    exit(1)

# Extract city names (node names) from the DataFrame index
nodes = df.index.tolist()

# Convert DataFrame to a NumPy array for easier access in the algorithm
adj_matrix = df.to_numpy()

# Haversine formula to calculate the distance between two points (lat1, lon1) and (lat2, lon2)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c  # Returns distance in kilometers

# All 50 states coordinates to use for the Euclidian heuristic
city_coordinates = {
    'Montgomery, AL': (32.3792, -86.3077),
    'Juneau, AK': (58.3019, -134.4197),
    'Phoenix, AZ': (33.4484, -112.0740),
    'Little Rock, AR': (34.7465, -92.2896),
    'Sacramento, CA': (38.5816, -121.4944),
    'Denver, CO': (39.7392, -104.9903),
    'Hartford, CT': (41.7658, -72.6734),
    'Dover, DE': (39.1582, -75.5244),
    'Tallahassee, FL': (30.4383, -84.2807),
    'Atlanta, GA': (33.7490, -84.3880),
    'Honolulu, HI': (21.3070, -157.8583),
    'Boise, ID': (43.6150, -116.2023),
    'Springfield, IL': (39.7817, -89.6501),
    'Indianapolis, IN': (39.7684, -86.1581),
    'Des Moines, IA': (41.5868, -93.6250),
    'Topeka, KS': (39.0489, -95.6780),
    'Frankfort, KY': (38.2009, -84.8733),
    'Baton Rouge, LA': (30.4515, -91.1871),
    'Augusta, ME': (44.3106, -69.7795),
    'Annapolis, MD': (38.9784, -76.4922),
    'Boston, MA': (42.3601, -71.0589),
    'Lansing, MI': (42.7325, -84.5555),
    'St. Paul, MN': (44.9537, -93.0900),
    'Jackson, MS': (32.2988, -90.1848),
    'Jefferson City, MO': (38.5767, -92.1735),
    'Helena, MT': (46.5891, -112.0391),
    'Lincoln, NE': (40.8136, -96.7026),
    'Carson City, NV': (39.1638, -119.7674),
    'Concord, NH': (43.2081, -71.5376),
    'Trenton, NJ': (40.2171, -74.7429),
    'Santa Fe, NM': (35.6869, -105.9378),
    'Albany, NY': (42.6526, -73.7562),
    'Raleigh, NC': (35.7796, -78.6382),
    'Bismarck, ND': (46.8083, -100.7837),
    'Columbus, OH': (39.9612, -82.9988),
    'Oklahoma City, OK': (35.4676, -97.5164),
    'Salem, OR': (44.9429, -123.0351),
    'Harrisburg, PA': (40.2732, -76.8867),
    'Providence, RI': (41.8240, -71.4128),
    'Columbia, SC': (34.0007, -81.0348),
    'Pierre, SD': (44.3683, -100.3510),
    'Nashville, TN': (36.1627, -86.7816),
    'Austin, TX': (30.2672, -97.7431),
    'Salt Lake City, UT': (40.7608, -111.8910),
    'Montpelier, VT': (44.2601, -72.5754),
    'Richmond, VA': (37.5407, -77.4360),
    'Olympia, WA': (47.0379, -122.9007),
    'Washington DC': (38.9072, -77.0369),
    'Charleston, WV': (38.3498, -81.6326),
    'Madison, WI': (43.0731, -89.4012),
    'Cheyenne, WY': (41.1400, -104.8202)
}

# Create a heuristic using the Haversine formula (straight-line distance to the goal)
def create_heuristic(end_city):
    heuristic = []
    end_lat, end_lon = city_coordinates[end_city]
    
    for node in nodes:
        # Get the coordinates for each city
        node_lat, node_lon = city_coordinates[node]
        # Calculate the straight-line distance to the goal (using Haversine)
        distance = haversine(node_lat, node_lon, end_lat, end_lon)
        heuristic.append(distance)  # Use the Haversine distance as the heuristic
    
    return heuristic

# A* algorithm to find the shortest path from start_city to end_city
def a_star(adj_matrix, start_city, end_city, heuristic):
    n = len(adj_matrix)
    start_idx = nodes.index(start_city)
    end_idx = nodes.index(end_city)

    # Initialize costs and priority queue
    g = [float('inf')] * n  # Actual cost from start to current node
    f = [float('inf')] * n  # Estimated cost from start to goal (f = g + h)
    previous_nodes = [None] * n
    g[start_idx] = 0
    f[start_idx] = heuristic[start_idx]

    pq = [(f[start_idx], start_idx)]  # Priority queue (f(n), node_index)

    while pq:
        _, current_idx = heapq.heappop(pq)

        if current_idx == end_idx:
            path = []
            while current_idx is not None:
                path.append(nodes[current_idx])
                current_idx = previous_nodes[current_idx]
            path.reverse()
            print(f"A* Path: {' -> '.join(path)}")
            print(f"A* Total Distance: {g[end_idx]}")
            return

        for neighbor_idx in range(n):
            distance = adj_matrix[current_idx][neighbor_idx]
            if distance != -1:  # Ignore -1 (no direct path)
                tentative_g = g[current_idx] + distance
                if tentative_g < g[neighbor_idx]:
                    g[neighbor_idx] = tentative_g
                    f[neighbor_idx] = tentative_g + heuristic[neighbor_idx]
                    previous_nodes[neighbor_idx] = current_idx
                    heapq.heappush(pq, (f[neighbor_idx], neighbor_idx))

    print(f"No path exists from {start_city} to {end_city}.")

# Main program loop
start_city = "Montgomery, AL"  # Example start city
end_city = "Cheyenne, WY"     # Example end city

# Create the heuristic for the destination city
heuristic = create_heuristic(end_city)

# Run A* algorithm with the new heuristic
print(f"\nRunning A* from {start_city} to {end_city}:")
a_star(adj_matrix, start_city, end_city, heuristic)
