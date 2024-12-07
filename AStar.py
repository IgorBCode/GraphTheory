import pandas as pd
import math
import heapq

# Load the distance matrix
file_path = 'distances_matrix.csv'
df = pd.read_csv(file_path, index_col=0)

# Convert DataFrame to a NumPy array and get node names as a list
adj_matrix = df.to_numpy() # This converts adjacenty matrix to a NumPy array which helps us later in the code to work with
nodes = df.index.tolist() # List of node names (our 50 cities) from the DataFrame. used for easier reference.

# 50 states coordinates
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
    'Washington, DC': (38.9072, -77.0369),
    'Charleston, WV': (38.3498, -81.6326),
    'Madison, WI': (43.0731, -89.4012),
    'Cheyenne, WY': (41.1400, -104.8202)
}

# Haversine formula to calculate distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c # Calculates the great-circle distance between two points - Should mostly be accurate

# Generate heuristic with dynamic scaling
def create_heuristic(adj_matrix, end_city, city_coordinates):
    heuristic = [] # Heuristic array stores estimated distance to the destination city for each city in the network
    end_lat, end_lon = city_coordinates[end_city]

    # Calculate average edge weight in the adjacency matrix
    valid_distances = [dist for row in adj_matrix for dist in row if dist != -1]
    avg_adj_distance = sum(valid_distances) / len(valid_distances) if valid_distances else 1 # Basically computing the average distance in the adjacency matrix (not including invalid edges -- only valid)

    # Calculate average Haversine distance
    haversine_distances = []
    for node in nodes:
        node_lat, node_lon = city_coordinates.get(node, (0, 0)) #  Retreiving the coordinates of a city (Default is set to 0)
        haversine_distances.append(haversine(node_lat, node_lon, end_lat, end_lon)) # Adds city's Haversine distance to destination
    avg_haversine_distance = sum(haversine_distances) / len(haversine_distances) if haversine_distances else 1 # Average of every single Haversine distance

    # Compute scaling factor
    scaling_factor = avg_adj_distance / avg_haversine_distance # Essentially helps balance the adjacency matrix with the Haversine distances. Useful for making adjustments in heuristic

    # Generate heuristic
    for node in nodes:
        node_lat, node_lon = city_coordinates.get(node, (0, 0))
        h = haversine(node_lat, node_lon, end_lat, end_lon) * scaling_factor # Using scaling factor to scale the distance for each city
        heuristic.append(h)

    return heuristic

# A* Algorithm
def a_star(adj_matrix, start_city, end_city, heuristic):
    n = len(adj_matrix)
    g = [float('inf')] * n # g stores cost to reach city. Initially infinity. 
    f = [float('inf')] * n # f stores estimated cost ( f(n) = g(n) + h(n). Initally infinity.
    previous_nodes = [None] * n # Keeps track of the path by storing previous node for each city
    start_idx = nodes.index(start_city)
    end_idx = nodes.index(end_city)
    g[start_idx] = 0
    f[start_idx] = heuristic[start_idx]
    pq = [(f[start_idx], start_idx)] # Use of priority queue for the A* process.
    while pq:
        _, current_idx = heapq.heappop(pq) # Pop city with lowest f value ( g + h ) 
        if current_idx == end_idx:
            path = [] # If we've reahed the destination, we backtrack to reconstruct path
            while current_idx is not None:
                path.append(nodes[current_idx])
                current_idx = previous_nodes[current_idx]
            path.reverse() # Reverse path to get it in correct order from start to end node
            return path, g[end_idx]
        for neighbor_idx in range(n): # Check the neighbors. if a path does exist (anything != -1) calculate costs.
            distance = adj_matrix[current_idx][neighbor_idx]
            if distance != -1:
                tentative_g = g[current_idx] + distance # Cost of reaching this neighbor
                if tentative_g < g[neighbor_idx]:
                    g[neighbor_idx] = tentative_g
                    f[neighbor_idx] = tentative_g + heuristic[neighbor_idx]
                    previous_nodes[neighbor_idx] = current_idx
                    heapq.heappush(pq, (f[neighbor_idx], neighbor_idx)) # Push neighbor onto priority queue for further testing
    return [], float('inf') # If no path is found, return list as empty and infinite

# Main program
if __name__ == "__main__":
    print("\n********* A* Algorithm *********")
    start_city = input("Enter starting city (e.g., Montgomery, AL): ").strip()
    end_city = input("Enter destination city (e.g., Cheyenne, WY): ").strip()

    if start_city not in nodes or end_city not in nodes:
        print("Invalid city names. Please ensure they exist in the dataset.")
    else:
        # Create heuristic
        heuristic = create_heuristic(adj_matrix, end_city, city_coordinates)

        # Run A* algorithm
        path, distance = a_star(adj_matrix, start_city, end_city, heuristic)

        # Print results
        if distance == float('inf'):
            print(f"No path exists from {start_city} to {end_city}.")
        else:
            print(f"Path: {' -> '.join(path)}")
            print(f"Total Distance: {distance}")
