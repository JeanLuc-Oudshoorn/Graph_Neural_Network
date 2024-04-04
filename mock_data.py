import tensorflow as tf
from tensorflow import keras
from tensorflow.gnn import GraphTensor
import pandas as pd
import networkx as nx
import random

# Define locations (nodes)
locations = ["City A", "City B", "City C", "Town 1", "Town 2"]

# Create a random graph (can be replaced with a specific network structure)
G = nx.random_geometric_graph(len(locations), radius=50)

# Add node features (population, healthcare)
population = {loc: random.randint(10000, 100000) for loc in locations}
hospitals = {loc: random.randint(1, 5) for loc in locations}
node_features = pd.DataFrame({"Location": locations, "Population": population, "Hospitals": hospitals})

# Add edge features (travel volume)
travel_volume = {}
for i, node1 in enumerate(locations):
    travel_volume[node1] = {}
    for j, node2 in enumerate(locations):
        if i != j and G.has_edge(node1, node2):
            travel_volume[node1][node2] = random.randint(100, 1000)
        else:
            travel_volume[node1][node2] = 0

# Add edges and edge features to the graph
for node1, neighbors in G.adj.items():
    for node2, _ in neighbors.items():
        G.add_edge(node1, node2, weight=travel_volume[node1][node2])

# Combine node features and graph data
data = pd.merge(node_features, G.nodes(data=True), how="inner", on="Location")

# Define node labels (replace with logic for your task)
# Example: Simulate initial cases for outbreak prediction
num_infected = int(0.1 * len(locations))  # Adjust percentage of initially infected nodes
infected_nodes = random.sample(data["Location"].tolist(), num_infected)
data["Label"] = 0
data.loc[data["Location"].isin(infected_nodes), "Label"] = 1

# Convert data to GraphTensor objects
node_features_tensor = tf.convert_to_tensor(data[["Population", "Hospitals"]].values, dtype=tf.float32)
edge_features_tensor = tf.constant([[d['weight']] for u, v, d in G.edges(data=True)], dtype=tf.float32)
adjacency_matrix = nx.adjacency_matrix(G).todense()  # Convert to dense matrix

# Create GraphTensor object
graph = GraphTensor(node_features_tensor, edge_features_tensor, adjacency_matrix=adjacency_matrix)

# Print sample data (optional)
print("Sample Locations (Nodes):")
print(data[["Location"]].head())

print("\nSample Node Features:")
print(data[["Location", "Population", "Hospitals"]].head())

print("\nSample Edges and Travel Volume:")
for u, v, d in G.edges(data=True):
    print(f"({u}, {v}) - Travel Volume: {d['weight']}")
