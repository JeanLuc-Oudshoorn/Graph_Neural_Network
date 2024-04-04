import tensorflow as tf
from tensorflow import keras
from tensorflow.gnn import GraphTensor

# Sample data (modify as needed for your data)
node_features = tf.constant([[1, 0.8], [0.5, 0.2], [0.7, 0.9], [0.3, 0.1]], dtype=tf.float32)
edge_features = tf.constant([[1], [1], [0.5]], dtype=tf.float32)
node_labels = tf.constant([0, 1, 0, 2])

# Adjacency matrix (alternative: create edge list using tf.SparseTensor)
adjacency_matrix = tf.constant([[0, 1, 1, 0],
                                 [1, 0, 1, 0.5],
                                 [1, 1, 0, 0],
                                 [0, 0.5, 0, 0]], dtype=tf.float32)

# Create GraphTensor object
graph = GraphTensor(node_features, edge_features, adjacency_matrix=adjacency_matrix)

# Define GNN model (replace with layers for your task)
model = keras.Sequential([
  tfgnn.layers.Dense(units=32, activation='relu')(graph),
  tfgnn.layers.GraphConv(units=16)(graph),
  keras.layers.Dense(units=3, activation='softmax')  # Output for 3 class prediction
])

# Compile the model (categorical crossentropy for multi-class classification)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (adjust epochs as needed)
model.fit(graph, node_labels, epochs=10)
