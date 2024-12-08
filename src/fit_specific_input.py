import tensorflow as tf
import numpy as np

# Load pre-trained offline model
offline_model = tf.keras.models.load_model('../models/offline_model.h5',
                                           custom_objects={'offline_loss': offline_loss})

# Placeholder for cost matrix and input vector creation
C = cost_matrix
n = C.shape[0]
C_flat = C.flatten()
C_flat[np.isinf(C_flat)] = 1e6

# Create input vector with source and destination nodes
def create_input_vector(C_flat, origin, destination, n):
    """
    Creates an input vector for a neural network model representing a graph.

    This function takes a flattened cost matrix and marks the origin and destination nodes
    in the graph, forming an input vector for model training.

    Args:
        C_flat (np.ndarray): Flattened cost matrix of the graph.
        origin (int): Index of the origin node in the graph.
        destination (int): Index of the destination node in the graph.
        n (int): Number of nodes in the graph.

    Returns:
        np.ndarray: Input vector for the neural network model, combining the cost matrix
                    and one-hot encoded origin and destination nodes.
    """
    input_vector = np.zeros(len(C_flat) + 2 * n)
    input_vector[:len(C_flat)] = C_flat
    input_vector[len(C_flat) + origin] = 1
    input_vector[len(C_flat) + n + destination] = 1
    return input_vector

# Fine-tuning for specific source and destination nodes
def energy_loss_with_input_vectors(y_true, y_pred):
    arc_values = y_pred[:, :n * n]
    origin_vector = y_pred[:, n * n:n * n + n]
    destination_vector = y_pred[:, n * n + n:]
    term1 = tf.reduce_sum(C_flat * arc_values)
    row_sums = tf.reduce_sum(tf.reshape(arc_values, (-1, n, n)), axis=2) - 1
    term2 = tf.reduce_sum(tf.square(row_sums))
    col_sums = tf.reduce_sum(tf.reshape(arc_values, (-1, n, n)), axis=1) - 1
    term3 = tf.reduce_sum(tf.square(col_sums))
    term4 = tf.reduce_sum(arc_values * (1 - arc_values))
    s = tf.argmax(origin_vector, axis=1)
    arc_matrix = tf.reshape(arc_values, (-1, n, n))
    source_out = tf.gather(arc_matrix, s, batch_dims=1)
    source_constraint = tf.reduce_sum(source_out, axis=1) - 1
    term5 = tf.reduce_sum(tf.square(source_constraint))
    d = tf.argmax(destination_vector, axis=1)
    dest_in = tf.gather(tf.transpose(arc_matrix, perm=[0, 2, 1]), d, batch_dims=1)
    dest_constraint = tf.reduce_sum(dest_in, axis=1) - 1
    term6 = tf.reduce_sum(tf.square(dest_constraint))
    loss = term1 + 10 * term2 + 10 * term3 + 10 * term4 + 10 * term5 + 10 * term6
    return loss

# Define full model for fine-tuning
input_dim = len(C_flat) + 2 * n
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(input_dim,)),
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
])

# Load weights from offline model and adjust
offline_weights = offline_model.layers[0].get_weights()
offline_kernel, offline_bias = offline_weights

# Adjust weights to match the new input dimension
new_kernel = np.zeros((input_dim, input_dim))  # Create new kernel with the correct size
new_bias = np.zeros(input_dim)  # Adjust bias size

# Copy offline weights for the C_flat portion
new_kernel[:len(C_flat), :len(C_flat)] = offline_kernel
new_bias[:len(C_flat)] = offline_bias

# Set adjusted weights into the fine-tuning model
model.layers[0].set_weights([new_kernel, new_bias])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=energy_loss_with_input_vectors)

# Create input vector
origin = 0
destination = 3
input_vector = create_input_vector(C_flat, origin, destination, n)
input_data = np.expand_dims(input_vector, axis=0)

# Fine-tune the model
model.fit(input_data, input_data, epochs=500, verbose=1)

# Predictions and results
predictions = model.predict(input_data)[0]
arc_values = predictions[:n * n]
arc_matrix = arc_values.reshape((n, n)) > 0.5

print("Matriz de arcos seleccionados:")
print(arc_matrix.astype(int))