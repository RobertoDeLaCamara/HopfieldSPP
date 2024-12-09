import pandas as pd
import numpy as np
import tensorflow as tf

def calculate_cost_matrix():   
    """
    Calculates the cost matrix for a network given in a CSV file.

    The CSV file should have columns 'origin', 'destination', and 'weight'. The function
    will exit with an error message if the file is not found, is empty, or is
    invalid in any way.

    The cost matrix is a matrix where the element at row i and column j
    represents the cost of going from node i to node j. The diagonal of the
    matrix is set to 0, representing the cost of going from a node to itself.

    The cost matrix is saved as a numpy array in a file named 'cost_matrix.npy'
    in the 'data/synthetic' directory.

    """
    # Load data from CSV file
    try:
        df = pd.read_csv('../data/network.csv', usecols=['origin', 'destination', 'weight'])
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        exit()
    except pd.errors.EmptyDataError:
        print("Error: The file is empty or invalid.")
        exit()

    # Identify unique nodes and create a mapping for indices
    nodos = pd.unique(df[['origin', 'destination']].values.ravel())
    node_to_index = pd.Series(range(len(nodos)), index=nodos)
    n = len(nodos)

    # Initialize the cost matrix with infinity
    cost_matrix = np.full((n, n), np.inf, dtype=float)

    # Set diagonal to 0 (self-costs)
    np.fill_diagonal(cost_matrix, 0)

    # Fill the cost matrix with the values from the CSV
    df['origin_idx'] = df['origin'].map(node_to_index)
    df['destination_idx'] = df['destination'].map(node_to_index)
    df.apply(lambda row: cost_matrix.__setitem__((row['origin_idx'], row['destination_idx']), row['weight']), axis=1)

    # Save the cost matrix
    np.save('../data/cost_matrix.npy', cost_matrix)

def offline_loss(y_true, y_pred, num_nodes, flattened_cost_matrix):
    """
    Computes the loss function for a predicted path in a graph.

    This function calculates the loss based on the predicted arc values, ensuring
    constraints for path cost, outgoing edges, incoming edges, and binary values are
    maintained. It applies penalties to deviations from these constraints, aiming to
    optimize the path in terms of cost and validity.

    Args:
        y_true: Ground truth values (not utilized in this computation).
        y_pred: Predicted values, containing arc values for the graph.
        num_nodes: Number of nodes in the graph.
        flattened_cost_matrix: A flattened version of the cost matrix, where the element
            at index i*num_nodes + j represents the cost of going from node i to node j.

    Returns:
        tf.Tensor: The computed loss value as a TensorFlow tensor.
    """
    predicted_arc_values = y_pred[:, :num_nodes * num_nodes]
    reshaped_values = tf.reshape(predicted_arc_values, (-1, num_nodes, num_nodes))

    # Path cost
    cost_term = tf.reduce_sum(flattened_cost_matrix * predicted_arc_values)

    # Outgoing edge constraint
    outgoing_sums = tf.reduce_sum(reshaped_values, axis=2)
    outgoing_edge_penalty = tf.reduce_sum(tf.square(outgoing_sums - 1))

    # Incoming edge constraint
    incoming_sums = tf.reduce_sum(reshaped_values, axis=1)
    incoming_edge_penalty = tf.reduce_sum(tf.square(incoming_sums - 1))

    # Binary values constraint
    binary_penalty = tf.reduce_sum(predicted_arc_values * (1 - predicted_arc_values))

    loss = cost_term + 100 * outgoing_edge_penalty + 100 * incoming_edge_penalty + 10 * binary_penalty
    return loss

   
def train_offline_model():
    """
    Train a model to learn graph properties without source/destination nodes.

    This function is responsible for training a model on the cost matrix of a graph
    without considering any source or destination nodes. The model is trained using
    the offline_loss function, which is defined above.

    The model is a simple feedforward neural network with one hidden layer, and
    is trained using the Adam optimizer with a learning rate of 0.01. The model is
    saved as a TensorFlow SavedModel after training is complete.
    """
    cost_matrix = calculate_cost_matrix()
    num_nodes = cost_matrix.shape[0]
    flattened_cost_matrix = cost_matrix.flatten()
    flattened_cost_matrix[np.isinf(flattened_cost_matrix)] = 1e6

    # Define offline training model
    offline_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(num_nodes * num_nodes,)),
        tf.keras.layers.Dense(num_nodes * num_nodes, activation='sigmoid')
    ])

    # Compile offline model
    offline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                          loss=lambda y_true, y_pred: offline_loss(y_true, y_pred, num_nodes, flattened_cost_matrix))

    # Train offline model with early stopping
    offline_input = np.expand_dims(flattened_cost_matrix, axis=0)
    offline_model.fit(
        offline_input, 
        offline_input, 
        epochs=500, 
        verbose=0, 
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
    )

    # Save pre-trained model
    offline_model.save('../models/offline_model.h5')
