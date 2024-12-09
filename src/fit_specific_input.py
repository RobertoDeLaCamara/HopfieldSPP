import tensorflow as tf
import numpy as np

def create_input_vector(cost_matrix, origin_index, destination_index, num_nodes):
    """
    Creates an input vector for a neural network model representing a graph.

    This function takes a flattened cost matrix and marks the origin and destination nodes
    in the graph, forming an input vector for model training.

    Args:
        cost_matrix (np.ndarray): Flattened cost matrix of the graph.
        origin_index (int): Index of the origin node in the graph.
        destination_index (int): Index of the destination node in the graph.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        np.ndarray: Input vector for the neural network model, combining the cost matrix
                    and one-hot encoded origin and destination nodes.
    """
    input_vector = np.concatenate((cost_matrix, np.eye(num_nodes)[[origin_index, destination_index]]))
    return input_vector

def energy_loss_with_input_vectors(y_true, y_pred, C_flat, n):
    """
    Computes the loss for a predicted path in a graph with specified input vectors.

    This function is similar to the offline loss, but it takes into account the input vectors
    for the origin and destination nodes. It applies penalties to deviations from the
    constraints for path cost, outgoing edges, incoming edges, binary values, and the
    source and destination nodes.

    Args:
        y_true: Ground truth values (not utilized in this computation).
        y_pred: Predicted values, containing arc values, origin vector, and destination vector.

    Returns:
        tf.Tensor: The computed loss value as a TensorFlow tensor.
    """
    predicted_arcs = y_pred[:, :n * n]
    origin_vec = y_pred[:, n * n:n * n + n]
    destination_vec = y_pred[:, n * n + n:]

    # Path cost
    path_cost = tf.reduce_sum(C_flat * predicted_arcs)

    # Outgoing and Incoming edge constraints
    reshaped_arcs = tf.reshape(predicted_arcs, (-1, n, n))
    outgoing_edges = tf.reduce_sum(reshaped_arcs, axis=2) - 1
    incoming_edges = tf.reduce_sum(reshaped_arcs, axis=1) - 1
    edge_constraints = tf.reduce_sum(tf.square(outgoing_edges)) + tf.reduce_sum(tf.square(incoming_edges))

    # Binary values constraint
    binary_constraint = tf.reduce_sum(predicted_arcs * (1 - predicted_arcs))

    # Source and Destination constraints
    source_index = tf.argmax(origin_vec, axis=1)
    source_outflow = tf.gather(reshaped_arcs, source_index, batch_dims=1)
    source_constraint = tf.reduce_sum(source_outflow, axis=1) - 1

    destination_index = tf.argmax(destination_vec, axis=1)
    destination_inflow = tf.gather(tf.transpose(reshaped_arcs, perm=[0, 2, 1]), destination_index, batch_dims=1)
    destination_constraint = tf.reduce_sum(destination_inflow, axis=1) - 1

    # Compute total loss
    loss = (path_cost 
            + 10 * edge_constraints 
            + 10 * binary_constraint 
            + 100 * tf.reduce_sum(tf.square(source_constraint)) 
            + 100 * tf.reduce_sum(tf.square(destination_constraint)))
    
    return loss

def fit_specific_input(cost_matrix, origin_index, destination_index, num_nodes):
    """
    Fits the neural network model to a specific input vector.

    Args:
        cost_matrix (np.ndarray): Flattened cost matrix of the graph.
        origin_index (int): Index of the origin node in the graph.
        destination_index (int): Index of the destination node in the graph.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        None
    """
    input_vector = create_input_vector(cost_matrix, origin_index, destination_index, num_nodes)  
    # TODO: Fit the model     
    return