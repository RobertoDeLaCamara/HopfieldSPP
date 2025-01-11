import tensorflow as tf
from src.train_model import HopfieldLayer, HopfieldModel
from tensorflow.keras.saving import custom_object_scope
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

def calculate_real_path_cost(real_cost_matrix, path):
    """
    Calculates the cost of the shortest path given the real cost matrix and a path.

    Args:
        real_cost_matrix (list of list of float): A 2D list representing the cost matrix of the network
            without normalization.
        path (list of int): A list of node indices representing the shortest path.

    Returns:
        float: The cost of the shortest path.

    """
    cost = 0
    for i in range(len(path) - 1):
        cost += real_cost_matrix[path[i], path[i + 1]]
    return cost


def calculate_shortest_path(cost_matrix, origin, destination):
    """
    Calculates the shortest path between two nodes in the network using a pre-trained Hopfield model.
    Args: 
        cost_matrix (list of list of float): A 2D list representing the cost matrix of the network.
        origin (int): The starting node.
        destination (int): The destination node.
    Returns:
        dict: A dictionary containing the predicted path and the cost of the shortest path.

    Raises:
        ValueError: If cost_matrix, origin, or destination is None.
        RuntimeError: If the model prediction returns an empty path or if any error occurs during the calculation.
    """
    print("Calculating the shortest path")
    if cost_matrix is None or origin is None or destination is None:
        raise ValueError("Cost matrix, origin, or destination cannot be None.")

    try:
        with custom_object_scope({'HopfieldModel': HopfieldModel, 'HopfieldLayer': HopfieldLayer}):
            loaded_model = load_model('../models/trained_model_without_source_dest.keras', custom_objects={'HopfieldModel': HopfieldModel, 'HopfieldLayer': HopfieldLayer})
        loaded_model.compile(optimizer=Adam(learning_rate=0.01))

        path = loaded_model.predict(origin, destination)
        if not path:
            raise RuntimeError("Model prediction returned an empty path.")

        print("Predicted Path:", path)
        path_cost = calculate_real_path_cost(cost_matrix, path)
        print("Cost of the Shortest Path (Real Costs):", path_cost)
        result = {
            "path": path,
            "cost": path_cost
        }
        return result
    except Exception as e:
        raise RuntimeError(f"An error occurred while calculating the shortest path: {str(e)}")
