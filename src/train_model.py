import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_cost_matrix(adjacency_matrix):
    """
    Calculates the cost matrix for an adjacency matrix of a network given in a CSV file.

    The CSV file should have columns 'origin', 'destination', and 'weight'. The function
    will exit with an error message if the file is not found, is empty, or is
    invalid in any way.

    Args:
        adjacency_matrix (str): The path to the CSV file containing the adjacency matrix.
    Returns:
        cost_matrix: A 2D numpy array representing the cost matrix.
        node_mapping: Dictionary mapping original node IDs to matrix indices.

    """
    try:
        df = pd.read_csv(adjacency_matrix, usecols=['origin', 'destination', 'weight'],
                         dtype={'origin': str, 'destination': str, 'weight': float})
    except FileNotFoundError:
        logger.error("File not found. Please check the file path.")
        raise
    except pd.errors.EmptyDataError:
        logger.error("The file is empty or invalid.")
        raise

    nodos = sorted(pd.unique(df[['origin', 'destination']].values.ravel()))
    node_to_index = {node: idx for idx, node in enumerate(nodos)}
    n = len(nodos)

    cost_matrix = np.full((n, n), 1e6, dtype=float)
    np.fill_diagonal(cost_matrix, 0)

    for _, row in df.iterrows():
        try:
            origen = row['origin']
            destino = row['destination']
            costo = float(row['weight'])
            cost_matrix[node_to_index[origen], node_to_index[destino]] = costo
        except KeyError:
            logger.error("Missing columns 'origin', 'destination', or 'weight'.")
            raise
        except ValueError:
            logger.error("Invalid cost value on row %s.", _)
            raise

    return cost_matrix, node_to_index


@register_keras_serializable()
class HopfieldLayer(Layer):
    def __init__(self, n, distance_matrix, **kwargs):
        super(HopfieldLayer, self).__init__(**kwargs)
        self.n = n
        self.distance_matrix = tf.constant(distance_matrix, dtype=tf.float32)
        self.x = self.add_weight(
            name="x",
            shape=(n, n),
            initializer="random_uniform",
            trainable=True
        )
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.valid_arcs = tf.constant(
            (self.distance_matrix.numpy() != 0).astype(np.float32), dtype=tf.float32
        )

    def energy(self):
        mu1 = 1.0
        mu2 = 10.0
        mu3 = 10.0
        path_cost = tf.reduce_sum(self.distance_matrix * self.x)
        row_constraint = tf.reduce_sum(tf.square(tf.reduce_sum(self.x, axis=1) - 1))
        col_constraint = tf.reduce_sum(tf.square(tf.reduce_sum(self.x, axis=0) - 1))
        binary_constraint = tf.reduce_sum(tf.square(self.x * (1 - self.x)))
        invalid_arcs_penalty = tf.reduce_sum(tf.square(self.x * (1 - self.valid_arcs)))
        return (
            (mu1 / 2) * path_cost + (mu2 / 2) * row_constraint
            + (mu2 / 2) * col_constraint + (mu3 / 2) * binary_constraint
            + (mu2 * 5) * invalid_arcs_penalty
        )

    def fine_tune_with_constraints(self, source, destination, iterations=500):
        logger.info("Initial Energy: %s", self.energy().numpy())
        for i in range(iterations):
            with tf.GradientTape() as tape:
                source_out = tf.reduce_sum(self.x[source, :]) - 1
                term5 = (10.0 / 2) * tf.square(source_out)
                dest_in = tf.reduce_sum(self.x[:, destination]) - 1
                term6 = (10.0 / 2) * tf.square(dest_in)
                energy = self.energy() + term5 + term6

            gradients = tape.gradient(energy, [self.x])
            self.optimizer.apply_gradients(zip(gradients, [self.x]))
            self.x.assign(tf.clip_by_value(self.x, 0.0, 1.0))
            # Mask invalid arcs
            self.x.assign(self.x * self.valid_arcs)
            if i % 100 == 0:
                logger.info("Fine-Tuning Iteration %d, Energy: %s", i, energy.numpy())
        logger.info("Final Energy: %s", self.energy().numpy())
        return self.x

    def call(self, inputs, training=False):
        return self.x

    def compile(self, optimizer):
        super(HopfieldLayer, self).compile()

    def get_config(self):
        config = super(HopfieldLayer, self).get_config()
        config.update({
            'n': self.n,
            'distance_matrix': self.distance_matrix.numpy().tolist(),
            'x': self.x.numpy().tolist(),
            'valid_arcs': self.valid_arcs.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        x = config.pop('x')
        valid_arcs = config.pop('valid_arcs')
        distance_matrix = config.pop('distance_matrix')
        n = config.pop('n')
        instance = cls(n=n, distance_matrix=distance_matrix)
        instance.x.assign(tf.constant(x, dtype=tf.float32))
        instance.valid_arcs.assign(tf.constant(valid_arcs, dtype=tf.float32))
        return instance


@register_keras_serializable()
class HopfieldModel(Model):
    def __init__(self, n, distance_matrix, **kwargs):
        super(HopfieldModel, self).__init__(**kwargs)
        self.hopfield_layer = HopfieldLayer(n, distance_matrix)
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.cost_matrix = None

    def set_cost_matrix(self, cost_matrix):
        self.cost_matrix = cost_matrix

    def get_cost_matrix(self):
        return self.cost_matrix

    def _dijkstra_path(self, source, destination):
        """Dijkstra's algorithm for fallback and validation."""
        n = len(self.cost_matrix)
        dist = np.full(n, np.inf)
        dist[source] = 0
        parent = np.full(n, -1)
        visited = set()

        for _ in range(n):
            u = None
            for i in range(n):
                if i not in visited and (u is None or dist[i] < dist[u]):
                    u = i
            if u is None or dist[u] == np.inf:
                break
            visited.add(u)

            for v in range(n):
                if self.cost_matrix[u][v] < 1e6:
                    if dist[u] + self.cost_matrix[u][v] < dist[v]:
                        dist[v] = dist[u] + self.cost_matrix[u][v]
                        parent[v] = u

        # Reconstruct path
        if dist[destination] == np.inf:
            return None, np.inf

        path = []
        current = destination
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()

        return path, dist[destination]

    def _calculate_path_cost(self, path):
        """Calculate total cost of a path."""
        cost = 0
        for i in range(len(path) - 1):
            edge_cost = self.cost_matrix[path[i]][path[i + 1]]
            if edge_cost >= 1e6:
                return np.inf
            cost += edge_cost
        return cost

    def validate_path(self, path, source, destination):
        """Validate path correctness and compare with Dijkstra."""
        if not path or path[0] != source or path[-1] != destination:
            return False, "Invalid path endpoints"

        # Check connectivity
        for i in range(len(path) - 1):
            if self.cost_matrix[path[i]][path[i + 1]] >= 1e6:
                return False, f"Invalid edge between {path[i]} and {path[i+1]}"

        # Compare with Dijkstra
        dijkstra_path, dijkstra_cost = self._dijkstra_path(source, destination)
        hopfield_cost = self._calculate_path_cost(path)

        if dijkstra_cost == np.inf:
            return False, "No path exists between nodes"

        accuracy = (dijkstra_cost / hopfield_cost * 100) if hopfield_cost > 0 else 0
        is_optimal = abs(hopfield_cost - dijkstra_cost) < 1e-6

        return True, {
            "is_optimal": is_optimal,
            "hopfield_cost": float(hopfield_cost),
            "dijkstra_cost": float(dijkstra_cost),
            "accuracy_percent": float(accuracy)
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.hopfield_layer.energy()
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def call(self, inputs, training=False):
        return self.hopfield_layer(inputs, training=training)

    def predict_path(self, source, destination, validate=True):
        if source == destination:
            return [source]

        self.hopfield_layer.fine_tune_with_constraints(source, destination)
        state_matrix = tf.round(self.hopfield_layer.x).numpy()

        def extract_path(state_matrix, source, destination):
            path = [source]
            current = source
            visited = {source}
            max_steps = len(state_matrix)

            for _ in range(max_steps):
                next_node = np.argmax(state_matrix[current])

                # Check if there's a valid edge
                if state_matrix[current][next_node] < 0.5:
                    logger.warning("No valid edge from node %s", current)
                    return None

                # Check if we reached destination
                if next_node == destination:
                    path.append(destination)
                    return path

                # Check for cycles
                if next_node in visited:
                    logger.warning("Cycle detected at node %s", next_node)
                    return None

                path.append(next_node)
                visited.add(next_node)
                current = next_node

            logger.warning("Max steps reached without finding destination")
            return None

        best_path = extract_path(state_matrix, source, destination)

        # Fallback to Dijkstra
        if validate and self.cost_matrix is not None:
            dijkstra_path, dijkstra_cost = self._dijkstra_path(source, destination)

            if dijkstra_path is None:
                raise ValueError(f"No path exists from {source} to {destination}")

            if best_path is None:
                logger.warning("Hopfield failed, using Dijkstra")
                best_path = dijkstra_path
            else:
                is_valid, _ = self.validate_path(best_path, source, destination)
                if not is_valid:
                    logger.warning("Hopfield invalid, using Dijkstra")
                    best_path = dijkstra_path

        if best_path is None:
            raise ValueError(f"Failed to find path from {source} to {destination}")

        return best_path

    def get_config(self):
        config = super(HopfieldModel, self).get_config()
        config.update({
            'n': self.hopfield_layer.n,
            'distance_matrix': self.hopfield_layer.distance_matrix.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        n = config['n']
        distance_matrix = config['distance_matrix']
        return cls(n=n, distance_matrix=distance_matrix)


def train_offline_model(adjacency_matrix_path: str) -> None:
    logger.info("Training offline model")
    logger.info("Adjacency matrix path: %s", adjacency_matrix_path)

    logger.info("Calculating cost matrix")
    try:
        cost_matrix, node_mapping = calculate_cost_matrix(adjacency_matrix_path)
    except Exception as e:
        logger.error("Error calculating cost matrix: %s", str(e))
        raise

    # Normalize cost matrix
    cost_matrix_normalized = (
        (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)
    )
    distance_matrix = cost_matrix_normalized

    n = distance_matrix.shape[0]
    logger.info("Number of nodes: %d", n)

    logger.info("Creating model")
    model = HopfieldModel(n, distance_matrix)
    logger.info("Setting cost matrix")
    logger.debug(cost_matrix)
    model.set_cost_matrix(cost_matrix)
    logger.debug(model.get_cost_matrix())
    logger.info("Compiling model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

    dummy_target = tf.zeros((1, n, n), dtype=tf.float32)

    logger.info("Training model")
    model(dummy_target)
    model.fit(dummy_target, epochs=1000)

    if 'PYTEST_CURRENT_TEST' in os.environ:
        logger.info("Function is being called by a pytest test")
        model_save_path = "data/synthetic/tests/"
    else:
        logger.info("Function is being called during real execution")
        model_save_path = "models/"

    os.makedirs(model_save_path, exist_ok=True)
    logger.info("Saving model to: %s", model_save_path)
    model.save(model_save_path + 'trained_model.keras')

    with open(model_save_path + 'cost_matrix.pkl', 'wb') as f:
        pickle.dump(cost_matrix, f)
    logger.info("Cost matrix saved")

    with open(model_save_path + 'cost_matrix.pkl', 'rb') as f:
        loaded_cost_matrix = pickle.load(f)

    if np.array_equal(loaded_cost_matrix, cost_matrix):
        logger.info("Cost matrix has been correctly saved and verified.")
    else:
        logger.error("Mismatch in the saved cost matrix.")

    if os.path.exists(model_save_path):
        logger.info("Model successfully saved to %s", model_save_path)
    else:
        logger.error("Failed to save the model to %s", model_save_path)
