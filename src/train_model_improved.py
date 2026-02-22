import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
import pickle
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_cost_matrix(adjacency_matrix):
    """
    Calculates the cost matrix for an adjacency matrix of a network given in a CSV file.
    """
    try:
        df = pd.read_csv(adjacency_matrix, usecols=['origin', 'destination', 'weight'])
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
class ImprovedHopfieldLayer(Layer):
    def __init__(self, n, distance_matrix, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.distance_matrix = tf.constant(distance_matrix, dtype=tf.float32)
        self.valid_arcs = tf.constant((distance_matrix < 1e6).astype(np.float32), dtype=tf.float32)

        # Use logits for better optimization
        self.logits = self.add_weight(
            name="logits",
            shape=(n, n),
            initializer=tf.keras.initializers.RandomNormal(mean=-2.0, stddev=0.5),
            trainable=True
        )

    def energy(self, source, destination, temperature=0.5):
        """
        Corrected energy function for shortest path with flow conservation.
        """
        # Sigmoid with temperature for soft binary decisions
        x = tf.nn.sigmoid(self.logits / temperature)
        x = x * self.valid_arcs

        # Flow conservation constraints (FIXED)
        flow_penalty = 0.0
        for i in range(self.n):
            out_flow = tf.reduce_sum(x[i, :])
            in_flow = tf.reduce_sum(x[:, i])

            if i == source:
                # Source: out_flow - in_flow = 1
                flow_penalty += tf.square(out_flow - in_flow - 1.0)
            elif i == destination:
                # Destination: in_flow - out_flow = 1
                flow_penalty += tf.square(in_flow - out_flow - 1.0)
            else:
                # Intermediate: in_flow = out_flow (or both 0)
                flow_penalty += tf.square(out_flow - in_flow)

        # Path cost
        path_cost = tf.reduce_sum(self.distance_matrix * x)

        # Binary constraint
        binary_penalty = tf.reduce_sum(x * (1.0 - x))

        # Connectivity penalty - ensure destination reachable from source
        connectivity_penalty = self._connectivity_penalty(x, source, destination)

        # Normalized combination
        n_edges = tf.reduce_sum(self.valid_arcs)
        normalized_cost = path_cost / (n_edges + 1e-6)
        normalized_flow = flow_penalty / self.n
        normalized_binary = binary_penalty / (self.n * self.n)

        return (normalized_cost +
                10.0 * normalized_flow +
                5.0 * normalized_binary +
                20.0 * connectivity_penalty)

    def _connectivity_penalty(self, x, source, destination):
        """Penalize if destination is not reachable from source."""
        # Compute reachability via matrix powers
        reachability = x
        for _ in range(self.n - 1):
            reachability = tf.minimum(
                reachability + tf.matmul(reachability, x),
                1.0
            )

        # Strong penalty if destination unreachable
        return tf.square(1.0 - reachability[source, destination])

    def optimize(self, source, destination, iterations=300, tolerance=1e-6):
        """
        Optimize for specific source-destination pair with early stopping.
        """
        # Fresh optimizer for each query (FIXED)
        optimizer = tf.optimizers.Adam(learning_rate=0.02)

        # Reinitialize logits
        self.logits.assign(
            tf.random.normal((self.n, self.n), mean=-2.0, stddev=0.5)
        )

        prev_energy = float('inf')
        patience = 20
        no_improvement = 0

        logger.info(f"Optimizing path from {source} to {destination}")

        for i in range(iterations):
            # Anneal temperature for sharper decisions
            temperature = max(0.1, 1.0 - i / iterations)

            with tf.GradientTape() as tape:
                energy = self.energy(source, destination, temperature)

            gradients = tape.gradient(energy, [self.logits])
            optimizer.apply_gradients(zip(gradients, [self.logits]))

            # Early stopping
            if abs(prev_energy - energy.numpy()) < tolerance:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.info(f"Converged at iteration {i}, Energy: {energy.numpy():.4f}")
                    break
            else:
                no_improvement = 0
            prev_energy = energy.numpy()

            if i % 50 == 0:
                logger.info(f"Iteration {i}, Energy: {energy.numpy():.4f}, Temp: {temperature:.3f}")

        # Return final soft assignment
        x_final = tf.nn.sigmoid(self.logits / 0.1) * self.valid_arcs
        return x_final

    def call(self, inputs, training=False):
        return tf.nn.sigmoid(self.logits) * self.valid_arcs

    def get_config(self):
        config = super().get_config()
        config.update({
            'n': self.n,
            'distance_matrix': self.distance_matrix.numpy().tolist(),
            'logits': self.logits.numpy().tolist(),
            'valid_arcs': self.valid_arcs.numpy().tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        logits = config.pop('logits')
        valid_arcs = config.pop('valid_arcs')
        distance_matrix = config.pop('distance_matrix')
        n = config.pop('n')

        instance = cls(n=n, distance_matrix=distance_matrix)
        instance.logits.assign(tf.constant(logits, dtype=tf.float32))
        instance.valid_arcs = tf.constant(valid_arcs, dtype=tf.float32)
        return instance


@register_keras_serializable()
class ImprovedHopfieldModel(Model):
    def __init__(self, n, distance_matrix, **kwargs):
        super().__init__(**kwargs)
        self.hopfield_layer = ImprovedHopfieldLayer(n, distance_matrix)
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

    def _extract_path_robust(self, state_matrix, source, destination, threshold=0.5):
        """
        Robust path extraction using BFS on edges above threshold.
        """
        n = state_matrix.shape[0]

        # Build adjacency list
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if state_matrix[i, j] > threshold and self.hopfield_layer.valid_arcs[i, j] > 0:
                    adj[i].append((j, state_matrix[i, j]))

        # BFS to find path
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            node, path = queue.popleft()

            if node == destination:
                return path

            # Sort neighbors by edge weight (prefer higher confidence)
            for next_node, weight in sorted(adj[node], key=lambda x: -x[1]):
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))

        return None

    def _calculate_path_cost(self, path):
        """Calculate total cost of a path."""
        if not path:
            return np.inf
        cost = 0
        for i in range(len(path) - 1):
            edge_cost = self.cost_matrix[path[i]][path[i + 1]]
            if edge_cost >= 1e6:
                return np.inf
            cost += edge_cost
        return cost

    def predict(self, source, destination, num_restarts=3, validate=True):
        """
        Predict shortest path with multi-start optimization and fallback.
        """
        n = self.hopfield_layer.n

        # Validate node indices
        if source < 0 or source >= n:
            raise ValueError(f"Source node {source} is out of range [0, {n-1}]")
        if destination < 0 or destination >= n:
            raise ValueError(f"Destination node {destination} is out of range [0, {n-1}]")

        if source == destination:
            return [source]

        best_path = None
        best_cost = float('inf')

        # Multi-start optimization
        for restart in range(num_restarts):
            logger.info(f"Restart {restart + 1}/{num_restarts}")

            try:
                # Optimize
                state_matrix = self.hopfield_layer.optimize(source, destination).numpy()

                # Extract path
                path = self._extract_path_robust(state_matrix, source, destination)

                if path:
                    cost = self._calculate_path_cost(path)
                    logger.info(f"Found path with cost: {cost:.2f}")

                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
            except Exception as e:
                logger.warning(f"Restart {restart} failed: {str(e)}")
                continue

        # Fallback to Dijkstra if Hopfield fails or produces suboptimal result
        if validate and self.cost_matrix is not None:
            dijkstra_path, dijkstra_cost = self._dijkstra_path(source, destination)

            if dijkstra_path is None:
                raise ValueError(f"No path exists from {source} to {destination}")

            if best_path is None:
                logger.warning("Hopfield failed, using Dijkstra")
                return dijkstra_path

            # Check if Hopfield solution is acceptable (within 5% of optimal)
            accuracy = (dijkstra_cost / best_cost * 100) if best_cost > 0 else 0
            logger.info(
                f"Hopfield cost: {best_cost:.2f}, Dijkstra cost: {dijkstra_cost:.2f}, "
                f"Accuracy: {accuracy:.1f}%"
            )

            if accuracy < 95:
                logger.warning(f"Hopfield suboptimal ({accuracy:.1f}%), using Dijkstra")
                return dijkstra_path

        if best_path is None:
            raise ValueError(f"Failed to find path from {source} to {destination}")

        return best_path

    def call(self, inputs, training=False):
        return self.hopfield_layer(inputs, training=training)

    def get_config(self):
        config = super().get_config()
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


def train_improved_model(adjacency_matrix_path: str) -> None:
    """
    Train improved Hopfield model (no meaningless offline training).
    """
    logger.info("Training improved Hopfield model")
    logger.info(f"Adjacency matrix path: {adjacency_matrix_path}")

    try:
        cost_matrix, node_mapping = calculate_cost_matrix(adjacency_matrix_path)
    except Exception as e:
        logger.error(f"Error calculating cost matrix: {str(e)}")
        raise

    # Normalize cost matrix for numerical stability
    cost_matrix_normalized = (
        (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)
    )
    distance_matrix = cost_matrix_normalized

    n = distance_matrix.shape[0]
    logger.info(f"Number of nodes: {n}")

    # Create model (no offline training needed)
    model = ImprovedHopfieldModel(n, distance_matrix)
    model.set_cost_matrix(cost_matrix)

    # Compile for saving
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02))

    # Build model by calling it once
    dummy_input = tf.zeros((1, n, n), dtype=tf.float32)
    model(dummy_input)

    # Determine save path
    if 'PYTEST_CURRENT_TEST' in os.environ:
        model_save_path = "data/synthetic/tests/"
    else:
        model_save_path = "models/"

    os.makedirs(model_save_path, exist_ok=True)

    logger.info(f"Saving model to: {model_save_path}")
    model.save(model_save_path + 'trained_model_improved.keras')

    with open(model_save_path + 'cost_matrix_improved.pkl', 'wb') as f:
        pickle.dump(cost_matrix, f)
    logger.info("Cost matrix saved")

    # Verify save
    with open(model_save_path + 'cost_matrix_improved.pkl', 'rb') as f:
        loaded_cost_matrix = pickle.load(f)

    if np.array_equal(loaded_cost_matrix, cost_matrix):
        logger.info("Cost matrix correctly saved and verified")
    else:
        logger.error("Mismatch in saved cost matrix")

    logger.info("Model training complete")
