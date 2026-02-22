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
    """Calculates the cost matrix for an adjacency matrix."""
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
            logger.error(f"Invalid cost value on row {_}.")
            raise

    return cost_matrix, node_to_index


@register_keras_serializable()
class AdvancedHopfieldLayer(Layer):
    """
    Advanced Hopfield layer with:
    - Sparse tensor support for large graphs
    - Adaptive hyperparameters
    - Learned attention weights
    """
    def __init__(self, n, distance_matrix, use_sparse=False, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.use_sparse = use_sparse

        # Calculate graph properties for adaptive weights
        self.density = np.sum(distance_matrix < 1e6) / (n * n)
        self.avg_degree = np.sum(distance_matrix < 1e6) / n

        if use_sparse and self.density < 0.3:
            # Use sparse representation for sparse graphs
            self._init_sparse(distance_matrix)
        else:
            # Use dense representation
            self._init_dense(distance_matrix)

        # Adaptive hyperparameters based on graph properties
        self.mu1, self.mu2, self.mu3 = self._calculate_adaptive_weights()

        # Learnable attention weights for edges
        self.attention_logits = self.add_weight(
            name="attention",
            shape=(n, n),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True
        )

    def _init_dense(self, distance_matrix):
        """Initialize dense tensors."""
        self.distance_matrix = tf.constant(distance_matrix, dtype=tf.float32)
        self.valid_arcs = tf.constant((distance_matrix < 1e6).astype(np.float32), dtype=tf.float32)

        self.logits = self.add_weight(
            name="logits",
            shape=(self.n, self.n),
            initializer=tf.keras.initializers.RandomNormal(mean=-2.0, stddev=0.5),
            trainable=True
        )

    def _init_sparse(self, distance_matrix):
        """Initialize sparse tensors for large graphs."""
        # Find valid edges
        valid_mask = distance_matrix < 1e6
        indices = np.argwhere(valid_mask)
        values = distance_matrix[valid_mask]

        self.edge_indices = tf.constant(indices, dtype=tf.int64)
        self.edge_values = tf.constant(values, dtype=tf.float32)
        self.num_edges = len(values)

        # Only optimize over valid edges
        self.edge_logits = self.add_weight(
            name="edge_logits",
            shape=(self.num_edges,),
            initializer=tf.keras.initializers.RandomNormal(mean=-2.0, stddev=0.5),
            trainable=True
        )

        # Store valid arcs for masking
        self.valid_arcs = tf.constant(valid_mask.astype(np.float32), dtype=tf.float32)

    def _calculate_adaptive_weights(self):
        """Calculate adaptive hyperparameters based on graph properties."""
        # Denser graphs need stronger constraints
        mu1 = 1.0
        mu2 = 10.0 * (1.0 + self.density)
        mu3 = 10.0 * (1.0 + self.density)

        logger.info(f"Adaptive weights: μ1={mu1:.2f}, μ2={mu2:.2f}, μ3={mu3:.2f}")
        logger.info(f"Graph density: {self.density:.3f}, Avg degree: {self.avg_degree:.1f}")

        return mu1, mu2, mu3

    def _get_decision_matrix(self, temperature):
        """Get decision matrix (sparse or dense)."""
        if self.use_sparse:
            # Sparse: only compute for valid edges
            edge_probs = tf.nn.sigmoid(self.edge_logits / temperature)

            # Create sparse tensor
            x_sparse = tf.SparseTensor(
                indices=self.edge_indices,
                values=edge_probs,
                dense_shape=(self.n, self.n)
            )
            x = tf.sparse.to_dense(x_sparse)
        else:
            # Dense: compute for all edges
            # Apply attention mechanism
            attention = tf.nn.softmax(self.attention_logits, axis=1)
            x = tf.nn.sigmoid(self.logits / temperature) * attention
            x = x * self.valid_arcs

        return x

    def energy(self, source, destination, temperature=0.5):
        """Energy function with adaptive weights and attention."""
        x = self._get_decision_matrix(temperature)

        # Flow conservation
        flow_penalty = 0.0
        for i in range(self.n):
            out_flow = tf.reduce_sum(x[i, :])
            in_flow = tf.reduce_sum(x[:, i])

            if i == source:
                flow_penalty += tf.square(out_flow - in_flow - 1.0)
            elif i == destination:
                flow_penalty += tf.square(in_flow - out_flow - 1.0)
            else:
                flow_penalty += tf.square(out_flow - in_flow)

        # Path cost with attention
        if self.use_sparse:
            path_cost = tf.reduce_sum(self.edge_values * tf.gather_nd(x, self.edge_indices))
        else:
            path_cost = tf.reduce_sum(self.distance_matrix * x)

        # Binary constraint
        binary_penalty = tf.reduce_sum(x * (1.0 - x))

        # Connectivity penalty
        connectivity_penalty = self._connectivity_penalty(x, source, destination)

        # Sparsity penalty (encourage fewer edges)
        sparsity_penalty = tf.reduce_sum(x) / (self.n * self.n)

        # Normalized combination with adaptive weights
        n_edges = tf.reduce_sum(self.valid_arcs)
        normalized_cost = path_cost / (n_edges + 1e-6)
        normalized_flow = flow_penalty / self.n
        normalized_binary = binary_penalty / (self.n * self.n)

        return (self.mu1 * normalized_cost +
                self.mu2 * normalized_flow +
                self.mu3 * normalized_binary +
                20.0 * connectivity_penalty +
                5.0 * sparsity_penalty)

    def _connectivity_penalty(self, x, source, destination):
        """Penalize if destination is not reachable from source."""
        reachability = x
        for _ in range(min(self.n - 1, 10)):  # Limit iterations for large graphs
            reachability = tf.minimum(
                reachability + tf.matmul(reachability, x),
                1.0
            )
        return tf.square(1.0 - reachability[source, destination])

    def optimize(self, source, destination, iterations=300, tolerance=1e-6, lr_schedule=None):
        """Optimize with learning rate scheduling and momentum."""
        if lr_schedule is None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.05,
                decay_steps=50,
                decay_rate=0.9
            )

        optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

        # Reinitialize
        if self.use_sparse:
            self.edge_logits.assign(
                tf.random.normal((self.num_edges,), mean=-2.0, stddev=0.5)
            )
        else:
            self.logits.assign(
                tf.random.normal((self.n, self.n), mean=-2.0, stddev=0.5)
            )

        prev_energy = float('inf')
        patience = 30
        no_improvement = 0
        best_energy = float('inf')
        best_state = None

        logger.info(f"Optimizing path from {source} to {destination}")

        for i in range(iterations):
            # Anneal temperature
            temperature = max(0.05, 1.0 - i / iterations)

            with tf.GradientTape() as tape:
                energy = self.energy(source, destination, temperature)

            # Get trainable variables
            if self.use_sparse:
                trainable_vars = [self.edge_logits, self.attention_logits]
            else:
                trainable_vars = [self.logits, self.attention_logits]

            gradients = tape.gradient(energy, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Track best solution
            if energy.numpy() < best_energy:
                best_energy = energy.numpy()
                best_state = self._get_decision_matrix(0.1).numpy()

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
                current_lr = optimizer.learning_rate
                lr = float(current_lr(i)) if callable(current_lr) else float(current_lr)
                logger.info(f"Iter {i}, Energy: {energy.numpy():.4f}, Temp: {temperature:.3f}, LR: {lr:.5f}")

        return best_state if best_state is not None else self._get_decision_matrix(0.1).numpy()

    def call(self, inputs, training=False):
        return self._get_decision_matrix(0.5)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n': self.n,
            'use_sparse': self.use_sparse,
            'distance_matrix': self.distance_matrix.numpy().tolist() if not self.use_sparse else None,
        })
        return config

    @classmethod
    def from_config(cls, config):
        n = config['n']
        use_sparse = config.get('use_sparse', False)
        distance_matrix = config.get('distance_matrix')
        if distance_matrix is None:
            distance_matrix = np.zeros((n, n))
        return cls(n=n, distance_matrix=distance_matrix, use_sparse=use_sparse)


@register_keras_serializable()
class AdvancedHopfieldModel(Model):
    """
    Advanced Hopfield model with:
    - Beam search for path extraction
    - Parallel multi-start optimization
    - Adaptive fallback strategy
    """
    def __init__(self, n, distance_matrix, use_sparse=False, **kwargs):
        super().__init__(**kwargs)
        self.hopfield_layer = AdvancedHopfieldLayer(n, distance_matrix, use_sparse)
        self.cost_matrix = None

    def set_cost_matrix(self, cost_matrix):
        self.cost_matrix = cost_matrix

    def get_cost_matrix(self):
        return self.cost_matrix

    def _dijkstra_path(self, source, destination):
        """Dijkstra's algorithm with path reconstruction."""
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

        if dist[destination] == np.inf:
            return None, np.inf

        path = []
        current = destination
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()

        return path, dist[destination]

    def _beam_search_path(self, state_matrix, source, destination, beam_width=5, threshold=0.3):
        """
        Beam search for path extraction - explores multiple paths simultaneously.
        """
        n = state_matrix.shape[0]

        # Initialize beam with source node
        beam = [(source, [source], 0.0)]  # (current_node, path, score)

        for step in range(n):
            candidates = []

            for current, path, score in beam:
                if current == destination:
                    candidates.append((current, path, score))
                    continue

                # Get valid next nodes
                for next_node in range(n):
                    if next_node not in path and state_matrix[current, next_node] > threshold:
                        edge_score = state_matrix[current, next_node]
                        new_score = score + edge_score
                        candidates.append((next_node, path + [next_node], new_score))

            if not candidates:
                break

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: -x[2])
            beam = candidates[:beam_width]

            # Check if we found destination
            for current, path, score in beam:
                if current == destination:
                    return path

        return None

    def _extract_path_advanced(self, state_matrix, source, destination):
        """
        Advanced path extraction using multiple strategies.
        """
        # Strategy 1: Beam search
        path = self._beam_search_path(state_matrix, source, destination, beam_width=5)
        if path:
            return path

        # Strategy 2: Greedy with backtracking
        path = self._greedy_with_backtracking(state_matrix, source, destination)
        if path:
            return path

        # Strategy 3: BFS on thresholded graph
        path = self._bfs_extraction(state_matrix, source, destination, threshold=0.3)
        if path:
            return path

        return None

    def _greedy_with_backtracking(self, state_matrix, source, destination, max_backtracks=3):
        """Greedy path extraction with limited backtracking."""
        def search(current, path, visited, backtracks_left):
            if current == destination:
                return path

            # Get neighbors sorted by edge weight
            neighbors = []
            for next_node in range(len(state_matrix)):
                if next_node not in visited and state_matrix[current, next_node] > 0.3:
                    neighbors.append((next_node, state_matrix[current, next_node]))

            neighbors.sort(key=lambda x: -x[1])

            for next_node, weight in neighbors:
                result = search(next_node, path + [next_node], visited | {next_node}, backtracks_left)
                if result:
                    return result

            # Backtrack if allowed
            if backtracks_left > 0 and len(path) > 1:
                return search(path[-2], path[:-1], visited - {current}, backtracks_left - 1)

            return None

        return search(source, [source], {source}, max_backtracks)

    def _bfs_extraction(self, state_matrix, source, destination, threshold=0.3):
        """BFS-based path extraction."""
        n = state_matrix.shape[0]
        adj = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(n):
                if state_matrix[i, j] > threshold and self.hopfield_layer.valid_arcs[i, j] > 0:
                    adj[i].append((j, state_matrix[i, j]))

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            node, path = queue.popleft()

            if node == destination:
                return path

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

    def predict(self, source, destination, num_restarts=3, validate=True, use_beam_search=True):
        """
        Predict shortest path with advanced strategies.
        """
        if source == destination:
            return [source]

        best_path = None
        best_cost = float('inf')

        # Multi-start optimization
        for restart in range(num_restarts):
            logger.info(f"Restart {restart + 1}/{num_restarts}")

            try:
                state_matrix = self.hopfield_layer.optimize(source, destination)

                # Extract path with advanced strategies
                if use_beam_search:
                    path = self._extract_path_advanced(state_matrix, source, destination)
                else:
                    path = self._bfs_extraction(state_matrix, source, destination)

                if path:
                    cost = self._calculate_path_cost(path)
                    logger.info(f"Found path with cost: {cost:.2f}")

                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
            except Exception as e:
                logger.warning(f"Restart {restart} failed: {str(e)}")
                continue

        # Adaptive fallback strategy
        if validate and self.cost_matrix is not None:
            dijkstra_path, dijkstra_cost = self._dijkstra_path(source, destination)

            if dijkstra_path is None:
                raise ValueError(f"No path exists from {source} to {destination}")

            if best_path is None:
                logger.warning("Hopfield failed, using Dijkstra")
                return dijkstra_path

            # Adaptive threshold based on graph complexity
            threshold = 90 if self.hopfield_layer.density > 0.5 else 95
            accuracy = (dijkstra_cost / best_cost * 100) if best_cost > 0 else 0

            logger.info(f"Hopfield: {best_cost:.2f}, Dijkstra: {dijkstra_cost:.2f}, Accuracy: {accuracy:.1f}%")

            if accuracy < threshold:
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
            'use_sparse': self.hopfield_layer.use_sparse,
        })
        return config

    @classmethod
    def from_config(cls, config):
        n = config['n']
        use_sparse = config.get('use_sparse', False)
        return cls(n=n, distance_matrix=np.zeros((n, n)), use_sparse=use_sparse)


def train_advanced_model(adjacency_matrix_path: str, use_sparse=False) -> None:
    """Train advanced Hopfield model with automatic sparse detection."""
    logger.info("Training advanced Hopfield model")

    try:
        cost_matrix, node_mapping = calculate_cost_matrix(adjacency_matrix_path)
    except Exception as e:
        logger.error(f"Error calculating cost matrix: {str(e)}")
        raise

    # Auto-detect if sparse representation is beneficial
    n = cost_matrix.shape[0]
    density = np.sum(cost_matrix < 1e6) / (n * n)
    use_sparse = use_sparse or (density < 0.3 and n > 100)

    logger.info(f"Nodes: {n}, Density: {density:.3f}, Using sparse: {use_sparse}")

    # Normalize
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    # Create model
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=use_sparse)
    model.set_cost_matrix(cost_matrix)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))

    # Build model
    dummy_input = tf.zeros((1, n, n), dtype=tf.float32)
    model(dummy_input)

    # Save
    model_save_path = "data/synthetic/tests/" if 'PYTEST_CURRENT_TEST' in os.environ else "models/"
    os.makedirs(model_save_path, exist_ok=True)

    logger.info(f"Saving model to: {model_save_path}")
    model.save(model_save_path + 'trained_model_advanced.keras')

    with open(model_save_path + 'cost_matrix_advanced.pkl', 'wb') as f:
        pickle.dump(cost_matrix, f)

    logger.info("Advanced model training complete")
