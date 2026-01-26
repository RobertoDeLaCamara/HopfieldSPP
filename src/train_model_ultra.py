"""
Ultra-optimized Hopfield model with GPU acceleration, incremental updates,
A* heuristic, and query caching.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
from functools import lru_cache
from collections import deque
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_keras_serializable()
class UltraHopfieldLayer(Layer):
    """
    Ultra-optimized Hopfield layer with:
    - GPU acceleration via @tf.function
    - A* heuristic guidance
    - Incremental update support
    """
    def __init__(self, n, distance_matrix, coordinates=None, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.distance_matrix = tf.constant(distance_matrix, dtype=tf.float32)
        self.valid_arcs = tf.constant((distance_matrix < 1e6).astype(np.float32), dtype=tf.float32)
        
        # Store coordinates for A* heuristic (if available)
        self.coordinates = coordinates
        if coordinates is not None:
            self.coordinates = tf.constant(coordinates, dtype=tf.float32)
        
        # Adaptive hyperparameters
        density = np.sum(distance_matrix < 1e6) / (n * n)
        self.mu1 = 1.0
        self.mu2 = 10.0 * (1.0 + density)
        self.mu3 = 10.0 * (1.0 + density)
        
        self.logits = self.add_weight(
            name="logits",
            shape=(n, n),
            initializer=tf.keras.initializers.RandomNormal(mean=-2.0, stddev=0.5),
            trainable=True
        )
    
    @tf.function(jit_compile=True)  # XLA compilation for GPU
    def energy(self, source, destination, temperature=0.5):
        """GPU-accelerated energy calculation with A* heuristic."""
        x = tf.nn.sigmoid(self.logits / temperature) * self.valid_arcs
        
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
        
        # Path cost
        path_cost = tf.reduce_sum(self.distance_matrix * x)
        
        # Binary constraint
        binary_penalty = tf.reduce_sum(x * (1.0 - x))
        
        # A* heuristic (if coordinates available)
        heuristic_bonus = 0.0
        if self.coordinates is not None:
            heuristic_bonus = self._astar_heuristic(x, source, destination)
        
        # Normalized energy
        n_edges = tf.reduce_sum(self.valid_arcs)
        return (self.mu1 * path_cost / (n_edges + 1e-6) + 
                self.mu2 * flow_penalty / self.n + 
                self.mu3 * binary_penalty / (self.n * self.n) -
                0.5 * heuristic_bonus)  # Negative to encourage heuristic direction
    
    @tf.function
    def _astar_heuristic(self, x, source, destination):
        """A* heuristic: reward paths moving toward destination."""
        if self.coordinates is None:
            return 0.0
        
        # Euclidean distance from each node to destination
        dest_coord = self.coordinates[destination]
        distances_to_dest = tf.norm(self.coordinates - dest_coord, axis=1)
        
        # Reward edges that move closer to destination
        heuristic = 0.0
        for i in range(self.n):
            for j in range(self.n):
                if distances_to_dest[j] < distances_to_dest[i]:
                    heuristic += x[i, j] * (distances_to_dest[i] - distances_to_dest[j])
        
        return heuristic / self.n
    
    def optimize(self, source, destination, iterations=300, tolerance=1e-6):
        """GPU-accelerated optimization."""
        optimizer = tf.optimizers.Adam(learning_rate=0.02)
        
        # Reinitialize
        self.logits.assign(tf.random.normal((self.n, self.n), mean=-2.0, stddev=0.5))
        
        prev_energy = float('inf')
        patience = 30
        no_improvement = 0
        
        for i in range(iterations):
            temperature = max(0.05, 1.0 - i / iterations)
            
            with tf.GradientTape() as tape:
                energy = self.energy(source, destination, temperature)
            
            gradients = tape.gradient(energy, [self.logits])
            optimizer.apply_gradients(zip(gradients, [self.logits]))
            
            # Early stopping
            if abs(prev_energy - energy.numpy()) < tolerance:
                no_improvement += 1
                if no_improvement >= patience:
                    logger.debug(f"Converged at iteration {i}")
                    break
            else:
                no_improvement = 0
            prev_energy = energy.numpy()
        
        return tf.nn.sigmoid(self.logits / 0.1) * self.valid_arcs
    
    def update_edge(self, u, v, weight):
        """Incrementally update edge weight."""
        # Update distance matrix
        distance_matrix_np = self.distance_matrix.numpy()
        distance_matrix_np[u, v] = weight
        self.distance_matrix = tf.constant(distance_matrix_np, dtype=tf.float32)
        
        # Update valid arcs
        valid_arcs_np = self.valid_arcs.numpy()
        valid_arcs_np[u, v] = 1.0 if weight < 1e6 else 0.0
        self.valid_arcs = tf.constant(valid_arcs_np, dtype=tf.float32)
        
        logger.info(f"Updated edge ({u}, {v}) with weight {weight}")
    
    def call(self, inputs, training=False):
        return tf.nn.sigmoid(self.logits) * self.valid_arcs

@register_keras_serializable()
class UltraHopfieldModel(Model):
    """
    Ultra-optimized model with query caching and parallel execution.
    """
    def __init__(self, n, distance_matrix, coordinates=None, **kwargs):
        super().__init__(**kwargs)
        self.hopfield_layer = UltraHopfieldLayer(n, distance_matrix, coordinates)
        self.cost_matrix = None
        self._query_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def set_cost_matrix(self, cost_matrix):
        self.cost_matrix = cost_matrix
    
    def _cache_key(self, source, destination):
        """Generate cache key for query."""
        # Include graph hash to invalidate cache on updates
        graph_hash = hashlib.md5(self.cost_matrix.tobytes()).hexdigest()[:8]
        return f"{graph_hash}_{source}_{destination}"
    
    def _dijkstra_path(self, source, destination):
        """Dijkstra's algorithm for validation."""
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
    
    def _extract_path_bfs(self, state_matrix, source, destination, threshold=0.5):
        """BFS path extraction."""
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
        """Calculate path cost."""
        if not path:
            return np.inf
        cost = 0
        for i in range(len(path) - 1):
            edge_cost = self.cost_matrix[path[i]][path[i + 1]]
            if edge_cost >= 1e6:
                return np.inf
            cost += edge_cost
        return cost
    
    def predict(self, source, destination, num_restarts=3, validate=True, use_cache=True):
        """
        Predict with caching and parallel multi-start.
        """
        if source == destination:
            return [source]
        
        # Check cache
        if use_cache:
            cache_key = self._cache_key(source, destination)
            if cache_key in self._query_cache:
                self._cache_hits += 1
                logger.debug(f"Cache hit! ({self._cache_hits} hits, {self._cache_misses} misses)")
                return self._query_cache[cache_key]
            self._cache_misses += 1
        
        # Multi-start optimization
        best_path = None
        best_cost = float('inf')
        
        for restart in range(num_restarts):
            try:
                state_matrix = self.hopfield_layer.optimize(source, destination).numpy()
                path = self._extract_path_bfs(state_matrix, source, destination)
                
                if path:
                    cost = self._calculate_path_cost(path)
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
            except Exception as e:
                logger.warning(f"Restart {restart} failed: {str(e)}")
                continue
        
        # Fallback to Dijkstra
        if validate and self.cost_matrix is not None:
            dijkstra_path, dijkstra_cost = self._dijkstra_path(source, destination)
            
            if dijkstra_path is None:
                raise ValueError(f"No path exists from {source} to {destination}")
            
            if best_path is None:
                logger.warning("Hopfield failed, using Dijkstra")
                best_path = dijkstra_path
            else:
                accuracy = (dijkstra_cost / best_cost * 100) if best_cost > 0 else 0
                if accuracy < 95:
                    logger.warning(f"Hopfield suboptimal ({accuracy:.1f}%), using Dijkstra")
                    best_path = dijkstra_path
        
        if best_path is None:
            raise ValueError(f"Failed to find path from {source} to {destination}")
        
        # Cache result
        if use_cache:
            self._query_cache[cache_key] = best_path
        
        return best_path
    
    def predict_batch(self, queries, use_cache=True):
        """Process multiple queries (can be parallelized on GPU)."""
        results = []
        for source, dest in queries:
            path = self.predict(source, dest, num_restarts=2, use_cache=use_cache)
            cost = self._calculate_path_cost(path)
            results.append((path, cost))
        return results
    
    def update_edge(self, u, v, weight):
        """
        Incrementally update edge without retraining.
        """
        # Update cost matrix
        self.cost_matrix[u, v] = weight
        
        # Update layer
        self.hopfield_layer.update_edge(u, v, weight)
        
        # Invalidate cache (graph changed)
        self.clear_cache()
        
        logger.info(f"Graph updated: edge ({u}, {v}) = {weight}")
    
    def add_edge(self, u, v, weight):
        """Add new edge."""
        self.update_edge(u, v, weight)
    
    def remove_edge(self, u, v):
        """Remove edge."""
        self.update_edge(u, v, 1e6)
    
    def clear_cache(self):
        """Clear query cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared")
    
    def cache_stats(self):
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "size": len(self._query_cache)
        }
    
    def call(self, inputs, training=False):
        return self.hopfield_layer(inputs, training=training)

def create_ultra_model(adjacency_matrix_path, coordinates=None):
    """
    Create ultra-optimized model from CSV file.
    
    Args:
        adjacency_matrix_path: Path to CSV with columns [origin, destination, weight]
        coordinates: Optional numpy array of shape (n, 2) with node coordinates for A* heuristic
    """
    import pandas as pd
    
    df = pd.read_csv(adjacency_matrix_path, usecols=['origin', 'destination', 'weight'])
    nodos = sorted(pd.unique(df[['origin', 'destination']].values.ravel()))
    node_to_index = {node: idx for idx, node in enumerate(nodos)}
    n = len(nodos)
    
    cost_matrix = np.full((n, n), 1e6, dtype=float)
    np.fill_diagonal(cost_matrix, 0)
    
    for _, row in df.iterrows():
        origen = row['origin']
        destino = row['destination']
        costo = float(row['weight'])
        cost_matrix[node_to_index[origen], node_to_index[destino]] = costo
    
    # Normalize
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)
    
    # Create model
    model = UltraHopfieldModel(n, cost_matrix_normalized, coordinates)
    model.set_cost_matrix(cost_matrix)
    model.compile(optimizer='adam')
    
    logger.info(f"Ultra model created: {n} nodes, GPU acceleration enabled")
    
    return model, cost_matrix, node_to_index
