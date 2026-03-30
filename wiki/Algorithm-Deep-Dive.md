# Algorithm Deep Dive

## Input: CSV to Cost Matrix

CSV format (required columns: `origin`, `destination`, `weight`):

```csv
origin,destination,weight
0,2,762
0,3,863
0,43,30
43,9,30
```

Parsing:

```python
df = pd.read_csv(path, usecols=['origin', 'destination', 'weight'],
                  dtype={'origin': str, 'destination': str, 'weight': float})

# Auto-generate node mapping
nodes = sorted(pd.unique(df[['origin', 'destination']].values.ravel()))
node_to_index = {node: idx for idx, node in enumerate(nodes)}
n = len(nodes)

# Initialize cost matrix with 1e6 (infinity) for missing edges
cost_matrix = np.full((n, n), 1e6, dtype=float)
np.fill_diagonal(cost_matrix, 0)

# Fill from CSV
for _, row in df.iterrows():
    i = node_to_index[row['origin']]
    j = node_to_index[row['destination']]
    cost_matrix[i][j] = float(row['weight'])
```

The graph is directed. Missing edges are represented as 1e6 (treated as infinite cost). `valid_arcs[i][j] = 1 if cost_matrix[i][j] < 1e6 else 0`.

## Normalization

Before optimization:

```python
cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (
    np.max(cost_matrix) - np.min(cost_matrix) + 1e-6
)
```

The +1e-6 prevents division by zero when all costs are identical. The original cost matrix is preserved for final cost calculation.

## Initialization Per Query

```python
def optimize(self, source, destination, iterations=300, tolerance=1e-6):
    # CRITICAL: fresh optimizer prevents momentum carryover between queries
    optimizer = tf.optimizers.Adam(learning_rate=0.02)

    # CRITICAL: reinitialize logits for fresh start
    self.logits.assign(
        tf.random.normal((self.n, self.n), mean=-2.0, stddev=0.5)
    )
```

Mean=-2.0 initializes neurons with negative logits, so `sigmoid(-2) ≈ 0.12` — most edges start "off", which reflects the expectation that most edges won't be in the shortest path.

## Gradient Descent with Temperature Annealing

```python
for i in range(iterations):
    # Temperature decreases from 1.0 to 0.1 linearly
    temperature = max(0.1, 1.0 - i / iterations)

    with tf.GradientTape() as tape:
        energy = self.energy(source, destination, temperature)

    gradients = tape.gradient(energy, [self.logits])
    optimizer.apply_gradients(zip(gradients, [self.logits]))

    # Early stopping
    if abs(prev_energy - energy.numpy()) < tolerance:
        no_improvement += 1
        if no_improvement >= patience:  # patience=20
            break
    else:
        no_improvement = 0
    prev_energy = energy.numpy()
```

At high temperature (T=1.0), sigmoid is smooth — the energy landscape is smooth, gradients are informative, and the network explores broadly. At low temperature (T=0.1), sigmoid is sharp — activations are pushed toward 0 or 1, and the network commits to a specific path.

## Connectivity Penalty

The connectivity penalty ensures the network can reach the destination:

```python
def _connectivity_penalty(self, x, source, destination):
    reachability = x
    # n-1 matrix multiplications: computes transitive closure
    for _ in range(self.n - 1):
        reachability = tf.minimum(
            reachability + tf.matmul(reachability, x),
            1.0
        )
    return tf.square(1.0 - reachability[source, destination])
```

After n-1 iterations, `reachability[i][j] ≈ 1` if there exists any path from i to j using selected edges, 0 otherwise. The squared penalty makes dead-end configurations (source can't reach destination) energetically very unfavorable.

## Multi-Start Optimization

```python
best_path, best_cost = None, float('inf')

for restart in range(num_restarts):    # 3 restarts
    state_matrix = self.hopfield_layer.optimize(source, destination).numpy()
    path = self._extract_path_robust(state_matrix, source, destination)
    if path:
        cost = self._calculate_path_cost(path)
        if cost < best_cost:
            best_path = path
            best_cost = cost
```

Each restart reinitializes logits and creates a fresh optimizer. Different random starting points explore different regions of the energy landscape.

## Path Extraction: BFS (Improved) vs Beam Search (Advanced)

### Improved Model: BFS

```python
# Build adjacency from edges with activation > threshold (0.5)
adj = {}
for i, j:
    if state_matrix[i][j] > 0.5 and valid_arcs[i][j] > 0:
        adj[i].append((j, state_matrix[i][j]))  # include edge weight

# BFS, prioritize high-activation edges
queue = deque([(source, [source])])
visited = {source}

while queue:
    node, path = queue.popleft()
    if node == destination:
        return path
    for next_node, weight in sorted(adj[node], key=lambda x: -x[1]):
        if next_node not in visited:
            queue.append((next_node, path + [next_node]))
```

BFS guarantees finding the path if one exists in the selected edges. Sorting neighbors by activation weight means high-confidence edges are explored first.

### Advanced Model: Beam Search

```python
def _beam_search_path(self, state_matrix, source, destination, beam_width=5):
    beam = [(source, [source], 0.0)]  # (node, path, score)

    for step in range(n):
        candidates = []
        for current, path, score in beam:
            if current == destination:
                return path  # found!
            for next_node in range(n):
                if next_node not in path and state_matrix[current][next_node] > 0.3:
                    new_score = score + state_matrix[current][next_node]
                    candidates.append((next_node, path + [next_node], new_score))

        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]   # keep top 5

    return None
```

Beam search maintains the top-5 partial paths at each step, exploring more options than greedy but fewer than exhaustive BFS.

## Dijkstra Fallback

```python
def _dijkstra_path(self, source, destination):
    dist = np.full(n, np.inf)
    dist[source] = 0
    parent = np.full(n, -1, dtype=int)
    visited = set()

    for _ in range(n):
        # Find unvisited node with minimum distance
        u = min((i for i in range(n) if i not in visited), key=lambda i: dist[i])
        if dist[u] == np.inf:
            break
        visited.add(u)
        for v in range(n):
            if cost_matrix[u][v] < 1e6:   # valid edge
                if dist[u] + cost_matrix[u][v] < dist[v]:
                    dist[v] = dist[u] + cost_matrix[u][v]
                    parent[v] = u

    # Reconstruct path
    path, current = [], destination
    while current != -1:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path, dist[destination]
```

Fallback is triggered when:
1. Hopfield finds no valid path after all restarts
2. Hopfield path cost > Dijkstra cost × 1.05 (>5% suboptimal for Improved, adaptive for Advanced)

## Numerical Stability: Why Separate Positive/Negative Paths in Sigmoid

```python
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + exp(-x))    # exp(-large) → 0, no overflow
    else:
        return exp(x) / (1 + exp(x)) # exp(large) avoided
```

For `x = 1000`: the naive `1/(1+exp(-1000))` computes `exp(1000)` which overflows. The stable version computes `exp(-1000)` which underflows to 0 (safe).

For `x = -1000`: `1/(1+exp(1000))` overflows. The stable version computes `exp(-1000) / (1 + exp(-1000))` which safely evaluates to ≈ 0.
