# Architecture & Model Tiers

## Three Model Tiers

| Model | File | Nodes | Memory | Reliability | Optimal Rate |
|-------|------|-------|--------|-------------|-------------|
| Original (deprecated) | `train_model.py` | ≤100 | O(n²) | 80–90% | 40–60% |
| ImprovedHopfieldModel | `train_model_improved.py` | ≤500 | O(n²) | 100% | 95–100% |
| AdvancedHopfieldModel | `train_model_advanced.py` | ≤5000+ | O(E) sparse | 100% | 98–100% |

## Why the Original Is Deprecated

The original model (`train_model.py`) encodes **Hamiltonian cycle constraints** — the same constraints used to solve the Traveling Salesman Problem. These constraints force each node to have exactly one incoming and one outgoing edge, producing a cycle rather than a path.

Shortest path requires **flow conservation constraints**:
- Source: `out_flow - in_flow = 1`
- Destination: `in_flow - out_flow = 1`
- Intermediate nodes: `out_flow = in_flow`

The original model also wastes 1000 training epochs on synthetic data that has no bearing on the query-time optimization. Both issues are fixed in the Improved model.

## ImprovedHopfieldModel Architecture

```
Input: n×n cost matrix C (float32)
       valid_arcs mask (1 where edge exists, 0 where C[i][j] ≥ 1e6)

Variables:
  logits: tf.Variable shape (n, n), initialized N(-2.0, 0.5) per query
  (reset at the start of each optimize() call)

Per query:
  1. Fresh Adam optimizer (lr=0.02) — prevents momentum carryover
  2. 300 iterations of gradient descent with early stopping (patience=20)
  3. Temperature annealing: T = max(0.1, 1.0 - iter/300)
  4. Multi-start: 3 restarts, keep best solution
  5. BFS path extraction on edges with activation > 0.5
  6. Dijkstra fallback if Hopfield fails or suboptimal (>5% of Dijkstra cost)
```

**Energy function**:
```
E = path_cost/num_edges
  + 10.0 × flow_penalty/n
  + 5.0 × binary_penalty/n²
  + 20.0 × connectivity_penalty

flow_penalty = Σ_i [
  (out[src] - in[src] - 1)²   ← source constraint
  (in[dst] - out[dst] - 1)²   ← destination constraint
  (out[i] - in[i])²            ← intermediate nodes
]
```

## AdvancedHopfieldModel Architecture

All features of Improved, plus:

```
Auto-sparse mode (when density < 0.3 AND n > 100):
  edge_logits: tf.Variable shape (E,)   ← only valid edges
  vs. dense:   tf.Variable shape (n,n)  ← all n² cells

Attention mechanism:
  attention_logits: tf.Variable shape (n, n)
  attention = softmax(attention_logits, axis=1)  ← row-normalized
  x = sigmoid(logits / T) * attention * valid_arcs

Adaptive hyperparameters:
  μ2 = 10.0 × (1 + density)   ← flow penalty scales with density
  μ3 = 10.0 × (1 + density)   ← binary penalty scales with density

Learning rate scheduling:
  ExponentialDecay(initial_lr=0.05, decay_steps=50, decay_rate=0.9)

Path extraction (3 strategies in priority order):
  1. Beam search (beam_width=5)
  2. Greedy with backtracking
  3. BFS on thresholded graph (threshold=0.3)

Best-state tracking:
  Stores lowest-energy state seen during all 300 iterations
  Returns best state, not final state
```

## Auto-Selection Logic

```python
# In train_advanced_model()
density = np.sum(cost_matrix < 1e6) / (n * n)
use_sparse = use_sparse or (density < 0.3 and n > 100)
```

For a 1000-node graph with 30% density:
- Dense: 1,000,000 logit parameters = 4 MB
- Sparse: 300,000 edge logits = 1.2 MB (3.3× smaller)
- At 10% density: 100,000 edge logits = 400 KB (10× smaller)

## API Architecture (main_improved.py)

```
POST /loadNetwork (CSV upload)
  │
  ├─ Validate file type (text/csv)
  ├─ Parse: origin, destination, weight columns
  ├─ train_improved_model(path)  ← computes cost matrix, saves .keras
  └─ invalidate_model_cache()    ← forces next predict to reload

GET /calculateShortestPath?origin=0&destination=9
  │
  ├─ Parse and validate: origin, destination as int
  ├─ get_cached_model()
  │   ├─ Cache hit → return (model, cost_matrix)
  │   └─ Cache miss → load_model() from disk → set cache
  └─ model.predict_path(origin, destination)
      ├─ Hopfield (3 restarts, 300 iterations each)
      ├─ Compare vs Dijkstra (if validate=True)
      └─ Return best path
```

## Model Persistence

```
models/
├── trained_model_improved.keras    TF SavedModel format (ImprovedHopfieldModel)
├── trained_model_improved.pkl      Cost matrix + node mapping (joblib)
├── trained_model_advanced.keras    (AdvancedHopfieldModel)
└── trained_model_advanced.pkl
```

Loading requires `custom_object_scope` to register both model and layer classes:

```python
with custom_object_scope({
    'ImprovedHopfieldModel': ImprovedHopfieldModel,
    'ImprovedHopfieldLayer': ImprovedHopfieldLayer
}):
    model = load_model('trained_model_improved.keras', custom_objects={...})
```
