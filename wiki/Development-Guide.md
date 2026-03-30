# Development Guide

## Prerequisites

- Python 3.10+
- TensorFlow (CPU) — `tensorflow-cpu` (no GPU needed)
- NumPy, pandas, FastAPI, joblib

## Setup

```bash
pip install -r requirements.txt
python3 src/main_improved.py   # API on :63235
```

## Running

```bash
# Improved API (recommended)
python3 src/main_improved.py

# Original API (deprecated, port 63234)
python3 src/main.py

# Train model directly (without API)
python3 src/train_model_improved.py   # loads data/synthetic/synthetic_network.csv

# Advanced model
python3 src/train_model_advanced.py
```

## Testing

```bash
pytest tests/ -v

# Specific tests
pytest tests/test_improved_model.py::test_calculate_cost_matrix -v
pytest tests/test_improved_model.py::test_solve_simple_path -v
pytest tests/test_advanced_model.py -v
pytest tests/test_api_improved.py -v
```

## Examples and Benchmarks

```bash
python3 examples/demo_improvements.py     # All 7 improvements with explanations
python3 examples/benchmark_all.py         # Compare all 3 tiers on same graphs
```

## Key Environment Considerations

- **`tensorflow-cpu`** — scripts use CPU-only TensorFlow. GPU is not required and GPU builds are not tested.
- **TF memory growth** — not configured (not needed for CPU builds)
- **First load is slow** — TensorFlow JIT tracing on the first `optimize()` call adds ~1s overhead

## Project Structure

```
src/
├── train_model.py              Original (deprecated) — Hamiltonian cycle constraints
├── train_model_improved.py     Phase 1: Flow conservation, zero offline training
├── train_model_advanced.py     Phase 2: Sparse tensors, attention, beam search
├── train_model_ultra.py        Additional optimizations
├── main.py                     Original FastAPI API (port 63234)
├── main_improved.py            Improved FastAPI API (port 63235, caching)
└── utils/visualize_graph.py    Graph visualization (matplotlib)

data/synthetic/
└── synthetic_network.csv       Example network (origin, destination, weight)

models/                         Saved models after training
tests/
├── test_improved_model.py      12 algorithm tests
├── test_advanced_model.py      Advanced model tests
├── test_api_improved.py        API endpoint tests
└── conftest.py                 pytest configuration

examples/
├── demo_improvements.py        7 improvements with before/after comparison
└── benchmark_all.py            Performance benchmarks
```

## Debugging Common Issues

**"No model loaded" on first /calculateShortestPath**:
Call `POST /loadNetwork` first to generate the `.keras` and `.pkl` files.

**Path not found (None returned)**:
The graph may have no path from origin to destination. Dijkstra will also confirm this and the API returns 404.

**Slow convergence**:
Reduce `num_restarts` or `iterations` in `predict_path()` for faster (less optimal) results.

**Wrong node indices**:
Node indices are assigned by sorted order of string node IDs. "10" sorts before "9" lexicographically. Print `node_to_index` dict from the .pkl file to verify mapping.

```python
import joblib
data = joblib.load('models/trained_model_improved.pkl')
print(data['node_mapping'])  # {'0': 0, '10': 1, '2': 2, ...}
```

## Comparing Model Tiers

Run `examples/benchmark_all.py` to measure:
- Solution quality (% of Dijkstra optimal)
- Query time per (origin, destination) pair
- Convergence iterations
- Memory usage

All three tiers solve the same set of random (origin, destination) pairs on the same graph, enabling direct comparison.
