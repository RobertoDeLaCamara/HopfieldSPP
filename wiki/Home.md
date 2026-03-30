# HopfieldSPP — Wiki

Solves the **shortest path problem** (not assignment) with Hopfield energy minimization. Three model tiers: `ImprovedHopfieldModel` (best for <500 nodes, zero training epochs), `AdvancedHopfieldModel` (sparse tensors for >500 nodes, auto-enabled when graph density <30%), and a deprecated original. Dijkstra fallback for 100% reliability. FastAPI on port 63235.

## Quick Start

```bash
pip install -r requirements.txt

# Start API
python3 src/main_improved.py           # API on :63235

# Load network and find path
curl -X POST http://localhost:63235/loadNetwork \
  -F "file=@data/synthetic/synthetic_network.csv"

curl "http://localhost:63235/calculateShortestPath?origin=0&destination=9"
# Returns: {"path": [0, 43, 9], "cost": 60.0}

# Demo and benchmarks
python3 examples/demo_improvements.py
python3 examples/benchmark_all.py
```

## Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI (port 63235) |
| ML framework | TensorFlow 2.x (`tensorflow-cpu`) |
| Algorithm | Hopfield energy minimization + Dijkstra fallback |
| Input | CSV (origin, destination, weight columns) |
| Model persistence | Keras `.keras` format + joblib |

## Wiki Pages

- [Architecture and Model Tiers](Architecture.md)
- [Algorithm Deep Dive](Algorithm-Deep-Dive.md)
- [API Reference](API-Reference.md)
- [Development Guide](Development-Guide.md)

## Key Layout

```
src/
├── train_model.py              Original model (deprecated — TSP constraints, not SPP)
├── train_model_improved.py     ImprovedHopfieldModel (Phase 1, ≤500 nodes)
├── train_model_advanced.py     AdvancedHopfieldModel (Phase 2, sparse, >500 nodes)
├── main.py                     Original API (port 63234, deprecated)
├── main_improved.py            Improved API (port 63235, model caching)
└── utils/visualize_graph.py    Graph visualization

examples/
├── demo_improvements.py        Phase 1 feature demo
└── benchmark_all.py            Performance comparison across all tiers
```

## Non-Obvious Facts

- **Zero offline training** — `ImprovedHopfieldModel` and `AdvancedHopfieldModel` do no offline training. All optimization happens at query time via gradient descent.
- **Fresh optimizer per query** — Each `/calculateShortestPath` call creates a new `Adam` optimizer. Reusing the same optimizer between queries causes momentum state pollution that degrades solution quality.
- **Dijkstra is the fallback, not the primary** — The API always tries Hopfield first. Dijkstra is invoked only if Hopfield fails (no valid path found) or produces a suboptimal solution (>5% worse than optimal).
- **Sparse mode is auto-selected** — `AdvancedHopfieldModel` enables sparse tensors automatically when graph density <30% AND nodes >100. No manual configuration required.
- **Original model solves the wrong problem** — `train_model.py` uses Hamiltonian cycle (TSP) constraints instead of flow conservation. It is kept for reference only.
- **Model caching is critical** — First `/calculateShortestPath` call loads the model from disk (~2–3s). Subsequent calls use the in-memory cache (~50–100ms).
