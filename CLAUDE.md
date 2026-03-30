# HopfieldSPP

Pure Python shortest path solver using Hopfield Networks + TensorFlow 2.18. Three model versions.

## Model Versions

| Version | Status | File |
|---|---|---|
| Original | Deprecated | `src/train_model.py` |
| Improved | **Recommended** | `src/train_model_improved.py` |
| Advanced | Research | `src/train_model_advanced.py` |

## Key Commands

```bash
# Train
python src/train_model_improved.py

# API server
python src/main_improved.py

# Benchmark all versions
python benchmark_all.py

# Tests
pytest tests/
```

## API

FastAPI REST with caching. Endpoint: `POST /solve`

## Remotes

- `origin` → Gitea (192.168.1.62:9090)
- `github` → GitHub (RobertoDeLaCamara/HopfieldSPP)
- License: Apache 2.0

## Thesis Connection

This project traces back to Roberto's 1998 thesis (Hopfield NN for SPP, U. Valladolid).
