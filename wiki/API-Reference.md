# API Reference

Base URL: `http://localhost:63235`

Framework: FastAPI (auto-generates `/docs` Swagger UI)

---

## POST /loadNetwork

Upload a CSV file and train/initialize the Hopfield model.

**Request**: `multipart/form-data` with field `file` (CSV)

**CSV requirements**:
- Columns: `origin`, `destination`, `weight`
- `origin` and `destination`: any string (auto-mapped to indices)
- `weight`: float ≥ 0
- Directed graph (each row is a one-way edge)

**Example**:
```bash
curl -X POST http://localhost:63235/loadNetwork \
  -F "file=@data/synthetic/synthetic_network.csv"
```

**Response 200**:
```json
{
  "message": "Network loaded successfully",
  "status": "success"
}
```

**Response 400**: Invalid file type or empty CSV

**Response 500**: Processing error

**Side effects**:
- Saves `models/trained_model_improved.keras` and `models/trained_model_improved.pkl`
- Invalidates in-memory model cache (next `/calculateShortestPath` reloads from disk)

---

## GET /calculateShortestPath

Find shortest path between two nodes using Hopfield + Dijkstra fallback.

**Query parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `origin` | int | Yes | Starting node index |
| `destination` | int | Required | Destination node index |

**Example**:
```bash
curl "http://localhost:63235/calculateShortestPath?origin=0&destination=9"
```

**Response 200**:
```json
{
  "path": [0, 43, 9],
  "cost": 60.0
}
```

`path`: Ordered list of node indices from origin to destination.
`cost`: Total edge weight sum along the path (uses original, non-normalized costs).

**Response 400**: Invalid parameters or nodes out of range

**Response 404**: No path exists from origin to destination

**Response 500**: Internal error

---

## Auto-generated Docs

```
GET /docs      Swagger UI
GET /redoc     ReDoc
GET /openapi.json
```

---

## Node Indexing

Node IDs in the CSV (strings like "0", "43", "node_A") are mapped to consecutive integer indices 0..n-1 in sorted order. The `/loadNetwork` response does not expose this mapping. Use the same integer indices in `/calculateShortestPath` that correspond to sorted node IDs.

For a CSV with node IDs "0", "3", "43", "9":
- Sorted: ["0", "3", "43", "9"] → after lexicographic sort: ["0", "3", "43", "9"]
- Index 0 = node "0", Index 1 = node "3", Index 2 = node "43", Index 3 = node "9"

**Note**: Python's default string sort is lexicographic. "43" sorts before "9". If your node IDs are integers, ensure the mapping matches your expectations.

---

## Performance

| Graph size | First call (cache miss) | Subsequent calls (cache hit) |
|------------|------------------------|------------------------------|
| 50 nodes | ~1–2s | ~50–100ms |
| 200 nodes | ~2–5s | ~100–300ms |
| 500 nodes | ~5–15s | ~300ms–1s |

Times include 3-restart Hopfield optimization + optional Dijkstra validation.
