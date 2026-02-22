# Developer Guide

## Getting Started

### Prerequisites
- Python 3.10+
- TensorFlow 2.18
- FastAPI
- NumPy, Pandas

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/HopfieldSPP.git
cd HopfieldSPP

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
HopfieldSPP/
├── src/                          # Source code
│   ├── train_model.py            # Original (deprecated)
│   ├── train_model_improved.py   # Phase 1: Critical fixes
│   ├── train_model_advanced.py   # Phase 2: Advanced features
│   ├── main.py                   # Original API
│   ├── main_improved.py          # Improved API
│   └── utils/                    # Utilities
│       └── visualize_graph.py
├── tests/                        # Test suite
│   ├── test_train_model.py
│   ├── test_main.py
│   └── test_improved_model.py
├── data/                         # Data files
│   ├── synthetic/                # Test networks
│   └── openAPIs/                 # API specs
├── models/                       # Saved models
├── notebooks/                    # Jupyter notebooks
├── demo_*.py                     # Demo scripts
├── benchmark_*.py                # Benchmark scripts
└── *.md                          # Documentation
```

---

## Core Components

### 1. Cost Matrix Calculation

```python
def calculate_cost_matrix(adjacency_matrix_path):
    """
    Load CSV and build cost matrix.
    
    Args:
        adjacency_matrix_path: Path to CSV with columns [origin, destination, weight]
    
    Returns:
        cost_matrix: n×n numpy array (1e6 for non-existent edges)
        node_mapping: Dict mapping node IDs to matrix indices
    """
```

### 2. Hopfield Layer

```python
class ImprovedHopfieldLayer(Layer):
    """
    Implements energy minimization with flow conservation.
    
    Key methods:
        - energy(): Calculate energy function
        - optimize(): Run gradient descent
        - _connectivity_penalty(): Ensure reachability
    """
```

### 3. Hopfield Model

```python
class ImprovedHopfieldModel(Model):
    """
    High-level interface for shortest path prediction.
    
    Key methods:
        - predict(): Find shortest path
        - _dijkstra_path(): Fallback algorithm
        - _extract_path_robust(): Path extraction
    """
```

---

## Development Workflow

### 1. Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes
# ... edit files ...

# Run tests
python3 -m pytest tests/ -v

# Run demos to verify
python3 demo_improvements.py
```

### 2. Adding New Features

#### Example: Add New Extraction Strategy

```python
# In train_model_improved.py

def _extract_path_new_strategy(self, state_matrix, source, destination):
    """Your new extraction strategy."""
    # Implementation
    pass

# Update predict() to use it
def predict(self, source, destination, ...):
    # Try new strategy first
    path = self._extract_path_new_strategy(state_matrix, source, destination)
    if path:
        return path
    
    # Fallback to existing strategies
    path = self._extract_path_robust(state_matrix, source, destination)
    # ...
```

### 3. Testing

```python
# In tests/test_improved_model.py

def test_new_extraction_strategy():
    """Test your new feature."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_norm = (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())
    
    model = ImprovedHopfieldModel(len(cost_matrix), cost_norm)
    model.set_cost_matrix(cost_matrix)
    
    # Test your feature
    path = model._extract_path_new_strategy(...)
    assert path is not None
    assert path[0] == source
    assert path[-1] == destination
```

---

## Code Style

### Naming Conventions
- Classes: `PascalCase` (e.g., `ImprovedHopfieldModel`)
- Functions: `snake_case` (e.g., `calculate_cost_matrix`)
- Private methods: `_snake_case` (e.g., `_dijkstra_path`)
- Constants: `UPPER_CASE` (e.g., `MAX_ITERATIONS`)

### Documentation
```python
def function_name(param1, param2):
    """
    Brief description.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something goes wrong
    """
```

### Logging
```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

---

## Testing Guide

### Running Tests

```bash
# All tests
python3 -m pytest tests/ -v

# Specific test file
python3 -m pytest tests/test_improved_model.py -v

# Specific test function
python3 -m pytest tests/test_improved_model.py::test_flow_conservation -v

# With coverage
python3 -m pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

```python
import pytest
import numpy as np
from src.train_model_improved import ImprovedHopfieldModel

def test_feature():
    """Test description."""
    # Setup
    model = create_test_model()
    
    # Execute
    result = model.some_method()
    
    # Assert
    assert result is not None
    assert len(result) > 0

def test_error_handling():
    """Test error cases."""
    model = create_test_model()
    
    with pytest.raises(ValueError):
        model.predict(source=-1, destination=0)
```

---

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

#### Issue: Model not converging
```python
# Increase iterations
model.hopfield_layer.optimize(source, dest, iterations=500)

# Try more restarts
path = model.predict(source, dest, num_restarts=5)
```

#### Issue: Out of memory
```python
# Use advanced model with sparse tensors
from src.train_model_advanced import AdvancedHopfieldModel

model = AdvancedHopfieldModel(n, cost_matrix, use_sparse=True)
```

#### Issue: Suboptimal solutions
```python
# Enable validation
path = model.predict(source, dest, validate=True)

# Check Dijkstra comparison
dijkstra_path, dijkstra_cost = model._dijkstra_path(source, dest)
hopfield_cost = model._calculate_path_cost(path)
print(f"Hopfield: {hopfield_cost}, Dijkstra: {dijkstra_cost}")
```

---

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile code
profiler = cProfile.Profile()
profiler.enable()

# Your code here
model.predict(source, dest)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Optimization Tips

1. **Use model caching** (already implemented in `main_improved.py`)
2. **Reduce restarts** for faster queries (trade quality for speed)
3. **Use sparse mode** for large graphs
4. **Adjust early stopping tolerance** for faster convergence
5. **Batch multiple queries** (future work)

---

## Adding New Models

### Template

```python
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

class YourHopfieldLayer(Layer):
    def __init__(self, n, distance_matrix, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.distance_matrix = tf.constant(distance_matrix, dtype=tf.float32)
        
        # Your initialization
        
    def energy(self, source, destination):
        """Calculate energy function."""
        # Your energy function
        pass
    
    def optimize(self, source, destination):
        """Optimize for specific query."""
        # Your optimization
        pass

class YourHopfieldModel(Model):
    def __init__(self, n, distance_matrix, **kwargs):
        super().__init__(**kwargs)
        self.hopfield_layer = YourHopfieldLayer(n, distance_matrix)
    
    def predict(self, source, destination):
        """Predict shortest path."""
        # Your prediction logic
        pass
```

---

## API Development

### Adding New Endpoints

```python
# In main_improved.py

@app.get("/newEndpoint")
async def new_endpoint(param: str = Query(...)):
    """
    New endpoint description.
    """
    try:
        # Your logic
        result = process(param)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Testing API

```bash
# Start server
python3 src/main_improved.py

# Test with curl
curl "http://localhost:63235/newEndpoint?param=value"

# Test with Python
import requests
response = requests.get("http://localhost:63235/newEndpoint", params={"param": "value"})
print(response.json())
```

---

## Documentation

### Updating Documentation

When adding features:
1. Update relevant `.md` files
2. Add docstrings to new functions/classes
3. Update `README.md` if user-facing
4. Add examples to demo scripts

### Building Documentation

```bash
# Generate API docs (if using Sphinx)
cd docs
make html

# View docs
open _build/html/index.html
```

---

## Release Process

### Version Numbering
- Major: Breaking changes (e.g., 1.0.0 → 2.0.0)
- Minor: New features (e.g., 1.0.0 → 1.1.0)
- Patch: Bug fixes (e.g., 1.0.0 → 1.0.1)

### Release Checklist
1. ✅ All tests pass
2. ✅ Documentation updated
3. ✅ CHANGELOG.md updated
4. ✅ Version bumped in `__init__.py`
5. ✅ Tag release in git
6. ✅ Build and test package
7. ✅ Deploy to PyPI (if applicable)

---

## Contributing

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Update documentation
5. Submit PR with description
6. Address review comments
7. Merge after approval

### Code Review Checklist
- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

---

## Resources

### Internal
- [INDEX.md](INDEX.md) - Documentation index
- [COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md) - Full improvement summary
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - API reference

### External
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Hopfield Networks](https://en.wikipedia.org/wiki/Hopfield_network)
- [Network Flow Theory](https://en.wikipedia.org/wiki/Flow_network)

---

## Support

For development questions:
- GitHub Issues: Technical issues and bugs
- GitHub Discussions: General questions and ideas
- Email: dev@hopfieldspp.com (if applicable)
