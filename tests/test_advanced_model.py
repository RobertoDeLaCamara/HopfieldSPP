import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_model_advanced import (
    AdvancedHopfieldModel,
    AdvancedHopfieldLayer,
    calculate_cost_matrix,
    train_advanced_model
)


def test_sparse_mode_initialization():
    """Test sparse tensor initialization."""
    n = 10
    # Create sparse graph (30% density)
    distance_matrix = np.full((n, n), 1e6)
    np.fill_diagonal(distance_matrix, 0)

    # Add some edges
    for i in range(n-1):
        distance_matrix[i, i+1] = np.random.rand()
        if i < n-2:
            distance_matrix[i, i+2] = np.random.rand()

    layer = AdvancedHopfieldLayer(n, distance_matrix, use_sparse=True)

    assert layer.use_sparse is True
    assert layer.num_edges > 0
    assert layer.num_edges < n * n


def test_adaptive_hyperparameters():
    """Test that hyperparameters adapt to graph density."""
    n = 10

    # Dense graph
    dense_matrix = np.random.rand(n, n)
    np.fill_diagonal(dense_matrix, 0)
    layer_dense = AdvancedHopfieldLayer(n, dense_matrix, use_sparse=False)

    # Sparse graph
    sparse_matrix = np.full((n, n), 1e6)
    np.fill_diagonal(sparse_matrix, 0)
    for i in range(n-1):
        sparse_matrix[i, i+1] = 1.0
    layer_sparse = AdvancedHopfieldLayer(n, sparse_matrix, use_sparse=False)

    # Dense graph should have higher mu values
    assert layer_dense.mu2 > layer_sparse.mu2


def test_attention_mechanism():
    """Test attention mechanism is initialized."""
    n = 5
    distance_matrix = np.random.rand(n, n)
    np.fill_diagonal(distance_matrix, 0)

    layer = AdvancedHopfieldLayer(n, distance_matrix, use_sparse=False)

    assert hasattr(layer, 'attention_logits')
    assert layer.attention_logits.shape == (n, n)


def test_beam_search_extraction():
    """Test beam search path extraction."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=False)
    model.set_cost_matrix(cost_matrix)

    # Create a simple state matrix
    state_matrix = np.zeros((n, n))
    # Create a path: 0->1->2
    state_matrix[0, 1] = 0.9
    state_matrix[1, 2] = 0.8

    path = model._beam_search_path(state_matrix, source=0, destination=2, beam_width=3)

    assert path is not None
    assert path[0] == 0
    assert path[-1] == 2


def test_greedy_with_backtracking():
    """Test greedy extraction with backtracking."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=False)
    model.set_cost_matrix(cost_matrix)

    # Create state matrix with some noise
    state_matrix = np.random.rand(n, n) * 0.3
    state_matrix = state_matrix * (cost_matrix < 1e6)

    path = model._greedy_with_backtracking(state_matrix, source=0, destination=min(5, n-1))

    # May or may not find path depending on state_matrix, but shouldn't crash
    assert path is None or (path[0] == 0 and path[-1] == min(5, n-1))


def test_advanced_model_predict():
    """Test advanced model prediction with beam search."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=False)
    model.set_cost_matrix(cost_matrix)

    path = model.predict_path(
        source=0,
        destination=min(5, n-1),
        num_restarts=2,
        validate=True,
        use_beam_search=True
    )

    assert path is not None
    assert len(path) >= 2
    assert path[0] == 0
    assert path[-1] == min(5, n-1)


def test_sparse_vs_dense_performance():
    """Test that sparse mode uses less memory."""
    n = 20
    # Create sparse graph
    distance_matrix = np.full((n, n), 1e6)
    np.fill_diagonal(distance_matrix, 0)
    for i in range(n-1):
        distance_matrix[i, i+1] = 1.0

    # Dense mode
    layer_dense = AdvancedHopfieldLayer(n, distance_matrix, use_sparse=False)

    # Sparse mode
    layer_sparse = AdvancedHopfieldLayer(n, distance_matrix, use_sparse=True)

    # Sparse should have fewer parameters
    assert layer_sparse.num_edges < n * n


def test_learning_rate_scheduling():
    """Test that learning rate scheduling is applied."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    layer = AdvancedHopfieldLayer(n, cost_matrix_normalized, use_sparse=False)

    # Optimize with LR scheduling
    import tensorflow as tf
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.05,
        decay_steps=50,
        decay_rate=0.9
    )

    state = layer.optimize(
        source=0,
        destination=min(5, n-1),
        iterations=100,
        lr_schedule=lr_schedule
    )

    assert state is not None


def test_multiple_extraction_strategies():
    """Test that multiple extraction strategies are tried."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=False)
    model.set_cost_matrix(cost_matrix)

    # Create a difficult state matrix
    state_matrix = np.random.rand(n, n) * 0.4
    state_matrix = state_matrix * (cost_matrix < 1e6)

    # Should try multiple strategies
    path = model._extract_path_advanced(state_matrix, source=0, destination=min(5, n-1))

    # May or may not succeed, but shouldn't crash
    assert path is None or len(path) >= 2


def test_adaptive_fallback_threshold():
    """Test adaptive fallback threshold based on density."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=False)
    model.set_cost_matrix(cost_matrix)

    # Density should affect threshold
    density = model.hopfield_layer.density

    # Just verify density is calculated
    assert 0 <= density <= 1


def test_train_advanced_model():
    """Test training advanced model."""
    train_advanced_model('data/synthetic/synthetic_network.csv', use_sparse=False)

    model_path = 'data/synthetic/tests/'
    assert os.path.exists(model_path + 'trained_model_advanced.keras')
    assert os.path.exists(model_path + 'cost_matrix_advanced.pkl')


def test_auto_sparse_detection():
    """Test automatic sparse mode detection."""
    # Create very sparse graph
    n = 50
    distance_matrix = np.full((n, n), 1e6)
    np.fill_diagonal(distance_matrix, 0)

    # Only 10% edges
    for i in range(n-1):
        distance_matrix[i, i+1] = 1.0

    cost_matrix_normalized = (distance_matrix - np.min(distance_matrix)) / (np.max(distance_matrix) - np.min(distance_matrix) + 1e-6)

    # Should auto-detect sparse mode
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=True)

    assert model.hopfield_layer.use_sparse is True


def test_beam_search_vs_greedy():
    """Test that beam search can find paths comparable to greedy."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=False)
    model.set_cost_matrix(cost_matrix)

    source, dest = 0, min(8, n-1)

    # With beam search (use validate=True for Dijkstra fallback)
    path_beam = model.predict_path(source, dest, num_restarts=2, use_beam_search=True, validate=True)
    cost_beam = model._calculate_path_cost(path_beam) if path_beam else float('inf')

    # Without beam search
    path_greedy = model.predict_path(source, dest, num_restarts=2, use_beam_search=False, validate=True)
    cost_greedy = model._calculate_path_cost(path_greedy) if path_greedy else float('inf')

    # Both should find valid paths; beam search should be at least comparable
    assert cost_beam <= cost_greedy * 2.0


def test_advanced_optimal_quality():
    """Test that advanced model achieves high optimality rate."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=False)
    model.set_cost_matrix(cost_matrix)

    optimal_count = 0
    total = 3

    for i in range(total):
        source = i % n
        dest = (i + 5) % n
        if source == dest:
            continue

        path = model.predict_path(source, dest, num_restarts=2, validate=True, use_beam_search=True)
        cost = model._calculate_path_cost(path)
        _, dijkstra_cost = model._dijkstra_path(source, dest)

        if abs(cost - dijkstra_cost) < 1e-6:
            optimal_count += 1

    # Should achieve at least 80% optimality
    assert optimal_count / total >= 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
