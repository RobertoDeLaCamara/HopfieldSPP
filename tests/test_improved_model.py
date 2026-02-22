import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_model_improved import (
    calculate_cost_matrix,
    ImprovedHopfieldLayer,
    ImprovedHopfieldModel,
    train_improved_model
)


def test_calculate_cost_matrix():
    """Test cost matrix calculation."""
    cost_matrix, node_mapping = calculate_cost_matrix('data/synthetic/synthetic_network.csv')

    assert cost_matrix is not None
    assert cost_matrix.shape[0] == cost_matrix.shape[1]
    assert len(node_mapping) == cost_matrix.shape[0]

    # Check diagonal is zero
    assert np.all(np.diag(cost_matrix) == 0)

    # Check non-existent edges are 1e6
    assert np.any(cost_matrix == 1e6)


def test_improved_hopfield_layer():
    """Test improved Hopfield layer initialization and energy calculation."""
    n = 5
    distance_matrix = np.random.rand(n, n)
    distance_matrix[distance_matrix > 0.5] = 1e6  # Some invalid edges
    np.fill_diagonal(distance_matrix, 0)

    layer = ImprovedHopfieldLayer(n, distance_matrix)

    assert layer.n == n
    assert layer.logits.shape == (n, n)
    assert layer.valid_arcs.shape == (n, n)

    # Test energy calculation
    energy = layer.energy(source=0, destination=4, temperature=0.5)
    assert energy.numpy() > 0  # Energy should be positive


def test_flow_conservation():
    """Test that flow conservation is correctly enforced."""
    n = 4
    # Simple linear graph: 0->1->2->3
    distance_matrix = np.full((n, n), 1e6)
    np.fill_diagonal(distance_matrix, 0)
    for i in range(n-1):
        distance_matrix[i, i+1] = 1.0

    layer = ImprovedHopfieldLayer(n, distance_matrix)

    # Manually set perfect solution: 0->1->2->3
    perfect_logits = np.full((n, n), -10.0)
    perfect_logits[0, 1] = 10.0
    perfect_logits[1, 2] = 10.0
    perfect_logits[2, 3] = 10.0
    layer.logits.assign(perfect_logits)

    energy = layer.energy(source=0, destination=3, temperature=0.1)

    # Energy should be low for perfect solution
    assert energy.numpy() < 5.0


def test_improved_model_predict():
    """Test improved model prediction."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')

    # Normalize
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)

    # Test prediction
    source, destination = 0, min(5, n-1)
    path = model.predict_path(source, destination, num_restarts=2, validate=True)

    assert path is not None
    assert len(path) >= 2
    assert path[0] == source
    assert path[-1] == destination

    # Check path is connected
    for i in range(len(path) - 1):
        assert cost_matrix[path[i], path[i+1]] < 1e6


def test_dijkstra_fallback():
    """Test that Dijkstra fallback works correctly."""
    # Create simple graph where Dijkstra should find optimal path
    n = 5
    cost_matrix = np.full((n, n), 1e6)
    np.fill_diagonal(cost_matrix, 0)

    # Create path: 0->1->2->3->4 with costs
    cost_matrix[0, 1] = 1.0
    cost_matrix[1, 2] = 1.0
    cost_matrix[2, 3] = 1.0
    cost_matrix[3, 4] = 1.0

    # Add longer alternative: 0->4 direct
    cost_matrix[0, 4] = 10.0

    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)

    # Dijkstra should find optimal path
    path, cost = model._dijkstra_path(0, 4)

    assert path == [0, 1, 2, 3, 4]
    assert cost == 4.0


def test_multi_start_optimization():
    """Test that multi-start finds reasonable solutions."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)

    source, destination = 0, min(8, n-1)

    # Multiple restarts should find a valid path
    path_multi = model.predict_path(source, destination, num_restarts=3, validate=True)
    assert path_multi is not None
    assert path_multi[0] == source
    assert path_multi[-1] == destination


def test_train_improved_model():
    """Test training improved model."""
    train_improved_model('data/synthetic/synthetic_network.csv')

    # Check that model files were created
    model_path = 'data/synthetic/tests/'
    assert os.path.exists(model_path + 'trained_model_improved.keras')
    assert os.path.exists(model_path + 'cost_matrix_improved.pkl')


def test_same_source_destination():
    """Test edge case: source equals destination."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)

    path = model.predict_path(source=0, destination=0, validate=True)

    assert path == [0]


def test_invalid_node_indices():
    """Test error handling for invalid node indices."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)

    # Test out of range
    with pytest.raises(ValueError):
        model.predict_path(source=-1, destination=0, validate=True)

    with pytest.raises(ValueError):
        model.predict_path(source=0, destination=n+10, validate=True)


def test_disconnected_graph():
    """Test handling of disconnected graphs."""
    n = 6
    cost_matrix = np.full((n, n), 1e6)
    np.fill_diagonal(cost_matrix, 0)

    # Create two disconnected components: 0-1-2 and 3-4-5
    cost_matrix[0, 1] = 1.0
    cost_matrix[1, 2] = 1.0
    cost_matrix[3, 4] = 1.0
    cost_matrix[4, 5] = 1.0

    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)

    # Should raise error or return None for disconnected nodes
    with pytest.raises(ValueError):
        model.predict_path(source=0, destination=5, validate=True)


def test_path_cost_calculation():
    """Test path cost calculation."""
    n = 4
    cost_matrix = np.full((n, n), 1e6)
    np.fill_diagonal(cost_matrix, 0)
    cost_matrix[0, 1] = 2.5
    cost_matrix[1, 2] = 3.0
    cost_matrix[2, 3] = 1.5

    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)

    path = [0, 1, 2, 3]
    cost = model._calculate_path_cost(path)

    assert abs(cost - 7.0) < 1e-6  # 2.5 + 3.0 + 1.5 = 7.0


def test_early_stopping():
    """Test that early stopping works."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    layer = ImprovedHopfieldLayer(n, cost_matrix_normalized)

    # Optimize with early stopping
    import time
    start = time.time()
    layer.optimize(source=0, destination=min(5, n-1), iterations=500, tolerance=1e-4)
    elapsed = time.time() - start

    # Should converge reasonably quickly in CI
    assert elapsed < 120.0


def test_model_caching():
    """Test that model state is preserved between predictions."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)

    # First prediction
    path1 = model.predict_path(source=0, destination=min(5, n-1), num_restarts=1, validate=True)

    # Second prediction (should work independently)
    path2 = model.predict_path(source=1, destination=min(6, n-1), num_restarts=1, validate=True)

    assert path1 is not None
    assert path2 is not None
    assert path1 != path2  # Different queries should give different paths


def test_optimal_solution_quality():
    """Test that model finds optimal or near-optimal solutions."""
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)

    n = cost_matrix.shape[0]
    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)

    # Test multiple queries
    optimal_count = 0
    total_queries = 5

    for i in range(total_queries):
        source = i % n
        destination = (i + 5) % n
        if source == destination:
            continue

        path = model.predict_path(source, destination, num_restarts=2, validate=True)
        hopfield_cost = model._calculate_path_cost(path)
        _, dijkstra_cost = model._dijkstra_path(source, destination)

        if abs(hopfield_cost - dijkstra_cost) < 1e-6:
            optimal_count += 1

    # Should find optimal solution in at least 60% of cases (stochastic model)
    assert optimal_count / total_queries >= 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
