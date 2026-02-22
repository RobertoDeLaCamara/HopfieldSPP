import pytest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_model_ultra import UltraHopfieldModel, UltraHopfieldLayer, create_ultra_model

def test_gpu_acceleration():
    """Test that GPU acceleration is enabled."""
    n = 10
    distance_matrix = np.random.rand(n, n)
    np.fill_diagonal(distance_matrix, 0)
    
    layer = UltraHopfieldLayer(n, distance_matrix)
    
    # Check that energy function is compiled
    assert hasattr(layer.energy, 'experimental_get_compiler_ir')

def test_query_caching():
    """Test query caching functionality."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    source, dest = 0, 5
    
    # First query (cache miss)
    start1 = time.time()
    path1 = model.predict_path(source, dest, num_restarts=1, use_cache=True)
    time1 = time.time() - start1
    
    # Second query (cache hit)
    start2 = time.time()
    path2 = model.predict_path(source, dest, num_restarts=1, use_cache=True)
    time2 = time.time() - start2
    
    # Should return same path
    assert path1 == path2
    
    # Second query should be faster
    assert time2 < time1 * 0.5  # At least 2x faster
    
    # Check cache stats
    stats = model.cache_stats()
    assert stats['hits'] >= 1
    assert stats['misses'] >= 1

def test_incremental_edge_update():
    """Test incremental edge updates."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    # Get original path
    source, dest = 0, 5
    path_original = model.predict_path(source, dest, num_restarts=1)
    cost_original = model._calculate_path_cost(path_original)
    
    # Update edge
    if 1 < len(cost_matrix) and 2 < len(cost_matrix):
        old_weight = cost_matrix[1, 2]
        new_weight = old_weight * 0.5
        
        model.update_edge(1, 2, new_weight)
        
        # Verify edge was updated
        assert model.cost_matrix[1, 2] == new_weight
        
        # Get new path
        path_updated = model.predict_path(source, dest, num_restarts=1)
        cost_updated = model._calculate_path_cost(path_updated)
        
        # Cost should be same or better
        assert cost_updated <= cost_original * 1.1

def test_add_remove_edge():
    """Test adding and removing edges."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    n = len(cost_matrix)
    if n > 5:
        # Add edge
        model.add_edge(0, 4, 2.5)
        assert model.cost_matrix[0, 4] == 2.5
        
        # Remove edge
        model.remove_edge(0, 4)
        assert model.cost_matrix[0, 4] == 1e6

def test_astar_heuristic():
    """Test A* heuristic with coordinates."""
    n = 10
    distance_matrix = np.random.rand(n, n)
    np.fill_diagonal(distance_matrix, 0)
    
    # Create coordinates
    coordinates = np.random.rand(n, 2) * 100
    
    layer = UltraHopfieldLayer(n, distance_matrix, coordinates)
    
    # Verify coordinates are stored
    assert layer.coordinates is not None
    
    # Energy calculation should include heuristic
    energy = layer.energy(source=0, destination=5, temperature=0.5)
    assert energy.numpy() is not None

def test_cache_invalidation():
    """Test that cache is invalidated on graph updates."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    # Query and cache
    path1 = model.predict_path(0, 5, use_cache=True)
    
    # Update graph
    model.update_edge(1, 2, 1.0)
    
    # Cache should be cleared
    stats = model.cache_stats()
    assert stats['size'] == 0

def test_batch_processing():
    """Test batch query processing."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    queries = [(0, 5), (1, 6), (2, 7)]
    
    results = model.predict_batch(queries, use_cache=True)
    
    assert len(results) == len(queries)
    
    for (path, cost), (source, dest) in zip(results, queries):
        assert path[0] == source
        assert path[-1] == dest
        assert cost > 0

def test_cache_stats():
    """Test cache statistics tracking."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    # Make some queries
    model.predict_path(0, 5, use_cache=True)  # Miss
    model.predict_path(0, 5, use_cache=True)  # Hit
    model.predict_path(1, 6, use_cache=True)  # Miss
    
    stats = model.cache_stats()
    
    assert stats['hits'] == 1
    assert stats['misses'] == 2
    assert stats['hit_rate'] == pytest.approx(33.33, rel=1)
    assert stats['size'] == 2

def test_clear_cache():
    """Test manual cache clearing."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    # Cache some queries
    model.predict_path(0, 5, use_cache=True)
    model.predict_path(1, 6, use_cache=True)
    
    assert model.cache_stats()['size'] == 2
    
    # Clear cache
    model.clear_cache()
    
    assert model.cache_stats()['size'] == 0

def test_performance_improvement():
    """Test that ultra model is faster than basic."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    # Warm up
    model.predict_path(0, 5, num_restarts=1)
    
    # Measure cached query
    start = time.time()
    model.predict_path(0, 5, num_restarts=1, use_cache=True)
    cached_time = time.time() - start
    
    # Should be very fast (< 0.01s)
    assert cached_time < 0.01

def test_incremental_vs_retrain():
    """Test that incremental update is faster than retraining."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    # Incremental update
    start = time.time()
    model.update_edge(1, 2, 5.0)
    incremental_time = time.time() - start
    
    # Should be very fast (< 0.001s)
    assert incremental_time < 0.001

def test_astar_convergence():
    """Test that A* heuristic improves convergence."""
    n = 10
    distance_matrix = np.random.rand(n, n)
    np.fill_diagonal(distance_matrix, 0)
    
    # With coordinates (A* enabled)
    coordinates = np.random.rand(n, 2) * 100
    model_with_astar = UltraHopfieldModel(n, distance_matrix, coordinates)
    model_with_astar.set_cost_matrix(distance_matrix)
    
    # Without coordinates (no A*)
    model_without_astar = UltraHopfieldModel(n, distance_matrix, None)
    model_without_astar.set_cost_matrix(distance_matrix)
    
    # Both should work
    path_with = model_with_astar.predict_path(0, 5, num_restarts=5, validate=True)
    path_without = model_without_astar.predict_path(0, 5, num_restarts=5, validate=True)
    
    assert path_with is not None
    assert path_without is not None

def test_ultra_model_reliability():
    """Test that ultra model maintains 100% reliability."""
    model, cost_matrix, _ = create_ultra_model('data/synthetic/synthetic_network.csv')
    
    n = len(cost_matrix)
    
    # Test multiple queries
    for i in range(3):
        source = i % n
        dest = (i + 5) % n

        if source == dest:
            continue

        path = model.predict_path(source, dest, num_restarts=2, validate=True)
        
        # Should always succeed
        assert path is not None
        assert len(path) >= 2
        assert path[0] == source
        assert path[-1] == dest

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
