"""
Comparison script between original and improved Hopfield models.
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.train_model import calculate_cost_matrix as calc_cost_original
from src.train_model import HopfieldModel as OriginalModel
from src.train_model_improved import calculate_cost_matrix as calc_cost_improved
from src.train_model_improved import ImprovedHopfieldModel as ImprovedModel

def compare_models(network_file='data/synthetic/synthetic_network.csv', num_queries=10):
    """
    Compare original vs improved Hopfield models.
    """
    print("="*80)
    print("HOPFIELD MODEL COMPARISON")
    print("="*80)
    
    # Load network
    print(f"\nLoading network from: {network_file}")
    cost_matrix, _ = calc_cost_original(network_file)
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)
    
    n = cost_matrix.shape[0]
    print(f"Network size: {n} nodes")
    print(f"Number of edges: {np.sum(cost_matrix < 1e6)}")
    
    # Initialize models
    print("\n" + "-"*80)
    print("INITIALIZING MODELS")
    print("-"*80)
    
    print("\n[Original Model]")
    start = time.time()
    original_model = OriginalModel(n, cost_matrix_normalized)
    original_model.set_cost_matrix(cost_matrix)
    original_model.compile(optimizer='adam')
    print(f"Initialization time: {time.time() - start:.2f}s")
    
    print("\n[Improved Model]")
    start = time.time()
    improved_model = ImprovedModel(n, cost_matrix_normalized)
    improved_model.set_cost_matrix(cost_matrix)
    improved_model.compile(optimizer='adam')
    print(f"Initialization time: {time.time() - start:.2f}s")
    
    # Generate random queries
    print("\n" + "-"*80)
    print(f"TESTING {num_queries} RANDOM QUERIES")
    print("-"*80)
    
    queries = []
    for _ in range(num_queries):
        source, dest = np.random.choice(n, size=2, replace=False)
        queries.append((source, dest))
    
    results = {
        'original': {'times': [], 'costs': [], 'optimal': [], 'failures': 0},
        'improved': {'times': [], 'costs': [], 'optimal': [], 'failures': 0}
    }
    
    for i, (source, dest) in enumerate(queries):
        print(f"\n--- Query {i+1}: {source} → {dest} ---")
        
        # Get Dijkstra optimal
        dijkstra_path, dijkstra_cost = improved_model._dijkstra_path(source, dest)
        print(f"Dijkstra optimal cost: {dijkstra_cost:.2f}")
        
        # Test original model
        print("\n[Original Model]")
        try:
            start = time.time()
            path_orig = original_model.predict(source, dest, validate=False)
            time_orig = time.time() - start
            cost_orig = original_model._calculate_path_cost(path_orig)
            
            results['original']['times'].append(time_orig)
            results['original']['costs'].append(cost_orig)
            results['original']['optimal'].append(abs(cost_orig - dijkstra_cost) < 1e-6)
            
            print(f"Time: {time_orig:.3f}s")
            print(f"Cost: {cost_orig:.2f}")
            print(f"Optimal: {abs(cost_orig - dijkstra_cost) < 1e-6}")
            print(f"Path length: {len(path_orig)}")
        except Exception as e:
            print(f"FAILED: {str(e)}")
            results['original']['failures'] += 1
        
        # Test improved model
        print("\n[Improved Model]")
        try:
            start = time.time()
            path_impr = improved_model.predict(source, dest, num_restarts=2, validate=True)
            time_impr = time.time() - start
            cost_impr = improved_model._calculate_path_cost(path_impr)
            
            results['improved']['times'].append(time_impr)
            results['improved']['costs'].append(cost_impr)
            results['improved']['optimal'].append(abs(cost_impr - dijkstra_cost) < 1e-6)
            
            print(f"Time: {time_impr:.3f}s")
            print(f"Cost: {cost_impr:.2f}")
            print(f"Optimal: {abs(cost_impr - dijkstra_cost) < 1e-6}")
            print(f"Path length: {len(path_impr)}")
        except Exception as e:
            print(f"FAILED: {str(e)}")
            results['improved']['failures'] += 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n[Original Model]")
    if results['original']['times']:
        print(f"Average time: {np.mean(results['original']['times']):.3f}s")
        print(f"Average cost: {np.mean(results['original']['costs']):.2f}")
        print(f"Optimal solutions: {sum(results['original']['optimal'])}/{len(results['original']['optimal'])}")
    print(f"Failures: {results['original']['failures']}/{num_queries}")
    
    print("\n[Improved Model]")
    if results['improved']['times']:
        print(f"Average time: {np.mean(results['improved']['times']):.3f}s")
        print(f"Average cost: {np.mean(results['improved']['costs']):.2f}")
        print(f"Optimal solutions: {sum(results['improved']['optimal'])}/{len(results['improved']['optimal'])}")
    print(f"Failures: {results['improved']['failures']}/{num_queries}")
    
    # Improvements
    print("\n" + "-"*80)
    print("IMPROVEMENTS")
    print("-"*80)
    
    if results['original']['times'] and results['improved']['times']:
        time_improvement = (1 - np.mean(results['improved']['times']) / np.mean(results['original']['times'])) * 100
        print(f"Time: {time_improvement:+.1f}%")
        
        cost_improvement = (1 - np.mean(results['improved']['costs']) / np.mean(results['original']['costs'])) * 100
        print(f"Cost: {cost_improvement:+.1f}%")
        
        optimal_improvement = sum(results['improved']['optimal']) - sum(results['original']['optimal'])
        print(f"Optimal solutions: {optimal_improvement:+d}")
        
        failure_improvement = results['original']['failures'] - results['improved']['failures']
        print(f"Fewer failures: {failure_improvement:+d}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    compare_models(num_queries=5)
