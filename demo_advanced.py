#!/usr/bin/env python3
"""
Demo of advanced features: sparse tensors, beam search, adaptive hyperparameters.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.train_model_advanced import (
    AdvancedHopfieldModel,
    calculate_cost_matrix
)

def demo_advanced_features():
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                    ADVANCED HOPFIELD MODEL DEMO                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    # Load network
    print("\n1. Loading Network...")
    print("-"*80)
    network_file = 'data/synthetic/synthetic_network.csv'
    cost_matrix, node_mapping = calculate_cost_matrix(network_file)
    n = cost_matrix.shape[0]
    num_edges = np.sum(cost_matrix < 1e6)
    density = num_edges / (n * n)
    
    print(f"   Nodes:       {n}")
    print(f"   Edges:       {num_edges}")
    print(f"   Density:     {density:.1%}")
    print(f"   Avg Degree:  {num_edges / n:.1f}")
    
    # Normalize
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)
    
    # Create model with advanced features
    print("\n2. Creating Advanced Model...")
    print("-"*80)
    
    # Auto-detect sparse mode
    use_sparse = density < 0.3 and n > 100
    print(f"   Sparse mode: {'Enabled' if use_sparse else 'Disabled'} (auto-detected)")
    
    model = AdvancedHopfieldModel(n, cost_matrix_normalized, use_sparse=use_sparse)
    model.set_cost_matrix(cost_matrix)
    model.compile(optimizer='adam')
    
    print(f"   ✓ Model initialized")
    print(f"   ✓ Adaptive hyperparameters: μ₁={model.hopfield_layer.mu1:.2f}, μ₂={model.hopfield_layer.mu2:.2f}, μ₃={model.hopfield_layer.mu3:.2f}")
    print(f"   ✓ Attention mechanism: Enabled")
    print(f"   ✓ Beam search: Enabled")
    
    # Test queries
    print("\n3. Testing Advanced Features...")
    print("="*80)
    
    test_queries = [
        (0, min(5, n-1)),
        (0, min(10, n-1)),
    ]
    
    for i, (source, dest) in enumerate(test_queries):
        print(f"\n   Query {i+1}: Node {source} → Node {dest}")
        print("   " + "-"*76)
        
        # Get Dijkstra optimal
        dijkstra_path, dijkstra_cost = model._dijkstra_path(source, dest)
        print(f"\n   [Dijkstra - Optimal]")
        print(f"     Path:  {dijkstra_path}")
        print(f"     Cost:  {dijkstra_cost:.2f}")
        
        # Test with beam search
        print(f"\n   [Advanced Hopfield - Beam Search]")
        try:
            path_beam = model.predict(
                source, dest,
                num_restarts=2,
                validate=True,
                use_beam_search=True
            )
            cost_beam = model._calculate_path_cost(path_beam)
            
            is_optimal = abs(cost_beam - dijkstra_cost) < 1e-6
            accuracy = (dijkstra_cost / cost_beam * 100) if cost_beam > 0 else 0
            
            print(f"     Path:     {path_beam}")
            print(f"     Cost:     {cost_beam:.2f}")
            print(f"     Optimal:  {'✓ YES' if is_optimal else f'✗ NO ({accuracy:.1f}%)'}")
            print(f"     Length:   {len(path_beam)} nodes")
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)}")
        
        # Test without beam search (for comparison)
        print(f"\n   [Advanced Hopfield - Without Beam Search]")
        try:
            path_no_beam = model.predict(
                source, dest,
                num_restarts=2,
                validate=True,
                use_beam_search=False
            )
            cost_no_beam = model._calculate_path_cost(path_no_beam)
            
            is_optimal = abs(cost_no_beam - dijkstra_cost) < 1e-6
            accuracy = (dijkstra_cost / cost_no_beam * 100) if cost_no_beam > 0 else 0
            
            print(f"     Path:     {path_no_beam}")
            print(f"     Cost:     {cost_no_beam:.2f}")
            print(f"     Optimal:  {'✓ YES' if is_optimal else f'✗ NO ({accuracy:.1f}%)'}")
            print(f"     Length:   {len(path_no_beam)} nodes")
            
            # Compare
            if cost_beam < cost_no_beam:
                improvement = ((cost_no_beam - cost_beam) / cost_no_beam * 100)
                print(f"\n     💡 Beam search found {improvement:.1f}% better solution!")
            
        except Exception as e:
            print(f"     ✗ Error: {str(e)}")
    
    # Feature summary
    print("\n" + "="*80)
    print("ADVANCED FEATURES DEMONSTRATED")
    print("="*80)
    print("""
✓ Sparse Tensor Support
  • Automatically enabled for sparse graphs (density < 30%)
  • Memory: O(E) instead of O(n²)
  • Enables graphs with 1000+ nodes

✓ Adaptive Hyperparameters
  • μ values automatically tuned based on graph density
  • No manual hyperparameter tuning needed
  • Better generalization across graph types

✓ Attention Mechanism
  • Learns which edges are more important
  • Improves focus on promising paths
  • Better solution quality

✓ Beam Search Path Extraction
  • Explores multiple paths simultaneously
  • Finds better solutions than greedy extraction
  • More robust to noisy optimization results

✓ Learning Rate Scheduling
  • Exponential decay for better convergence
  • Fast initial progress, fine-grained final tuning
  • Better final solutions

✓ Greedy with Backtracking
  • Escapes local dead ends
  • More robust path finding
  • Fallback when beam search fails

✓ Best State Tracking
  • Never loses good solutions during optimization
  • Returns best state seen across all iterations
  • More stable results

✓ Adaptive Fallback Threshold
  • Adjusts based on graph complexity
  • Better balance between speed and optimality
  • Context-aware decision making

✓ Multiple Extraction Strategies
  • Tries beam search → backtracking → BFS
  • Higher success rate
  • Graceful degradation
""")
    print("="*80)
    
    # Memory comparison
    if use_sparse:
        dense_memory = n * n * 4 / 1024 / 1024  # MB
        sparse_memory = num_edges * 4 / 1024 / 1024  # MB
        reduction = (1 - sparse_memory / dense_memory) * 100
        
        print("\nMEMORY SAVINGS (Sparse Mode)")
        print("="*80)
        print(f"Dense representation:  {dense_memory:.2f} MB")
        print(f"Sparse representation: {sparse_memory:.2f} MB")
        print(f"Reduction:             {reduction:.1f}%")
        print("="*80)

if __name__ == "__main__":
    demo_advanced_features()
