#!/usr/bin/env python3
"""
Quick demo of the improved Hopfield model.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.train_model_improved import (
    calculate_cost_matrix,
    ImprovedHopfieldModel,
    train_improved_model
)

def demo():
    print("="*80)
    print("HOPFIELD SPP - IMPROVED MODEL DEMO")
    print("="*80)
    
    # Load network
    print("\n1. Loading network...")
    network_file = 'data/synthetic/synthetic_network.csv'
    cost_matrix, node_mapping = calculate_cost_matrix(network_file)
    n = cost_matrix.shape[0]
    print(f"   ✓ Loaded {n} nodes, {np.sum(cost_matrix < 1e6)} edges")
    
    # Normalize
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)
    
    # Create model
    print("\n2. Creating improved model...")
    model = ImprovedHopfieldModel(n, cost_matrix_normalized)
    model.set_cost_matrix(cost_matrix)
    model.compile(optimizer='adam')
    print("   ✓ Model initialized")
    
    # Test queries
    print("\n3. Testing shortest path queries...")
    print("-"*80)
    
    test_queries = [
        (0, 5),
        (0, 9),
        (3, 7),
    ]
    
    for source, dest in test_queries:
        if source >= n or dest >= n:
            continue
            
        print(f"\n   Query: Node {source} → Node {dest}")
        print("   " + "-"*40)
        
        # Get Dijkstra optimal
        dijkstra_path, dijkstra_cost = model._dijkstra_path(source, dest)
        print(f"   Dijkstra (optimal):")
        print(f"     Path: {dijkstra_path}")
        print(f"     Cost: {dijkstra_cost:.2f}")
        
        # Get Hopfield solution
        try:
            hopfield_path = model.predict(source, dest, num_restarts=2, validate=True)
            hopfield_cost = model._calculate_path_cost(hopfield_path)
            
            print(f"\n   Hopfield (improved):")
            print(f"     Path: {hopfield_path}")
            print(f"     Cost: {hopfield_cost:.2f}")
            
            # Check optimality
            is_optimal = abs(hopfield_cost - dijkstra_cost) < 1e-6
            accuracy = (dijkstra_cost / hopfield_cost * 100) if hopfield_cost > 0 else 0
            
            print(f"\n   Result:")
            if is_optimal:
                print(f"     ✓ OPTIMAL solution found!")
            else:
                print(f"     ⚠ Near-optimal: {accuracy:.1f}% accuracy")
                
        except Exception as e:
            print(f"\n   ✗ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS:")
    print("="*80)
    print("✓ Correct flow conservation (not Hamiltonian cycle)")
    print("✓ No meaningless offline training")
    print("✓ Multi-start optimization for better solutions")
    print("✓ Dijkstra fallback for 100% reliability")
    print("✓ Early stopping for faster convergence")
    print("✓ Robust path extraction with BFS")
    print("✓ Fresh optimizer per query (no state pollution)")
    print("="*80)

if __name__ == "__main__":
    demo()
