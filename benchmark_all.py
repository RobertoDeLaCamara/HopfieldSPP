"""
Comprehensive benchmark: Original vs Improved vs Advanced
"""
import numpy as np
import time
import sys
import os
import traceback

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.train_model import HopfieldModel as OriginalModel, calculate_cost_matrix
from src.train_model_improved import ImprovedHopfieldModel as ImprovedModel
from src.train_model_advanced import AdvancedHopfieldModel as AdvancedModel

def generate_test_graphs():
    """Generate different types of test graphs."""
    graphs = []
    
    # Small dense graph
    graphs.append({
        'name': 'Small Dense (20 nodes, 80% density)',
        'file': 'data/synthetic/synthetic_network.csv',
        'queries': [(0, 5), (0, 10), (3, 15)]
    })
    
    return graphs

def benchmark_model(model_class, model_name, cost_matrix, queries, **kwargs):
    """Benchmark a single model."""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*80}")
    
    n = cost_matrix.shape[0]
    cost_matrix_normalized = (cost_matrix - np.min(cost_matrix)) / (np.max(cost_matrix) - np.min(cost_matrix) + 1e-6)
    
    # Initialize model
    print("\n[Initialization]")
    start = time.time()
    try:
        model = model_class(n, cost_matrix_normalized, **kwargs)
        model.set_cost_matrix(cost_matrix)
        model.compile(optimizer='adam')
        init_time = time.time() - start
        print(f"✓ Time: {init_time:.3f}s")
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return None
    
    # Run queries
    results = {
        'times': [],
        'costs': [],
        'optimal': [],
        'failures': 0,
        'init_time': init_time
    }
    
    for i, (source, dest) in enumerate(queries):
        if source >= n or dest >= n:
            continue
            
        print(f"\n[Query {i+1}/{len(queries)}] {source} → {dest}")
        
        # Get optimal
        dijkstra_path, dijkstra_cost = model._dijkstra_path(source, dest)
        print(f"  Optimal cost: {dijkstra_cost:.2f}")
        
        # Test model
        try:
            start = time.time()
            
            if model_name == "Original":
                path = model.predict(source, dest, validate=False)
            elif model_name == "Improved":
                path = model.predict(source, dest, num_restarts=2, validate=True)
            else:  # Advanced
                path = model.predict(source, dest, num_restarts=2, validate=True, use_beam_search=True)
            
            query_time = time.time() - start
            cost = model._calculate_path_cost(path)
            
            is_optimal = abs(cost - dijkstra_cost) < 1e-6
            accuracy = (dijkstra_cost / cost * 100) if cost > 0 else 0
            
            results['times'].append(query_time)
            results['costs'].append(cost)
            results['optimal'].append(is_optimal)
            
            print(f"  Time: {query_time:.3f}s")
            print(f"  Cost: {cost:.2f}")
            print(f"  Optimal: {'✓' if is_optimal else '✗'} ({accuracy:.1f}%)")
            print(f"  Path length: {len(path)}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            results['failures'] += 1
    
    return results

def print_comparison(results_dict):
    """Print comparison table."""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Original':<15} {'Improved':<15} {'Advanced':<15}")
    print("-"*80)
    
    for model_name in ['Original', 'Improved', 'Advanced']:
        if model_name not in results_dict or results_dict[model_name] is None:
            continue
        
        r = results_dict[model_name]
        
        if model_name == 'Original':
            print(f"\n{'Init Time (s)':<25} {r['init_time']:.3f}", end='')
        elif model_name == 'Improved':
            print(f"           {r['init_time']:.3f}", end='')
        else:
            print(f"           {r['init_time']:.3f}")
    
    print()
    
    for model_name in ['Original', 'Improved', 'Advanced']:
        if model_name not in results_dict or results_dict[model_name] is None:
            continue
        
        r = results_dict[model_name]
        
        if not r['times']:
            continue
        
        if model_name == 'Original':
            print(f"{'Avg Query Time (s)':<25} {np.mean(r['times']):.3f}", end='')
        elif model_name == 'Improved':
            print(f"           {np.mean(r['times']):.3f}", end='')
        else:
            print(f"           {np.mean(r['times']):.3f}")
    
    print()
    
    for model_name in ['Original', 'Improved', 'Advanced']:
        if model_name not in results_dict or results_dict[model_name] is None:
            continue
        
        r = results_dict[model_name]
        
        if not r['costs']:
            continue
        
        if model_name == 'Original':
            print(f"{'Avg Cost':<25} {np.mean(r['costs']):.2f}", end='')
        elif model_name == 'Improved':
            print(f"           {np.mean(r['costs']):.2f}", end='')
        else:
            print(f"           {np.mean(r['costs']):.2f}")
    
    print()
    
    for model_name in ['Original', 'Improved', 'Advanced']:
        if model_name not in results_dict or results_dict[model_name] is None:
            continue
        
        r = results_dict[model_name]
        
        if not r['optimal']:
            continue
        
        optimal_count = sum(r['optimal'])
        total = len(r['optimal'])
        
        if model_name == 'Original':
            print(f"{'Optimal Solutions':<25} {optimal_count}/{total}", end='')
        elif model_name == 'Improved':
            print(f"           {optimal_count}/{total}", end='')
        else:
            print(f"           {optimal_count}/{total}")
    
    print()
    
    for model_name in ['Original', 'Improved', 'Advanced']:
        if model_name not in results_dict or results_dict[model_name] is None:
            continue
        
        r = results_dict[model_name]
        total_queries = len(r['times']) + r['failures']
        
        if model_name == 'Original':
            print(f"{'Failures':<25} {r['failures']}/{total_queries}", end='')
        elif model_name == 'Improved':
            print(f"           {r['failures']}/{total_queries}", end='')
        else:
            print(f"           {r['failures']}/{total_queries}")
    
    print("\n" + "="*80)
    
    # Calculate improvements
    if 'Original' in results_dict and 'Advanced' in results_dict:
        orig = results_dict['Original']
        adv = results_dict['Advanced']
        
        if orig and adv and orig['times'] and adv['times']:
            print("\nIMPROVEMENTS (Original → Advanced):")
            print("-"*80)
            
            time_improvement = (1 - np.mean(adv['times']) / np.mean(orig['times'])) * 100
            print(f"Speed:              {time_improvement:+.1f}%")
            
            cost_improvement = (1 - np.mean(adv['costs']) / np.mean(orig['costs'])) * 100
            print(f"Solution Quality:   {cost_improvement:+.1f}%")
            
            optimal_improvement = sum(adv['optimal']) - sum(orig['optimal'])
            print(f"Optimal Solutions:  {optimal_improvement:+d}")
            
            failure_improvement = orig['failures'] - adv['failures']
            print(f"Fewer Failures:     {failure_improvement:+d}")
            
            print("="*80)

def main():
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║           HOPFIELD SPP - COMPREHENSIVE BENCHMARK                           ║")
    print("║           Original vs Improved vs Advanced                                 ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    # Load test graph
    print("\n[Loading Test Graph]")
    cost_matrix, _ = calculate_cost_matrix('data/synthetic/synthetic_network.csv')
    n = cost_matrix.shape[0]
    num_edges = np.sum(cost_matrix < 1e6)
    density = num_edges / (n * n)
    
    print(f"Nodes: {n}")
    print(f"Edges: {num_edges}")
    print(f"Density: {density:.1%}")
    
    # Generate queries
    queries = [(0, min(5, n-1)), (0, min(10, n-1)), (3, min(15, n-1))]
    print(f"Queries: {len(queries)}")
    
    results = {}
    
    # Benchmark Original
    try:
        results['Original'] = benchmark_model(
            OriginalModel, 
            "Original",
            cost_matrix,
            queries
        )
    except Exception as e:
        print(f"\n✗ Original model failed: {str(e)}")
        traceback.print_exc()
        results['Original'] = None
    
    # Benchmark Improved
    try:
        results['Improved'] = benchmark_model(
            ImprovedModel,
            "Improved", 
            cost_matrix,
            queries
        )
    except Exception as e:
        print(f"\n✗ Improved model failed: {str(e)}")
        traceback.print_exc()
        results['Improved'] = None
    
    # Benchmark Advanced
    try:
        results['Advanced'] = benchmark_model(
            AdvancedModel,
            "Advanced",
            cost_matrix,
            queries,
            use_sparse=(density < 0.3)
        )
    except Exception as e:
        print(f"\n✗ Advanced model failed: {str(e)}")
        traceback.print_exc()
        results['Advanced'] = None
    
    # Print comparison
    print_comparison(results)
    
    # Feature comparison
    print("\n" + "="*80)
    print("FEATURE COMPARISON")
    print("="*80)
    
    features = [
        ("Correct Algorithm", "✗", "✓", "✓"),
        ("Flow Conservation", "✗", "✓", "✓"),
        ("Offline Training", "1000 epochs", "0 epochs", "0 epochs"),
        ("Optimizer Reset", "✗", "✓", "✓"),
        ("Connectivity Check", "✗", "✓", "✓"),
        ("Dijkstra Fallback", "✗", "✓", "✓"),
        ("Model Caching", "✗", "✓", "✓"),
        ("Early Stopping", "✗", "✓", "✓"),
        ("Multi-Start", "✗", "✓", "✓"),
        ("Sparse Tensors", "✗", "✗", "✓"),
        ("Adaptive Hyperparams", "✗", "✗", "✓"),
        ("Attention Mechanism", "✗", "✗", "✓"),
        ("Beam Search", "✗", "✗", "✓"),
        ("LR Scheduling", "✗", "✗", "✓"),
        ("Backtracking", "✗", "✗", "✓"),
        ("Best State Tracking", "✗", "✗", "✓"),
    ]
    
    print(f"\n{'Feature':<25} {'Original':<15} {'Improved':<15} {'Advanced':<15}")
    print("-"*80)
    for feature, orig, impr, adv in features:
        print(f"{feature:<25} {orig:<15} {impr:<15} {adv:<15}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
For Production Use:
  • Small graphs (< 100 nodes):     Use Improved
  • Medium graphs (100-500 nodes):  Use Improved or Advanced
  • Large graphs (500+ nodes):      Use Advanced with sparse=True
  • Sparse graphs (density < 30%):  Use Advanced with sparse=True
  
For Research:
  • Use Advanced for maximum flexibility and performance
  
Never Use:
  • Original - has fundamental algorithmic flaws
""")
    print("="*80)

if __name__ == "__main__":
    main()
