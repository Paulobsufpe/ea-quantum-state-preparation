# qext_batch_test.py
import time
from typing import List, Dict
import numpy as np

try:
    from qext_visualize import VisualQuantumOptimizer
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

def batch_optimization_test():
    """Run multiple optimization tests with different parameters"""
    if not HAS_EXT:
        print("C++ extension not available. Please compile first.")
        return
    
    test_cases = [
        {"qubits": 2, "depth": 8, "population": 30, "generations": 50},
        {"qubits": 3, "depth": 12, "population": 50, "generations": 100},
        {"qubits": 4, "depth": 15, "population": 80, "generations": 150},
    ]
    
    results = []
    
    for i, config in enumerate(test_cases):
        print(f"\n{'#'*80}")
        print(f"TEST CASE {i+1}: {config['qubits']} qubits, depth {config['depth']}")
        print(f"{'#'*80}")
        
        try:
            optimizer = VisualQuantumOptimizer(
                num_qubits=config['qubits'],
                target_depth=config['depth'],
                population_size=config['population'],
                generations=config['generations'],
                use_hybrid=True
            )
            
            start_time = time.time()
            best_circuit = optimizer.run_optimization(verbose=True)
            end_time = time.time()
            
            if best_circuit:
                # Calculate metrics
                target_unitary = optimizer.target_circuit.circuit_to_unitary()
                best_unitary = best_circuit.circuit_to_unitary()
                fidelity = calculate_fidelity(best_unitary, target_unitary)
                
                results.append({
                    'test_case': i+1,
                    'qubits': config['qubits'],
                    'target_depth': config['depth'],
                    'optimized_depth': best_circuit.depth,
                    'depth_reduction': ((config['depth'] - best_circuit.depth) / config['depth']) * 100,
                    'fidelity': fidelity,
                    'fitness': best_circuit.fitness,
                    'optimization_time': end_time - start_time,
                    'gate_reduction': ((optimizer.target_circuit.count_non_id_gates() - best_circuit.count_non_id_gates()) / 
                                      optimizer.target_circuit.count_non_id_gates()) * 100
                })
                
        except Exception as e:
            print(f"Error in test case {i+1}: {e}")
            continue
    
    # Print summary of all tests
    if results:
        print(f"\n{'#'*80}")
        print("BATCH TEST SUMMARY")
        print(f"{'#'*80}")
        print(f"{'Case':<6} {'Qubits':<8} {'Target Depth':<12} {'Opt Depth':<10} {'Depth Red %':<12} {'Fidelity':<10} {'Time (s)':<10}")
        print(f"{'-'*80}")
        
        for result in results:
            print(f"{result['test_case']:<6} {result['qubits']:<8} {result['target_depth']:<12} "
                  f"{result['optimized_depth']:<10} {result['depth_reduction']:<12.1f} "
                  f"{result['fidelity']:<10.4f} {result['optimization_time']:<10.2f}")

if __name__ == "__main__":
    batch_optimization_test()
