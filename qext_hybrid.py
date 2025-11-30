# qext_simple.py
import numpy as np
import time
from typing import List, Dict

try:
    from qext import Gate, GateType, CircuitIndividual, QuantumEvolutionaryOptimizer
    from qext import calculate_fidelity, calculate_fitness, identity_matrix
    HAS_EXT = True
except ImportError as e:
    print(f"C++ extension not available: {e}")
    HAS_EXT = False

class SimpleQuantumOptimizer:
    """Simplified interface for the hybrid quantum optimizer"""
    
    def __init__(self, num_qubits: int, target_depth: int = 20):
        if not HAS_EXT:
            raise RuntimeError("C++ extension not available")
            
        self.num_qubits = num_qubits
        self.target_depth = target_depth
        
        # Create a random target circuit
        gate_set = [GateType.ID, GateType.X, GateType.SX, GateType.RZ, GateType.CX]
        temp_optimizer = QuantumEvolutionaryOptimizer(
            num_qubits, 1, 1, 0.85, 0.85, 0.3, 0.3, 10.0, 1.0, target_depth, gate_set
        )
        self.target_circuit = temp_optimizer.create_random_circuit(target_depth)
        self.target_unitary = self.target_circuit.circuit_to_unitary()
        
        # Initialize main optimizer
        self.optimizer = QuantumEvolutionaryOptimizer(
            num_qubits=num_qubits,
            population_size=100,
            generations=200,
            crossover_rate=0.85,
            mutation_rate=0.85,
            offspring_rate=0.3,
            replace_rate=0.3,
            alpha=10.0,
            beta=1.0,
            target_depth=target_depth,
            gate_set=gate_set,
            param_freq=25,
            param_rate=0.1
        )
        
        self.optimizer.set_target_unitary(self.target_unitary)
        self.optimizer.set_fitness_function(
            lambda circuit: calculate_fitness(circuit, self.target_unitary, 10.0, 1.0, target_depth)
        )
    
    def run_optimization(self):
        """Run the hybrid optimization"""
        print(f"Starting optimization for {self.num_qubits} qubits...")
        start_time = time.time()
        
        best_circuit = self.optimizer.run_evolution(from_scratch=True)
        
        optimization_time = time.time() - start_time
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        
        return best_circuit
    
    def analyze_results(self, circuit: CircuitIndividual):
        """Analyze optimization results"""
        unitary = circuit.circuit_to_unitary()
        fidelity = calculate_fidelity(unitary, self.target_unitary)
        
        print(f"\nResults:")
        print(f"  Circuit depth: {circuit.depth} (target: {self.target_depth})")
        print(f"  Fitness: {circuit.fitness:.6f}")
        print(f"  Fidelity: {fidelity:.6f}")
        print(f"  Non-ID gates: {circuit.count_non_id_gates()}")
        
        gate_counts = circuit.gate_counts()
        print("  Gate counts:")
        for gate_type, count in gate_counts.items():
            if count > 0:
                print(f"    {gate_type.name}: {count}")

# Example usage
if __name__ == "__main__":
    if HAS_EXT:
        print("=== Hybrid Quantum Circuit Optimization ===\n")
        
        optimizer = SimpleQuantumOptimizer(num_qubits=3, target_depth=15)
        
        print("Target circuit created")
        optimizer.analyze_results(optimizer.target_circuit)
        
        print("\nRunning optimization...")
        best_circuit = optimizer.run_optimization()
        
        print("\nOptimized circuit:")
        optimizer.analyze_results(best_circuit)
        
    else:
        print("Please compile the C++ extension first:")
        print("python setup.py build_ext --inplace")
