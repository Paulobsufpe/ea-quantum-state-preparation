import numpy as np
from typing import List, Dict, Optional, Callable
import time

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, Operator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

# Import the C++ extension
try:
    from qext import Gate, GateType, CircuitIndividual, QuantumEvolutionaryOptimizer
except ImportError:
    print("C++ extension not available. Please compile with: python setup.py build_ext --inplace")
    raise

class OptimizedQuantumEvolver:
    """
    Optimized wrapper that uses the C++ extension for high-performance
    quantum circuit optimization.
    """
    
    def __init__(self, 
                 num_qubits: int,
                 target_unitary: Optional[np.ndarray] = None,
                 target_circuit: Optional[CircuitIndividual] = None,
                 population_size: int = 200,
                 generations: int = 1000,
                 crossover_rate: float = 0.85,
                 mutation_rate: float = 0.85,
                 offspring_rate: float = 0.3,
                 replace_rate: float = 0.3,
                 gate_set: Optional[List[GateType]] = None,
                 alpha: float = 10.0,
                 beta: float = 1.0):
        
        self.num_qubits = num_qubits
        
        # Set target
        if target_unitary is not None:
            self.target_unitary = target_unitary
            self.target_depth = 20
        elif target_circuit is not None:
            self.target_unitary = self.circuit_to_unitary(target_circuit)
            self.target_depth = target_circuit.depth
        else:
            raise ValueError("Either target_unitary or target_circuit must be provided")
        
        # Default gate set
        if gate_set is None:
            gate_set = [GateType.ID, GateType.X, GateType.SX, GateType.RZ, GateType.CX]
        
        # Initialize C++ optimizer
        self.cpp_optimizer = QuantumEvolutionaryOptimizer(
            num_qubits, population_size, generations, crossover_rate,
            mutation_rate, offspring_rate, replace_rate, alpha, beta,
            self.target_depth, gate_set
        )
        
        # Set fitness function
        self.cpp_optimizer.set_fitness_function(self._fitness_wrapper)
        
    def _fitness_wrapper(self, circuit: CircuitIndividual) -> float:
        """Wrapper for fitness function that C++ can call"""
        fidelity = self.calculate_fidelity(circuit, self.target_unitary)
        
        # Normalize depth
        if self.target_depth > 1:
            normalized_depth = (circuit.depth - 1) / (self.target_depth - 1)
        else:
            normalized_depth = 0.0
            
        # Multi-objective fitness
        return 10.0 * fidelity - 1.0 * normalized_depth
    
    def circuit_to_unitary(self, circuit: CircuitIndividual) -> np.ndarray:
        """Convert C++ circuit to unitary matrix using Qiskit"""
        if not HAS_QISKIT:
            return np.eye(2 ** self.num_qubits)
            
        qc = QuantumCircuit(self.num_qubits)
        
        for layer in circuit.layers:
            for gate in layer:
                if gate.type == GateType.ID:
                    continue
                    
                if gate.type == GateType.X:
                    for qubit in gate.qubits:
                        qc.x(qubit)
                elif gate.type == GateType.SX:
                    for qubit in gate.qubits:
                        qc.sx(qubit)
                elif gate.type == GateType.RZ:
                    for qubit in gate.qubits:
                        qc.rz(gate.angle, qubit)
                elif gate.type == GateType.H:
                    for qubit in gate.qubits:
                        qc.h(qubit)
                elif gate.type == GateType.CX and len(gate.qubits) == 2:
                    qc.cx(gate.qubits[0], gate.qubits[1])
        
        return Operator(qc).data
    
    def calculate_fidelity(self, circuit: CircuitIndividual, target_unitary: np.ndarray) -> float:
        """Calculate fidelity between circuit and target unitary"""
        circuit_unitary = self.circuit_to_unitary(circuit)
        fidelity = np.abs(np.trace(circuit_unitary.conj().T @ target_unitary)) ** 2
        fidelity /= (2 ** self.num_qubits) ** 2
        return float(fidelity)
    
    def run_evolution(self, from_scratch: bool = True, selection_method: str = "tournament") -> CircuitIndividual:
        """Run the evolutionary optimization"""
        print("Starting optimized C++ evolution...")
        start_time = time.time()
        
        best_circuit = self.cpp_optimizer.run_evolution(from_scratch, selection_method)
        
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        return best_circuit
    
    def get_population_stats(self) -> Dict:
        """Get statistics about the current population"""
        population = self.cpp_optimizer.get_population()
        
        fitnesses = [ind.fitness for ind in population]
        depths = [ind.depth for ind in population]
        
        return {
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'avg_depth': np.mean(depths),
            'min_depth': np.min(depths),
            'max_depth': np.max(depths),
            'population_size': len(population)
        }

# Helper function to create random circuits using C++
def create_random_circuit_cpp(num_qubits: int, depth: int) -> CircuitIndividual:
    """Create a random circuit using the C++ implementation"""
    gate_set = [GateType.ID, GateType.X, GateType.SX, GateType.RZ, GateType.CX]
    temp_optimizer = QuantumEvolutionaryOptimizer(
        num_qubits, 1, 1, 0.85, 0.85, 0.3, 0.3, 10.0, 1.0, depth, gate_set
    )
    return temp_optimizer.create_random_circuit(depth)

# Example usage
def benchmark_optimization():
    """Benchmark the optimized C++ implementation"""
    print("Benchmarking C++ implementation...")
    
    # Create a simple target circuit using C++
    num_qubits = 3
    target_depth = 8
    
    target_circuit = create_random_circuit_cpp(num_qubits, target_depth)
    
    # Test configurations
    configs = [
        {"pop_size": 50, "generations": 100},
        {"pop_size": 100, "generations": 200},
    ]
    
    for config in configs:
        print(f"\nTesting with population {config['pop_size']}, generations {config['generations']}")
        
        # C++ implementation
        cpp_evolver = OptimizedQuantumEvolver(
            num_qubits=num_qubits,
            target_circuit=target_circuit,
            population_size=config['pop_size'],
            generations=config['generations']
        )
        
        start_time = time.time()
        best_cpp = cpp_evolver.run_evolution()
        cpp_time = time.time() - start_time
        
        print(f"C++ implementation: {cpp_time:.2f}s")
        print(f"Best fitness: {best_cpp.fitness:.4f}")
        print(f"Best depth: {best_cpp.depth}")

if __name__ == "__main__":
    benchmark_optimization()
