# qext_visualize_fixed.py
import numpy as np
import time
from typing import List, Dict, Optional

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, Operator
    from qiskit.circuit import Parameter
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Warning: Qiskit not available. Circuit visualization disabled.")

import sys
if (len(sys.argv) > 1 and sys.argv[1] == "omp"):
    try:
        from qext_omp import Gate, GateType, CircuitIndividual, QuantumEvolutionaryOptimizer
        from qext_omp import calculate_fidelity, calculate_fitness, identity_matrix
        HAS_EXT = True
    except ImportError as e:
        print(f"C++ extension not available: {e}")
        HAS_EXT = False
else:
    try:
        from qext import Gate, GateType, CircuitIndividual, QuantumEvolutionaryOptimizer
        from qext import calculate_fidelity, calculate_fitness, identity_matrix
        HAS_EXT = True
    except ImportError as e:
        print(f"C++ extension not available: {e}")
        HAS_EXT = False
    pass

class VisualQuantumOptimizer:
    """
    Enhanced quantum circuit optimizer with visualization capabilities.
    """
    
    def __init__(self, 
                 num_qubits: int, 
                 target_depth: int = 20,
                 population_size: int = 100,
                 generations: int = 200,
                 use_hybrid: bool = True):
        
        if not HAS_EXT:
            raise RuntimeError("C++ extension not available. Please compile first.")
        if not HAS_QISKIT:
            print("Warning: Qiskit not available. Visualization will be limited.")
            
        self.num_qubits = num_qubits
        self.target_depth = target_depth
        self.use_hybrid = use_hybrid
        
        # Default gate set matching the research papers
        self.gate_set = [GateType.ID, GateType.X, GateType.SX, GateType.RZ, GateType.CX, GateType.H]
        
        # Create a random target circuit
        temp_optimizer = QuantumEvolutionaryOptimizer(
            num_qubits, 1, 1, 0.85, 0.85, 0.3, 0.3, 10.0, 1.0, target_depth, self.gate_set
        )
        self.target_circuit = temp_optimizer.create_random_circuit(target_depth)
        self.target_unitary = self.target_circuit.circuit_to_unitary()
        
        # Set hybrid optimization parameters
        param_freq = 25 if use_hybrid else 1000  # Disable if not hybrid
        param_rate = 0.1 if use_hybrid else 0.0
        
        # Initialize main optimizer
        self.optimizer = QuantumEvolutionaryOptimizer(
            num_qubits=num_qubits,
            population_size=population_size,
            generations=generations,
            crossover_rate=0.85,
            mutation_rate=0.85,
            offspring_rate=0.3,
            replace_rate=0.3,
            alpha=10.0,
            beta=1.0,
            target_depth=target_depth,
            gate_set=self.gate_set,
            param_freq=param_freq,
            param_rate=param_rate
        )
        
        self.optimizer.set_target_unitary(self.target_unitary)
        self.optimizer.set_fitness_function(
            lambda circuit: calculate_fitness(circuit, self.target_unitary, 10.0, 1.0, target_depth)
        )
        
        self.best_circuit = None
        self.optimization_stats = {}
    
    def circuit_to_qiskit(self, circuit: CircuitIndividual) -> QuantumCircuit:
        """Convert C++ CircuitIndividual to Qiskit QuantumCircuit"""
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required for circuit conversion")
            
        qc = QuantumCircuit(circuit.num_qubits)
        
        for layer_idx, layer in enumerate(circuit.layers):
            for gate in layer:
                if gate.type == GateType.ID:
                    continue  # Skip identity gates
                    
                qubits = gate.qubits
                
                if gate.type == GateType.X:
                    for qubit in qubits:
                        qc.x(qubit)
                elif gate.type == GateType.SX:
                    for qubit in qubits:
                        qc.sx(qubit)
                elif gate.type == GateType.RZ:
                    for qubit in qubits:
                        # Use the actual angle value from the gate
                        qc.rz(gate.angle, qubit)
                elif gate.type == GateType.H:
                    for qubit in qubits:
                        qc.h(qubit)
                elif gate.type == GateType.CX and len(qubits) >= 2:
                    # For CX gates, use first two qubits as control and target
                    qc.cx(qubits[0], qubits[1])
            
            # Add barrier between layers for better visualization
            if layer_idx < len(circuit.layers) - 1:
                qc.barrier()
        
        return qc
    
    def visualize_circuit(self, 
                         circuit: CircuitIndividual, 
                         title: str = "Circuit",
                         show_unitary: bool = False,
                         show_statevector: bool = False):
        """Visualize a circuit using Qiskit's draw method"""
        if not HAS_QISKIT:
            print(f"Cannot visualize {title}: Qiskit not available")
            return
            
        try:
            qc = self.circuit_to_qiskit(circuit)
            
            print(f"\n{'='*60}")
            print(f"{title}")
            print(f"{'='*60}")
            
            # Basic circuit info
            unitary = circuit.circuit_to_unitary()
            fidelity = calculate_fidelity(unitary, self.target_unitary)
            
            print(f"Depth: {circuit.depth}")
            print(f"Qubits: {circuit.num_qubits}")
            print(f"Fitness: {circuit.fitness:.6f}")
            print(f"Fidelity: {fidelity:.6f}")
            print(f"Non-ID gates: {circuit.count_non_id_gates()}")
            
            # Gate distribution
            gate_counts = circuit.gate_counts()
            print("Gate distribution:")
            for gate_type, count in gate_counts.items():
                if count > 0:
                    print(f"  {gate_type.name}: {count}")
            
            print(f"\nCircuit diagram:")
            print(qc.draw(output='text'))
            
            # Optional: Show unitary matrix
            if show_unitary:
                print(f"\nUnitary matrix (first 4x4 block):")
                dim = min(4, unitary.rows())
                for i in range(dim):
                    row = "  ".join([f"{unitary(i,j).real:.3f}{unitary(i,j).imag:+.3f}j" for j in range(dim)])
                    print(f"  [{row}]")
            
            # Optional: Show statevector
            if show_statevector:
                try:
                    # Apply circuit to |0⟩ state
                    initial_state = Statevector.from_int(0, 2**circuit.num_qubits)
                    final_state = initial_state.evolve(qc)
                    print(f"\nFinal statevector (first 8 amplitudes):")
                    for i in range(min(8, len(final_state))):
                        amp = final_state[i]
                        print(f"  |{i:0{circuit.num_qubits}b}⟩: {amp.real:.3f}{amp.imag:+.3f}j")
                except Exception as e:
                    print(f"  Could not compute statevector: {e}")
                    
        except Exception as e:
            print(f"Error visualizing circuit {title}: {e}")
    
    def run_optimization(self, verbose: bool = True) -> CircuitIndividual:
        """Run the hybrid optimization with progress tracking"""
        if verbose:
            mode = "Hybrid" if self.use_hybrid else "Standard"
            print(f"=== {mode} Quantum Circuit Optimization ===")
            print(f"Qubits: {self.num_qubits}, Target depth: {self.target_depth}")
            
            # FIXED: Use direct attribute access instead of getter methods
            print(f"Population: {self.optimizer.population_size}")
            print(f"Generations: {self.optimizer.generations}")
            
            if self.use_hybrid:
                print("Parameter optimization: Enabled (every 25 generations)")
            else:
                print("Parameter optimization: Disabled")
        
        start_time = time.time()
        
        # Visualize target circuit before optimization
        if verbose and HAS_QISKIT:
            self.visualize_circuit(self.target_circuit, "TARGET CIRCUIT", show_unitary=True)
        
        # Run optimization
        self.best_circuit = self.optimizer.run_evolution(from_scratch=True, selection_method="roulette")
        optimization_time = time.time() - start_time
        
        # Collect optimization statistics
        population = self.optimizer.get_population()
        fitness_history = self.optimizer.get_fitness_history()
        
        if population:
            fitnesses = [ind.fitness for ind in population]
            depths = [ind.depth for ind in population]
            
            self.optimization_stats = {
                'best_fitness': max(fitnesses) if fitnesses else 0.0,
                'avg_fitness': np.mean(fitnesses) if fitnesses else 0.0,
                'best_depth': min(depths) if depths else 0,
                'avg_depth': np.mean(depths) if depths else 0,
                'convergence_generation': len(fitness_history),
                'optimization_time': optimization_time,
                'population_size': len(population)
            }
        
        if verbose:
            print(f"\nOptimization completed in {optimization_time:.2f} seconds")
            
            # Visualize best circuit
            if HAS_QISKIT and self.best_circuit:
                self.visualize_circuit(self.best_circuit, "BEST OPTIMIZED CIRCUIT", show_unitary=True)
            
            self.print_optimization_summary()
        
        return self.best_circuit
    
    def print_optimization_summary(self):
        """Print comprehensive optimization summary"""
        if not self.best_circuit:
            print("No optimization results available.")
            return
            
        target_unitary = self.target_circuit.circuit_to_unitary()
        best_unitary = self.best_circuit.circuit_to_unitary()
        fidelity = calculate_fidelity(best_unitary, target_unitary)
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"Target Circuit:")
        print(f"  Depth: {self.target_circuit.depth}")
        print(f"  Non-ID gates: {self.target_circuit.count_non_id_gates()}")
        
        print(f"\nOptimized Circuit:")
        print(f"  Depth: {self.best_circuit.depth} ({self._get_depth_change():+.1f}%)")
        print(f"  Non-ID gates: {self.best_circuit.count_non_id_gates()} ({self._get_gate_change():+.1f}%)")
        print(f"  Final fidelity: {fidelity:.6f}")
        print(f"  Final fitness: {self.best_circuit.fitness:.6f}")
        
        print(f"\nOptimization Process:")
        print(f"  Generations: {self.optimization_stats.get('convergence_generation', 0)}")
        print(f"  Best fitness: {self.optimization_stats.get('best_fitness', 0):.6f}")
        print(f"  Average fitness: {self.optimization_stats.get('avg_fitness', 0):.6f}")
        print(f"  Optimization time: {self.optimization_stats.get('optimization_time', 0):.2f}s")
        
        # Show gate distribution comparison
        self._compare_gate_distributions()
    
    def _get_depth_change(self) -> float:
        """Calculate depth change percentage"""
        if not self.best_circuit:
            return 0.0
        change = ((self.best_circuit.depth - self.target_circuit.depth) / self.target_circuit.depth) * 100
        return change
    
    def _get_gate_change(self) -> float:
        """Calculate gate count change percentage"""
        if not self.best_circuit:
            return 0.0
        target_gates = self.target_circuit.count_non_id_gates()
        best_gates = self.best_circuit.count_non_id_gates()
        if target_gates == 0:
            return 0.0
        change = ((best_gates - target_gates) / target_gates) * 100
        return change
    
    def _compare_gate_distributions(self):
        """Compare gate distributions between target and optimized circuits"""
        target_counts = self.target_circuit.gate_counts()
        best_counts = self.best_circuit.gate_counts()
        
        print(f"\nGate Distribution Comparison:")
        print(f"  Gate       |  Target  |  Optimized |  Change")
        print(f"  -----------|----------|------------|---------")
        
        all_gate_types = set(target_counts.keys()) | set(best_counts.keys())
        for gate_type in sorted(all_gate_types, key=lambda x: x.name):
            target_count = target_counts.get(gate_type, 0)
            best_count = best_counts.get(gate_type, 0)
            change = best_count - target_count
            change_str = f"{change:+.1f}" if change != 0 else " 0"
            print(f"  {gate_type.name:10} | {target_count:8} | {best_count:10} | {change_str:>7}")

def create_comparison_optimizer():
    """Create an optimizer for comparison between hybrid and standard approaches"""
    print("Creating comparison optimizer with 3 qubits...")
    return VisualQuantumOptimizer(
        num_qubits=4,
        target_depth=20,
        population_size=200,
        generations=1000,
        use_hybrid=True
    )

def demo_optimization():
    """Run a demonstration of the quantum circuit optimization"""
    if not HAS_EXT:
        print("Please compile the C++ extension first:")
        print("python setup_fixed.py build_ext --inplace")
        return
        
    print("=== Quantum Circuit Optimization Demo ===")
    print("This demo will:")
    print("1. Create a random target quantum circuit")
    print("2. Run hybrid evolutionary optimization")
    print("3. Show the original and optimized circuits")
    print("4. Display optimization statistics\n")
    
    try:
        # Create and run optimizer
        optimizer = create_comparison_optimizer()
        
        # Run optimization with visualization
        best_circuit = optimizer.run_optimization(verbose=True)
        
        # Additional detailed analysis
        if best_circuit and HAS_QISKIT:
            print(f"\n{'='*60}")
            print("DETAILED CIRCUIT ANALYSIS")
            print(f"{'='*60}")
            
            # Show both circuits side by side in a compact form
            print("\nTarget Circuit (Compact):")
            target_qc = optimizer.circuit_to_qiskit(optimizer.target_circuit)
            print(target_qc.draw(output='text', fold=80))  # Compact output
            
            print("\nOptimized Circuit (Compact):")
            best_qc = optimizer.circuit_to_qiskit(best_circuit)
            print(best_qc.draw(output='text', fold=80))  # Compact output
            
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_optimization()
