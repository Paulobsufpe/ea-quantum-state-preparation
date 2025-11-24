import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import itertools
from scipy.optimize import minimize

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector, Operator
    from qiskit.circuit import Parameter
    from qiskit import transpile
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Warning: Qiskit not available. Some functionality will be limited.")

class GateType(Enum):
    """Quantum gate types supported by the evolutionary algorithm"""
    ID = "id"      # Identity
    X = "x"        # Pauli X
    SX = "sx"      # Square root of X
    RZ = "rz"      # Rotation around Z-axis
    CX = "cx"      # Controlled X (CNOT)
    H = "h"        # Hadamard (added for completeness)

@dataclass
class Gate:
    """Represents a quantum gate in the circuit"""
    gate_type: GateType
    qubits: List[int]           # Target qubit(s)
    control_qubits: List[int] = None  # For controlled gates
    angle: float = 0.0          # For parameterized gates (RZ)
    
    def __post_init__(self):
        if self.control_qubits is None:
            self.control_qubits = []

class CircuitIndividual:
    """Represents an individual quantum circuit in the population"""
    
    def __init__(self, num_qubits: int, depth: int, gates: List[List[Gate]] = None):
        self.num_qubits = num_qubits
        self.depth = depth
        
        # 2D grid: gates[qubit][time_step]
        if gates is None:
            self.gates = [[Gate(GateType.ID, [q]) for _ in range(depth)] for q in range(num_qubits)]
        else:
            self.gates = gates
            
        self.fitness: float = -float('inf')
        self.fidelity: float = 0.0
        self.normalized_depth: float = 0.0
        
    def to_qiskit_circuit(self) -> 'QuantumCircuit':
        """Convert to Qiskit QuantumCircuit"""
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required for this operation")
            
        qc = QuantumCircuit(self.num_qubits)
        
        for time_step in range(self.depth):
            for qubit in range(self.num_qubits):
                gate = self.gates[qubit][time_step]
                
                if gate.gate_type == GateType.ID:
                    continue  # Skip identity gates
                    
                if gate.gate_type == GateType.X:
                    qc.x(qubit)
                elif gate.gate_type == GateType.SX:
                    qc.sx(qubit)
                elif gate.gate_type == GateType.RZ:
                    qc.rz(gate.angle, qubit)
                elif gate.gate_type == GateType.H:
                    qc.h(qubit)
                elif gate.gate_type == GateType.CX:
                    if len(gate.control_qubits) > 0:
                        control = gate.control_qubits[0]
                        qc.cx(control, qubit)
        
        return qc
    
    def calculate_unitary(self) -> np.ndarray:
        """Calculate the unitary matrix of the circuit"""
        if not HAS_QISKIT:
            # Fallback: return identity matrix
            return np.eye(2 ** self.num_qubits)
            
        qc = self.to_qiskit_circuit()
        return Operator(qc).data
    
    def calculate_fidelity(self, target_unitary: np.ndarray) -> float:
        """Calculate fidelity with target unitary"""
        circuit_unitary = self.calculate_unitary()
        fidelity = np.abs(np.trace(circuit_unitary.conj().T @ target_unitary)) ** 2
        fidelity /= (2 ** self.num_qubits) ** 2
        return float(fidelity)
    
    def get_parameters(self) -> List[float]:
        """Get all parameterized gate angles"""
        parameters = []
        for qubit_gates in self.gates:
            for gate in qubit_gates:
                if gate.gate_type == GateType.RZ:
                    parameters.append(gate.angle)
        return parameters
    
    def set_parameters(self, parameters: List[float]):
        """Set parameterized gate angles"""
        param_idx = 0
        for qubit_gates in self.gates:
            for gate in qubit_gates:
                if gate.gate_type == GateType.RZ:
                    if param_idx < len(parameters):
                        gate.angle = parameters[param_idx]
                        param_idx += 1

class CrossoverType(Enum):
    SINGLE_POINT = "single_point"
    UNIFORM = "uniform"
    MULTI_POINT = "multi_point"
    BLOCKWISE = "blockwise"

class MutationType(Enum):
    SINGLE_GATE = "single_gate"
    GATE_SWAP = "gate_swap"
    COLUMN_SWAP = "column_swap"
    CTRL_TARGET_SWAP = "ctrl_target_swap"
    ADD_RANDOM_COLUMN = "add_random_column"
    DELETE_COLUMN = "delete_column"
    ADD_CX_GATE = "add_cx_gate"
    ADD_SINGLE_GATE = "add_single_gate"
    MUTATE_PARAMETERS = "mutate_parameters"

class QuantumEvolutionaryOptimizer:
    """
    Main class for quantum circuit optimization using evolutionary algorithms
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
                 gate_set: List[GateType] = None,
                 alpha: float = 10.0,
                 beta: float = 1.0):
        
        self.num_qubits = num_qubits
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.offspring_rate = offspring_rate
        self.replace_rate = replace_rate
        self.alpha = alpha
        self.beta = beta
        
        # Set target
        if target_unitary is not None:
            self.target_unitary = target_unitary
            self.target_depth = 20  # Default
        elif target_circuit is not None:
            self.target_unitary = target_circuit.calculate_unitary()
            self.target_depth = target_circuit.depth
        else:
            raise ValueError("Either target_unitary or target_circuit must be provided")
        
        # Default gate set from papers
        if gate_set is None:
            self.gate_set = [GateType.ID, GateType.X, GateType.SX, GateType.RZ, GateType.CX]
        else:
            self.gate_set = gate_set
            
        self.population: List[CircuitIndividual] = []
        self.best_individual: Optional[CircuitIndividual] = None
        self.fitness_history: List[float] = []
        
    def initialize_population(self, initial_depth: int = None, from_target: bool = False):
        """Initialize population with random circuits or from target"""
        if initial_depth is None:
            initial_depth = self.target_depth
            
        self.population = []
        
        for _ in range(self.population_size):
            if from_target:
                # Start with target circuit (for optimization)
                # This would require cloning the target circuit
                # For now, create random circuit
                individual = self._create_random_circuit(initial_depth)
            else:
                # Create from scratch
                individual = self._create_random_circuit(initial_depth)
                
            self.population.append(individual)
    
    def _create_random_circuit(self, depth: int) -> CircuitIndividual:
        """Create a random quantum circuit"""
        gates = []
        
        for qubit in range(self.num_qubits):
            qubit_gates = []
            for time_step in range(depth):
                gate_type = random.choice(self.gate_set)
                
                if gate_type == GateType.CX:
                    # Find available control qubit
                    available_qubits = [q for q in range(self.num_qubits) if q != qubit]
                    if available_qubits:
                        control = random.choice(available_qubits)
                        gate = Gate(gate_type, [qubit], [control])
                    else:
                        # Fallback to single-qubit gate
                        gate_type = random.choice([gt for gt in self.gate_set if gt != GateType.CX])
                        gate = Gate(gate_type, [qubit])
                elif gate_type == GateType.RZ:
                    angle = random.uniform(0, 2 * np.pi)
                    gate = Gate(gate_type, [qubit], angle=angle)
                else:
                    gate = Gate(gate_type, [qubit])
                    
                qubit_gates.append(gate)
            gates.append(qubit_gates)
            
        return CircuitIndividual(self.num_qubits, depth, gates)
    
    def calculate_fitness(self, individual: CircuitIndividual) -> float:
        """Calculate fitness based on fidelity and circuit depth"""
        fidelity = individual.calculate_fidelity(self.target_unitary)
        
        # Normalize depth (equation from paper)
        if self.target_depth > 1:
            normalized_depth = (individual.depth - 1) / (self.target_depth - 1)
        else:
            normalized_depth = 0
            
        # Multi-objective fitness (equation 9 from paper)
        fitness = self.alpha * fidelity - self.beta * normalized_depth
        return fitness
    
    def evaluate_population(self):
        """Evaluate fitness for all individuals in population"""
        for individual in self.population:
            individual.fidelity = individual.calculate_fidelity(self.target_unitary)
            individual.fitness = self.calculate_fitness(individual)
    
    def select_parents(self, method: str = "tournament") -> Tuple[CircuitIndividual, CircuitIndividual]:
        """Select parents for crossover"""
        if method == "tournament":
            return self._tournament_selection()
        elif method == "roulette":
            return self._roulette_selection()
        else:  # random
            return random.sample(self.population, 2)
    
    def _tournament_selection(self, tournament_size: int = 3) -> Tuple[CircuitIndividual, CircuitIndividual]:
        """Tournament selection"""
        tournament1 = random.sample(self.population, tournament_size)
        tournament2 = random.sample(self.population, tournament_size)
        
        parent1 = max(tournament1, key=lambda ind: ind.fitness)
        parent2 = max(tournament2, key=lambda ind: ind.fitness)
        
        return parent1, parent2
    
    def _roulette_selection(self) -> Tuple[CircuitIndividual, CircuitIndividual]:
        """Roulette wheel selection"""
        fitnesses = [ind.fitness for ind in self.population]
        min_fitness = min(fitnesses)
        
        # Shift fitnesses to be positive
        shifted_fitnesses = [f - min_fitness + 1e-6 for f in fitnesses]
        total_fitness = sum(shifted_fitnesses)
        
        # Select first parent
        pick1 = random.uniform(0, total_fitness)
        current1 = 0
        for ind, fitness in zip(self.population, shifted_fitnesses):
            current1 += fitness
            if current1 >= pick1:
                parent1 = ind
                break
                
        # Select second parent (different from first)
        pick2 = random.uniform(0, total_fitness - shifted_fitnesses[self.population.index(parent1)])
        current2 = 0
        for ind, fitness in zip(self.population, shifted_fitnesses):
            if ind == parent1:
                continue
            current2 += fitness
            if current2 >= pick2:
                parent2 = ind
                break
                
        return parent1, parent2
    
    def crossover(self, parent1: CircuitIndividual, parent2: CircuitIndividual, 
                  method: CrossoverType = CrossoverType.UNIFORM) -> CircuitIndividual:
        """Perform crossover between two parents"""
        if random.random() > self.crossover_rate:
            # No crossover, return random parent
            return random.choice([parent1, parent2])
            
        if method == CrossoverType.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif method == CrossoverType.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif method == CrossoverType.MULTI_POINT:
            return self._multi_point_crossover(parent1, parent2)
        else:  # BLOCKWISE
            return self._blockwise_crossover(parent1, parent2)
    
    def _single_point_crossover(self, parent1: CircuitIndividual, parent2: CircuitIndividual) -> CircuitIndividual:
        """Single-point crossover"""
        # Inherit depth from random parent
        child_depth = random.choice([parent1.depth, parent2.depth])
        crossover_point = random.randint(1, child_depth - 1)
        
        child_gates = []
        for qubit in range(self.num_qubits):
            qubit_gates = []
            for time_step in range(child_depth):
                if time_step < crossover_point:
                    # Take from smaller parent or pad
                    if time_step < parent1.depth:
                        qubit_gates.append(parent1.gates[qubit][time_step])
                    else:
                        qubit_gates.append(parent2.gates[qubit][time_step])
                else:
                    if time_step < parent2.depth:
                        qubit_gates.append(parent2.gates[qubit][time_step])
                    else:
                        qubit_gates.append(Gate(GateType.ID, [qubit]))
            child_gates.append(qubit_gates)
            
        return CircuitIndividual(self.num_qubits, child_depth, child_gates)
    
    def _uniform_crossover(self, parent1: CircuitIndividual, parent2: CircuitIndividual) -> CircuitIndividual:
        """Uniform column crossover"""
        child_depth = random.choice([parent1.depth, parent2.depth])
        
        child_gates = []
        for qubit in range(self.num_qubits):
            qubit_gates = []
            for time_step in range(child_depth):
                if random.random() < 0.5 and time_step < parent1.depth:
                    qubit_gates.append(parent1.gates[qubit][time_step])
                elif time_step < parent2.depth:
                    qubit_gates.append(parent2.gates[qubit][time_step])
                else:
                    qubit_gates.append(Gate(GateType.ID, [qubit]))
            child_gates.append(qubit_gates)
            
        return CircuitIndividual(self.num_qubits, child_depth, child_gates)
    
    def _multi_point_crossover(self, parent1: CircuitIndividual, parent2: CircuitIndividual) -> CircuitIndividual:
        """Multi-point crossover"""
        child_depth = random.choice([parent1.depth, parent2.depth])
        num_points = random.randint(2, 5)
        points = sorted([0, child_depth] + [random.randint(1, child_depth-1) for _ in range(num_points)])
        
        child_gates = []
        use_parent1 = True
        
        for qubit in range(self.num_qubits):
            qubit_gates = []
            for i in range(len(points)-1):
                start, end = points[i], points[i+1]
                parent = parent1 if use_parent1 else parent2
                
                for time_step in range(start, end):
                    if time_step < parent.depth:
                        qubit_gates.append(parent.gates[qubit][time_step])
                    else:
                        qubit_gates.append(Gate(GateType.ID, [qubit]))
                        
                use_parent1 = not use_parent1
            child_gates.append(qubit_gates)
            
        return CircuitIndividual(self.num_qubits, child_depth, child_gates)
    
    def _blockwise_crossover(self, parent1: CircuitIndividual, parent2: CircuitIndividual) -> CircuitIndividual:
        """Block-wise crossover (2D)"""
        child_depth = random.choice([parent1.depth, parent2.depth])
        row_point = random.randint(1, self.num_qubits - 1)
        col_point = random.randint(1, child_depth - 1)
        
        child_gates = []
        for qubit in range(self.num_qubits):
            qubit_gates = []
            for time_step in range(child_depth):
                if (qubit < row_point and time_step < col_point) or (qubit >= row_point and time_step >= col_point):
                    # Take from parent1
                    if time_step < parent1.depth:
                        qubit_gates.append(parent1.gates[qubit][time_step])
                    else:
                        qubit_gates.append(Gate(GateType.ID, [qubit]))
                else:
                    # Take from parent2
                    if time_step < parent2.depth:
                        qubit_gates.append(parent2.gates[qubit][time_step])
                    else:
                        qubit_gates.append(Gate(GateType.ID, [qubit]))
            child_gates.append(qubit_gates)
            
        return CircuitIndividual(self.num_qubits, child_depth, child_gates)
    
    def mutate(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Apply mutation to an individual"""
        if random.random() > self.mutation_rate:
            return individual
            
        mutation_type = random.choice(list(MutationType))
        
        if mutation_type == MutationType.SINGLE_GATE:
            return self._mutate_single_gate(individual)
        elif mutation_type == MutationType.GATE_SWAP:
            return self._mutate_gate_swap(individual)
        elif mutation_type == MutationType.COLUMN_SWAP:
            return self._mutate_column_swap(individual)
        elif mutation_type == MutationType.CTRL_TARGET_SWAP:
            return self._mutate_ctrl_target_swap(individual)
        elif mutation_type == MutationType.ADD_RANDOM_COLUMN:
            return self._mutate_add_random_column(individual)
        elif mutation_type == MutationType.DELETE_COLUMN:
            return self._mutate_delete_column(individual)
        elif mutation_type == MutationType.ADD_CX_GATE:
            return self._mutate_add_cx_gate(individual)
        elif mutation_type == MutationType.ADD_SINGLE_GATE:
            return self._mutate_add_single_gate(individual)
        else:  # MUTATE_PARAMETERS
            return self._mutate_parameters(individual)
    
    def _mutate_single_gate(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Mutate a single gate"""
        qubit = random.randint(0, self.num_qubits - 1)
        time_step = random.randint(0, individual.depth - 1)
        
        new_gate_type = random.choice(self.gate_set)
        if new_gate_type == GateType.RZ:
            angle = random.uniform(0, 2 * np.pi)
            individual.gates[qubit][time_step] = Gate(new_gate_type, [qubit], angle=angle)
        else:
            individual.gates[qubit][time_step] = Gate(new_gate_type, [qubit])
            
        return individual
    
    def _mutate_gate_swap(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Swap two gates"""
        qubit1, time_step1 = random.randint(0, self.num_qubits - 1), random.randint(0, individual.depth - 1)
        qubit2, time_step2 = random.randint(0, self.num_qubits - 1), random.randint(0, individual.depth - 1)
        
        individual.gates[qubit1][time_step1], individual.gates[qubit2][time_step2] = \
            individual.gates[qubit2][time_step2], individual.gates[qubit1][time_step1]
            
        return individual
    
    def _mutate_column_swap(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Swap two columns"""
        col1, col2 = random.sample(range(individual.depth), 2)
        
        for qubit in range(self.num_qubits):
            individual.gates[qubit][col1], individual.gates[qubit][col2] = \
                individual.gates[qubit][col2], individual.gates[qubit][col1]
                
        return individual
    
    def _mutate_ctrl_target_swap(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Swap control and target of CX gates"""
        for qubit in range(self.num_qubits):
            for time_step in range(individual.depth):
                gate = individual.gates[qubit][time_step]
                if gate.gate_type == GateType.CX and gate.control_qubits:
                    # Swap control and target
                    gate.control_qubits, gate.qubits = gate.qubits, gate.control_qubits
                    
        return individual
    
    def _mutate_add_random_column(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Add a random column"""
        new_depth = individual.depth + 1
        new_gates = []
        
        for qubit in range(self.num_qubits):
            qubit_gates = individual.gates[qubit].copy()
            # Add random gate at the end
            gate_type = random.choice(self.gate_set)
            if gate_type == GateType.RZ:
                angle = random.uniform(0, 2 * np.pi)
                qubit_gates.append(Gate(gate_type, [qubit], angle=angle))
            else:
                qubit_gates.append(Gate(gate_type, [qubit]))
            new_gates.append(qubit_gates)
            
        return CircuitIndividual(self.num_qubits, new_depth, new_gates)
    
    def _mutate_delete_column(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Delete a random column (if depth > 1)"""
        if individual.depth <= 1:
            return individual
            
        col_to_delete = random.randint(0, individual.depth - 1)
        new_depth = individual.depth - 1
        new_gates = []
        
        for qubit in range(self.num_qubits):
            qubit_gates = [gate for i, gate in enumerate(individual.gates[qubit]) if i != col_to_delete]
            new_gates.append(qubit_gates)
            
        return CircuitIndividual(self.num_qubits, new_depth, new_gates)
    
    def _mutate_add_cx_gate(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Add a CX gate at random position"""
        qubit = random.randint(0, self.num_qubits - 1)
        time_step = random.randint(0, individual.depth - 1)
        
        # Find available control qubit
        available_qubits = [q for q in range(self.num_qubits) if q != qubit]
        if available_qubits:
            control = random.choice(available_qubits)
            individual.gates[qubit][time_step] = Gate(GateType.CX, [qubit], [control])
            
        return individual
    
    def _mutate_add_single_gate(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Add a single-qubit gate at random position"""
        qubit = random.randint(0, self.num_qubits - 1)
        time_step = random.randint(0, individual.depth - 1)
        
        single_qubit_gates = [gt for gt in self.gate_set if gt != GateType.CX]
        if single_qubit_gates:
            gate_type = random.choice(single_qubit_gates)
            if gate_type == GateType.RZ:
                angle = random.uniform(0, 2 * np.pi)
                individual.gates[qubit][time_step] = Gate(gate_type, [qubit], angle=angle)
            else:
                individual.gates[qubit][time_step] = Gate(gate_type, [qubit])
                
        return individual
    
    def _mutate_parameters(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Mutate parameters of parameterized gates"""
        for qubit_gates in individual.gates:
            for gate in qubit_gates:
                if gate.gate_type == GateType.RZ:
                    # Small perturbation
                    gate.angle += random.uniform(-0.1, 0.1)
                    # Keep angle in [0, 2π]
                    gate.angle %= 2 * np.pi
                    
        return individual
    
    def optimize_parameters(self, individual: CircuitIndividual, max_iter: int = 1000) -> CircuitIndividual:
        """Optimize gate parameters using COBYLA (hybrid approach)"""
        if not HAS_QISKIT:
            return individual
            
        initial_params = individual.get_parameters()
        
        if not initial_params:
            return individual  # No parameters to optimize
            
        def objective_function(params):
            individual.set_parameters(params)
            fidelity = individual.calculate_fidelity(self.target_unitary)
            return 1 - fidelity  # Minimize infidelity
            
        result = minimize(objective_function, initial_params, method='COBYLA', 
                         options={'maxiter': max_iter})
        
        individual.set_parameters(result.x)
        return individual
    
    def optimize_circuit_structure(self, individual: CircuitIndividual) -> CircuitIndividual:
        """Heuristic circuit optimization (remove identities, combine gates)"""
        # Remove identity gates from the end of each qubit's gates
        optimized_gates = []
        max_depth = 0
        
        for qubit in range(self.num_qubits):
            qubit_gates = []
            for gate in individual.gates[qubit]:
                if gate.gate_type != GateType.ID:
                    qubit_gates.append(gate)
            optimized_gates.append(qubit_gates)
            max_depth = max(max_depth, len(qubit_gates))
        
        # Pad with identities to maintain rectangular structure
        for qubit in range(self.num_qubits):
            while len(optimized_gates[qubit]) < max_depth:
                optimized_gates[qubit].append(Gate(GateType.ID, [qubit]))
                
        return CircuitIndividual(self.num_qubits, max_depth, optimized_gates)
    
    def run_evolution(self, from_scratch: bool = True, use_hybrid: bool = True, 
                      hybrid_interval: int = 25) -> CircuitIndividual:
        """Run the complete evolutionary algorithm"""
        
        print("Initializing population...")
        self.initialize_population(from_target=not from_scratch)
        self.evaluate_population()
        
        num_offspring = int(self.population_size * self.offspring_rate)
        num_replace = int(self.population_size * self.replace_rate)
        
        for generation in range(self.generations):
            # Create offspring
            offspring = []
            for _ in range(num_offspring):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                
                # Hybrid parameter optimization
                if use_hybrid and generation % hybrid_interval == 0 and random.random() < 0.1:
                    child = self.optimize_parameters(child)
                    
                # Circuit structure optimization
                child = self.optimize_circuit_structure(child)
                
                child.fidelity = child.calculate_fidelity(self.target_unitary)
                child.fitness = self.calculate_fitness(child)
                offspring.append(child)
            
            # Replace worst individuals with best offspring
            self.population.sort(key=lambda ind: ind.fitness)
            offspring.sort(key=lambda ind: ind.fitness, reverse=True)
            
            for i in range(num_replace):
                if i < len(offspring):
                    self.population[i] = offspring[i]
            
            # Track best individual
            current_best = max(self.population, key=lambda ind: ind.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best
                
            self.fitness_history.append(current_best.fitness)
            
            if generation % 100 == 0:
                print(f"Generation {generation}: Best fitness = {current_best.fitness:.4f}, "
                      f"Fidelity = {current_best.fidelity:.4f}, Depth = {current_best.depth}")
        
        return self.best_individual

# Example usage and helper functions
def create_random_target_circuit(num_qubits: int, depth: int) -> CircuitIndividual:
    """Create a random target circuit for testing"""
    optimizer = QuantumEvolutionaryOptimizer(num_qubits, 
                                           target_unitary=np.eye(2**num_qubits))  # Dummy target
    return optimizer._create_random_circuit(depth)

def calculate_circuit_metrics(circuit: CircuitIndividual, target_unitary: np.ndarray) -> Dict:
    """Calculate various metrics for a circuit"""
    fidelity = circuit.calculate_fidelity(target_unitary)
    
    # Count different gate types
    gate_counts = {gt: 0 for gt in GateType}
    for qubit_gates in circuit.gates:
        for gate in qubit_gates:
            gate_counts[gate.gate_type] += 1
            
    return {
        'fidelity': fidelity,
        'depth': circuit.depth,
        'num_qubits': circuit.num_qubits,
        'gate_counts': gate_counts,
        'num_cx_gates': gate_counts[GateType.CX]
    }

# Example usage
if __name__ == "__main__":
    # Create a random target circuit
    num_qubits = 3
    target_depth = 10
    target_circuit = create_random_target_circuit(num_qubits, target_depth)
    target_unitary = target_circuit.calculate_unitary()
    
    print("Target circuit created")
    print(f"Target depth: {target_depth}")
    print(f"Target unitary shape: {target_unitary.shape}")
    
    # Initialize optimizer
    optimizer = QuantumEvolutionaryOptimizer(
        num_qubits=num_qubits,
        target_unitary=target_unitary,
        population_size=50,  # Smaller for demo
        generations=100,     # Smaller for demo
        crossover_rate=0.85,
        mutation_rate=0.85,
        offspring_rate=0.3,
        replace_rate=0.3
    )
    
    # Run evolution
    print("Starting evolution...")
    best_circuit = optimizer.run_evolution(from_scratch=True, use_hybrid=True)
    
    # Results
    print("\nOptimization completed!")
    print(f"Best circuit fitness: {best_circuit.fitness:.4f}")
    print(f"Best circuit fidelity: {best_circuit.fidelity:.4f}")
    print(f"Best circuit depth: {best_circuit.depth}")
    
    # Compare with target
    target_metrics = calculate_circuit_metrics(target_circuit, target_unitary)
    best_metrics = calculate_circuit_metrics(best_circuit, target_unitary)
    
    print(f"\nTarget vs Optimized:")
    print(f"Fidelity: {target_metrics['fidelity']:.4f} -> {best_metrics['fidelity']:.4f}")
    print(f"Depth: {target_metrics['depth']} -> {best_metrics['depth']}")
    print(f"CX gates: {target_metrics['num_cx_gates']} -> {best_metrics['num_cx_gates']}")