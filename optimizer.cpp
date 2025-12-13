#include "random_generator.hpp"
#include "optimizer.hpp"

// Initialize thread-local random generator
thread_local std::mt19937 RandomGenerator::gen;
thread_local RandomGenerator random_generator;

CircuitIndividual crossover_single_point(const CircuitIndividual& p1, const CircuitIndividual& p2) {
    int child_depth = std::max(p1.depth, p2.depth);
    if (child_depth <= 1) return p1;
    
    int crossover_point = random_generator.random_int(1, child_depth - 1);
    
    std::vector<std::vector<Gate>> child_layers;
    for (int i = 0; i < child_depth; ++i) {
        if (i < crossover_point) {
            child_layers.push_back(i < static_cast<int>(p1.layers.size()) ? p1.layers[i] : std::vector<Gate>());
        } else {
            child_layers.push_back(i < static_cast<int>(p2.layers.size()) ? p2.layers[i] : std::vector<Gate>());
        }
    }
    
    return CircuitIndividual(p1.num_qubits, child_depth, std::move(child_layers));
}

CircuitIndividual crossover_uniform(const CircuitIndividual& p1, const CircuitIndividual& p2) {
    const int child_depth = std::max(p1.depth, p2.depth);
    
    std::vector<std::vector<Gate>> child_layers;
    for (int i = 0; i < child_depth; ++i) {
        if (random_generator.random_bool()) {
            child_layers.push_back(i < static_cast<int>(p1.layers.size()) ? p1.layers[i] : std::vector<Gate>());
        } else {
            child_layers.push_back(i < static_cast<int>(p2.layers.size()) ? p2.layers[i] : std::vector<Gate>());
        }
    }
    
    return CircuitIndividual(p1.num_qubits, child_depth, std::move(child_layers));
}

CircuitIndividual crossover_multi_point(const CircuitIndividual& p1, const CircuitIndividual& p2) {
    int child_depth = std::max(p1.depth, p2.depth);
    int num_points = random_generator.random_int(2, std::min(5, child_depth));
    
    std::vector<int> points = {0, child_depth};
    for (int i = 0; i < num_points - 2; ++i) {
        points.push_back(random_generator.random_int(1, child_depth - 1));
    }
    std::sort(points.begin(), points.end());
    
    std::vector<std::vector<Gate>> child_layers;
    bool use_p1 = true;
    
    for (size_t i = 0; i < points.size() - 1; ++i) {
        int start = points[i], end = points[i + 1];
        const auto& parent = use_p1 ? p1 : p2;
        
        for (int j = start; j < end; ++j) {
            if (j < static_cast<int>(parent.layers.size())) {
                child_layers.push_back(parent.layers[j]);
            } else {
                child_layers.emplace_back();
            }
        }
        use_p1 = !use_p1;
    }
    
    return CircuitIndividual(p1.num_qubits, child_depth, std::move(child_layers));
}

CircuitIndividual crossover_blockwise(const CircuitIndividual& p1, const CircuitIndividual& p2) {
    int child_depth = std::max(p1.depth, p2.depth);
    if (child_depth <= 0 || p1.num_qubits <= 1) return p1;
    
    int depth_split = random_generator.random_int(1, child_depth - 1);
    int qubit_split = random_generator.random_int(1, p1.num_qubits - 1);
    
    std::vector<std::vector<Gate>> child_layers;
    for (int i = 0; i < child_depth; ++i) {
        std::vector<Gate> layer_gates;
        const auto& parent = (i < depth_split) ? p1 : p2;
        
        if (i < static_cast<int>(parent.layers.size())) {
            for (const auto& gate : parent.layers[i]) {
                bool all_match = true;
                for (int q : gate.qubits) {
                    if ((i < depth_split && q >= qubit_split) ||
                        (i >= depth_split && q < qubit_split)) {
                        all_match = false;
                        break;
                    }
                }
                if (all_match) {
                    layer_gates.push_back(gate);
                }
            }
        }
        child_layers.push_back(std::move(layer_gates));
    }
    
    return CircuitIndividual(p1.num_qubits, child_depth, std::move(child_layers));
}

CircuitIndividual QuantumEvolutionaryOptimizer::mutate_single_gate(CircuitIndividual ind) const {
    if (ind.layers.empty()) return ind;
    
    int layer_idx = random_generator.random_int(0, static_cast<int>(ind.layers.size()) - 1);
    if (ind.layers[layer_idx].empty()) return ind;
    
    int gate_idx = random_generator.random_int(0, static_cast<int>(ind.layers[layer_idx].size()) - 1);
    GateType new_type = random_generator.random_choice(gate_set);
    const auto& old_gate = ind.layers[layer_idx][gate_idx];
    
    if (new_type == GateType::CX && !old_gate.qubits.empty()) {
        auto used = ind.get_used_qubits(layer_idx);
        auto available = ind.get_available_qubits(used);
        if (available.size() >= 1) {
            std::vector<int> new_qubits = {old_gate.qubits[0], random_generator.random_choice(available)};
            ind.layers[layer_idx][gate_idx] = Gate(GateType::CX, new_qubits);
        }
    } else if (new_type == GateType::RZ) {
        double angle = random_generator.random_double(0.0, 2.0 * M_PI);
        std::vector<int> qubits = old_gate.qubits.empty() ? std::vector<int>{0} : 
                                 std::vector<int>{old_gate.qubits[0]};
        ind.layers[layer_idx][gate_idx] = Gate(GateType::RZ, qubits, angle);
    } else {
        std::vector<int> qubits = old_gate.qubits.empty() ? std::vector<int>{0} : 
                                 std::vector<int>{old_gate.qubits[0]};
        ind.layers[layer_idx][gate_idx] = Gate(new_type, qubits);
    }
    
    return ind;
}

CircuitIndividual QuantumEvolutionaryOptimizer::mutate_gate_swap(CircuitIndividual ind) {
    if (ind.layers.size() < 2) return ind;
    
    std::vector<int> indices(ind.layers.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto layers_idx = random_generator.random_sample(indices, 2);
    int layer1 = layers_idx[0], layer2 = layers_idx[1];
    
    if (!ind.layers[layer1].empty() && !ind.layers[layer2].empty()) {
        int gate1 = random_generator.random_int(0, static_cast<int>(ind.layers[layer1].size()) - 1);
        int gate2 = random_generator.random_int(0, static_cast<int>(ind.layers[layer2].size()) - 1);
        std::swap(ind.layers[layer1][gate1], ind.layers[layer2][gate2]);
    }
    
    return ind;
}

CircuitIndividual QuantumEvolutionaryOptimizer::mutate_column_swap(CircuitIndividual ind) {
    if (ind.layers.size() >= 2) {
        std::vector<int> indices(ind.layers.size());
        std::iota(indices.begin(), indices.end(), 0);
        auto selected = random_generator.random_sample(indices, 2);
        std::swap(ind.layers[selected[0]], ind.layers[selected[1]]);
    }
    return ind;
}

CircuitIndividual QuantumEvolutionaryOptimizer::mutate_delete_column(CircuitIndividual ind) {
    if (ind.depth > 1) {
        int idx = random_generator.random_int(0, static_cast<int>(ind.layers.size()) - 1);
        ind.layers.erase(ind.layers.begin() + idx);
        ind.depth = static_cast<int>(ind.layers.size());
    }
    return ind;
}

CircuitIndividual QuantumEvolutionaryOptimizer::mutate_add_cx_gate(CircuitIndividual ind) {
    if (!ind.layers.empty()) {
        int layer_idx = random_generator.random_int(0, static_cast<int>(ind.layers.size()) - 1);
        auto used = ind.get_used_qubits(layer_idx);
        auto available = ind.get_available_qubits(used);
        
        if (available.size() >= 2) {
            auto qubit_pair = random_generator.random_sample(available, 2);
            ind.layers[layer_idx].emplace_back(GateType::CX, qubit_pair);
        }
    }
    return ind;
}

CircuitIndividual QuantumEvolutionaryOptimizer::mutate_add_single_gate(CircuitIndividual ind) const {
    if (!ind.layers.empty()) {
        int layer_idx = random_generator.random_int(0, static_cast<int>(ind.layers.size()) - 1);
        auto used = ind.get_used_qubits(layer_idx);
        auto available = ind.get_available_qubits(used);
        
        if (!available.empty()) {
            std::vector<GateType> single_qubit_gates;
            for (auto gt : gate_set) {
                if (gt != GateType::CX) single_qubit_gates.push_back(gt);
            }
            
            if (!single_qubit_gates.empty()) {
                GateType gate_type = random_generator.random_choice(single_qubit_gates);
                int qubit = random_generator.random_choice(available);
                
                if (gate_type == GateType::RZ) {
                    double angle = random_generator.random_double(0.0, 2.0 * M_PI);
                    ind.layers[layer_idx].emplace_back(GateType::RZ, std::vector<int>{qubit}, angle);
                } else {
                    ind.layers[layer_idx].emplace_back(gate_type, std::vector<int>{qubit});
                }
            }
        }
    }
    return ind;
}

CircuitIndividual QuantumEvolutionaryOptimizer::mutate_parameters(CircuitIndividual ind) {
    for (auto& layer : ind.layers) {
        for (auto& gate : layer) {
            if (gate.type == GateType::RZ) {
                gate.angle += random_generator.random_double(-0.1, 0.1);
                gate.angle = std::fmod(gate.angle, 2.0 * M_PI);
                if (gate.angle < 0) gate.angle += 2.0 * M_PI;
            }
        }
    }
    return ind;
}

CircuitIndividual QuantumEvolutionaryOptimizer::mutate_ctrl_target_swap(CircuitIndividual ind) {
    for (auto& layer : ind.layers) {
        for (auto& gate : layer) {
            if (gate.type == GateType::CX && gate.qubits.size() == 2) {
                std::swap(gate.qubits[0], gate.qubits[1]);
            }
        }
    }
    return ind;
}
CircuitIndividual QuantumEvolutionaryOptimizer::mutate_add_random_column(CircuitIndividual ind) const {
    auto new_circuit = create_random_circuit(1);
    if (!new_circuit.layers.empty()) {
        ind.layers.push_back(new_circuit.layers[0]);
        ind.depth = static_cast<int>(ind.layers.size());
    }
    return ind;
}

// Apply parameter optimization to population subset
void QuantumEvolutionaryOptimizer::apply_parameter_optimization() const {
    const int num_to_optimize = std::max(1, static_cast<int>(population_size * param_optimization_rate));
    auto candidates = random_generator.random_sample(population, num_to_optimize);
    
    // Parallel parameter optimization with OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < candidates.size(); ++i) {
        optimize_parameters(candidates[i]);
        // Update fitness after parameter optimization
        if (fitness_func) {
            candidates[i].fitness = fitness_func(candidates[i]);
        }
    }
}

// FIX: this function must have a bug
// Create random circuit
CircuitIndividual QuantumEvolutionaryOptimizer::create_random_circuit(int depth) const {
    std::vector<std::vector<Gate>> layers;
    
    for (int i = 0; i < depth; ++i) {
        std::vector<Gate> layer;
        std::vector<bool> used(num_qubits, false);
        
        // Try to add some gates to this layer
        int attempts = 0;
        while (attempts < num_qubits * 2) {
            auto available = CircuitIndividual::get_available_qubits(used);
            if (available.empty()) break;
            
            GateType gate_type = random_generator.random_choice(gate_set);
            
            if (gate_type == GateType::CX && available.size() >= 2) {
                // Add CX gate
                auto qubit_pair = random_generator.random_sample(available, 2);
                layer.emplace_back(GateType::CX, qubit_pair);
                used[qubit_pair[0]] = true;
                used[qubit_pair[1]] = true;
            } else if (gate_type != GateType::ID && !available.empty()) {
                // Add single-qubit gate
                int qubit = random_generator.random_choice(available);
                if (gate_type == GateType::RZ) {
                    double angle = random_generator.random_double(0.0, 2.0 * M_PI);
                    layer.emplace_back(GateType::RZ, std::vector<int>{qubit}, angle);
                } else {
                    layer.emplace_back(gate_type, std::vector<int>{qubit});
                }
                used[qubit] = true;
            }
            attempts++;
        }
        layers.push_back(std::move(layer));
    }
    
    return CircuitIndividual(num_qubits, depth, std::move(layers));
}

// Tournament selection
std::pair<CircuitIndividual, CircuitIndividual> QuantumEvolutionaryOptimizer::tournament_selection(int tournament_size) const {
    const auto tournament1 = random_generator.random_sample(population, tournament_size);
    const auto tournament2 = random_generator.random_sample(population, tournament_size);
    
    if (tournament1.empty() || tournament2.empty()) {
        if (population.size() >= 2) {
            return {population[0], population[1]};
        } else if (population.size() == 1) {
            return {population[0], population[0]};
        } else {
            throw std::runtime_error("Empty population");
        }
    }
    
    const auto best1 = *std::max_element(tournament1.begin(), tournament1.end(),
                                 [](const CircuitIndividual& a, const CircuitIndividual& b) {
                                     return a.fitness < b.fitness;
                                 });
    const auto best2 = *std::max_element(tournament2.begin(), tournament2.end(),
                                 [](const CircuitIndividual& a, const CircuitIndividual& b) {
                                     return a.fitness < b.fitness;
                                 });
    
    return {best1, best2};
}

// Roulette wheel selection
std::pair<CircuitIndividual, CircuitIndividual> QuantumEvolutionaryOptimizer::roulette_selection() const {
    // Calculate total fitness
    double total_fitness = 0.0;
    for (const auto& individual : population) {
        total_fitness += individual.fitness;
    }
    
    if (total_fitness <= 0.0) {
        // Fall back to tournament selection if all fitness values are non-positive
        return tournament_selection();
    }
    
    // Calculate selection probabilities
    std::vector<double> probabilities;
    probabilities.reserve(population.size());
    for (const auto& individual : population) {
        probabilities.push_back(individual.fitness / total_fitness);
    }
    
    // Select first parent
    double rand1 = random_generator.random_double(0.0, 1.0);
    double cumulative_prob = 0.0;
    CircuitIndividual parent1 = population[0]; // fallback
    
    for (size_t i = 0; i < population.size(); ++i) {
        cumulative_prob += probabilities[i];
        if (rand1 <= cumulative_prob) {
            parent1 = population[i];
            break;
        }
    }
    
    // Select second parent (ensure different from first)
    CircuitIndividual parent2 = parent1;
    int attempts = 0;
    while (parent2.fitness == parent1.fitness && attempts < 10) {
        double rand2 = random_generator.random_double(0.0, 1.0);
        cumulative_prob = 0.0;
        
        for (size_t i = 0; i < population.size(); ++i) {
            cumulative_prob += probabilities[i];
            if (rand2 <= cumulative_prob) {
                parent2 = population[i];
                break;
            }
        }
        attempts++;
    }
    
    return {parent1, parent2};
}

// Crossover operations
CircuitIndividual QuantumEvolutionaryOptimizer::crossover(const CircuitIndividual& parent1, const CircuitIndividual& parent2,
                           CrossoverType method) {
    if (random_generator.random_double() > crossover_rate || parent1.layers.empty() || parent2.layers.empty()) {
        return random_generator.random_bool() ? parent1 : parent2;
    }
    
    switch (method) {
        case CrossoverType::SINGLE_POINT:
            return crossover_single_point(parent1, parent2);
        case CrossoverType::UNIFORM:
            return crossover_uniform(parent1, parent2);
        case CrossoverType::MULTI_POINT:
            return crossover_multi_point(parent1, parent2);
        case CrossoverType::BLOCKWISE:
            return crossover_blockwise(parent1, parent2);
        default:
            return parent1;
    }
}

// Mutation operations
CircuitIndividual QuantumEvolutionaryOptimizer::mutate(CircuitIndividual individual) {
    if (random_generator.random_double() > mutation_rate) {
        return individual;
    }
    
    const static std::vector<MutationType> mutation_types = {
        MutationType::SINGLE_GATE, MutationType::GATE_SWAP, MutationType::COLUMN_SWAP,
        MutationType::CTRL_TARGET_SWAP, MutationType::ADD_RANDOM_COLUMN,
        MutationType::DELETE_COLUMN, MutationType::ADD_CX_GATE,
        MutationType::ADD_SINGLE_GATE, MutationType::MUTATE_PARAMETERS
    };
    
    const MutationType mutation_type = random_generator.random_choice(mutation_types);
    
    switch (mutation_type) {
        case MutationType::SINGLE_GATE:
            return mutate_single_gate(std::move(individual));
        case MutationType::GATE_SWAP:
            return mutate_gate_swap(std::move(individual));
        case MutationType::COLUMN_SWAP:
            return mutate_column_swap(std::move(individual));
        case MutationType::CTRL_TARGET_SWAP:
            return mutate_ctrl_target_swap(std::move(individual));
        case MutationType::ADD_RANDOM_COLUMN:
            return mutate_add_random_column(std::move(individual));
        case MutationType::DELETE_COLUMN:
            return mutate_delete_column(std::move(individual));
        case MutationType::ADD_CX_GATE:
            return mutate_add_cx_gate(std::move(individual));
        case MutationType::ADD_SINGLE_GATE:
            return mutate_add_single_gate(std::move(individual));
        case MutationType::MUTATE_PARAMETERS:
            return mutate_parameters(std::move(individual));
        default:
            return individual;
    }
}
