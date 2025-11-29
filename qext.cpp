#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <numeric>

namespace py = pybind11;

// Gate types
enum class GateType {
    ID, X, SX, RZ, CX, H
};

// Gate structure
struct Gate {
    GateType type;
    std::vector<int> qubits;
    double angle;
    
    Gate(GateType t, std::vector<int> q, double a = 0.0) 
        : type(t), qubits(std::move(q)), angle(a) {
        // Remove duplicates and sort
        std::sort(qubits.begin(), qubits.end());
        qubits.erase(std::unique(qubits.begin(), qubits.end()), qubits.end());
    }
    
    bool is_single_qubit() const { return qubits.size() == 1; }
    bool is_two_qubit() const { return qubits.size() == 2; }
};

// Circuit individual
class CircuitIndividual {
public:
    int num_qubits;
    int depth;
    std::vector<std::vector<Gate>> layers;
    double fitness;
    double fidelity;
    double normalized_depth;
    
    CircuitIndividual(int nq, int d, std::vector<std::vector<Gate>> l = {})
        : num_qubits(nq), depth(d), layers(std::move(l)), fitness(-1e9), fidelity(0.0), normalized_depth(0.0) {
        if (layers.empty()) {
            layers.resize(depth);
        }
    }
    
    // Get used qubits in a layer
    std::vector<bool> get_used_qubits(int layer_idx) const {
        std::vector<bool> used(num_qubits, false);
        for (const auto& gate : layers[layer_idx]) {
            for (int q : gate.qubits) {
                if (q < num_qubits) used[q] = true;
            }
        }
        return used;
    }
    
    // Get available qubits
    std::vector<int> get_available_qubits(const std::vector<bool>& used) const {
        std::vector<int> available;
        for (int q = 0; q < num_qubits; ++q) {
            if (!used[q]) available.push_back(q);
        }
        return available;
    }
    
    // Get parameters for optimization
    std::vector<double> get_parameters() const {
        std::vector<double> params;
        for (const auto& layer : layers) {
            for (const auto& gate : layer) {
                if (gate.type == GateType::RZ) {
                    params.push_back(gate.angle);
                }
            }
        }
        return params;
    }
    
    // Set parameters
    void set_parameters(const std::vector<double>& params) {
        size_t param_idx = 0;
        for (auto& layer : layers) {
            for (auto& gate : layer) {
                if (gate.type == GateType::RZ) {
                    if (param_idx < params.size()) {
                        gate.angle = params[param_idx++];
                    }
                }
            }
        }
    }
    
    // Count gates by type
    std::unordered_map<GateType, int> gate_counts() const {
        std::unordered_map<GateType, int> counts;
        for (const auto& layer : layers) {
            for (const auto& gate : layer) {
                counts[gate.type]++;
            }
        }
        return counts;
    }
};

// Random number generator
class RandomGenerator {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> real_dist;
    
public:
    RandomGenerator() : gen(std::random_device{}()), real_dist(0.0, 1.0) {}
    
    double random_double(double min = 0.0, double max = 1.0) {
        return min + (max - min) * real_dist(gen);
    }
    
    int random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }
    
    bool random_bool() {
        return random_double() < 0.5;
    }
    
    template<typename T>
    T random_choice(const std::vector<T>& items) {
        if (items.empty()) throw std::runtime_error("Cannot choose from empty list");
        return items[random_int(0, items.size() - 1)];
    }
    
    template<typename T>
    std::vector<T> random_sample(const std::vector<T>& population, int k) {
        if (k >= population.size()) return population;
        
        std::vector<T> result;
        std::vector<int> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        for (int i = 0; i < k; ++i) {
            int idx = random_int(i, indices.size() - 1);
            result.push_back(population[indices[idx]]);
            std::swap(indices[i], indices[idx]);
        }
        
        return result;
    }
};

// Crossover types
enum class CrossoverType {
    SINGLE_POINT, UNIFORM, MULTI_POINT, BLOCKWISE
};

// Mutation types
enum class MutationType {
    SINGLE_GATE, GATE_SWAP, COLUMN_SWAP, CTRL_TARGET_SWAP,
    ADD_RANDOM_COLUMN, DELETE_COLUMN, ADD_CX_GATE, ADD_SINGLE_GATE, MUTATE_PARAMETERS
};

// Main optimizer class
class QuantumEvolutionaryOptimizer {
private:
    int num_qubits;
    int population_size;
    int generations;
    double crossover_rate;
    double mutation_rate;
    double offspring_rate;
    double replace_rate;
    double alpha;
    double beta;
    int target_depth;
    
    std::vector<GateType> gate_set;
    std::vector<CircuitIndividual> population;
    std::shared_ptr<CircuitIndividual> best_individual;
    std::vector<double> fitness_history;
    
    RandomGenerator rng;
    
    // Fitness function callback
    std::function<double(const CircuitIndividual&)> fitness_func;
    
public:
    QuantumEvolutionaryOptimizer(int nq, int pop_size, int gen, double cross_rate,
                               double mut_rate, double off_rate, double rep_rate,
                               double a, double b, int t_depth,
                               std::vector<GateType> g_set)
        : num_qubits(nq), population_size(pop_size), generations(gen),
          crossover_rate(cross_rate), mutation_rate(mut_rate),
          offspring_rate(off_rate), replace_rate(rep_rate),
          alpha(a), beta(b), target_depth(t_depth), gate_set(std::move(g_set)) {
    }
    
    void set_fitness_function(std::function<double(const CircuitIndividual&)> func) {
        fitness_func = std::move(func);
    }
    
    // Create random circuit - PUBLIC METHOD
    CircuitIndividual create_random_circuit(int depth) {
        std::vector<std::vector<Gate>> layers;
        
        for (int i = 0; i < depth; ++i) {
            std::vector<Gate> layer;
            std::vector<bool> used(num_qubits, false);
            
            while (true) {
                auto available = get_available_qubits(used);
                if (available.empty()) break;
                
                GateType gate_type = rng.random_choice(gate_set);
                
                if (gate_type == GateType::CX && available.size() >= 2) {
                    // Add CX gate
                    auto qubit_pair = rng.random_sample(available, 2);
                    layer.emplace_back(GateType::CX, qubit_pair);
                    used[qubit_pair[0]] = true;
                    used[qubit_pair[1]] = true;
                } else if (gate_type != GateType::ID && !available.empty()) {
                    // Add single-qubit gate
                    int qubit = rng.random_choice(available);
                    if (gate_type == GateType::RZ) {
                        double angle = rng.random_double(0.0, 2.0 * M_PI);
                        layer.emplace_back(GateType::RZ, std::vector<int>{qubit}, angle);
                    } else {
                        layer.emplace_back(gate_type, std::vector<int>{qubit});
                    }
                    used[qubit] = true;
                } else {
                    break;
                }
            }
            layers.push_back(std::move(layer));
        }
        
        return CircuitIndividual(num_qubits, depth, std::move(layers));
    }
    
    void initialize_population(int initial_depth, bool from_target = false) {
        population.clear();
        for (int i = 0; i < population_size; ++i) {
            population.push_back(create_random_circuit(initial_depth));
        }
    }
    
    // Tournament selection
    std::pair<CircuitIndividual, CircuitIndividual> tournament_selection(int tournament_size = 3) {
        auto tournament1 = rng.random_sample(population, tournament_size);
        auto tournament2 = rng.random_sample(population, tournament_size);
        
        auto best1 = *std::max_element(tournament1.begin(), tournament1.end(),
                                     [](const CircuitIndividual& a, const CircuitIndividual& b) {
                                         return a.fitness < b.fitness;
                                     });
        auto best2 = *std::max_element(tournament2.begin(), tournament2.end(),
                                     [](const CircuitIndividual& a, const CircuitIndividual& b) {
                                         return a.fitness < b.fitness;
                                     });
        
        return {best1, best2};
    }
    
    // Roulette wheel selection
    std::pair<CircuitIndividual, CircuitIndividual> roulette_selection() {
        double total_fitness = 0.0;
        double min_fitness = 1e9;
        
        for (const auto& ind : population) {
            total_fitness += ind.fitness;
            if (ind.fitness < min_fitness) min_fitness = ind.fitness;
        }
        
        // Shift to positive
        double shift = -min_fitness + 1e-6;
        total_fitness = 0.0;
        for (const auto& ind : population) {
            total_fitness += ind.fitness + shift;
        }
        
        // Select first parent
        double pick1 = rng.random_double(0.0, total_fitness);
        double current = 0.0;
        CircuitIndividual* parent1 = nullptr;
        
        for (auto& ind : population) {
            current += ind.fitness + shift;
            if (current >= pick1) {
                parent1 = &ind;
                break;
            }
        }
        
        if (!parent1) parent1 = &population.back();
        
        // Select second parent
        double remaining_fitness = total_fitness - (parent1->fitness + shift);
        double pick2 = rng.random_double(0.0, remaining_fitness);
        current = 0.0;
        CircuitIndividual* parent2 = nullptr;
        
        for (auto& ind : population) {
            if (&ind == parent1) continue;
            current += ind.fitness + shift;
            if (current >= pick2) {
                parent2 = &ind;
                break;
            }
        }
        
        if (!parent2) {
            for (auto& ind : population) {
                if (&ind != parent1) {
                    parent2 = &ind;
                    break;
                }
            }
        }
        
        return {*parent1, *parent2};
    }
    
    // Crossover operations
    CircuitIndividual crossover(const CircuitIndividual& parent1, const CircuitIndividual& parent2,
                               CrossoverType method = CrossoverType::UNIFORM) {
        if (rng.random_double() > crossover_rate) {
            return rng.random_bool() ? parent1 : parent2;
        }
        
        switch (method) {
            case CrossoverType::SINGLE_POINT:
                return single_point_crossover(parent1, parent2);
            case CrossoverType::UNIFORM:
                return uniform_crossover(parent1, parent2);
            case CrossoverType::MULTI_POINT:
                return multi_point_crossover(parent1, parent2);
            case CrossoverType::BLOCKWISE:
                return blockwise_crossover(parent1, parent2);
        }
        return parent1;
    }
    
private:
    std::vector<int> get_available_qubits(const std::vector<bool>& used) {
        std::vector<int> available;
        for (int i = 0; i < used.size(); ++i) {
            if (!used[i]) available.push_back(i);
        }
        return available;
    }
    
    CircuitIndividual single_point_crossover(const CircuitIndividual& p1, const CircuitIndividual& p2) {
        int child_depth = rng.random_bool() ? p1.depth : p2.depth;
        int crossover_point = rng.random_int(1, child_depth - 1);
        
        std::vector<std::vector<Gate>> child_layers;
        for (int i = 0; i < child_depth; ++i) {
            if (i < crossover_point) {
                child_layers.push_back(i < p1.layers.size() ? p1.layers[i] : p2.layers[i]);
            } else {
                child_layers.push_back(i < p2.layers.size() ? p2.layers[i] : p1.layers[i]);
            }
        }
        
        return CircuitIndividual(num_qubits, child_depth, std::move(child_layers));
    }
    
    CircuitIndividual uniform_crossover(const CircuitIndividual& p1, const CircuitIndividual& p2) {
        int child_depth = rng.random_bool() ? p1.depth : p2.depth;
        std::vector<std::vector<Gate>> child_layers;
        
        for (int i = 0; i < child_depth; ++i) {
            if (rng.random_bool() && i < p1.layers.size()) {
                child_layers.push_back(p1.layers[i]);
            } else if (i < p2.layers.size()) {
                child_layers.push_back(p2.layers[i]);
            } else {
                child_layers.emplace_back();
            }
        }
        
        return CircuitIndividual(num_qubits, child_depth, std::move(child_layers));
    }
    
    CircuitIndividual multi_point_crossover(const CircuitIndividual& p1, const CircuitIndividual& p2) {
        int child_depth = rng.random_bool() ? p1.depth : p2.depth;
        int num_points = rng.random_int(2, std::min(5, child_depth));
        
        std::vector<int> points = {0, child_depth};
        for (int i = 0; i < num_points - 2; ++i) {
            points.push_back(rng.random_int(1, child_depth - 1));
        }
        std::sort(points.begin(), points.end());
        
        std::vector<std::vector<Gate>> child_layers;
        bool use_p1 = true;
        
        for (size_t i = 0; i < points.size() - 1; ++i) {
            int start = points[i], end = points[i + 1];
            const auto& parent = use_p1 ? p1 : p2;
            
            for (int j = start; j < end; ++j) {
                if (j < parent.layers.size()) {
                    child_layers.push_back(parent.layers[j]);
                } else {
                    child_layers.emplace_back();
                }
            }
            use_p1 = !use_p1;
        }
        
        return CircuitIndividual(num_qubits, child_depth, std::move(child_layers));
    }
    
    CircuitIndividual blockwise_crossover(const CircuitIndividual& p1, const CircuitIndividual& p2) {
        int child_depth = rng.random_bool() ? p1.depth : p2.depth;
        int depth_split = rng.random_int(1, child_depth - 1);
        int qubit_split = rng.random_int(1, num_qubits - 1);
        
        std::vector<std::vector<Gate>> child_layers;
        for (int i = 0; i < child_depth; ++i) {
            std::vector<Gate> layer_gates;
            const auto& parent = (i < depth_split) ? p1 : p2;
            
            if (i < parent.layers.size()) {
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
        
        return CircuitIndividual(num_qubits, child_depth, std::move(child_layers));
    }
    
public:
    // Mutation operations
    CircuitIndividual mutate(CircuitIndividual individual) {
        if (rng.random_double() > mutation_rate) {
            return individual;
        }
        
        static std::vector<MutationType> mutation_types = {
            MutationType::SINGLE_GATE, MutationType::GATE_SWAP, MutationType::COLUMN_SWAP,
            MutationType::CTRL_TARGET_SWAP, MutationType::ADD_RANDOM_COLUMN,
            MutationType::DELETE_COLUMN, MutationType::ADD_CX_GATE,
            MutationType::ADD_SINGLE_GATE, MutationType::MUTATE_PARAMETERS
        };
        
        MutationType mutation_type = rng.random_choice(mutation_types);
        
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
        }
        return individual;
    }
    
private:
    CircuitIndividual mutate_single_gate(CircuitIndividual ind) {
        if (ind.layers.empty()) return ind;
        
        int layer_idx = rng.random_int(0, ind.layers.size() - 1);
        if (ind.layers[layer_idx].empty()) return ind;
        
        int gate_idx = rng.random_int(0, ind.layers[layer_idx].size() - 1);
        GateType new_type = rng.random_choice(gate_set);
        const auto& old_gate = ind.layers[layer_idx][gate_idx];
        
        if (new_type == GateType::CX && old_gate.qubits.size() >= 1) {
            auto used = ind.get_used_qubits(layer_idx);
            auto available = ind.get_available_qubits(used);
            if (available.size() >= 1) {
                std::vector<int> new_qubits = {old_gate.qubits[0], rng.random_choice(available)};
                ind.layers[layer_idx][gate_idx] = Gate(GateType::CX, new_qubits);
            }
        } else if (new_type == GateType::RZ) {
            double angle = rng.random_double(0.0, 2.0 * M_PI);
            ind.layers[layer_idx][gate_idx] = Gate(GateType::RZ, old_gate.qubits, angle);
        } else {
            ind.layers[layer_idx][gate_idx] = Gate(new_type, old_gate.qubits);
        }
        
        return ind;
    }
    
    CircuitIndividual mutate_gate_swap(CircuitIndividual ind) {
        if (ind.layers.size() < 2) return ind;
        
        auto layers = rng.random_sample(std::vector<int>(ind.layers.size()), 2);
        int layer1 = layers[0], layer2 = layers[1];
        
        if (!ind.layers[layer1].empty() && !ind.layers[layer2].empty()) {
            int gate1 = rng.random_int(0, ind.layers[layer1].size() - 1);
            int gate2 = rng.random_int(0, ind.layers[layer2].size() - 1);
            std::swap(ind.layers[layer1][gate1], ind.layers[layer2][gate2]);
        }
        
        return ind;
    }
    
    CircuitIndividual mutate_column_swap(CircuitIndividual ind) {
        if (ind.layers.size() >= 2) {
            auto indices = rng.random_sample(std::vector<int>(ind.layers.size()), 2);
            std::swap(ind.layers[indices[0]], ind.layers[indices[1]]);
        }
        return ind;
    }
    
    CircuitIndividual mutate_ctrl_target_swap(CircuitIndividual ind) {
        for (auto& layer : ind.layers) {
            for (auto& gate : layer) {
                if (gate.type == GateType::CX && gate.qubits.size() == 2) {
                    std::swap(gate.qubits[0], gate.qubits[1]);
                }
            }
        }
        return ind;
    }
    
    CircuitIndividual mutate_add_random_column(CircuitIndividual ind) {
        auto new_layer = create_random_circuit(1).layers[0];
        ind.layers.push_back(std::move(new_layer));
        ind.depth++;
        return ind;
    }
    
    CircuitIndividual mutate_delete_column(CircuitIndividual ind) {
        if (ind.depth > 1) {
            int idx = rng.random_int(0, ind.layers.size() - 1);
            ind.layers.erase(ind.layers.begin() + idx);
            ind.depth--;
        }
        return ind;
    }
    
    CircuitIndividual mutate_add_cx_gate(CircuitIndividual ind) {
        if (!ind.layers.empty()) {
            int layer_idx = rng.random_int(0, ind.layers.size() - 1);
            auto used = ind.get_used_qubits(layer_idx);
            auto available = ind.get_available_qubits(used);
            
            if (available.size() >= 2) {
                auto qubit_pair = rng.random_sample(available, 2);
                ind.layers[layer_idx].emplace_back(GateType::CX, qubit_pair);
            }
        }
        return ind;
    }
    
    CircuitIndividual mutate_add_single_gate(CircuitIndividual ind) {
        if (!ind.layers.empty()) {
            int layer_idx = rng.random_int(0, ind.layers.size() - 1);
            auto used = ind.get_used_qubits(layer_idx);
            auto available = ind.get_available_qubits(used);
            
            if (!available.empty()) {
                std::vector<GateType> single_qubit_gates;
                for (auto gt : gate_set) {
                    if (gt != GateType::CX) single_qubit_gates.push_back(gt);
                }
                
                if (!single_qubit_gates.empty()) {
                    GateType gate_type = rng.random_choice(single_qubit_gates);
                    int qubit = rng.random_choice(available);
                    
                    if (gate_type == GateType::RZ) {
                        double angle = rng.random_double(0.0, 2.0 * M_PI);
                        ind.layers[layer_idx].emplace_back(GateType::RZ, std::vector<int>{qubit}, angle);
                    } else {
                        ind.layers[layer_idx].emplace_back(gate_type, std::vector<int>{qubit});
                    }
                }
            }
        }
        return ind;
    }
    
    CircuitIndividual mutate_parameters(CircuitIndividual ind) {
        for (auto& layer : ind.layers) {
            for (auto& gate : layer) {
                if (gate.type == GateType::RZ) {
                    gate.angle += rng.random_double(-0.1, 0.1);
                    gate.angle = std::fmod(gate.angle, 2.0 * M_PI);
                    if (gate.angle < 0) gate.angle += 2.0 * M_PI;
                }
            }
        }
        return ind;
    }
    
public:
    // Circuit optimization
    CircuitIndividual optimize_circuit_structure(CircuitIndividual ind) {
        // Remove empty layers
        std::vector<std::vector<Gate>> non_empty_layers;
        for (const auto& layer : ind.layers) {
            if (!layer.empty()) {
                non_empty_layers.push_back(layer);
            }
        }
        
        if (non_empty_layers.empty()) {
            non_empty_layers.emplace_back();
        }
        
        return CircuitIndividual(ind.num_qubits, non_empty_layers.size(), std::move(non_empty_layers));
    }
    
    // Main evolution loop
    CircuitIndividual run_evolution(bool from_scratch = true, const std::string& selection_method = "tournament") {
        initialize_population(target_depth, !from_scratch);
        
        // Evaluate initial population
        for (auto& ind : population) {
            if (fitness_func) {
                ind.fitness = fitness_func(ind);
            }
        }
        
        int num_offspring = static_cast<int>(population_size * offspring_rate);
        int num_replace = static_cast<int>(population_size * replace_rate);
        
        for (int generation = 0; generation < generations; ++generation) {
            // Create offspring
            std::vector<CircuitIndividual> offspring;
            for (int i = 0; i < num_offspring; ++i) {
                auto [parent1, parent2] = (selection_method == "roulette") ? 
                                          roulette_selection() : tournament_selection();
                
                auto child = crossover(parent1, parent2);
                child = mutate(std::move(child));
                child = optimize_circuit_structure(std::move(child));
                
                if (fitness_func) {
                    child.fitness = fitness_func(child);
                }
                offspring.push_back(std::move(child));
            }
            
            // Replace worst individuals
            std::sort(population.begin(), population.end(),
                     [](const CircuitIndividual& a, const CircuitIndividual& b) {
                         return a.fitness < b.fitness;
                     });
            std::sort(offspring.begin(), offspring.end(),
                     [](const CircuitIndividual& a, const CircuitIndividual& b) {
                         return a.fitness > b.fitness;
                     });
            
            for (int i = 0; i < num_replace && i < offspring.size(); ++i) {
                population[i] = std::move(offspring[i]);
            }
            
            // Track best individual
            auto best_it = std::max_element(population.begin(), population.end(),
                                          [](const CircuitIndividual& a, const CircuitIndividual& b) {
                                              return a.fitness < b.fitness;
                                          });
            
            if (!best_individual || best_it->fitness > best_individual->fitness) {
                best_individual = std::make_shared<CircuitIndividual>(*best_it);
            }
            
            fitness_history.push_back(best_it->fitness);
            
            if (generation % 100 == 0) {
                std::cout << "Generation " << generation << ": Best fitness = " << best_it->fitness
                         << ", Depth = " << best_it->depth << std::endl;
            }
        }
        
        return *best_individual;
    }
    
    // Getters
    const std::vector<CircuitIndividual>& get_population() const { return population; }
    const std::vector<double>& get_fitness_history() const { return fitness_history; }
    const CircuitIndividual& get_best_individual() const { return *best_individual; }
};

// Pybind11 module
PYBIND11_MODULE(qext, m) {
    m.doc() = "High-performance quantum circuit optimization using evolutionary algorithms";
    
    // GateType enum
    py::enum_<GateType>(m, "GateType")
        .value("ID", GateType::ID)
        .value("X", GateType::X)
        .value("SX", GateType::SX)
        .value("RZ", GateType::RZ)
        .value("CX", GateType::CX)
        .value("H", GateType::H)
        .export_values();
    
    // Gate class
    py::class_<Gate>(m, "Gate")
        .def(py::init<GateType, std::vector<int>, double>(),
             py::arg("type"), py::arg("qubits"), py::arg("angle") = 0.0)
        .def_readwrite("type", &Gate::type)
        .def_readwrite("qubits", &Gate::qubits)
        .def_readwrite("angle", &Gate::angle)
        .def("is_single_qubit", &Gate::is_single_qubit)
        .def("is_two_qubit", &Gate::is_two_qubit);
    
    // CircuitIndividual class
    py::class_<CircuitIndividual>(m, "CircuitIndividual")
        .def(py::init<int, int, std::vector<std::vector<Gate>>>(),
             py::arg("num_qubits"), py::arg("depth"), py::arg("layers") = std::vector<std::vector<Gate>>())
        .def_readwrite("num_qubits", &CircuitIndividual::num_qubits)
        .def_readwrite("depth", &CircuitIndividual::depth)
        .def_readwrite("layers", &CircuitIndividual::layers)
        .def_readwrite("fitness", &CircuitIndividual::fitness)
        .def_readwrite("fidelity", &CircuitIndividual::fidelity)
        .def_readwrite("normalized_depth", &CircuitIndividual::normalized_depth)
        .def("get_parameters", &CircuitIndividual::get_parameters)
        .def("set_parameters", &CircuitIndividual::set_parameters)
        .def("gate_counts", &CircuitIndividual::gate_counts);
    
    // QuantumEvolutionaryOptimizer class
    py::class_<QuantumEvolutionaryOptimizer>(m, "QuantumEvolutionaryOptimizer")
        .def(py::init<int, int, int, double, double, double, double, double, double, int, std::vector<GateType>>(),
             py::arg("num_qubits"), py::arg("population_size"), py::arg("generations"),
             py::arg("crossover_rate"), py::arg("mutation_rate"), py::arg("offspring_rate"),
             py::arg("replace_rate"), py::arg("alpha"), py::arg("beta"), py::arg("target_depth"),
             py::arg("gate_set"))
        .def("set_fitness_function", &QuantumEvolutionaryOptimizer::set_fitness_function)
        .def("create_random_circuit", &QuantumEvolutionaryOptimizer::create_random_circuit) // ADDED THIS LINE
        .def("initialize_population", &QuantumEvolutionaryOptimizer::initialize_population,
             py::arg("initial_depth"), py::arg("from_target") = false)
        .def("run_evolution", &QuantumEvolutionaryOptimizer::run_evolution,
             py::arg("from_scratch") = true, py::arg("selection_method") = "tournament")
        .def("get_population", &QuantumEvolutionaryOptimizer::get_population)
        .def("get_fitness_history", &QuantumEvolutionaryOptimizer::get_fitness_history)
        .def("get_best_individual", &QuantumEvolutionaryOptimizer::get_best_individual);
}
