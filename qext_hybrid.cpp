// qext_hybrid_fixed.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <memory>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <numeric>
#include <string>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#define ModuleName qext_omp
#else
#define omp_get_thread_num() 0
#define ModuleName qext
#endif // _OPENMP

namespace py = pybind11;
using Complex = std::complex<double>;
using MatrixXcd = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXcd = Eigen::Vector<Complex, Eigen::Dynamic>;

// COBYLA implementation for parameter optimization
namespace cobyla {
    
class COBYLA {
private:
    std::function<double(const std::vector<double>&)> objective_;
    int n_;
    int m_;
    double rhobeg_;
    double rhoend_;
    int maxfun_;
    
public:
    COBYLA(int n, int m, double rhobeg = 0.5, double rhoend = 1e-6, int maxfun = 1000)
        : n_(n), m_(m), rhobeg_(rhobeg), rhoend_(rhoend), maxfun_(maxfun) {}
    
    void set_objective(std::function<double(const std::vector<double>&)> objective) {
        objective_ = objective;
    }
    
    struct Result {
        std::vector<double> x;
        double fun;
        bool success;
        int nfev;
    };
    
    Result minimize(const std::vector<double>& x0) {
        Result result;
        result.x = x0;
        result.nfev = 0;
        result.success = false;
        
        if (!objective_) {
            std::cerr << "Objective function not set!" << std::endl;
            return result;
        }
        
        // Simplified COBYLA implementation
        std::vector<double> x = x0;
        double current_rho = rhobeg_;
        double best_fval = objective_(x);
        result.nfev++;
        
        int iterations = 0;
        while (current_rho > rhoend_ && iterations < maxfun_ / 10) {
            bool improved = false;
            
            // Generate trial points
            for (int i = 0; i < 2 * n_; ++i) {
                std::vector<double> trial = x;
                
                // Perturb parameters
                for (size_t j = 0; j < trial.size(); ++j) {
                    double perturbation = ((j % 2 == 0) ? 1 : -1) * current_rho;
                    trial[j] += perturbation;
                    
                    // Keep angles in [0, 2π]
                    if (trial[j] < 0) trial[j] += 2 * M_PI;
                    if (trial[j] > 2 * M_PI) trial[j] -= 2 * M_PI;
                }
                
                double trial_fval = objective_(trial);
                result.nfev++;
                
                if (trial_fval < best_fval) {
                    best_fval = trial_fval;
                    x = trial;
                    improved = true;
                }
                
                if (result.nfev >= maxfun_) break;
            }
            
            if (!improved) {
                current_rho *= 0.5;
            }
            
            iterations++;
            if (result.nfev >= maxfun_) break;
        }
        
        result.x = x;
        result.fun = best_fval;
        result.success = true;
        
        return result;
    }
};

} // namespace cobyla

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
        } else {
            // Ensure depth matches layers size
            depth = static_cast<int>(layers.size());
        }
    }
    
    // Default constructor for pair compatibility
    CircuitIndividual() : num_qubits(0), depth(0), fitness(-1e9), fidelity(0.0), normalized_depth(0.0) {}
    
    // Get used qubits in a layer
    std::vector<bool> get_used_qubits(int layer_idx) const {
        std::vector<bool> used(num_qubits, false);
        if (layer_idx >= 0 && layer_idx < static_cast<int>(layers.size())) {
            for (const auto& gate : layers[layer_idx]) {
                for (int q : gate.qubits) {
                    if (q < num_qubits) used[q] = true;
                }
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
    
    // Get number of non-identity gates
    int count_non_id_gates() const {
        int count = 0;
        for (const auto& layer : layers) {
            for (const auto& gate : layer) {
                if (gate.type != GateType::ID) {
                    count++;
                }
            }
        }
        return count;
    }
    
    // Optimize circuit structure - remove empty layers and redundant gates
    CircuitIndividual optimize_structure() const {
        std::vector<std::vector<Gate>> new_layers;
        
        for (const auto& layer : layers) {
            std::vector<Gate> new_layer;
            for (const auto& gate : layer) {
                if (gate.type != GateType::ID) {
                    new_layer.push_back(gate);
                }
            }
            if (!new_layer.empty()) {
                new_layers.push_back(std::move(new_layer));
            }
        }
        
        if (new_layers.empty()) {
            new_layers.emplace_back();
        }
        
        return CircuitIndividual(num_qubits, static_cast<int>(new_layers.size()), std::move(new_layers));
    }
    
    // Convert circuit to unitary matrix using Eigen
    MatrixXcd circuit_to_unitary() const {
        int dim = 1 << num_qubits; // 2^n
        MatrixXcd unitary = MatrixXcd::Identity(dim, dim);
        
        for (const auto& layer : layers) {
            MatrixXcd layer_matrix = MatrixXcd::Identity(dim, dim);
            
            for (const auto& gate : layer) {
                if (gate.type == GateType::ID) continue;
                
                MatrixXcd gate_matrix = get_gate_matrix(gate, num_qubits);
                layer_matrix = gate_matrix * layer_matrix;
            }
            
            unitary = layer_matrix * unitary;
        }
        
        return unitary;
    }
    
private:
    MatrixXcd get_gate_matrix(const Gate& gate, int total_qubits) const {
        int dim = 1 << total_qubits;
        
        switch (gate.type) {
            case GateType::X:
                return get_pauli_x_matrix(gate.qubits[0], total_qubits);
            case GateType::SX:
                return get_sx_matrix(gate.qubits[0], total_qubits);
            case GateType::RZ:
                return get_rz_matrix(gate.angle, gate.qubits[0], total_qubits);
            case GateType::H:
                return get_hadamard_matrix(gate.qubits[0], total_qubits);
            case GateType::CX:
                if (gate.qubits.size() >= 2) {
                    return get_cx_matrix(gate.qubits[0], gate.qubits[1], total_qubits);
                }
                // Fallthrough
            default:
                return MatrixXcd::Identity(dim, dim);
        }
    }
    
    MatrixXcd get_pauli_x_matrix(int target_qubit, int total_qubits) const {
        int dim = 1 << total_qubits;
        MatrixXcd result = MatrixXcd::Zero(dim, dim);
        
        for (int i = 0; i < dim; ++i) {
            int flipped = i ^ (1 << target_qubit);
            result(flipped, i) = 1.0;
        }
        
        return result;
    }
    
    MatrixXcd get_sx_matrix(int target_qubit, int total_qubits) const {
        int dim = 1 << total_qubits;
        MatrixXcd result = MatrixXcd::Zero(dim, dim);
        Complex half_plus_half_i = Complex(0.5, 0.5);
        Complex half_minus_half_i = Complex(0.5, -0.5);
        
        for (int i = 0; i < dim; ++i) {
            int basis_with_zero = i & ~(1 << target_qubit);
            int basis_with_one = i | (1 << target_qubit);
            
            if (i == basis_with_zero) {
                // |0⟩ component
                result(basis_with_zero, i) = half_plus_half_i;
                result(basis_with_one, i) = half_minus_half_i;
            } else {
                // |1⟩ component
                result(basis_with_zero, i) = half_minus_half_i;
                result(basis_with_one, i) = half_plus_half_i;
            }
        }
        
        return result;
    }
    
    MatrixXcd get_rz_matrix(double angle, int target_qubit, int total_qubits) const {
        int dim = 1 << total_qubits;
        MatrixXcd result = MatrixXcd::Zero(dim, dim);
        Complex phase0 = std::exp(Complex(0, -angle/2));
        Complex phase1 = std::exp(Complex(0, angle/2));
        
        for (int i = 0; i < dim; ++i) {
            if (i & (1 << target_qubit)) {
                result(i, i) = phase1;
            } else {
                result(i, i) = phase0;
            }
        }
        
        return result;
    }
    
    MatrixXcd get_hadamard_matrix(int target_qubit, int total_qubits) const {
        int dim = 1 << total_qubits;
        MatrixXcd result = MatrixXcd::Zero(dim, dim);
        double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        
        for (int i = 0; i < dim; ++i) {
            int basis_with_zero = i & ~(1 << target_qubit);
            int basis_with_one = i | (1 << target_qubit);
            
            if (i == basis_with_zero) {
                result(basis_with_zero, i) = inv_sqrt2;
                result(basis_with_one, i) = inv_sqrt2;
            } else {
                result(basis_with_zero, i) = inv_sqrt2;
                result(basis_with_one, i) = -inv_sqrt2;
            }
        }
        
        return result;
    }
    
    MatrixXcd get_cx_matrix(int control_qubit, int target_qubit, int total_qubits) const {
        int dim = 1 << total_qubits;
        MatrixXcd result = MatrixXcd::Identity(dim, dim);
        
        for (int i = 0; i < dim; ++i) {
            if (i & (1 << control_qubit)) {
                int flipped = i ^ (1 << target_qubit);
                result(i, i) = 0.0;
                result(flipped, i) = 1.0;
            }
        }
        
        return result;
    }
};

// Native fidelity calculation
double calculate_fidelity(const MatrixXcd& U1, const MatrixXcd& U2) {
    if (U1.rows() != U2.rows() || U1.cols() != U2.cols()) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
    
    MatrixXcd product = U1.adjoint() * U2;
    Complex trace = product.trace();
    double fidelity = std::norm(trace) / (U1.rows() * U1.rows());
    
    return fidelity;
}

// Fast fitness calculation
double calculate_fitness(const CircuitIndividual& circuit, const MatrixXcd& target_unitary, 
                        double alpha = 10.0, double beta = 1.0, int target_depth = 20) {
    try {
        MatrixXcd circuit_unitary = circuit.circuit_to_unitary();
        double fid = calculate_fidelity(circuit_unitary, target_unitary);
        
        // Multi-objective fitness with depth penalty
        double normalized_depth = (target_depth > 1) ? 
            static_cast<double>(circuit.depth - 1) / (target_depth - 1) : 0.0;
        
        double fitness = alpha * fid - beta * normalized_depth;
        return std::max(fitness, 0.0);
    } catch (const std::exception& e) {
        std::cerr << "Error in fitness calculation: " << e.what() << std::endl;
        return 0.0;
    }
}

// Thread-safe random number generator
class RandomGenerator {
private:
    // Thread-local storage for random generators
    static thread_local std::mt19937 gen;
    std::uniform_real_distribution<double> real_dist;
    
public:
    RandomGenerator() : real_dist(0.0, 1.0) {
        // Initialize thread-local generator if not already initialized
        if (gen == std::mt19937{}) {
            std::random_device rd;
            gen.seed(rd() + omp_get_thread_num() * 1000); // Different seed per thread
        }
    }
    
    // Seed constructor for thread safety
    RandomGenerator(int seed) : real_dist(0.0, 1.0) {
        gen.seed(seed);
    }
    
    double random_double(double min = 0.0, double max = 1.0) {
        return min + (max - min) * real_dist(gen);
    }
    
    int random_int(int min, int max) {
        if (min > max) std::swap(min, max);
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }
    
    bool random_bool() {
        return random_double() < 0.5;
    }
    
    template<typename T>
    T random_choice(const std::vector<T>& items) {
        if (items.empty()) throw std::runtime_error("Cannot choose from empty list");
        return items[random_int(0, static_cast<int>(items.size()) - 1)];
    }
    
    template<typename T>
    std::vector<T> random_sample(const std::vector<T>& population, int k) {
        if (k <= 0) return {};
        if (k >= static_cast<int>(population.size())) return population;
        
        std::vector<T> result;
        std::vector<int> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        for (int i = 0; i < k; ++i) {
            int idx = random_int(i, static_cast<int>(indices.size()) - 1);
            result.push_back(population[indices[idx]]);
            std::swap(indices[i], indices[idx]);
        }
        
        return result;
    }
};

// Initialize thread-local random generator
thread_local std::mt19937 RandomGenerator::gen;
static thread_local RandomGenerator rng;

// Enhanced QuantumEvolutionaryOptimizer with hybrid optimization
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
    int param_optimization_frequency;
    double param_optimization_rate;
    
    std::vector<GateType> gate_set;
    std::vector<CircuitIndividual> population;
    std::shared_ptr<CircuitIndividual> best_individual;
    std::vector<double> fitness_history;
    
    MatrixXcd target_unitary;  // Store target unitary
    
    std::function<double(const CircuitIndividual&)> fitness_func;
    
    // Helper function to get available qubits
    std::vector<int> get_available_qubits(const std::vector<bool>& used) const {
        std::vector<int> available;
        for (int i = 0; i < static_cast<int>(used.size()); ++i) {
            if (!used[i]) available.push_back(i);
        }
        return available;
    }
    
public:
    QuantumEvolutionaryOptimizer(int nq, int pop_size, int gen, double cross_rate,
                               double mut_rate, double off_rate, double rep_rate,
                               double a, double b, int t_depth,
                               std::vector<GateType> g_set,
                               int param_freq = 25, double param_rate = 0.1)
        : num_qubits(nq), population_size(pop_size), generations(gen),
          crossover_rate(cross_rate), mutation_rate(mut_rate),
          offspring_rate(off_rate), replace_rate(rep_rate),
          alpha(a), beta(b), target_depth(t_depth), 
          param_optimization_frequency(param_freq), param_optimization_rate(param_rate),
          gate_set(std::move(g_set)) {
    }
    
    // Set target unitary for native fitness calculation
    void set_target_unitary(const MatrixXcd& target) {
        target_unitary = target;
    }
    
    void set_fitness_function(std::function<double(const CircuitIndividual&)> func) {
        fitness_func = std::move(func);
    }
    
    // Hybrid parameter optimization using COBYLA
    void optimize_parameters(CircuitIndividual& individual) {
        std::vector<double> current_params = individual.get_parameters();
        if (current_params.empty()) return;
        
        cobyla::COBYLA optimizer(static_cast<int>(current_params.size()), 0, 0.1, 1e-4, 100);
        
        optimizer.set_objective([&](const std::vector<double>& params) {
            CircuitIndividual temp = individual;
            temp.set_parameters(params);
            MatrixXcd circuit_unitary = temp.circuit_to_unitary();
            double fid = calculate_fidelity(circuit_unitary, target_unitary);
            return 1.0 - fid; // COBYLA minimizes, so we return 1 - fidelity
        });
        
        auto result = optimizer.minimize(current_params);
        if (result.success) {
            individual.set_parameters(result.x);
        }
    }
    
    // Apply parameter optimization to population subset
    void apply_parameter_optimization() {
        int num_to_optimize = std::max(1, static_cast<int>(population_size * param_optimization_rate));
        auto candidates = rng.random_sample(population, num_to_optimize);
        
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
    
    // Create random circuit
    CircuitIndividual create_random_circuit(int depth) {
        std::vector<std::vector<Gate>> layers;
        
        for (int i = 0; i < depth; ++i) {
            std::vector<Gate> layer;
            std::vector<bool> used(num_qubits, false);
            
            // Try to add some gates to this layer
            int attempts = 0;
            while (attempts < num_qubits * 2) {
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
                }
                attempts++;
            }
            layers.push_back(std::move(layer));
        }
        
        return CircuitIndividual(num_qubits, depth, std::move(layers));
    }
    
    void initialize_population(int initial_depth, bool from_target = false) {
        population.clear();
        best_individual.reset();
        fitness_history.clear();
        
        for (int i = 0; i < population_size; ++i) {
            population.push_back(create_random_circuit(initial_depth));
        }
    }
    
    // Tournament selection
    std::pair<CircuitIndividual, CircuitIndividual> tournament_selection(int tournament_size = 3) {
        auto tournament1 = rng.random_sample(population, tournament_size);
        auto tournament2 = rng.random_sample(population, tournament_size);
        
        if (tournament1.empty() || tournament2.empty()) {
            if (population.size() >= 2) {
                return {population[0], population[1]};
            } else if (population.size() == 1) {
                return {population[0], population[0]};
            } else {
                throw std::runtime_error("Empty population");
            }
        }
        
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
        double rand1 = rng.random_double(0.0, 1.0);
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
            double rand2 = rng.random_double(0.0, 1.0);
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
    
    // Selection method dispatcher
    std::pair<CircuitIndividual, CircuitIndividual> select_parents(const std::string& method) {
        if (method == "roulette") {
            return roulette_selection();
        } else { // Default to tournament
            return tournament_selection();
        }
    }
    
    // Simple uniform crossover
    CircuitIndividual crossover(const CircuitIndividual& parent1, const CircuitIndividual& parent2) {
        if (rng.random_double() > crossover_rate || parent1.layers.empty() || parent2.layers.empty()) {
            return rng.random_bool() ? parent1 : parent2;
        }
        
        int child_depth = std::max(parent1.depth, parent2.depth);
        std::vector<std::vector<Gate>> child_layers;
        
        for (int i = 0; i < child_depth; ++i) {
            if (rng.random_bool()) {
                child_layers.push_back(i < static_cast<int>(parent1.layers.size()) ? 
                                      parent1.layers[i] : std::vector<Gate>());
            } else {
                child_layers.push_back(i < static_cast<int>(parent2.layers.size()) ? 
                                      parent2.layers[i] : std::vector<Gate>());
            }
        }
        
        return CircuitIndividual(parent1.num_qubits, child_depth, std::move(child_layers));
    }
    
    // Simple mutation
    CircuitIndividual mutate(CircuitIndividual individual) {
        if (rng.random_double() > mutation_rate || individual.layers.empty()) {
            return individual;
        }
        
        // Simple gate mutation
        int layer_idx = rng.random_int(0, static_cast<int>(individual.layers.size()) - 1);
        if (!individual.layers[layer_idx].empty()) {
            int gate_idx = rng.random_int(0, static_cast<int>(individual.layers[layer_idx].size()) - 1);
            GateType new_type = rng.random_choice(gate_set);
            const auto& old_gate = individual.layers[layer_idx][gate_idx];
            
            if (new_type == GateType::RZ) {
                double angle = rng.random_double(0.0, 2.0 * M_PI);
                std::vector<int> qubits = old_gate.qubits.empty() ? std::vector<int>{0} : 
                                         std::vector<int>{old_gate.qubits[0]};
                individual.layers[layer_idx][gate_idx] = Gate(GateType::RZ, qubits, angle);
            } else {
                std::vector<int> qubits = old_gate.qubits.empty() ? std::vector<int>{0} : 
                                         std::vector<int>{old_gate.qubits[0]};
                individual.layers[layer_idx][gate_idx] = Gate(new_type, qubits);
            }
        }
        
        return individual;
    }
    
    void evaluate_population_fitness(std::vector<CircuitIndividual>& individuals) {
        if (!fitness_func) return;
        
        // Parallel evaluation with OpenMP - safe and reliable
        #pragma omp parallel for
        for (size_t i = 0; i < individuals.size(); ++i) {
            individuals[i].fitness = fitness_func(individuals[i]);
        }
    }
    
    // Enhanced evolution with hybrid optimization
    CircuitIndividual run_evolution(bool from_scratch = true, const std::string& selection_method = "tournament") {
        initialize_population(target_depth, !from_scratch);
        
        // Evaluate initial population
        evaluate_population_fitness(population);
        
        int num_offspring = std::max(1, static_cast<int>(population_size * offspring_rate));
        int num_replace = std::max(1, static_cast<int>(population_size * replace_rate));
        
        for (int generation = 0; generation < generations; ++generation) {
            // Apply parameter optimization every N generations
            if (generation % param_optimization_frequency == 0 && generation > 0) {
                apply_parameter_optimization();
            }
            
            // Create offspring in parallel with OpenMP
            std::vector<CircuitIndividual> offspring(num_offspring);
            #pragma omp parallel for
            for (int i = 0; i < num_offspring; ++i) {
                auto parents = select_parents(selection_method);
                auto child = crossover(parents.first, parents.second);
                child = mutate(std::move(child));
                child = child.optimize_structure();
                
                if (fitness_func) {
                    child.fitness = fitness_func(child);
                }
                offspring[i] = std::move(child);
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
            
            for (int i = 0; i < num_replace && i < static_cast<int>(offspring.size()); ++i) {
                population[i] = std::move(offspring[i]);
            }
            
            // Track best individual
            if (!population.empty()) {
                auto best_it = std::max_element(population.begin(), population.end(),
                                              [](const CircuitIndividual& a, const CircuitIndividual& b) {
                                                  return a.fitness < b.fitness;
                                              });
                
                if (best_it != population.end()) {
                    if (!best_individual || best_it->fitness > best_individual->fitness) {
                        best_individual = std::make_shared<CircuitIndividual>(*best_it);
                    }
                    
                    fitness_history.push_back(best_it->fitness);
                    
                    // Log every 50 generations
                    #pragma omp single
                    if (generation % 50 == 0) {
                        int non_id_gates = best_it->count_non_id_gates();
                        std::cout << "Generation " << generation << ": Best fitness = " << best_it->fitness
                                 << ", Depth = " << best_it->depth 
                                 << ", Non-ID gates = " << non_id_gates 
                                 << ", Selection = " << selection_method << std::endl;
                    }
                }
            }
        }
        
        if (best_individual) {
            return *best_individual;
        } else {
            return create_random_circuit(target_depth);
        }
    }
    
    // Getters
    const std::vector<CircuitIndividual>& get_population() const { return population; }
    const std::vector<double>& get_fitness_history() const { return fitness_history; }
    const CircuitIndividual& get_best_individual() const { 
        if (best_individual) return *best_individual;
        throw std::runtime_error("No best individual available");
    }
    
    // Add getters for Python access
    int get_population_size() const { return population_size; }
    int get_generations() const { return generations; }
    int get_num_qubits() const { return num_qubits; }
    int get_target_depth() const { return target_depth; }
};

// Pybind11 module
PYBIND11_MODULE(ModuleName, m) {
    m.doc() = "High-performance hybrid quantum circuit optimization with COBYLA and OpenMP";
    
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
        .def("gate_counts", &CircuitIndividual::gate_counts)
        .def("count_non_id_gates", &CircuitIndividual::count_non_id_gates)
        .def("circuit_to_unitary", &CircuitIndividual::circuit_to_unitary)
        .def("optimize_structure", &CircuitIndividual::optimize_structure);
    
    // QuantumEvolutionaryOptimizer class
    py::class_<QuantumEvolutionaryOptimizer>(m, "QuantumEvolutionaryOptimizer")
        .def(py::init<int, int, int, double, double, double, double, double, double, int, std::vector<GateType>, int, double>(),
             py::arg("num_qubits"), py::arg("population_size"), py::arg("generations"),
             py::arg("crossover_rate"), py::arg("mutation_rate"), py::arg("offspring_rate"),
             py::arg("replace_rate"), py::arg("alpha"), py::arg("beta"), py::arg("target_depth"),
             py::arg("gate_set"), py::arg("param_freq") = 25, py::arg("param_rate") = 0.1)
        .def("set_fitness_function", &QuantumEvolutionaryOptimizer::set_fitness_function)
        .def("set_target_unitary", &QuantumEvolutionaryOptimizer::set_target_unitary)
        .def("optimize_parameters", &QuantumEvolutionaryOptimizer::optimize_parameters)
        .def("apply_parameter_optimization", &QuantumEvolutionaryOptimizer::apply_parameter_optimization)
        .def("create_random_circuit", &QuantumEvolutionaryOptimizer::create_random_circuit)
        .def("run_evolution", &QuantumEvolutionaryOptimizer::run_evolution, py::call_guard<py::gil_scoped_release>(),
             py::arg("from_scratch") = true, py::arg("selection_method") = "tournament")
        .def("get_population", &QuantumEvolutionaryOptimizer::get_population)
        .def("get_fitness_history", &QuantumEvolutionaryOptimizer::get_fitness_history)
        .def("get_best_individual", &QuantumEvolutionaryOptimizer::get_best_individual)
        .def_property_readonly("population_size", &QuantumEvolutionaryOptimizer::get_population_size)
        .def_property_readonly("generations", &QuantumEvolutionaryOptimizer::get_generations)
        .def_property_readonly("num_qubits", &QuantumEvolutionaryOptimizer::get_num_qubits)
        .def_property_readonly("target_depth", &QuantumEvolutionaryOptimizer::get_target_depth);
    
    // Standalone functions
    m.def("calculate_fidelity", &calculate_fidelity, 
          "Calculate fidelity between two unitary matrices",
          py::arg("U1"), py::arg("U2"));
    
    m.def("calculate_fitness", &calculate_fitness,
          "Calculate fitness for a circuit against target unitary",
          py::arg("circuit"), py::arg("target_unitary"), 
          py::arg("alpha") = 10.0, py::arg("beta") = 1.0, py::arg("target_depth") = 20);
    
    // Helper function to create identity matrix
    m.def("identity_matrix", [](int n_qubits) {
        int dim = 1 << n_qubits;
        return MatrixXcd::Identity(dim, dim);
    }, "Create identity matrix for n qubits");
}
