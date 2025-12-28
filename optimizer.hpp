#pragma once
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>

#include "cobyla.hpp"
#include "gate.hpp"
#include "circuit_individual.hpp"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif // _OPENMP

static inline constexpr double calculate_fidelity(const MatrixXcd& U1, const MatrixXcd& U2) noexcept {
    assert(U1.rows() == U2.rows() && U1.cols() == U2.cols() && "Matrix dimensions must match");
    
    MatrixXcd product = U1.adjoint() * U2;
    Complex trace = product.trace();
    double fidelity = std::norm(trace) / (U1.rows() * U1.rows());
    
    return fidelity;
}

enum class CrossoverType {
    SINGLE_POINT, UNIFORM, MULTI_POINT, BLOCKWISE
};

enum class MutationType {
    SINGLE_GATE, GATE_SWAP, COLUMN_SWAP, CTRL_TARGET_SWAP,
    ADD_RANDOM_COLUMN, DELETE_COLUMN, ADD_CX_GATE, ADD_SINGLE_GATE, MUTATE_PARAMETERS
};

class QuantumEvolutionaryOptimizer {
private:
    const int num_qubits;
    const int population_size;
    const int generations;
    const double crossover_rate;
    const double mutation_rate;
    const double offspring_rate;
    const double replace_rate;
    const double alpha;
    const double beta;
    const int target_depth;
    const int param_optimization_frequency;
    const double param_optimization_rate;

    const std::vector<GateType> gate_set;
    std::vector<CircuitIndividual> population;
    std::shared_ptr<CircuitIndividual> best_individual;
    std::vector<double> fitness_history;
    
    CircuitIndividual target_circuit;
    MatrixXcd target_unitary;
    
    std::function<double(const CircuitIndividual&)> fitness_func;
    
    CircuitIndividual mutate_single_gate(CircuitIndividual ind) const;
    CircuitIndividual mutate_gate_swap(CircuitIndividual ind);
    CircuitIndividual mutate_column_swap(CircuitIndividual ind);
    CircuitIndividual mutate_delete_column(CircuitIndividual ind);
    CircuitIndividual mutate_add_cx_gate(CircuitIndividual ind);
    CircuitIndividual mutate_add_single_gate(CircuitIndividual ind) const;
    CircuitIndividual mutate_parameters(CircuitIndividual ind);
    CircuitIndividual mutate_ctrl_target_swap(CircuitIndividual ind);
    CircuitIndividual mutate_add_random_column(CircuitIndividual ind) const;
    
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
    
    inline void set_target_circuit(const CircuitIndividual& target) {
        target_circuit = target;
    }

    // Set target unitary for native fitness calculation
    inline void set_target_unitary(const MatrixXcd& target) {
        target_unitary = target;
    }
    
    inline void set_fitness_function(std::function<double(const CircuitIndividual&)> func) {
        fitness_func = std::move(func);
    }
    
    // Parameter optimization using COBYLA
    inline void optimize_parameters(CircuitIndividual& individual) const {
        const std::vector<double> current_params = individual.get_parameters();
        if (current_params.empty()) return;
        
        Cobyla optimizer(static_cast<int>(current_params.size()), 0, 0.1, 1e-4, 100);
        
        optimizer.set_objective([&](const std::vector<double>& params) {
            CircuitIndividual temp = individual;
            temp.set_parameters(params);
            const MatrixXcd circuit_unitary = temp.circuit_to_unitary();
            const double fid = calculate_fidelity(circuit_unitary, target_unitary);
            return 1.0 - fid; // COBYLA minimizes, so we return 1 - fidelity
        });
        
        const auto result = optimizer.minimize(current_params);
        if (result.success) {
            individual.set_parameters(result.x);
        }
    }
    
    // Apply parameter optimization to population subset
    void apply_parameter_optimization() const;

    CircuitIndividual create_random_circuit(int depth) const;
    
    // TODO: make use of from_target
    inline void initialize_population(int initial_depth, [[maybe_unused]] bool from_target = false) {
        population.clear();
        best_individual.reset();
        fitness_history.clear();

        if (from_target) {
            population.push_back(target_circuit);
            population[0].unitary = target_unitary;
            for (int i = 1; i < population_size; ++i) {
                population.push_back(create_random_circuit(initial_depth));
            }
        } else {
            for (int i = 0; i < population_size; ++i) {
                population.push_back(create_random_circuit(initial_depth));
            }
        }
    }
    
    // Tournament selection
    std::pair<CircuitIndividual, CircuitIndividual> tournament_selection(int tournament_size = 3) const;
    // Roulette wheel selection
    std::pair<CircuitIndividual, CircuitIndividual> roulette_selection() const;
    // Selection method dispatcher
    inline std::pair<CircuitIndividual, CircuitIndividual> select_parents(const std::string& method) const {
        if (method == "roulette") {
            return roulette_selection();
        } else {
            return tournament_selection();
        }
    }
    
    CircuitIndividual crossover(const CircuitIndividual& parent1, const CircuitIndividual& parent2,
                               CrossoverType method = CrossoverType::UNIFORM);
    
    CircuitIndividual mutate(CircuitIndividual individual);
    
    inline void evaluate_population_fitness(std::vector<CircuitIndividual>& individuals) const {
        if (!fitness_func) return;
        
        #pragma omp parallel for
        for (size_t i = 0; i < individuals.size(); ++i) {
            individuals[i].fitness = fitness_func(individuals[i]);
        }
    }
    
    inline CircuitIndividual run_evolution(bool from_scratch = true, const std::string& selection_method = "roulette") {
        initialize_population(target_depth, !from_scratch);
        
        evaluate_population_fitness(population);
        
        const size_t num_offspring = std::max(1, static_cast<int>(population_size * offspring_rate));
        const size_t num_replace = std::max(1, static_cast<int>(population_size * replace_rate));
        std::vector<CircuitIndividual> offspring(num_offspring);
        
        for (int generation = 0; generation < generations; ++generation) {
            #pragma omp parallel for
            for (size_t i = 0; i < num_offspring; ++i) {
                auto parents = select_parents(selection_method);
                auto child = crossover(parents.first, parents.second);
                child = mutate(std::move(child));
                child = child.optimize_structure();
                
                if (fitness_func) {
                    child.fitness = fitness_func(child);
                }
                offspring[i] = std::move(child);
            }

            if (generation % param_optimization_frequency == 0 && generation > 0) {
                apply_parameter_optimization();
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
            
            for (size_t i = 0; i < offspring.size(); ++i) {
                if (i < num_replace) {
                    population[i] = std::move(offspring[i]);
                } else {
                    population[i] = population[i].optimize_structure();
                }
            }
            
            // Track best individual
            if (!population.empty()) {
                const auto best_it = std::max_element(population.begin(), population.end(),
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
