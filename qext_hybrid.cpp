#define EIGEN_USE_BLAS
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <pybind11/eigen.h>
#include <Eigen/Dense>
namespace py = pybind11;

#include <vector>
#ifdef _OPENMP
#define ModuleName qext_omp
#else
#define ModuleName qext
#endif // _OPENMP

#include "util.hpp"
#include "gate.hpp"
#include "circuit_individual.hpp"
#include "optimizer.hpp"

// Fast fitness calculation
static inline constexpr double calculate_fitness(CircuitIndividual& circuit, const MatrixXcd& target_unitary, 
                                                 double alpha = 10.0, double beta = 1.0, int target_depth = 20) {
    MatrixXcd circuit_unitary = circuit.circuit_to_unitary();
    double fid = calculate_fidelity(circuit_unitary, target_unitary);

    // Multi-objective fitness with depth penalty
    double normalized_depth = (target_depth > 1) ? 
        static_cast<double>(circuit.depth - 1) / (target_depth - 1) : 0.0;

    double fitness = alpha * fid - beta * normalized_depth;
    return fitness;
}

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
        .def("set_target_circuit", &QuantumEvolutionaryOptimizer::set_target_circuit)
        .def("set_target_unitary", &QuantumEvolutionaryOptimizer::set_target_unitary)
        .def("optimize_parameters", &QuantumEvolutionaryOptimizer::optimize_parameters)
        .def("apply_parameter_optimization", &QuantumEvolutionaryOptimizer::apply_parameter_optimization)
        .def("create_random_circuit", &QuantumEvolutionaryOptimizer::create_random_circuit)
#ifdef _OPENMP
        .def("run_evolution", &QuantumEvolutionaryOptimizer::run_evolution, py::call_guard<py::gil_scoped_release>(),
#else
        .def("run_evolution", &QuantumEvolutionaryOptimizer::run_evolution,
#endif // _OPENMP
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
