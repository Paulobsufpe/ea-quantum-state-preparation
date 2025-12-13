#include "circuit_individual.hpp"

CircuitIndividual CircuitIndividual::optimize_structure() const {
    std::vector<std::vector<Gate>> new_layers;
    
    for (const auto& layer : layers) {
        bool have_non_id_gate = false;
        for (const auto& gate : layer) {
            if (gate.type != GateType::ID) {
                have_non_id_gate = true;
                break;
            }
        }
        if (have_non_id_gate) {
            new_layers.push_back(std::move(layer));
        }
    }
    
    if (new_layers.empty()) {
        new_layers.emplace_back();
    }
    
    return CircuitIndividual(num_qubits, static_cast<int>(new_layers.size()), std::move(new_layers));
}

MatrixXcd CircuitIndividual::circuit_to_unitary() {
    if (unitary.has_value()) return unitary.value();
    int dim = 1 << num_qubits; // 2^n
    MatrixXcd unitary_matrix = MatrixXcd::Identity(dim, dim);
    MatrixXcd layer_matrix {};
    MatrixXcd gate_matrix {};
    
    for (const auto& layer : layers) {
        layer_matrix = MatrixXcd::Identity(dim, dim);
        
        for (const auto& gate : layer) {
            if (gate.type == GateType::ID) continue;
            
            gate_matrix = get_gate_matrix(gate, num_qubits);
            layer_matrix = gate_matrix * layer_matrix;
        }
        
        unitary_matrix = layer_matrix * unitary_matrix;
    }
    unitary = unitary_matrix;
    return unitary_matrix;
}
