#pragma once
#include "util.hpp"
#include "gate.hpp"
namespace rng = std::ranges;

// Circuit individual
class CircuitIndividual {
public:
    int num_qubits;
    int depth;
    std::vector<std::vector<Gate>> layers;
    double fitness;
    double fidelity;
    double normalized_depth;
    std::optional<MatrixXcd> unitary;
    
    inline CircuitIndividual(int nq, int d, std::vector<std::vector<Gate>> l = {})
        : num_qubits(nq), depth(d), layers(std::move(l)), fitness(-1e9), fidelity(0.0), normalized_depth(0.0) {
        if (layers.empty()) {
            layers.resize(depth);
        } else {
            // Ensure depth matches layers size
            depth = static_cast<int>(layers.size());
        }
    }
    
    // Default constructor for pair compatibility
    inline CircuitIndividual() : num_qubits(0), depth(0), fitness(-1e9), fidelity(0.0), normalized_depth(0.0) {}
    
    // Get used qubits in a layer
    inline constexpr std::vector<bool> get_used_qubits(int layer_idx) const {
        std::vector<bool> used(num_qubits, false);
        auto layer = layers[layer_idx];
        rng::transform(layer, used.begin(),
            [this](Gate gate) -> bool {
                return rng::max(gate.qubits) < num_qubits; });
        return used;
    }
    
    // Get available qubits
    static inline constexpr std::vector<int> get_available_qubits(const std::vector<bool>& used) {
        std::vector<int> available;
        for (size_t idx = 0; idx < used.size(); ++idx) {
            if (!used[idx]) available.push_back(idx);
        }
        return available;
    }
    
    // Get parameters for optimization
    inline constexpr std::vector<double> get_parameters() const {
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
    inline constexpr void set_parameters(const std::vector<double>& params) {
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
    inline constexpr std::unordered_map<GateType, int> gate_counts() const {
        std::unordered_map<GateType, int> counts;
        for (const auto& layer : layers) {
            for (const auto& gate : layer) {
                counts[gate.type]++;
            }
        }
        return counts;
    }
    
    // Get number of non-identity gates
    inline constexpr int count_non_id_gates() const {
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
    CircuitIndividual optimize_structure() const;
    
    // Convert circuit to unitary matrix using Eigen
    MatrixXcd circuit_to_unitary();
    
private:
    static inline constexpr MatrixXcd get_gate_matrix(const Gate& gate, int dim) {
        switch (gate.type) {
            case GateType::X:
                return get_pauli_x_matrix(gate.qubits[0], dim);
            case GateType::SX:
                return get_sx_matrix(gate.qubits[0], dim);
            case GateType::RZ:
                return get_rz_matrix(gate.angle, gate.qubits[0], dim);
            case GateType::H:
                return get_hadamard_matrix(gate.qubits[0], dim);
            case GateType::CX:
                if (gate.qubits.size() >= 2) {
                    return get_cx_matrix(gate.qubits[0], gate.qubits[1], dim);
                }
                // Fallthrough
            default:
                return MatrixXcd::Identity(dim, dim);
        }
    }
    
    static inline constexpr MatrixXcd get_pauli_x_matrix(int target_qubit, int dim) {
        MatrixXcd result = MatrixXcd::Zero(dim, dim);
        
        for (int i = 0; i < dim; ++i) {
            int flipped = i ^ (1 << target_qubit);
            result(flipped, i) = 1.0;
        }
        
        return result;
    }
    
    static inline constexpr MatrixXcd get_sx_matrix(int target_qubit, int dim) {
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
    
    static inline constexpr MatrixXcd get_rz_matrix(double angle, int target_qubit, int dim) {
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
    
    static inline constexpr MatrixXcd get_hadamard_matrix(int target_qubit, int dim) {
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
    
    static inline constexpr MatrixXcd get_cx_matrix(int control_qubit, int target_qubit, int dim) {
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

