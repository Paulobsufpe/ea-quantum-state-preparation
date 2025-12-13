#pragma once
#include <vector>

// Gate types
enum class GateType {
    ID, X, SX, RZ, CX, H
};

// Gate structure
struct Gate {
    GateType type;
    std::vector<int> qubits;
    double angle;

    Gate(GateType t, std::vector<int> q, double a = 0.0);
    
    inline constexpr bool is_single_qubit() const { return qubits.size() == 1; }
    inline constexpr bool is_two_qubit() const { return qubits.size() == 2; }
};

