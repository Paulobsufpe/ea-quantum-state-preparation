#include "gate.hpp"
#include <algorithm>

Gate::Gate(GateType t, std::vector<int> q, double a)
    : type(t), qubits(std::move(q)), angle(a) {
    // Remove duplicates and sort
    std::sort(qubits.begin(), qubits.end());
    qubits.erase(std::unique(qubits.begin(), qubits.end()), qubits.end());
}
