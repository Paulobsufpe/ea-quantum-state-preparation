#include <array>
#include <cstdint>
#include <random>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
 
#include <print>
#define QISKIT_C_PYTHON_INTERFACE
#include <qiskit.h>

enum class GateK {
  ID, X, SX, RZ, CX // NOTE: adicionar mais gates? talvez não seja necessário.
                    // O artigo tem R_X nas figuras. Pq RZ nessa lista?
};

struct Gate {
  GateK gate;
};

struct Circuit {
  uint32_t num_qubits;
  uint32_t depth;
  Gate** table; // NOTE: first dimention will be vertical. (need testing)
                // Will iterate by num_qubits
  QkCircuit* to_QkCircuit() {
    return qk_circuit_new(num_qubits, 0);
  }
};

constexpr size_t PopulationSize = 512;

static std::array<Circuit, PopulationSize> Population{};

enum class CrossoverK {
  SinglePoint,
  Uniform,
};

constexpr CrossoverK DefaultCrossoverKind = CrossoverK::Uniform;

// Hyperparameters
constexpr uint32_t Generations   = 1000;
constexpr uint32_t MaxCobylaIter = 1000;
constexpr double   CrossoverRate = 0.85;
constexpr double   MutationRate  = 0.85;
constexpr double   OffspringRate = 0.30;
constexpr double   ReplaceRate   = 0.30;
// end Hyperparameters

static std::random_device random_dev;

static inline constexpr uint64_t random_int(uint64_t a, uint64_t b) noexcept {
  std::mt19937_64 rng(random_dev());
  std::uniform_int_distribution<> dist(a, b);
  return dist(rng);
}

static inline constexpr double random_double(double a, double b) noexcept {
  std::mt19937_64 rng(random_dev());
  std::uniform_real_distribution<> dist(a, b);
  return dist(rng);
}

static inline constexpr bool random_bool(void) noexcept {
  return random_int(0, 1) ? true : false;
}

void cross(const Circuit&) {}

enum class MutationK {
  SingleGateMut,
  GateSwap,
  ColumnSwap,
  CtrlAndTargetSwap,
  AddRandomColumn,
  DelCollumn,
  AddCXGate,
  AddSingleGate,
};

namespace Mutation {
static void singleGateMut(const Circuit&) {}
static void gateSwap(const Circuit&) {}
static void columnSwap(const Circuit&) {}
static void ctrlAndTargetSwap(const Circuit&) {}
static void addRandomColumn(const Circuit&) {}
static void delCollumn(const Circuit&) {}
static void addCXGate(const Circuit&) {}
static void addSingleGate(const Circuit&) {}

static constexpr auto fTable = {
	&singleGateMut,
	&gateSwap,
	&columnSwap,
	&ctrlAndTargetSwap,
	&addRandomColumn,
	&delCollumn,
	&addCXGate,
	&addSingleGate,
};
} // namespace Mutation

void mutate(const Circuit&) {}

uint64_t fidelity(const Circuit&){
  return 0;
}

uint64_t fitness(const Circuit&) {
  return 0;
}

static inline QkObs *build_observable(void) {
  // build a 100-qubit empty observable
  u_int32_t num_qubits = 100;
  QkObs *obs = qk_obs_zero(num_qubits);

  // add the term 2 * (X0 Y1 Z2) to the observable
  QkComplex64 coeff {2,0};       // the coefficient
  QkBitTerm bit_terms[3]{
      QkBitTerm_X, 
      QkBitTerm_Y,
      QkBitTerm_Z
  };                             // bit terms: X Y Z
  uint32_t indices[3]{0, 1, 2};  // indices: 0 1 2
  QkObsTerm term{.coeff = coeff,
                 .len = 3,
                 .bit_terms = bit_terms,
                 .indices = indices,
                 .num_qubits = num_qubits};
  qk_obs_add_term(obs, &term);   // append the term

  return obs;
}

static inline QkCircuit* test() {
  for (int i = 0; i < 12; ++i) {
    std::println("{}", random_int(1, 100));
  }
  std::println("------------------------");
  for (int i = 0; i < 12; ++i) {
    std::println("{}", random_double(1, 100));
  }
  std::println("------------------------");
  for (int i = 0; i < 12; ++i) {
    std::println("{}", random_bool());
  }
  QkCircuit *qc = qk_circuit_new(3, 0);
  // H gate on qubit 0, putting this qubit in a superposition of |0> + |1>.
  qk_circuit_gate(qc, QkGate_H, (uint32_t[]){0}, NULL);
  // A CX (CNOT) gate on control qubit 0 and target qubit 1 generating a Bell state.
  qk_circuit_gate(qc, QkGate_CX, (uint32_t[]){0, 1}, NULL);
  // A CX (CNOT) gate on control qubit 0 and target qubit 2 generating a GHZ state.
  qk_circuit_gate(qc, QkGate_CX, (uint32_t[]){0, 2}, NULL);
  return qc;
}

static PyObject *ext_test(PyObject *self, PyObject *args) {
    (void) args;
    (void) self;
    QkCircuit *qc = test();
    PyObject *py_qc = qk_circuit_to_python(qc);
    return py_qc;
}

/// Define the Python function, which will internally build the QkObs using the
/// C function defined above, and then convert the C object to the Python equivalent:
/// a SparseObservable, handled as PyObject.
static PyObject *ext_build_observable(PyObject *self, PyObject *args) {
    (void) args;
    (void) self;
    // At this point, ``args`` could be parsed for arguments. See PyArg_ParseTuple for details.
    QkObs *obs = build_observable();           // call the C function to build the observable
    PyObject *py_obs = qk_obs_to_python(obs);  // convert QkObs to the Python-equivalent
    return py_obs;
}
 
/// Define the module methods.
static PyMethodDef ExtMethods[] {
    {"build_observable", ext_build_observable, METH_VARARGS, "Build an observable."},
    {"test", ext_test, METH_VARARGS, "A test"},
    {NULL, NULL, 0, NULL}, // sentinel
};
 
/// Define the module, which here is called ``cextension``.
static struct PyModuleDef cppextension {
    PyModuleDef_HEAD_INIT,
    "qext_cpp",     // module name
    NULL,       // docs
    -1,         // keep the module state in global variables
    ExtMethods,
    // to silence a warning
    0, 0, 0, 0
};
 
PyMODINIT_FUNC PyInit_qext_cpp(void) { return PyModule_Create(&cppextension); }
