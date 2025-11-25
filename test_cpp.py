import qiskit
import qext_cpp

obs = qext_cpp.build_observable()
print("SparseObservable instance?", isinstance(obs, qiskit.quantum_info.SparseObservable))
print(obs)

qc = qext_cpp.test()
# qc.draw(output="text")

print(type(obs))
print(type(qc))
