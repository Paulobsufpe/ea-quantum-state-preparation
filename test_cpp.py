import qext_cpp

print(qext_cpp.build_observable())
qc = qext_cpp.test()
qc.draw(output="mpl")
