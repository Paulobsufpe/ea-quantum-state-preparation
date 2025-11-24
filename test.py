import qext

print(qext.build_observable())
qc = qext.test()
qc.draw(output="mpl")
