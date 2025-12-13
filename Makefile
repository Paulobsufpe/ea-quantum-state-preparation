CC :=clang
CXX:=clang++

ARGS:=

PKG_CONFIG_INC:=$$(pkg-config --cflags python3 eigen3)
PKG_CONFIG_LIB:=-lopenblas
CFLAGS:=-fPIC -Wall -Wextra -fno-plt -fstack-protector-strong \
				-fno-math-errno -fno-trapping-math -fvisibility=hidden -gdwarf-5 \
				-fno-omit-frame-pointer
CXXFLAGS:=${CFLAGS} -std=c++23
OPTFLAGS:=-O3 -march=native -flto

# QISKIT_LDFLAGS:=-l:_accelerate.cpython-313-x86_64-linux-gnu.so -lqiskit -Wl,-z,now,-rpath,/usr/local/lib
LDFLAGS:=${QISKIT_LDFLAGS} -Wl,-z,now,-z,relro,-z,noexecstack,--gc-sections -fuse-ld=mold

PYBIND_INC:=$$(/usr/bin/env python3 -m pybind11 --includes)
# PYBIND_EXT:=$$(/usr/bin/env python3 -m pybind11 --extension-suffix)

# all: $(patsubst %.cpp, %.so, $(wildcard *.cpp))
all: qext.so qext_omp.so

.PHONY: all run clean

r: run
run: visualize.py .venv qext.so qext_omp.so
	uv run $< ${ARGS}

cl: clean
clean:
	rm *.so *.o *.tmp

qext.so: qext.o cobyla.o gate.o circuit_individual.o optimizer.o
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_LIB} ${LDFLAGS} $^ -shared -o $@

qext.o: qext_hybrid.cpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

qext_omp.so: qext_omp.o cobyla.o gate.o circuit_individual.o optimizer_omp.o
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_LIB} ${LDFLAGS} $^ -shared -o $@ -fopenmp

qext_omp.o: qext_hybrid.cpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@ -fopenmp

cobyla.o: cobyla.cpp cobyla.hpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

gate.o: gate.cpp gate.hpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

circuit_individual.o: circuit_individual.cpp circuit_individual.hpp gate.hpp util.hpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

optimizer.o: optimizer.cpp optimizer.hpp random_generator.hpp cobyla.hpp \
	gate.hpp circuit_individual.hpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

optimizer_omp.o: optimizer.cpp optimizer.hpp random_generator.hpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@ -fopenmp
