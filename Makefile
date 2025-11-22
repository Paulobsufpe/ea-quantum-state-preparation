CC :=clang
CXX:=clang++

PYTHON_PKG_CONFIG:=$$(pkg-config --cflags python)
CFLAGS:=-shared -fpic -Wall -Wextra
CXXFLAGS:=${CFLAGS} -std=c++17

QISKIT_LDFLAGS:=-l:_accelerate.cpython-313-x86_64-linux-gnu.so -lqiskit -Wl,-z,now,-rpath,/usr/local/lib
LDFLAGS:=${QISKIT_LDFLAGS}

PYBIND_INC:=$$(/usr/bin/env python3 -m pybind11 --includes)
# PYBIND_EXT:=$$(/usr/bin/env python3 -m pybind11 --extension-suffix)

all: $(patsubst %.cpp, %.so, $(wildcard *.cpp))

%.so: %.cpp
	${CXX} $< ${CXXFLAGS} ${PYTHON_PKG_CONFIG} ${LDFLAGS} -o $@
