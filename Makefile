CC :=clang
CXX:=clang++

PYTHON_PKG_CONFIG:=$$(pkg-config --cflags python)
CFLAGS:=-shared -fpic -Wall -Wextra -fno-plt -fstack-protector-strong
CXXFLAGS:=${CFLAGS} -std=c++23 -fno-rtti
OPTFLAGS:=-O0 -g

QISKIT_LDFLAGS:=-l:_accelerate.cpython-313-x86_64-linux-gnu.so -lqiskit -Wl,-z,now,-rpath,/usr/local/lib
LDFLAGS:=${QISKIT_LDFLAGS} -Wl,-z,now,-z,relro,-z,noexecstack,--gc-sections

# PYBIND_INC:=$$(/usr/bin/env python3 -m pybind11 --includes)
# PYBIND_EXT:=$$(/usr/bin/env python3 -m pybind11 --extension-suffix)

all: $(patsubst %.cpp, %.so, $(wildcard *.cpp))

.PHONY: all run clean

r: run
run: test.py all
	VIRTUAL_ENV=~/.virtualenvs/qiskit uv run $<

cl: clean
clean:
	fd -e so -I -X rm; rm compile_commands.json

%.so: %.cpp
	${CXX} $< ${CXXFLAGS} ${PYTHON_PKG_CONFIG} ${OPTFLAGS} ${LDFLAGS} -o $@
