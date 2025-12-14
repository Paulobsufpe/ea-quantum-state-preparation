CC :=clang
CXX:=clang++

ARGS:=

PKG_CONFIG_INC:=$$(pkg-config --cflags python3 eigen3)
PKG_CONFIG_LIB:=-lopenblas
CFLAGS:=-fPIC -Wall -Wextra -fno-plt -fstack-protector-strong \
				-fno-math-errno -fno-trapping-math -fvisibility=hidden -gdwarf-5 \
				-fno-omit-frame-pointer
CXXFLAGS:=${CFLAGS} -std=c++23
OPTFLAGS:=-O3 -march=native

ifeq ($(CXX),clang++)
    PCH_INC:=-include-pch pch.hpp.gch
else
    PCH_INC:=-include pch.hpp
endif

LDFLAGS:=-Wl,-z,now,-z,relro,-z,noexecstack,--gc-sections -fuse-ld=lld

PYBIND_INC:=$$(python3 -m pybind11 --includes)
# PYBIND_EXT:=$$(/usr/bin/env python3 -m pybind11 --extension-suffix)

default: qext.so
omp: qext_omp.so
all: default omp

.PHONY: default omp run clean

r: run
run: visualize.py .venv qext.so qext_omp.so
	uv run $< ${ARGS}

cl: clean
clean:
	rm *.so *.o *.tmp pch*

qext.so: qext.o cobyla.o gate.o circuit_individual.o optimizer.o pch.hpp.gch
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_LIB} ${LDFLAGS} $^ -shared -o $@

qext.o: qext_hybrid.cpp util.hpp cobyla.hpp gate.hpp circuit_individual.hpp optimizer.hpp pch.hpp.gch
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PCH_INC} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

qext_omp.so: qext_omp.o cobyla.o gate.o circuit_individual.o optimizer_omp.o
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_LIB} ${LDFLAGS} $^ -shared -o $@ -fopenmp

qext_omp.o: qext_hybrid.cpp util.hpp cobyla.hpp gate.hpp circuit_individual.hpp optimizer.hpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@ -fopenmp

cobyla.o: cobyla.cpp cobyla.hpp pch.hpp.gch
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PCH_INC} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

gate.o: gate.cpp gate.hpp pch.hpp.gch
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PCH_INC} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

circuit_individual.o: circuit_individual.cpp circuit_individual.hpp util.hpp gate.hpp pch.hpp.gch
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PCH_INC} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

optimizer.o: optimizer.cpp optimizer.hpp random_generator.hpp util.hpp cobyla.hpp \
	gate.hpp circuit_individual.hpp pch.hpp.gch
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PCH_INC} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@

optimizer_omp.o: optimizer.cpp optimizer.hpp random_generator.hpp util.hpp cobyla.hpp \
	gate.hpp circuit_individual.hpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -c $< -o $@ -fopenmp

pch.hpp.tmp: $(filter-out pch.hpp,$(wildcard *.cpp *.hpp))
	grep -E '\s*#include\s*<.*?>\s*' *.cpp *.hpp | cut -d ':' -f 2 | sort | uniq | grep -v 'pybind' > $@

pch.hpp: pch.hpp.tmp
	set -x; \
	if [ ! -f  "$@" ] || [ -n "$(cmp $@ $<)" ]; then \
		cat $< > $@; \
	fi

pch.hpp.gch: pch.hpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} ${PKG_CONFIG_INC} ${PYBIND_INC} -x c++-header $< -o $@
