# setup_fixed.py
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os

def find_eigen():
    paths = [
        '/usr/include/eigen3',
        '/usr/local/include/eigen3', 
        '/opt/homebrew/include/eigen3',
        os.path.join(os.path.expanduser('~'), 'eigen'),
    ]
    
    for path in paths:
        if os.path.exists(os.path.join(path, 'Eigen', 'Dense')):
            return path
    return None

eigen_include = find_eigen()
include_dirs = [pybind11.get_include()]

if eigen_include:
    include_dirs.append(eigen_include)
    print(f"Found Eigen at: {eigen_include}")
else:
    print("Warning: Eigen not found in standard locations")

ext_modules = [
    Pybind11Extension(
        "qext",
        ["qext_hybrid.cpp"],
        include_dirs=include_dirs,
        language='c++',
        cxx_std=17,
        extra_compile_args=['-O3', '-march=native', '-ffast-math', '-g'],
        extra_link_args=['-O3', '-march=native', '-ffast-math', '-g'],
    ),
]

setup(
    name="qext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
