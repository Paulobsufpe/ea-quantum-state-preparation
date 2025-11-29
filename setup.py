from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "qext",
        ["qext.cpp"],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        cxx_std=17,
        extra_compile_args=['-O3', '-march=native', '-ffast-math']
    ),
]

setup(
    name="qext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
