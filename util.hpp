#pragma once
#include <complex>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>

using Complex = std::complex<double>;
using MatrixXcd = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXcd = Eigen::Vector<Complex, Eigen::Dynamic>;
