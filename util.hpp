#pragma once
#include <complex>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>

namespace rng = std::ranges;
namespace vws = rng::views;

namespace py = pybind11;
using Complex = std::complex<double>;
using MatrixXcd = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXcd = Eigen::Vector<Complex, Eigen::Dynamic>;
