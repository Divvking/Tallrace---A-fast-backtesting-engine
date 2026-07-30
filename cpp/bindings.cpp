#include <cstring>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "simulation.h"

namespace py = pybind11;

py::array_t<double> py_compute_portfolio_returns(
    py::array_t<double, py::array::c_style | py::array::forcecast> returns,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights)
{
    auto buf_r = returns.request();
    auto buf_w = weights.request();

    if (buf_r.ndim != 2)
        throw py::value_error("returns must be a 2D array (T x N)");
    if (buf_w.ndim != 1)
        throw py::value_error("weights must be a 1D array (N)");

    int T = static_cast<int>(buf_r.shape[0]);
    int N = static_cast<int>(buf_r.shape[1]);

    if (T <= 0)
        throw py::value_error("returns must contain at least one row");
    if (N <= 0)
        throw py::value_error("returns must contain at least one asset");
    if (static_cast<int>(buf_w.shape[0]) != N)
        throw py::value_error("weights length must match number of assets");

    const double* r_ptr = static_cast<double*>(buf_r.ptr);
    const double* w_ptr = static_cast<double*>(buf_w.ptr);

    for (int i = 0; i < T * N; ++i) {
        if (!std::isfinite(r_ptr[i]))
            throw py::value_error("returns must contain only finite values");
    }
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(w_ptr[i]))
            throw py::value_error("weights must contain only finite values");
    }

    std::vector<double> result;
    {
        py::gil_scoped_release release;
        result = compute_portfolio_returns(r_ptr, T, N, w_ptr);
    }

    py::array_t<double> output(result.size());
    std::memcpy(output.mutable_data(), result.data(), result.size() * sizeof(double));
    return output;
}

py::array_t<double> py_simulate_bootstrap(
    py::array_t<double, py::array::c_style | py::array::forcecast> portfolio_returns,
    int num_scenarios, int horizon_days, unsigned int seed)
{
    auto buf = portfolio_returns.request();
    if (buf.ndim != 1)
        throw py::value_error("portfolio_returns must be a 1D array");

    int T = static_cast<int>(buf.shape[0]);

    if (T <= 0)
        throw py::value_error("portfolio_returns cannot be empty");
    if (num_scenarios <= 0)
        throw py::value_error("num_scenarios must be greater than zero");
    if (horizon_days <= 0)
        throw py::value_error("horizon_days must be greater than zero");

    const double* ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < T; ++i) {
        if (!std::isfinite(ptr[i]))
            throw py::value_error("portfolio_returns must contain only finite values");
    }

    std::vector<double> result;
    {
        py::gil_scoped_release release;
        result = simulate_bootstrap(ptr, T, num_scenarios, horizon_days, seed);
    }

    py::array_t<double> output(result.size());
    std::memcpy(output.mutable_data(), result.data(), result.size() * sizeof(double));
    return output;
}

PYBIND11_MODULE(quantforge_cpp, m) {
    m.doc() = "QuantForge C++ historical simulation engine";
    m.def("compute_portfolio_returns", &py_compute_portfolio_returns,
          "Compute daily portfolio returns from asset returns and weights");
    m.def("simulate_bootstrap", &py_simulate_bootstrap,
          "Run a bootstrap historical simulation of portfolio returns",
          py::arg("portfolio_returns"), py::arg("num_scenarios"),
          py::arg("horizon_days"), py::arg("seed") = 42);
}