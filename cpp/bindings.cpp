#include <cstring>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "simulation.h"

namespace py = pybind11;

// Below this many total elements (T*N), the fixed cost of a GIL
// release/reacquire (a mutex lock/unlock plus thread-state juggling)
// tends to exceed whatever parallelism it would buy — at microsecond-scale
// workloads, releasing the GIL can slow things down rather than speed
// them up. Above it, the compute is large enough that letting other
// Python threads run during the native call is worth the fixed cost.
// This is a heuristic, not a measured crossover point — if you have a
// workload that lives right around this boundary, benchmark it directly
// rather than trusting the constant.
constexpr long long GIL_RELEASE_ELEMENT_THRESHOLD = 50'000;

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

    // Weights are only N elements (small), so a dedicated pre-pass here is
    // cheap. The expensive T*N validation over `returns` is fused into the
    // compute loop in simulation.cpp instead — see compute_portfolio_returns.
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(w_ptr[i]))
            throw py::value_error("weights must contain only finite values");
    }

    // Allocate the output array and grab its raw pointer up front, before any
    // GIL release, so the native call can write straight into NumPy-owned
    // memory. This replaces the previous std::vector<double> + memcpy, which
    // paid for an extra T-length allocation and a full copy on every call.
    py::array_t<double> output(T);
    double* out_ptr = output.mutable_data();

    bool ok;
    const long long total_elements = static_cast<long long>(T) * N;
    if (total_elements > GIL_RELEASE_ELEMENT_THRESHOLD) {
        py::gil_scoped_release release;
        ok = compute_portfolio_returns(r_ptr, T, N, w_ptr, out_ptr);
    } else {
        // Below threshold: skip the release entirely. At this size the
        // compute is on the order of the GIL lock/unlock cost itself.
        ok = compute_portfolio_returns(r_ptr, T, N, w_ptr, out_ptr);
    }

    if (!ok)
        throw py::value_error("returns must contain only finite values");

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

    // Same direct-write pattern as above: allocate + get pointer before
    // release, write straight into NumPy's buffer inside the native call.
    py::array_t<double> output(num_scenarios);
    double* out_ptr = output.mutable_data();

    {
        // Always released here (unlike compute_portfolio_returns above) —
        // this loop does num_scenarios * horizon_days work, which is
        // reliably compute-bound at any realistic problem size, so the
        // GIL release consistently pays for itself.
        py::gil_scoped_release release;
        simulate_bootstrap(ptr, T, num_scenarios, horizon_days, seed, out_ptr);
    }

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