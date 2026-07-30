# Tallrace — High-Performance Historical Simulation Engine

A C++ historical/bootstrap simulation engine for portfolio risk estimation, exposed to Python via pybind11, benchmarked against naive and vectorized Python implementations.

## What it does

Given a historical matrix of daily asset returns and a set of portfolio weights, the engine:

1. Computes historical daily portfolio returns (weighted sum across assets)
2. Runs a **bootstrap historical simulation** — repeatedly resampling from actual historical daily returns (with replacement) to build thousands of simulated N-day forward paths
3. Derives **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** from the resulting distribution of simulated outcomes, reported as positive loss figures per standard risk-reporting convention

Unlike parametric VaR (which assumes a theoretical distribution, e.g. Gaussian), this is a **non-parametric, historical/bootstrap** approach — it makes no assumption about the shape of returns, only that history is a reasonable guide to plausible future scenarios.

**Known modeling limitation:** the bootstrap samples individual historical days independently (i.i.d.), which destroys temporal structure such as volatility clustering and crisis regimes. A block bootstrap (resampling contiguous historical windows) would preserve more realistic short-run dynamics — see Next Steps.

## Architecture

quantforge-sim/
├── cpp/
│ ├── simulation.h # Core engine interface
│ ├── simulation.cpp # Portfolio return calc + bootstrap simulation
│ └── bindings.cpp # pybind11 wrapper: validation, GIL release, NumPy I/O
├── python/
│ ├── test_run.py # Correctness smoke test with VaR/CVaR reporting
│ └── benchmark.py # Performance comparison: naive vs vectorized vs C++
├── tests/
│ └── test_simulation.py # pytest suite: correctness, determinism, input validation
└── quantforge_cpp.pyd # Compiled extension module (built via clang++/llvm-mingw)

## Build

Compiled with `llvm-mingw` (clang), statically linked to avoid runtime DLL dependencies:

```bash
clang++ -O3 -Wall -shared -std=c++17 -static -static-libgcc -static-libstdc++ \
  -I<pybind11_include_path> -I<python_include_path> \
  -L<python_libs_path> -lpython313 \
  cpp/bindings.cpp cpp/simulation.cpp -o quantforge_cpp.pyd
```

**Build note:** static linking was necessary — the initial dynamically-linked build failed to import with a Windows `DLL load failed` error, traced via `llvm-objdump -p` to missing `libc++.dll`/`libunwind.dll` runtime dependencies not present on PATH. Static linking removed the external dependency entirely, producing a self-contained, portable module.

## Testing

```bash
pip install pytest --user
python3 -m pytest tests/test_simulation.py -v
```

8 tests covering: correctness against NumPy matrix multiplication, deterministic reproducibility under a fixed seed, a closed-form compounding check (constant historical returns must produce an exact analytical result, independent of randomness), and rejection of invalid inputs (empty arrays, non-positive scenario/horizon counts, mismatched weight dimensions, and non-finite NaN/Inf values in returns or weights).

## Benchmark results

50,000 scenarios, 10-day horizon, 3-asset portfolio, median of 7 timed runs after 2 discarded warm-up runs:

| Implementation | Median time | Speedup vs naive | Speedup vs vectorized |
|---|---|---|---|
| Naive pure Python (loop) | ~2.35s | 1x | — |
| Vectorized NumPy | ~0.013s | ~184x | 1x |
| **C++ (pybind11)** | **~0.004s** | **~620x** | **~3.4x** |

### Why C++ beats vectorized NumPy specifically

The ~620x figure vs naive Python is expected — any compiled code beats a pure Python loop. The more interesting result is the **~3.4x speedup over vectorized NumPy**, since vectorized NumPy is what a competent developer would actually write as a baseline.

The gap comes down to **memory allocation, not raw arithmetic speed**: the vectorized NumPy path allocates a full `(num_scenarios × horizon_days)` index array, a same-sized sampled-returns array, and a broadcast temporary for `1.0 + sampled` — O(S×H) memory traffic. The C++ implementation only stores O(S) outputs, sampling and compounding incrementally with no intermediate allocation. At this problem size, avoiding memory traffic matters more than SIMD width.

Because NumPy's `RandomState` and C++'s `std::mt19937` do not transform RNG output into bounded integers identically, the three implementations are compared on **distributional statistics** (mean, std, percentiles) in `benchmark.py`, not on exact per-scenario equality.

## Engineering decisions worth noting

- **Input validation at the Python/C++ boundary**: rejects empty arrays, non-positive scenario/horizon counts, dimension mismatches, and non-finite (NaN/Inf) values before they reach the native engine — closing off a class of crash/undefined-behavior bugs (e.g. a negative scenario count converting to a huge unsigned allocation).
- **GIL released during native computation**: `py::gil_scoped_release` wraps only the C++ simulation call, allowing other Python threads to run during the compute-bound section, then reacquiring the GIL before constructing NumPy output.
- **Explicit NumPy output ownership**: results are copied into a Python-owned `py::array_t` via `memcpy` rather than relying on implicit pybind11 array-constructor semantics, making memory ownership unambiguous for future refactoring.
- **VaR/CVaR sign convention**: reported as positive loss figures (industry-standard convention), not raw negative percentile returns.

## Next steps

- [ ] Block bootstrap (resample contiguous historical windows, not individual days) for more realistic temporal structure
- [ ] OpenMP multithreading across scenarios (embarrassingly parallel — each scenario is independent; requires per-thread RNG streams to avoid contention)
- [ ] Direct-write output (native function writes straight into NumPy-allocated memory, eliminating the intermediate `std::vector` + `memcpy`)
- [ ] Full parametric Monte Carlo variant (simulate from a fitted distribution) for comparison against the historical/bootstrap method
- [ ] VaR backtesting (Kupiec proportion-of-failures test, Christoffersen independence test) to validate model calibration, not just compute a number
- [ ] Portfolio-level mean-variance optimization layer feeding into this engine
