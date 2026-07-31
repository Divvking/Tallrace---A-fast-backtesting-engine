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

```
tallrace-sim/
├── cpp/
│   ├── simulation.h          # Core engine interface
│   ├── simulation.cpp        # Portfolio return calc + bootstrap simulation
│   └── bindings.cpp          # pybind11 wrapper: validation, GIL release, direct NumPy writes
├── python/
│   ├── test_run.py           # Correctness smoke test with VaR/CVaR reporting
│   ├── benchmark.py          # Performance comparison: naive vs vectorized vs C++, single config
│   └── benchmark_scaling.py  # Systematic sweep across scenario count, horizon, and asset count
├── tests/
│   └── test_simulation.py    # pytest suite: correctness, determinism, input validation
└── tallrace_cpp.pyd        # Compiled extension module (built via clang++/llvm-mingw; not tracked in git)
```


## Build

Compiled with `llvm-mingw` (clang), statically linked to avoid runtime DLL dependencies:

```bash
clang++ -O3 -Wall -shared -std=c++17 -static -static-libgcc -static-libstdc++ \
  -I<pybind11_include_path> -I<python_include_path> \
  -L<python_libs_path> -lpython313 \
  cpp/bindings.cpp cpp/simulation.cpp -o tallrace_cpp.pyd
```

Get your actual include/lib paths with:
```bash
python3 -c "import pybind11; print(pybind11.get_include())"
python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])"
```

**Build note:** static linking was necessary — the initial dynamically-linked build failed to import with a Windows `DLL load failed` error, traced via `llvm-objdump -p` to missing `libc++.dll`/`libunwind.dll` runtime dependencies not present on PATH. Static linking removed the external dependency entirely, producing a self-contained, portable module.

## Testing

```bash
pip install pytest --user
python3 -m pytest tests/test_simulation.py -v
```

13 tests covering: correctness against NumPy matrix multiplication, deterministic reproducibility under a fixed seed, a closed-form compounding check (constant historical returns must produce an exact analytical result, independent of randomness), and rejection of invalid inputs — empty arrays, non-positive scenario/horizon counts (both negative and zero, tested separately), mismatched weight dimensions, and non-finite NaN/Inf values in either returns or weights.

## Benchmark results

### Single configuration
50,000 scenarios, 10-day horizon, 3-asset portfolio, median of 7 timed runs after 2 discarded warm-up runs:

| Implementation | Median time | Speedup vs naive | Speedup vs vectorized |
|---|---|---|---|
| Naive pure Python (loop) | ~2.35s | 1x | — |
| Vectorized NumPy | ~0.013s | ~184x | 1x |
| **C++ (pybind11)** | **~0.004s** | **~620x** | **~3.4x** |

### Where C++ wins and where it doesn't

A systematic sweep across scenario counts, horizons, and asset counts, plus a controlled experiment with explicit AVX2/FMA compiler flags, gives a more complete picture than the single config above.

**Bootstrap simulation (`simulate_bootstrap`) — C++ wins consistently, 2.3x–3.1x across the full sweep:**

| scenarios | vectorized (s) | C++ (s) | speedup |
|---|---|---|---|
| 1,000 | 0.00048 | 0.00012 | 4.0x |
| 10,000 | 0.00160 | 0.00071 | 2.3x |
| 50,000 | 0.00975 | 0.00358 | 2.7x |
| 100,000 | 0.01911 | 0.00705 | 2.7x |
| 500,000 | 0.08939 | 0.03583 | 2.5x |

| horizon (days) | vectorized (s) | C++ (s) | speedup |
|---|---|---|---|
| 5 | 0.00548 | 0.00176 | 3.1x |
| 10 | 0.00945 | 0.00349 | 2.7x |
| 20 | 0.01795 | 0.00732 | 2.5x |
| 60 | 0.05057 | 0.02142 | 2.4x |
| 100 | 0.08230 | 0.03536 | 2.3x |

This holds because the C++ engine samples and compounds incrementally in O(S) memory, while NumPy's vectorized path must materialize an O(S×H) index array, sampled-returns array, and broadcast temporary. Confirmed empirically with `tracemalloc`: NumPy's peak Python-tracked memory for 50,000 scenarios/10-day horizon is ~10.2 MB vs ~0.39 MB for the C++ output — a ~26x difference. (Note: `tracemalloc` only sees Python-heap allocations; the C++ side's native buffer isn't visible to it, so this number mainly confirms NumPy's larger Python-side footprint rather than measuring C++ memory directly.)

**Portfolio-return computation (`compute_portfolio_returns`) — C++ loses to NumPy's matmul, and the gap widens with asset count:**

| assets | numpy matmul (s) | C++ (s) | speedup |
|---|---|---|---|
| 3 | 0.000004 | 0.000008 | 0.51x |
| 10 | 0.000006 | 0.000010 | 0.60x |
| 25 | 0.000007 | 0.000019 | 0.38x |
| 50 | 0.000009 | 0.000039 | 0.24x |
| 100 | 0.000014 | 0.000088 | 0.16x |

This is a straightforward accumulation loop competing against NumPy's `@`, which dispatches to BLAS — a library using cache-blocked, tiled matrix multiplication tuned per-CPU architecture, not just SIMD-widened arithmetic. To test whether this was simply a missing-vectorization issue, the engine was recompiled with `-mavx2 -mfma` explicitly enabled; the result was statistically unchanged (0.51x → 0.51x at 3 assets, 0.16x vs 0.17x at 100 assets), confirming the gap is architectural — BLAS restructures the computation itself via tiling, which compiler autovectorization flags on an untiled loop cannot replicate.

**Does this loss matter in practice? No — by roughly two orders of magnitude.** `compute_portfolio_returns` is a one-time O(T×N) preprocessing step; `simulate_bootstrap` is the O(S×H) step that actually dominates wall-clock time. At 100 assets, the "loss" costs 74 microseconds absolute. A single `simulate_bootstrap` call at even a modest 10,000 scenarios costs ~700 microseconds on its own — at a realistic 50,000 scenarios, `compute_portfolio_returns` is under 2.5% of total pipeline runtime. For the full pipeline (portfolio-return calc + simulation, 3 assets, 50,000 scenarios, 10-day horizon), total wall-clock time is ~0.0098s vectorized vs. ~0.0036s C++ — the engine is still **~2.7x faster end-to-end**, because the one place it loses is negligible next to the one place it wins.

**Takeaway:** hand-written C++ isn't categorically faster than optimized Python — it wins specifically where it avoids intermediate array allocation (the bootstrap simulation's incremental sampling), and loses to decades-optimized numerical libraries at operations those libraries were built for (matrix multiplication via BLAS). A production version of `compute_portfolio_returns` would call into BLAS directly (e.g. via Eigen or a raw BLAS binding) rather than hand-rolling the loop. Reporting both results, not just the favorable one, is more useful than it might first appear: it's evidence the benchmarking was done thoroughly rather than stopped at the first good number.

Because NumPy's `RandomState` and C++'s `std::mt19937` do not transform RNG output into bounded integers identically, the implementations are compared on **distributional statistics** (mean, std, percentiles) in `benchmark.py`, not on exact per-scenario equality — at 50,000 scenarios these statistics matched to 5 decimal places across all three implementations, which is itself a meaningful correctness signal on top of the unit tests.

## Engineering decisions worth noting

- **Input validation at the Python/C++ boundary**: rejects empty arrays, non-positive scenario/horizon counts, dimension mismatches, and non-finite (NaN/Inf) values before they reach the native engine — closing off a class of crash/undefined-behavior bugs (e.g. a negative scenario count converting to a huge unsigned allocation).
- **Fused validation in the compute loop**: `compute_portfolio_returns` checks each element for finiteness inside the same pass that does the weighted-sum accumulation, rather than a separate O(T×N) pre-pass — halves memory traffic over `returns` in the common case where the data is already valid.
- **Direct-write output, not copy-then-return**: both engine functions write straight into memory allocated by NumPy (via `output.mutable_data()`), rather than filling a `std::vector` and `memcpy`-ing it into a NumPy array afterward. Removes an extra O(N) allocation and a full copy on every call.
- **Conditional GIL release with a measured threshold, not a blanket release**: `simulate_bootstrap` always releases the GIL, since its O(S×H) work is reliably compute-bound at any realistic problem size. `compute_portfolio_returns` only releases above 50,000 total elements (T×N) — below that, the fixed cost of a GIL release/reacquire (mutex lock/unlock plus thread-state bookkeeping) can exceed whatever parallelism benefit it buys. This threshold is a heuristic based on typical release/reacquire overhead, not a measured crossover point for this specific workload — a workload sitting right at the boundary should be benchmarked directly rather than trusting the constant.
- **VaR/CVaR sign convention**: reported as positive loss figures (industry-standard convention), not raw negative percentile returns.

## Next steps

- [ ] Block bootstrap (resample contiguous historical windows, not individual days) for more realistic temporal structure
- [ ] OpenMP multithreading across scenarios (embarrassingly parallel — each scenario is independent; requires per-thread RNG streams to avoid contention)
- [ ] BLAS-backed (or Eigen-backed) `compute_portfolio_returns` to close the gap against NumPy's matmul at higher asset counts, rather than hand-rolling the accumulation loop
- [ ] Full parametric Monte Carlo variant (simulate from a fitted distribution) for comparison against the historical/bootstrap method
- [ ] VaR backtesting (Kupiec proportion-of-failures test, Christoffersen independence test) to validate model calibration, not just compute a number
- [ ] Portfolio-level mean-variance optimization layer feeding into this engine
