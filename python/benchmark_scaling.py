
import sys, os, time, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import quantforge_cpp as qf


def measure(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"median": float(np.median(times)), "min": float(np.min(times))}


def vectorized_numpy_bootstrap(portfolio_returns, num_scenarios, horizon_days, seed):
    rng = np.random.RandomState(seed)
    T = len(portfolio_returns)
    idx = rng.randint(0, T, size=(num_scenarios, horizon_days))
    sampled = portfolio_returns[idx]
    return np.prod(1.0 + sampled, axis=1) - 1.0


def peak_memory_bytes(fn):
    """Peak *Python-allocated* memory during fn(). Does not see memory
    allocated inside the C++ extension (which allocates natively), so
    this under-counts the C++ side's true peak — but that's the point:
    it isolates how much memory traffic NumPy's own path generates."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak

rng = np.random.default_rng(2026)
BASE_RETURNS_500x3 = rng.normal(0.0005, 0.01, size=(500, 3))
BASE_WEIGHTS_3 = np.array([0.4, 0.35, 0.25])
port_returns_3asset = qf.compute_portfolio_returns(BASE_RETURNS_500x3, BASE_WEIGHTS_3)

SEED = 42

print("=" * 72)
print("1. SCENARIO-COUNT SWEEP (horizon=10 days, 3 assets)")
print("=" * 72)
print(f"{'scenarios':>10} | {'vectorized (s)':>15} | {'C++ (s)':>10} | {'speedup':>8}")
print("-" * 72)

for n_scenarios in [1_000, 10_000, 50_000, 100_000, 500_000]:
    vec = measure(lambda: vectorized_numpy_bootstrap(port_returns_3asset, n_scenarios, 10, SEED))
    cpp = measure(lambda: qf.simulate_bootstrap(port_returns_3asset, num_scenarios=n_scenarios, horizon_days=10, seed=SEED))
    speedup = vec["median"] / cpp["median"]
    print(f"{n_scenarios:>10,} | {vec['median']:>15.5f} | {cpp['median']:>10.5f} | {speedup:>7.1f}x")

print()
print("=" * 72)
print("2. HORIZON SWEEP (scenarios=50,000, 3 assets)")
print("=" * 72)
print(f"{'horizon_days':>12} | {'vectorized (s)':>15} | {'C++ (s)':>10} | {'speedup':>8}")
print("-" * 72)

for horizon in [5, 10, 20, 60, 100]:
    vec = measure(lambda: vectorized_numpy_bootstrap(port_returns_3asset, 50_000, horizon, SEED))
    cpp = measure(lambda: qf.simulate_bootstrap(port_returns_3asset, num_scenarios=50_000, horizon_days=horizon, seed=SEED))
    speedup = vec["median"] / cpp["median"]
    print(f"{horizon:>12} | {vec['median']:>15.5f} | {cpp['median']:>10.5f} | {speedup:>7.1f}x")

print()
print("=" * 72)
print("3. ASSET-COUNT SWEEP for compute_portfolio_returns (T=500 days)")
print("=" * 72)
print(f"{'assets':>8} | {'numpy matmul (s)':>17} | {'C++ (s)':>10} | {'speedup':>8}")
print("-" * 72)

for n_assets in [3, 10, 25, 50, 100]:
    returns_matrix = rng.normal(0.0005, 0.01, size=(500, n_assets))
    weights = np.full(n_assets, 1.0 / n_assets)

    npy = measure(lambda: returns_matrix @ weights)
    cpp = measure(lambda: qf.compute_portfolio_returns(returns_matrix, weights))
    speedup = npy["median"] / cpp["median"]
    print(f"{n_assets:>8} | {npy['median']:>17.6f} | {cpp['median']:>10.6f} | {speedup:>7.2f}x")


print()
print("=" * 72)
print("4. PEAK PYTHON-SIDE MEMORY (50,000 scenarios, 10-day horizon)")
print("=" * 72)

vec_mem = peak_memory_bytes(lambda: vectorized_numpy_bootstrap(port_returns_3asset, 50_000, 10, SEED))
cpp_mem = peak_memory_bytes(lambda: qf.simulate_bootstrap(port_returns_3asset, num_scenarios=50_000, horizon_days=10, seed=SEED))

print(f"Vectorized NumPy peak (Python-tracked): {vec_mem / 1024:.1f} KB")
print(f"C++ output peak (Python-tracked):       {cpp_mem / 1024:.1f} KB")
print("Note: tracemalloc only sees Python-heap allocations. The C++ side's")
print("intermediate O(S) std::vector is allocated natively and won't show")
print("up here — this number mainly confirms NumPy's larger Python-side")
print("footprint (index array + sampled array + broadcast temporary).")

print()
print("Done.")