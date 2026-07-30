import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import quantforge_cpp as qf


def measure(fn, repeats=7):
    for _ in range(2):
        fn()  # warm-up, discarded
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"median": float(np.median(times)), "min": float(np.min(times))}


def naive_python_bootstrap(portfolio_returns, num_scenarios, horizon_days, seed):
    rng = np.random.RandomState(seed)
    T = len(portfolio_returns)
    results = []
    for _ in range(num_scenarios):
        cumulative = 1.0
        for _ in range(horizon_days):
            idx = rng.randint(0, T)
            cumulative *= (1.0 + portfolio_returns[idx])
        results.append(cumulative - 1.0)
    return np.array(results)


def vectorized_numpy_bootstrap(portfolio_returns, num_scenarios, horizon_days, seed):
    rng = np.random.RandomState(seed)
    T = len(portfolio_returns)
    idx = rng.randint(0, T, size=(num_scenarios, horizon_days))
    sampled = portfolio_returns[idx]
    cumulative = np.prod(1.0 + sampled, axis=1) - 1.0
    return cumulative


def summary(x):
    pcts = np.percentile(x, [1, 5, 50, 95, 99])
    return f"mean={np.mean(x):.5f} std={np.std(x):.5f} p1={pcts[0]:.5f} p5={pcts[1]:.5f} p50={pcts[2]:.5f} p95={pcts[3]:.5f} p99={pcts[4]:.5f}"


rng = np.random.default_rng(2026)
returns = rng.normal(0.0005, 0.01, size=(500, 3))
weights = np.array([0.4, 0.35, 0.25])
port_returns = qf.compute_portfolio_returns(returns, weights)

NUM_SCENARIOS = 50000
HORIZON = 10

print(f"Running {NUM_SCENARIOS} scenarios, {HORIZON}-day horizon (median of 7 runs, 2 warm-up)...\n")

naive_stats = measure(lambda: naive_python_bootstrap(port_returns, NUM_SCENARIOS, HORIZON, seed=42))
print(f"Naive pure Python:    median={naive_stats['median']:.4f}s  min={naive_stats['min']:.4f}s")

vec_stats = measure(lambda: vectorized_numpy_bootstrap(port_returns, NUM_SCENARIOS, HORIZON, seed=42))
print(f"Vectorized NumPy:     median={vec_stats['median']:.4f}s  min={vec_stats['min']:.4f}s")

cpp_stats = measure(lambda: qf.simulate_bootstrap(port_returns, num_scenarios=NUM_SCENARIOS, horizon_days=HORIZON, seed=42))
print(f"C++ (pybind11):       median={cpp_stats['median']:.4f}s  min={cpp_stats['min']:.4f}s")

print(f"\nSpeedup vs naive Python (median):    {naive_stats['median'] / cpp_stats['median']:.1f}x")
print(f"Speedup vs vectorized NumPy (median): {vec_stats['median'] / cpp_stats['median']:.1f}x")

print("\n--- Distributional comparison (different RNGs, same seed will NOT match exactly) ---")
naive_result = naive_python_bootstrap(port_returns, NUM_SCENARIOS, HORIZON, seed=42)
vec_result = vectorized_numpy_bootstrap(port_returns, NUM_SCENARIOS, HORIZON, seed=42)
cpp_result = qf.simulate_bootstrap(port_returns, num_scenarios=NUM_SCENARIOS, horizon_days=HORIZON, seed=42)

print("Naive:      ", summary(naive_result))
print("Vectorized: ", summary(vec_result))
print("C++:        ", summary(cpp_result))