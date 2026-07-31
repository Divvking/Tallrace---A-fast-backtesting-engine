#include "simulation.h"
#include <random>
#include <cmath>

bool compute_portfolio_returns(
    const double* returns, int T, int N, const double* weights,
    double* output)
{
    for (int t = 0; t < T; ++t) {
        double sum = 0.0;
        const double* row = returns + static_cast<size_t>(t) * N;
        for (int n = 0; n < N; ++n) {
            double v = row[n];
            // Fused into the accumulation loop instead of a separate T*N
            // pre-pass — halves the memory traffic over `returns` for the
            // common case where the data is valid (the overwhelming majority
            // of calls). Bails out on the first bad value rather than
            // continuing to accumulate into a result that will be discarded.
            if (!std::isfinite(v)) {
                return false;
            }
            sum += v * weights[n];
        }
        output[t] = sum;
    }
    return true;
}

void simulate_bootstrap(
    const double* portfolio_returns, int T,
    int num_scenarios, int horizon_days, unsigned int seed,
    double* output)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, T - 1);

    for (int s = 0; s < num_scenarios; ++s) {
        double cumulative = 1.0;
        for (int d = 0; d < horizon_days; ++d) {
            int idx = dist(rng);
            cumulative *= (1.0 + portfolio_returns[idx]);
        }
        output[s] = cumulative - 1.0;
    }
}