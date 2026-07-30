#include "simulation.h"
#include <random>

std::vector<double> compute_portfolio_returns(
    const double* returns, int T, int N, const double* weights)
{
    std::vector<double> portfolio_returns(T, 0.0);
    for (int t = 0; t < T; ++t) {
        double sum = 0.0;
        const double* row = returns + static_cast<size_t>(t) * N;
        for (int n = 0; n < N; ++n) {
            sum += row[n] * weights[n];
        }
        portfolio_returns[t] = sum;
    }
    return portfolio_returns;
}

std::vector<double> simulate_bootstrap(
    const double* portfolio_returns, int T,
    int num_scenarios, int horizon_days, unsigned int seed)
{
    std::vector<double> results(num_scenarios);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, T - 1);

    for (int s = 0; s < num_scenarios; ++s) {
        double cumulative = 1.0;
        for (int d = 0; d < horizon_days; ++d) {
            int idx = dist(rng);
            cumulative *= (1.0 + portfolio_returns[idx]);
        }
        results[s] = cumulative - 1.0;
    }
    return results;
}