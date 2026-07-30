#pragma once
#include <vector>

/* Computes daily portfolio returns from an asset return matrix and weight vector.
 returns: T x N row-major flattened array (T days, N assets)
 weights: N-length array
 output: T-length array of portfolio returns*/
std::vector<double> compute_portfolio_returns(
    const double* returns, int T, int N, const double* weights);

// Runs a bootstrap historical simulation.
// portfolio_returns: T-length array of historical daily portfolio returns
// num_scenarios: number of Monte Carlo scenarios to generate
// horizon_days: number of days to simulate forward per scenario
// seed: RNG seed for reproducibility
// output: num_scenarios-length array of simulated cumulative returns
std::vector<double> simulate_bootstrap(
    const double* portfolio_returns, int T,
    int num_scenarios, int horizon_days, unsigned int seed);