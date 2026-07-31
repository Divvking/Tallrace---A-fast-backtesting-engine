#pragma once
#include <vector>

// Computes daily portfolio returns from an asset return matrix and weight vector,
// writing results directly into a caller-provided buffer (avoids the intermediate
// std::vector allocation + memcpy the previous version used).
//
// Validates `returns` for finiteness in the SAME pass as the compute loop, rather
// than a separate O(T*N) pre-pass over the data. Returns false and stops at the
// first non-finite value found; in that case `output` contents are undefined and
// must not be used by the caller.
//
// returns: T x N row-major flattened array (T days, N assets)
// weights: N-length array (validated separately by the caller — N is small,
//          so a dedicated pre-pass over just the weights is cheap)
// output:  caller-owned T-length buffer to write into
// Returns: true on success, false if a non-finite value was found in `returns`
bool compute_portfolio_returns(
    const double* returns, int T, int N, const double* weights,
    double* output);

// Runs a bootstrap historical simulation, writing results directly into a
// caller-provided buffer (avoids the intermediate std::vector allocation +
// memcpy the previous version used).
//
// portfolio_returns: T-length array of historical daily portfolio returns
// num_scenarios: number of Monte Carlo scenarios to generate
// horizon_days: number of days to simulate forward per scenario
// seed: RNG seed for reproducibility
// output: caller-owned num_scenarios-length buffer to write into
void simulate_bootstrap(
    const double* portfolio_returns, int T,
    int num_scenarios, int horizon_days, unsigned int seed,
    double* output);