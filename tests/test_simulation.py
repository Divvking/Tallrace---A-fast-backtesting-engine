import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import quantforge_cpp as qf


def test_portfolio_returns_matches_numpy():
    returns = np.array([
        [0.01, 0.02],
        [-0.01, 0.03],
        [0.005, -0.005],
    ])
    weights = np.array([0.6, 0.4])

    actual = qf.compute_portfolio_returns(returns, weights)
    expected = returns @ weights

    np.testing.assert_allclose(actual, expected)


def test_same_seed_produces_same_output():
    returns = np.array([0.01, -0.02, 0.005, 0.01, -0.01])

    first = qf.simulate_bootstrap(returns, num_scenarios=1000, horizon_days=10, seed=42)
    second = qf.simulate_bootstrap(returns, num_scenarios=1000, horizon_days=10, seed=42)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_change_output():
    returns = np.array([0.01, -0.02, 0.005, 0.01, -0.01])

    first = qf.simulate_bootstrap(returns, num_scenarios=1000, horizon_days=10, seed=1)
    second = qf.simulate_bootstrap(returns, num_scenarios=1000, horizon_days=10, seed=2)

    assert not np.array_equal(first, second)


def test_constant_returns_gives_closed_form_result():
    # If every historical day has the same return r, every bootstrap scenario
    # must compound to exactly (1+r)^horizon - 1, regardless of which days
    # get sampled. This removes randomness as a variable and gives a hard
    # correctness check on the compounding logic itself.
    r = 0.01
    horizon = 10
    returns = np.full(50, r)

    sim = qf.simulate_bootstrap(returns, num_scenarios=500, horizon_days=horizon, seed=7)
    expected = (1 + r) ** horizon - 1

    np.testing.assert_allclose(sim, expected, rtol=1e-10)


def test_empty_returns_raises():
    with pytest.raises(ValueError):
        qf.simulate_bootstrap(np.array([]), num_scenarios=100, horizon_days=10, seed=1)


def test_negative_num_scenarios_raises():
    returns = np.array([0.01, -0.02, 0.005])
    with pytest.raises(ValueError):
        qf.simulate_bootstrap(returns, num_scenarios=-10, horizon_days=10, seed=1)


def test_negative_horizon_raises():
    returns = np.array([0.01, -0.02, 0.005])
    with pytest.raises(ValueError):
        qf.simulate_bootstrap(returns, num_scenarios=100, horizon_days=-5, seed=1)


def test_zero_num_scenarios_raises():
    returns = np.array([0.01, -0.02, 0.005])
    with pytest.raises(ValueError):
        qf.simulate_bootstrap(returns, num_scenarios=0, horizon_days=10, seed=1)


def test_zero_horizon_raises():
    returns = np.array([0.01, -0.02, 0.005])
    with pytest.raises(ValueError):
        qf.simulate_bootstrap(returns, num_scenarios=100, horizon_days=0, seed=1)


def test_mismatched_weights_raises():
    returns = np.array([[0.01, 0.02, 0.03], [0.01, -0.01, 0.02]])
    weights = np.array([0.5, 0.5])  # wrong length — 3 assets, 2 weights
    with pytest.raises(ValueError):
        qf.compute_portfolio_returns(returns, weights)


def test_nan_in_returns_raises():
    returns = np.array([0.01, np.nan, -0.02])
    with pytest.raises(ValueError):
        qf.simulate_bootstrap(returns, num_scenarios=100, horizon_days=10, seed=1)


def test_inf_in_returns_raises():
    returns = np.array([0.01, np.inf, -0.02])
    with pytest.raises(ValueError):
        qf.simulate_bootstrap(returns, num_scenarios=100, horizon_days=10, seed=1)


def test_nan_in_weights_raises():
    returns = np.array([[0.01, 0.02], [-0.01, 0.03]])
    weights = np.array([0.5, np.nan])
    with pytest.raises(ValueError):
        qf.compute_portfolio_returns(returns, weights)