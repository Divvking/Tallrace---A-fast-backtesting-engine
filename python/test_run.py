import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import quantforge_cpp as qf

returns = np.random.normal(0.0005, 0.01, size=(500, 3))
weights = np.array([0.4, 0.35, 0.25])

port_returns = qf.compute_portfolio_returns(returns, weights)
print('portfolio returns shape:', port_returns.shape)

sim = qf.simulate_bootstrap(port_returns, num_scenarios=10000, horizon_days=10, seed=42)
print('simulated scenarios:', sim.shape)

q = np.percentile(sim, 5)
var_95 = -q
cvar_95 = -sim[sim <= q].mean()

print(f'5th-percentile return: {q:.4%}')
print(f'VaR at 95% confidence:  {var_95:.4%}')
print(f'Expected shortfall (CVaR): {cvar_95:.4%}')