"""
quantlib: shared research utilities for the quant-portfolio projects.

Each of the four project notebooks imports from this package instead of
re-implementing its own version of "compute Sharpe" or "run a decile
backtest". Keeping the math in one tested place means a fix or improvement
made here propagates to every project automatically.

Modules
-------
stats     : performance statistics and bootstrap confidence intervals
backtest  : cross-sectional decile backtests, pairs-trading backtests,
            cointegration testing
optimize  : efficient frontier, tangency portfolio, and risk parity via
            convex optimization (cvxpy)
"""
from . import stats, backtest, optimize

__all__ = ["stats", "backtest", "optimize"]
