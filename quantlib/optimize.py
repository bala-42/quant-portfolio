"""
Portfolio construction via convex optimization.

Every portfolio here is solved as a genuine convex program with cvxpy,
including the two that are easy to get away with hand-rolled
approximations:

- max_sharpe: naive approaches grid-search the efficient frontier and pick
  the best point. This is exact instead, via the Charnes-Cooper
  transformation, which turns Sharpe-ratio maximization into a QP.
- risk_parity: naive approaches use gradient descent toward equal risk
  contribution. This uses Spinu's (2013) convex reformulation instead,
  which cvxpy solves directly and exactly.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp


def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> tuple:
    """Return (expected_return, volatility, sharpe) for weights w."""
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Each asset's contribution to total portfolio volatility. Contributions
    sum to the portfolio volatility; for a risk-parity portfolio they
    should all be approximately equal.
    """
    port_vol = np.sqrt(w @ cov @ w)
    marginal_contrib = cov @ w / port_vol
    return w * marginal_contrib


def min_variance(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """Minimum-variance portfolio weights."""
    n = cov.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints)
    prob.solve()
    return w.value


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 40,
                        long_only: bool = True) -> tuple:
    """
    Trace the efficient frontier by solving a min-variance QP at each of
    `n_points` target return levels between the lowest and highest asset
    expected returns.

    Returns
    -------
    (vols, rets, weights_list) as parallel arrays/list
    """
    n = len(mu)
    targets = np.linspace(mu.min() * 0.9, mu.max() * 0.98, n_points)
    vols, rets, weights = [], [], []
    for target in targets:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, mu @ w >= target]
        if long_only:
            constraints.append(w >= 0)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints)
        prob.solve()
        if w.value is None:
            continue
        vols.append(float(np.sqrt(w.value @ cov @ w.value)))
        rets.append(float(mu @ w.value))
        weights.append(w.value)
    return np.array(vols), np.array(rets), weights


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0,
                long_only: bool = True) -> np.ndarray:
    """
    Exact maximum-Sharpe (tangency) portfolio via the Charnes-Cooper
    transformation, which converts the fractional Sharpe-maximization
    objective into a convex quadratic program:

        minimize    y' Sigma y
        subject to  (mu - rf)' y == 1
                     sum(y) == kappa
                     y >= 0  (kappa >= 0)   [if long_only]

    then w = y / kappa. This is the standard convex reformulation for
    long-only Sharpe maximization (see e.g. Cornuejols & Tutuncu,
    "Optimization Methods in Finance") and is exact, unlike grid-searching
    the frontier for the highest-Sharpe point.
    """
    n = len(mu)
    y = cp.Variable(n)
    kappa = cp.Variable(nonneg=True)
    excess = mu - rf
    constraints = [excess @ y == 1, cp.sum(y) == kappa]
    if long_only:
        constraints.append(y >= 0)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov))), constraints)
    prob.solve()
    if y.value is None or kappa.value is None or kappa.value <= 1e-9:
        raise RuntimeError("max_sharpe QP did not converge to a usable solution")
    return y.value / kappa.value


def risk_parity(cov: np.ndarray, budget: np.ndarray | None = None) -> np.ndarray:
    """
    Risk-parity portfolio via Spinu's (2013) convex reformulation:

        minimize   0.5 * y' Sigma y  -  sum_i budget_i * log(y_i)
        subject to y > 0

    then w = y / sum(y). At the optimum, y_i * (Sigma y)_i = budget_i for
    every i, which is exactly the equal (or budget-weighted) risk
    contribution condition — solved directly by a convex solver rather
    than approximated with iterative gradient descent.
    """
    n = cov.shape[0]
    if budget is None:
        budget = np.ones(n) / n
    y = cp.Variable(n, pos=True)
    objective = cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(cov)) - budget @ cp.log(y))
    prob = cp.Problem(objective)
    prob.solve()
    if y.value is None:
        raise RuntimeError("risk_parity QP did not converge to a usable solution")
    return y.value / y.value.sum()
