import numpy as np
import pytest

from quantlib.optimize import (
    min_variance, efficient_frontier, max_sharpe, risk_parity,
    portfolio_stats, risk_contributions,
)


def test_min_variance_diagonal_cov_is_inverse_variance():
    # For a diagonal covariance, the closed-form min-variance weights are
    # proportional to 1/variance.
    variances = np.array([0.04, 0.01, 0.09])
    cov = np.diag(variances)
    w = min_variance(cov)
    expected = (1 / variances) / (1 / variances).sum()
    np.testing.assert_allclose(w, expected, atol=1e-4)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-8).all()


def test_efficient_frontier_is_increasing_in_risk():
    mu = np.array([0.05, 0.10, 0.15])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    vols, rets, weights = efficient_frontier(mu, cov, n_points=15)
    assert len(vols) > 5
    # higher target return should never require less risk on an efficient frontier
    assert np.all(np.diff(vols) >= -1e-6)


def test_max_sharpe_beats_all_frontier_points():
    mu = np.array([0.05, 0.10, 0.15])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    rf = 0.02
    w_tan = max_sharpe(mu, cov, rf=rf)
    _, _, sharpe_tan = portfolio_stats(w_tan, mu, cov, rf=rf)

    vols, rets, weights = efficient_frontier(mu, cov, n_points=60)
    frontier_sharpes = [(r - rf) / v for r, v in zip(rets, vols) if v > 0]

    assert w_tan.sum() == pytest.approx(1.0, abs=1e-4)
    assert (w_tan >= -1e-6).all()
    # the exact tangency Sharpe should be at least as good as any frontier grid point
    assert sharpe_tan >= max(frontier_sharpes) - 1e-4


def test_risk_parity_equalizes_risk_contributions():
    cov = np.array([[0.09, 0.02, 0.01],
                     [0.02, 0.04, 0.015],
                     [0.01, 0.015, 0.16]])
    w = risk_parity(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    assert (w >= -1e-6).all()

    rc = risk_contributions(w, cov)
    # all three assets should contribute (roughly) equally to portfolio risk
    rc_pct = rc / rc.sum()
    np.testing.assert_allclose(rc_pct, [1 / 3] * 3, atol=1e-3)


def test_risk_parity_respects_custom_budget():
    cov = np.eye(2) * 0.04
    budget = np.array([0.8, 0.2])
    w = risk_parity(cov, budget=budget)
    rc = risk_contributions(w, cov)
    rc_pct = rc / rc.sum()
    np.testing.assert_allclose(rc_pct, budget, atol=1e-3)


def test_efficient_frontier_handles_all_negative_expected_returns():
    # Pre-fix the target grid was built by scaling mu's endpoints, which only
    # moves them in the intended direction when they are positive. With a
    # negative maximum, mu.max() * 0.98 sits ABOVE the highest attainable
    # return and those QPs are infeasible, so the sweep silently returned
    # fewer points than asked for.
    mu = np.array([-0.05, -0.02, -0.10])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    vols, rets, weights = efficient_frontier(mu, cov, n_points=15)
    assert len(vols) == 15
    assert np.all(np.diff(vols) >= -1e-6)


def test_efficient_frontier_degenerate_when_all_returns_equal():
    mu = np.array([0.07, 0.07, 0.07])
    cov = np.diag([0.04, 0.01, 0.09])
    vols, rets, weights = efficient_frontier(mu, cov, n_points=10)
    assert len(vols) >= 1
