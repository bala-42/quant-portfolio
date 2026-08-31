import numpy as np
import pandas as pd
import pytest

from quantlib.backtest import decile_backtest, decile_turnover, cointegration_test, pairs_backtest


def _make_synthetic_panel(n_dates=6, n_assets=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    assets = [f"A{i}" for i in range(n_assets)]
    # Signal: each asset has a fixed "quality" score plus noise each period.
    quality = rng.normal(0, 1, n_assets)
    signal = pd.DataFrame(
        quality[None, :] + rng.normal(0, 0.05, (n_dates, n_assets)),
        index=dates, columns=assets,
    )
    # Forward returns are driven by the SAME quality score, so top decile
    # should clearly outperform bottom decile.
    fwd_returns = pd.DataFrame(
        0.01 * quality[None, :] + rng.normal(0, 0.001, (n_dates, n_assets)),
        index=dates, columns=assets,
    )
    return signal, fwd_returns


def test_decile_backtest_top_beats_bottom():
    signal, fwd_returns = _make_synthetic_panel()
    result = decile_backtest(signal, fwd_returns, n_deciles=10, min_names=10)
    assert len(result) == len(signal)
    assert (result["long_short"] > 0).all()  # quality signal should always separate correctly here


def test_decile_backtest_respects_min_names():
    signal, fwd_returns = _make_synthetic_panel(n_assets=5)
    result = decile_backtest(signal, fwd_returns, n_deciles=10, min_names=50)
    assert len(result) == 0


def test_decile_turnover_is_between_0_and_1():
    signal, _ = _make_synthetic_panel(n_dates=8)
    turnover = decile_turnover(signal, n_deciles=10, min_names=10)
    assert 0.0 <= turnover <= 1.0


def test_cointegration_test_detects_cointegrated_series():
    rng = np.random.default_rng(0)
    n = 500
    b = np.cumsum(rng.normal(0, 1, n)) + 100  # random walk
    noise = rng.normal(0, 0.5, n)             # stationary noise
    a = 2.0 * b + noise                       # a is cointegrated with b, hedge ratio ~2
    dates = pd.date_range("2020-01-01", periods=n)
    result = cointegration_test(pd.Series(a, index=dates), pd.Series(b, index=dates))
    assert result["coint_pvalue"] < 0.05
    assert result["hedge_ratio"] == pytest.approx(2.0, abs=0.1)


def test_cointegration_test_rejects_independent_walks():
    rng = np.random.default_rng(1)
    n = 500
    a = np.cumsum(rng.normal(0, 1, n))
    b = np.cumsum(rng.normal(0, 1, n))  # independent random walk, not cointegrated
    dates = pd.date_range("2020-01-01", periods=n)
    result = cointegration_test(pd.Series(a, index=dates), pd.Series(b, index=dates))
    assert result["coint_pvalue"] > 0.05


def test_pairs_backtest_runs_and_returns_expected_keys():
    rng = np.random.default_rng(2)
    n = 300
    b = np.cumsum(rng.normal(0, 0.01, n)) + 5
    spread_noise = rng.normal(0, 0.02, n)
    a = 1.0 * b + spread_noise
    dates = pd.date_range("2020-01-01", periods=n)
    result = pairs_backtest(
        pd.Series(a, index=dates), pd.Series(b, index=dates),
        hedge_ratio=1.0, window=20, ann_factor=365,
    )
    for key in ["returns", "zscore", "position", "ann_return", "ann_vol",
                "sharpe", "max_drawdown", "n_position_changes", "pct_time_in_market"]:
        assert key in result
    assert 0.0 <= result["pct_time_in_market"] <= 1.0
