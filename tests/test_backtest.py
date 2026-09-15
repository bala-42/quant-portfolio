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


# ---------------------------------------------------------------------------
# Regression tests for the pairs-backtest economics fixes.
# ---------------------------------------------------------------------------

def _synthetic_pair(beta=3.0, n=400, seed=11):
    """A cointegrated log-price pair with a mean-reverting spread."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    log_b = pd.Series(np.cumsum(rng.normal(0.0005, 0.02, n)) + 5.0, index=idx)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.90 * spread[i - 1] + rng.normal(0, 0.05)
    log_a = pd.Series(beta * log_b.to_numpy() + spread + 2.0, index=idx)
    return log_a, log_b, beta


def test_pairs_backtest_returns_are_on_gross_notional_and_use_simple_returns():
    log_a, log_b, beta = _synthetic_pair()
    res = pairs_backtest(log_a, log_b, beta, window=30, cost_bps=0.0)
    assert (res["position"] != 0).any(), "fixture never trades; test proves nothing"

    gross = 1.0 + abs(beta)
    ret_a = np.expm1(log_a.diff())
    ret_b = np.expm1(log_b.diff())
    expected = (res["position"].shift(1) * (ret_a - beta * ret_b) / gross).dropna()

    pd.testing.assert_series_equal(res["returns"], expected, check_names=False)


def test_pairs_backtest_does_not_compound_log_returns_as_simple_ones():
    log_a, log_b, beta = _synthetic_pair()
    res = pairs_backtest(log_a, log_b, beta, window=30, cost_bps=0.0)

    gross = 1.0 + abs(beta)
    log_version = (
        res["position"].shift(1) * (log_a.diff() - beta * log_b.diff()) / gross
    ).dropna()

    # The pre-fix engine produced exactly `log_version`, then handed it to
    # perf_stats, which compounds with (1 + r).prod().
    assert not np.allclose(res["returns"].to_numpy(), log_version.to_numpy())


def test_pairs_backtest_cost_scales_with_the_hedge_ratio_not_by_two():
    log_a, log_b, beta = _synthetic_pair()
    free = pairs_backtest(log_a, log_b, beta, window=30, cost_bps=0.0)
    charged = pairs_backtest(log_a, log_b, beta, window=30, cost_bps=100.0)

    drag = float((free["returns"] - charged["returns"]).sum())
    turnover = float(
        free["position"].diff().abs().fillna(0).reindex(free["returns"].index).fillna(0).sum()
    )

    # Returns are per unit of gross notional, so a round trip costs cost_bps
    # on the position change -- not twice that, which is what `* 2` assumed
    # and which is only right when the hedge ratio happens to be 1.
    assert drag == pytest.approx(turnover * 100 / 1e4, rel=1e-9)


def test_pairs_backtest_records_the_sharpe_convention():
    log_a, log_b, beta = _synthetic_pair()
    res = pairs_backtest(log_a, log_b, beta, window=30)
    assert res["sharpe_convention"] == "arithmetic"
