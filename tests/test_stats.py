import numpy as np
import pandas as pd
import pytest

from quantlib.stats import perf_stats, max_drawdown, bootstrap_ci, sharpe_ci


def test_perf_stats_constant_return():
    # A constant monthly return of 1% should annualize to (1.01)^12 - 1
    r = pd.Series([0.01] * 24)
    stats = perf_stats(r, freq=12)
    assert stats["ann_return"] == pytest.approx(1.01 ** 12 - 1, rel=1e-9)
    assert stats["ann_vol"] == pytest.approx(0.0, abs=1e-12)
    assert stats["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert stats["hit_rate"] == 1.0


def test_perf_stats_with_name():
    r = pd.Series([0.01, -0.02, 0.03])
    stats = perf_stats(r, freq=12, name="my_strategy")
    assert stats["name"] == "my_strategy"


def test_perf_stats_empty_raises():
    with pytest.raises(ValueError):
        perf_stats(pd.Series(dtype=float), freq=12)


def test_max_drawdown_known_path():
    # Growth path: 1 -> 1.10 -> 0.99 -> 1.05  => returns are 10%, -10%, +6.06...%
    prices = pd.Series([1.0, 1.10, 0.99, 1.05])
    returns = prices.pct_change().dropna()
    dd = max_drawdown(returns)
    # trough is 0.99 vs peak 1.10 -> drawdown = 0.99/1.10 - 1
    assert dd == pytest.approx(0.99 / 1.10 - 1, rel=1e-9)


def test_bootstrap_ci_contains_point_estimate_direction():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.05, 200))
    result = bootstrap_ci(r, lambda s: s.mean(), n_boot=500, seed=1)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]
    assert result["ci_level"] == 0.90
    assert result["n_boot"] == 500


def test_bootstrap_ci_too_short_raises():
    with pytest.raises(ValueError):
        bootstrap_ci(pd.Series([0.01, 0.02]), lambda s: s.mean())


def test_sharpe_ci_wraps_perf_stats():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.02, 0.04, 100))
    result = sharpe_ci(r, freq=12, n_boot=300, seed=2)
    point_sharpe = perf_stats(r, freq=12)["sharpe"]
    assert result["point_estimate"] == pytest.approx(point_sharpe, rel=1e-9)


# ---------------------------------------------------------------------------
# Regression tests for the convention / silent-NaN fixes.
# Each of these fails against the pre-fix implementation.
# ---------------------------------------------------------------------------

def test_perf_stats_records_the_convention_it_used():
    r = pd.Series([0.01, -0.02, 0.03, 0.01] * 10)
    assert perf_stats(r, freq=12)["sharpe_convention"] == "arithmetic"
    assert perf_stats(r, freq=12, sharpe_convention="geometric")["sharpe_convention"] == "geometric"


def test_perf_stats_rejects_an_unknown_convention():
    r = pd.Series([0.01, -0.02, 0.03])
    with pytest.raises(ValueError, match="sharpe_convention"):
        perf_stats(r, freq=12, sharpe_convention="harmonic")


def test_conventions_diverge_wildly_on_a_short_window_with_large_moves():
    # Three periods of very large gains. The geometric numerator raises the
    # window's cumulative return to the power freq/len(r) and reports a year
    # that did not happen; the arithmetic one does not. This is the whole
    # reason the convention is a parameter.
    r = pd.Series([0.50, 0.40, 0.60])
    arith = perf_stats(r, freq=252, sharpe_convention="arithmetic")["sharpe"]
    geo = perf_stats(r, freq=252, sharpe_convention="geometric")["sharpe"]
    assert geo > 100 * arith


def test_on_a_long_sample_with_modest_returns_the_conventions_stay_close():
    # Counterpart to the test above. With realistic per-period returns over a
    # long sample the two conventions land within ~8% of each other, the
    # geometric one slightly lower because compounding a volatile series
    # gives up roughly sigma^2 / 2 per period against the arithmetic mean.
    #
    # The mean is set exactly rather than drawn, because on a sample whose
    # realized mean lands near zero the Sharpe lands near zero too and a
    # relative comparison between the two conventions says nothing at all.
    rng = np.random.default_rng(7)
    x = rng.normal(0, 0.01, 2000)
    r = pd.Series(x - x.mean() + 0.0004)   # 10%/yr arithmetic, 1%/day vol

    arith = perf_stats(r, freq=252, sharpe_convention="arithmetic")["sharpe"]
    geo = perf_stats(r, freq=252, sharpe_convention="geometric")["sharpe"]

    assert geo < arith
    assert abs(geo - arith) / abs(arith) < 0.15


def test_perf_stats_raises_on_a_return_at_or_below_minus_one():
    # Pre-fix this returned a silent NaN: (1 + r).prod() goes negative and a
    # negative base to a fractional power is NaN.
    r = pd.Series([0.01, -1.5, 0.02, 0.01])
    with pytest.raises(ValueError, match="below -100%"):
        perf_stats(r, freq=252)


def test_two_impossible_returns_do_not_cancel_into_a_plausible_number():
    # Two returns below -100% multiply to a positive product, so the pre-fix
    # code produced a finite, plausible-looking annualized return from a
    # series that cannot exist.
    r = pd.Series([-1.5, -1.5, 0.01, 0.02])
    with pytest.raises(ValueError, match="below -100%"):
        perf_stats(r, freq=252)


def test_bootstrap_ci_raises_rather_than_quietly_using_the_survivors():
    r = pd.Series(np.random.default_rng(3).normal(0.01, 0.05, 100))
    calls = {"n": 0}

    def flaky(s):
        calls["n"] += 1
        return 0.5 if calls["n"] == 1 else np.nan  # point estimate fine, every resample fails

    with pytest.raises(ValueError, match="no usable statistic"):
        bootstrap_ci(r, flaky, n_boot=200, seed=1)


def test_bootstrap_ci_warns_and_reports_a_small_number_of_failures():
    r = pd.Series(np.random.default_rng(4).normal(0.01, 0.05, 100))
    calls = {"n": 0}

    def occasionally_flaky(s):
        calls["n"] += 1
        return np.nan if calls["n"] == 5 else float(s.mean())

    with pytest.warns(RuntimeWarning, match="no usable statistic"):
        result = bootstrap_ci(r, occasionally_flaky, n_boot=500, seed=1)
    assert result["n_failed"] == 1


def test_bootstrap_ci_returns_the_draw_distribution():
    r = pd.Series(np.random.default_rng(5).normal(0.01, 0.05, 100))
    result = bootstrap_ci(r, lambda s: float(s.mean()), n_boot=250, seed=1)
    assert result["draws"].shape == (250,)
    assert result["n_failed"] == 0


def test_bootstrap_ci_carries_the_index_into_each_resample():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    r = pd.Series(np.random.default_rng(6).normal(0.01, 0.05, 100), index=idx)

    def needs_dates(s):
        assert isinstance(s.index, pd.DatetimeIndex), "resample lost its DatetimeIndex"
        return float(s.mean())

    result = bootstrap_ci(r, needs_dates, n_boot=50, seed=1)
    assert np.isfinite(result["point_estimate"])


def test_sharpe_ci_threads_the_convention_through():
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.02, 0.04, 100))
    for convention in ("arithmetic", "geometric"):
        result = sharpe_ci(r, freq=12, sharpe_convention=convention, n_boot=200, seed=2)
        point = perf_stats(r, freq=12, sharpe_convention=convention)["sharpe"]
        assert result["point_estimate"] == pytest.approx(point, rel=1e-9)
        assert result["sharpe_convention"] == convention
