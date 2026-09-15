"""
Performance statistics and bootstrap confidence intervals.

All four projects need the same handful of numbers (annualized return,
annualized vol, Sharpe, max drawdown, hit rate) computed from a return
series. Previously each notebook had its own copy of this function; now
there's exactly one, and it's tested.

Two things that used to be implicit are now explicit, because a
convention nobody states produces numbers that cannot be compared with
anything published anywhere else:

- ``sharpe_convention`` selects the numerator of the Sharpe ratio.
  ``"arithmetic"`` (the default) uses the annualized *mean* period
  return; ``"geometric"`` uses the annualized *compound* return. Over a
  long sample the two agree closely. Over a short window containing large
  moves they do not, because the geometric numerator raises the window's
  cumulative return to the power ``freq / len(r)`` — extrapolating a few
  months into a statement about a year that did not happen. Whichever
  convention produced a number is recorded alongside it.

- A period return of -100% or worse now raises instead of silently
  producing NaN. ``(1 + r).prod()`` is zero or negative in that case, so
  the annualized return is either undefined or — with two such returns —
  *positive* through sign cancellation, which is worse than missing.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

SHARPE_CONVENTIONS = ("arithmetic", "geometric")


def _check_representable(r: pd.Series, caller: str) -> None:
    """Reject returns at or below -100%, which make compounding meaningless."""
    bad = r <= -1
    if bool(bad.any()):
        n = int(bad.sum())
        worst = float(r.min())
        raise ValueError(
            f"{caller} received {n} return(s) at or below -100% (worst: {worst:.4f}). "
            "Cumulative growth is zero or negative from that point, so the annualized "
            "return is undefined. This is raised rather than returned as NaN because a "
            "NaN here propagates silently into every downstream statistic."
        )


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the cumulative growth curve."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    return float(drawdown.min())


def perf_stats(
    returns: pd.Series,
    freq: int,
    rf: float = 0.0,
    name: str | None = None,
    sharpe_convention: str = "arithmetic",
) -> dict:
    """
    Compute standard performance statistics from a periodic return series.

    Parameters
    ----------
    returns : pd.Series of periodic (not cumulative) simple returns
    freq    : periods per year used for annualization (12=monthly, 252=daily
              trading, 365=daily calendar/crypto)
    rf      : annualized risk-free rate, used in the Sharpe ratio
    name    : optional label, included in the returned dict as "name"
    sharpe_convention : "arithmetic" (default) or "geometric" — see module
              docstring. Recorded in the returned dict so a number can
              always be traced back to the convention that produced it.

    Returns
    -------
    dict with ann_return, ann_vol, sharpe, sharpe_convention, max_drawdown,
    hit_rate

    Raises
    ------
    ValueError
        if the series is empty, if the convention is unknown, or if any
        return is at or below -100%.
    """
    if sharpe_convention not in SHARPE_CONVENTIONS:
        raise ValueError(
            f"sharpe_convention must be one of {SHARPE_CONVENTIONS}, got {sharpe_convention!r}. "
            "It is not defaulted, because a silently-chosen convention is the bug this "
            "parameter exists to prevent."
        )

    r = returns.dropna()
    if len(r) == 0:
        raise ValueError("perf_stats received an empty return series")
    _check_representable(r, "perf_stats")

    ann_return = float((1 + r).prod() ** (freq / len(r)) - 1)
    ann_vol = float(r.std() * np.sqrt(freq))

    if sharpe_convention == "arithmetic":
        numerator = float(r.mean()) * freq - rf
    else:
        numerator = ann_return - rf
    sharpe = numerator / ann_vol if ann_vol > 0 else np.nan

    result = {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sharpe_convention": sharpe_convention,
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
    }
    if name is not None:
        result = {"name": name, **result}
    return result


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices of length n."""
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([
        (np.arange(start, start + block_size) % n) for start in starts
    ])
    return idx[:n]


def bootstrap_ci(
    returns: pd.Series,
    stat_fn,
    n_boot: int = 2000,
    ci: float = 0.90,
    block_size: int | None = None,
    seed: int | None = 42,
    max_failed_frac: float = 0.01,
) -> dict:
    """
    Moving-block bootstrap confidence interval for an arbitrary statistic.

    A plain i.i.d. bootstrap understates uncertainty for financial returns
    because they're autocorrelated (volatility clustering, momentum in the
    signal itself). The moving-block bootstrap resamples contiguous chunks
    of the series instead of individual points, which preserves short-run
    dependence structure.

    Resamples on which ``stat_fn`` fails or returns a non-finite value are
    counted and reported rather than dropped silently. An interval computed
    from "whichever resamples happened to survive" is not the interval it
    claims to be, so more than ``max_failed_frac`` of them failing raises.

    The original index is carried into each resample, so a date-aware
    statistic still works; and the full draw distribution is returned, which
    is what makes BCa intervals or one-sided p-values possible later without
    rewriting this function.

    Parameters
    ----------
    returns    : pd.Series of periodic returns
    stat_fn    : callable(pd.Series) -> float
    n_boot     : number of bootstrap resamples
    ci         : confidence level (0.90 -> 5th/95th percentile interval)
    block_size : block length; defaults to round(n**(1/3)), a standard
                 rule-of-thumb for block bootstrap on n observations
    seed       : RNG seed for reproducibility
    max_failed_frac : fraction of resamples allowed to fail before raising

    Returns
    -------
    dict with point_estimate, ci_low, ci_high, ci_level, n_boot, block_size,
    n_failed, draws
    """
    r = returns.dropna()
    n = len(r)
    if n < 8:
        raise ValueError("bootstrap_ci needs at least 8 observations")
    if block_size is None:
        block_size = max(2, round(n ** (1 / 3)))

    rng = np.random.default_rng(seed)
    point_estimate = stat_fn(r)
    if not np.isfinite(point_estimate):
        raise ValueError(
            "bootstrap_ci: the point estimate itself is not finite, so an interval "
            "around it would be meaningless."
        )

    boot_stats = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = _block_bootstrap_indices(n, block_size, rng)
        try:
            boot_stats[i] = stat_fn(r.iloc[idx])
        except Exception:
            boot_stats[i] = np.nan

    finite = np.isfinite(boot_stats)
    n_failed = int((~finite).sum())
    if n_failed:
        frac = n_failed / n_boot
        msg = (
            f"bootstrap_ci: {n_failed} of {n_boot} resamples ({frac:.1%}) produced no "
            "usable statistic."
        )
        if frac > max_failed_frac:
            raise ValueError(
                msg + " The interval would be computed only from the resamples that "
                "happened to succeed, which is a different quantity from the one "
                "requested. Raise max_failed_frac only if that is understood."
            )
        warnings.warn(msg + " The interval below excludes them.", RuntimeWarning, stacklevel=2)

    alpha = 1 - ci
    lo, hi = np.percentile(boot_stats[finite], [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": float(point_estimate),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
        "block_size": block_size,
        "n_failed": n_failed,
        "draws": boot_stats,
    }


def sharpe_ci(
    returns: pd.Series,
    freq: int,
    rf: float = 0.0,
    sharpe_convention: str = "arithmetic",
    **kwargs,
) -> dict:
    """
    Convenience wrapper: bootstrap CI specifically for the Sharpe ratio.

    Takes ``sharpe_convention`` for the same reason ``perf_stats`` does, and
    threads it through, so an interval and its own point estimate can never
    be computed under different conventions.
    """
    def _sharpe(r):
        return perf_stats(r, freq=freq, rf=rf, sharpe_convention=sharpe_convention)["sharpe"]

    result = bootstrap_ci(returns, _sharpe, **kwargs)
    result["sharpe_convention"] = sharpe_convention
    return result
