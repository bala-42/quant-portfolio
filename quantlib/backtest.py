"""
Reusable backtest engines.

decile_backtest   : generic cross-sectional signal backtest (used by the
                     momentum project, and reusable for any other
                     cross-sectional signal).
cointegration_test: Engle-Granger + ADF test with OLS hedge ratio.
pairs_backtest    : threshold mean-reversion backtest on a spread, used by
                     the statistical arbitrage project for any pair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm


def decile_backtest(signal: pd.DataFrame, fwd_returns: pd.DataFrame, n_deciles: int = 10,
                     min_names: int = 50) -> pd.DataFrame:
    """
    Generic cross-sectional decile backtest.

    At each date in `signal`, ranks the cross-section into `n_deciles`
    groups and computes the equal-weighted forward return of each group
    (using `fwd_returns` at the same date), plus the top-minus-bottom
    long-short spread and the equal-weight universe average.

    Parameters
    ----------
    signal      : DataFrame (date x asset) of the ranking signal
    fwd_returns : DataFrame (date x asset) of the return realized over the
                  period immediately following each signal date
    n_deciles   : number of groups to sort into
    min_names   : skip a date if fewer than this many names have a valid
                  signal (avoids noisy deciles early in a sample)

    Returns
    -------
    DataFrame indexed by date with columns decile_0 .. decile_{n-1},
    long_short (decile_{n-1} - decile_0), and universe_avg.
    """
    records = []
    for date in signal.index:
        sig_t = signal.loc[date].dropna()
        if len(sig_t) < min_names:
            continue
        fwd_t = fwd_returns.loc[date] if date in fwd_returns.index else None
        if fwd_t is None:
            continue
        fwd_t = fwd_t.dropna()

        ranks = pd.qcut(sig_t, n_deciles, labels=False, duplicates="drop")
        decile_rets = {}
        for d in range(n_deciles):
            names = ranks[ranks == d].index.intersection(fwd_t.index)
            if len(names) == 0:
                continue
            decile_rets[d] = fwd_t[names].mean()

        if (n_deciles - 1) in decile_rets and 0 in decile_rets:
            long_short = decile_rets[n_deciles - 1] - decile_rets[0]
        else:
            long_short = np.nan

        row = {"date": date, "long_short": long_short}
        row.update({f"decile_{d}": v for d, v in decile_rets.items()})
        row["universe_avg"] = fwd_t[ranks.index.intersection(fwd_t.index)].mean()
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["long_short", "universe_avg"]).rename_axis("date")

    out = pd.DataFrame(records).set_index("date").sort_index()
    return out.dropna(subset=["long_short"])


def decile_turnover(signal: pd.DataFrame, n_deciles: int = 10, min_names: int = 50) -> float:
    """
    Average one-sided turnover of the top+bottom decile membership between
    consecutive rebalances (0 = no names change, 1 = complete replacement).
    """
    prev_top, prev_bottom = None, None
    turnovers = []
    for date in signal.index:
        sig_t = signal.loc[date].dropna()
        if len(sig_t) < min_names:
            continue
        ranks = pd.qcut(sig_t, n_deciles, labels=False, duplicates="drop")
        top = set(ranks[ranks == n_deciles - 1].index)
        bottom = set(ranks[ranks == 0].index)
        if prev_top is not None:
            turn_top = 1 - len(top & prev_top) / max(len(top), 1)
            turn_bottom = 1 - len(bottom & prev_bottom) / max(len(bottom), 1)
            turnovers.append((turn_top + turn_bottom) / 2)
        prev_top, prev_bottom = top, bottom
    return float(np.mean(turnovers)) if turnovers else np.nan


def cointegration_test(series_a: pd.Series, series_b: pd.Series) -> dict:
    """
    Engle-Granger two-step cointegration test between two (typically
    log-price) series, plus the OLS hedge ratio and an ADF test on the
    resulting spread.

    Returns
    -------
    dict with coint_pvalue, adf_pvalue, hedge_ratio (beta), alpha, r_squared
    """
    aligned = pd.concat([series_a, series_b], axis=1).dropna()
    aligned.columns = ["a", "b"]

    score, pvalue, _ = coint(aligned["a"], aligned["b"])

    X = sm.add_constant(aligned["b"])
    model = sm.OLS(aligned["a"], X).fit()
    alpha, beta = model.params["const"], model.params["b"]

    spread = aligned["a"] - beta * aligned["b"]
    adf_result = adfuller(spread, result_object=True)
    adf_p = adf_result.pvalue

    return {
        "coint_pvalue": float(pvalue),
        "adf_pvalue": float(adf_p),
        "hedge_ratio": float(beta),
        "alpha": float(alpha),
        "r_squared": float(model.rsquared),
        "n_obs": len(aligned),
    }


def pairs_backtest(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    hedge_ratio: float,
    window: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    cost_bps: float = 10.0,
    ann_factor: int = 365,
) -> dict:
    """
    Threshold mean-reversion backtest on the spread log_price_a - hedge_ratio * log_price_b.

    Long the spread when its z-score drops below -entry_z (bet it reverts
    up), short when it rises above +entry_z, exit at +/-exit_z, and
    stop out beyond +/-stop_z. Net of a simple round-trip cost assumption
    charged on every position change (both legs).

    Returns
    -------
    dict with the net return series ("returns"), the z-score series
    ("zscore"), and summary stats (ann_return, ann_vol, sharpe,
    max_drawdown, n_position_changes, pct_time_in_market).
    """
    aligned = pd.concat([log_price_a, log_price_b], axis=1).dropna()
    aligned.columns = ["a", "b"]

    spread = aligned["a"] - hedge_ratio * aligned["b"]
    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    zscore = (spread - spread_mean) / spread_std

    position = np.zeros(len(zscore))
    pos = 0
    for i in range(1, len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            position[i] = pos
            continue
        if pos == 0:
            if z > entry_z:
                pos = -1
            elif z < -entry_z:
                pos = 1
        elif pos == 1 and (z >= -exit_z or z < -stop_z):
            pos = 0
        elif pos == -1 and (z <= exit_z or z > stop_z):
            pos = 0
        position[i] = pos
    position = pd.Series(position, index=zscore.index)

    ret_a = aligned["a"].diff()
    ret_b = aligned["b"].diff()
    spread_ret = ret_a - hedge_ratio * ret_b
    strat_ret = position.shift(1) * spread_ret

    trades = position.diff().abs().fillna(0)
    cost = trades * (cost_bps / 1e4) * 2
    net_ret = (strat_ret - cost.reindex(strat_ret.index).fillna(0)).dropna()

    from .stats import perf_stats
    stats = perf_stats(net_ret, freq=ann_factor)
    stats["n_position_changes"] = int((trades > 0).sum())
    stats["pct_time_in_market"] = float((position != 0).mean())

    return {"returns": net_ret, "zscore": zscore, "position": position, **stats}
