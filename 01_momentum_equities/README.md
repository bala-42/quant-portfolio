# Cross-Sectional Momentum on NYSE Equities

**Notebook:** [`momentum_strategy.ipynb`](./momentum_strategy.ipynb)

A backtest of the classic 12-1 month cross-sectional momentum anomaly
(Jegadeesh & Titman, 1993) on ~500 NYSE-listed stocks, 2010–2016 — built on
[`quantlib.backtest.decile_backtest`](../quantlib/backtest.py), the same
tested decile-sort engine used nowhere else as one-off code.

## Method

1. Filter to tickers with a complete daily history (avoids IPO/delisting
   noise in the cross-section).
2. Signal = cumulative return from month *t*-12 to *t*-1 (skips the most
   recent month to avoid short-term reversal contamination).
3. Each month: rank into deciles by the signal, go long the top decile and
   short the bottom decile, equal-weighted, rebalanced monthly.
4. A moving-block bootstrap (`quantlib.stats.sharpe_ci`) gives a confidence
   interval around the headline Sharpe ratio, instead of reporting a bare
   point estimate.

## Results

| Portfolio | Ann. return | Ann. vol | Sharpe | Max drawdown | Hit rate |
|---|---|---|---|---|---|
| Momentum L/S (D10−D1) | 4.2% | 15.3% | 0.28 | −27.0% | 53.5% |
| Top decile (long only) | 16.0% | 13.9% | 1.15 | −17.3% | 62.0% |
| Bottom decile (long only) | 9.1% | 19.5% | 0.47 | −34.3% | 60.6% |
| Equal-weight universe (benchmark) | 12.8% | 12.5% | 1.03 | −18.5% | 62.0% |

**Long-short Sharpe, with uncertainty:** point estimate 0.275, **90%
bootstrap confidence interval [-0.410, 1.056]** (moving-block bootstrap,
block size 4, 3,000 resamples).

Average one-sided monthly turnover on the long+short legs: **~27%**.

## Interpretation

The decile spread rises roughly monotonically from losers to winners —
the real signature of momentum, and confirmation the signal carries genuine
information. But the long-short spread's Sharpe (0.28) is well below the
long-only top decile's (1.15): shorting the bottom decile *detracted* from
performance here, consistent with the documented "momentum crash"
literature (Daniel & Moskowitz, 2016).

**The confidence interval is the more important addition.** A bare Sharpe
of 0.28 reads as "modest but real." The bootstrap CI — which spans from
clearly negative to clearly positive — reveals that with only 71 monthly
observations, this specific number isn't estimated precisely enough to
distinguish it from zero. That doesn't invalidate the top-decile or
monotonic-spread findings (both have larger, more robust effect sizes),
but it's exactly the kind of honest uncertainty that reporting only a
point estimate hides.

## Honest limitations

- Survivorship bias: only tickers with a complete 2010–2016 history are
  included.
- No borrow-cost modeling for the short leg.
- No sector or beta neutralization — a natural next step.
- Gross of transaction costs; at ~27% monthly turnover per leg, realistic
  costs would meaningfully erode the long-short spread's already-uncertain
  Sharpe ratio.

## Data

Daily split-adjusted OHLCV prices for NYSE-listed stocks, originally the
Kaggle "New York Stock Exchange" dataset (dgawlik), mirrored at
[kyi3081/stock-analysis](https://github.com/kyi3081/stock-analysis).
