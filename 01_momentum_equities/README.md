# Cross-Sectional Momentum on NYSE Equities

**Notebook:** [`momentum_strategy.ipynb`](./momentum_strategy.ipynb)

A backtest of the classic 12-1 month cross-sectional momentum anomaly
(Jegadeesh & Titman, 1993) on ~500 NYSE-listed stocks, 2010–2016.

## Method

1. Filter to tickers with a complete daily history (avoids IPO/delisting
   noise in the cross-section).
2. Signal = cumulative return from month *t*-12 to *t*-1 (skips the most
   recent month to avoid short-term reversal contamination).
3. Each month: rank into deciles by the signal, go long the top decile and
   short the bottom decile, equal-weighted, rebalanced monthly.

## Results

| Portfolio | Ann. return | Ann. vol | Sharpe | Max drawdown | Hit rate |
|---|---|---|---|---|---|
| Momentum L/S (D10−D1) | 4.2% | 15.3% | 0.28 | −27.0% | 53.5% |
| Top decile (long only) | 16.0% | 13.9% | 1.15 | −17.3% | 62.0% |
| Bottom decile (long only) | 9.1% | 19.5% | 0.47 | −34.3% | 60.6% |
| Equal-weight universe (benchmark) | 12.8% | 12.5% | 1.03 | −18.5% | 62.0% |

Average one-sided monthly turnover on the long+short legs: **~27%**.

## Interpretation

The decile spread rises roughly monotonically from losers to winners —
the real signature of momentum, and confirmation the signal carries genuine
information. But the long-short spread's Sharpe (0.28) is well below the
long-only top decile's (1.15): shorting the bottom decile *detracted* from
performance here. That's consistent with the documented "momentum crash"
literature (Daniel & Moskowitz, 2016) — post-2009-crash loser stocks staged
strong recoveries during parts of this window, which is exactly the
environment where naive momentum shorts get hurt.

## Honest limitations

- Survivorship bias: only tickers with a complete 2010–2016 history are
  included.
- No borrow-cost modeling for the short leg.
- No sector or beta neutralization — a natural next step.
- Gross of transaction costs; at ~27% monthly turnover per leg, realistic
  costs would meaningfully erode the long-short spread.

## Data

Daily split-adjusted OHLCV prices for NYSE-listed stocks, originally the
Kaggle "New York Stock Exchange" dataset (dgawlik), mirrored at
[kyi3081/stock-analysis](https://github.com/kyi3081/stock-analysis).
