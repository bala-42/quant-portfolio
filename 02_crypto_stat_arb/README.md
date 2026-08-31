# Statistical Arbitrage: Cointegration-Based Pairs Trading

**Notebook:** [`pairs_trading.ipynb`](./pairs_trading.ipynb)

Tests whether two crypto assets share a stable long-run (cointegrating)
relationship before trading a mean-reversion strategy on them — first on
BTC/ETH (which fails the test), then on a pair found through a systematic
scan of 21 candidate pairs across 7 major coins. Built on
[`quantlib.backtest.cointegration_test`](../quantlib/backtest.py) and
`pairs_backtest`, the same tested functions for any pair.

## Method

1. Engle-Granger cointegration test + ADF test on the spread, for BTC/ETH.
2. A systematic scan: cointegration test on all 21 pairs among BTC, ETH,
   LTC, XRP, DOGE, LINK, and ADA, ranked by p-value.
3. Backtest the strongest candidate (XRP/DOGE) with the same 30-day window
   convention used for BTC/ETH — which still loses money.
4. Diagnose why: measure the spread's actual mean-reversion half-life via
   an AR(1) fit, and recalibrate the window around that instead of
   grid-searching.
5. A moving-block bootstrap gives a confidence interval around the
   recalibrated Sharpe ratio.

## Results

**Pair 1 — BTC/ETH:**

| Metric | Value |
|---|---|
| Engle-Granger cointegration p-value | 0.409 |
| ADF test on spread p-value | 0.196 |
| OLS hedge ratio (β) | 0.548 (R² = 0.911) |
| Backtest Sharpe (net) | −0.73 |
| Backtest annualized return | −45.3% |
| Max drawdown | −92.9% |

**Pair 2 — XRP/DOGE**, found via the 21-pair scan:

| Metric | Value |
|---|---|
| Engle-Granger cointegration p-value | **0.0013** |
| ADF test on spread p-value | 0.0002 |
| R² | 0.939 (2,275 daily observations) |
| Naive backtest (30-day window) Sharpe | −0.53 |
| Measured mean-reversion half-life | 48.6 days |
| Half-life-calibrated backtest Sharpe | **+0.33** |
| 90% bootstrap CI on calibrated Sharpe | [-0.35, 1.82] |

## Interpretation

**BTC/ETH fails at the testing stage.** Both cointegration tests are well
above conventional significance thresholds — there's no statistically
reliable long-run equilibrium between BTC and ETH's log prices over this
sample, and the backtest loses money accordingly. That's the correct
outcome, not a bug: the test said don't trade it, so it isn't.

**XRP/DOGE passes the testing stage, but initially fails at calibration —
a genuinely different and more interesting failure mode.** The
cointegration is strong and well-supported (p=0.0013, R²=0.94), yet the
same 30-day window that was a reasonable default for BTC/ETH still loses
money on this pair. Measuring the spread's actual mean-reversion half-life
(48.6 days — well over the 30-day window) explains why: the z-score was
reacting to noise on a timescale shorter than the spread actually takes to
revert. Recalibrating the window around the measured half-life (not
grid-searched, *measured*) flips the Sharpe from -0.53 to +0.33.

**Even the improved result isn't proven.** The bootstrap confidence
interval on the calibrated Sharpe still spans from -0.35 to +1.82 — it
includes zero. And the max drawdown remains severe (~-92%) even in the
"working" version. The honest summary: correct diagnosis produces a
directionally better result, but "better than a losing strategy" isn't the
same as "a strategy with a proven edge."

## A real next step

A volatility-scaled position size (the current backtest risks the same
notional regardless of regime, which is almost certainly why the drawdown
stays severe even in the improved version), and testing the
half-life-calibration approach out-of-sample on a pair not used to
validate the method — to check the calibration approach itself isn't
overfit to this one case.

## Data

- BTC/USD, ETH/USD:
  [Habrador/Bitcoin-price-visualization](https://github.com/Habrador/Bitcoin-price-visualization),
  [blockchain-unica/ethereum-ponzi](https://github.com/blockchain-unica/ethereum-ponzi)
- BTC, ETH, LTC, XRP, DOGE, LINK, ADA (for the pair scan):
  [MainakRepositor/Datasets](https://github.com/MainakRepositor/Datasets) (Cryptocurrency folder)
