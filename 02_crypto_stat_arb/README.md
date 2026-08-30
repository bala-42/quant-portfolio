# Statistical Arbitrage: BTC/ETH Pairs Trading

**Notebook:** [`pairs_trading.ipynb`](./pairs_trading.ipynb)

Tests whether BTC and ETH share a stable long-run (cointegrating)
relationship, and — since the answer is no — demonstrates why that matters
before deploying a mean-reversion strategy on the pair.

## Method

1. Align BTC and ETH on their overlapping daily history (2015–2019).
2. Engle-Granger two-step cointegration test on log(BTC) vs. log(ETH), plus
   an ADF test on the OLS-fitted spread.
3. Build a rolling z-scored spread and backtest a threshold-based
   mean-reversion trading rule, net of a 10bps-per-leg transaction cost
   assumption.

## Results

| Metric | Value |
|---|---|
| Engle-Granger cointegration p-value | 0.409 |
| ADF test on spread p-value | 0.196 |
| OLS hedge ratio (β) | 0.548 |
| Annualized return (net) | −40.4% |
| Annualized volatility | 62.1% |
| Sharpe ratio | −0.65 |
| Max drawdown | −92.9% |
| Position changes | 68 |

## Interpretation

Both cointegration tests **fail to reject the null of no cointegration** at
conventional significance levels — there's no statistically reliable
long-run equilibrium between BTC and ETH's log prices over this sample. The
backtested strategy loses money accordingly, which is the correct outcome
given the test result, not a bug. I also tried a rolling 90-day hedge ratio
(letting the relationship adapt through the 2017 bubble and 2018 crash);
performance did not improve, which rules out "stale parameters" as the
explanation.

**This is included deliberately.** A large part of real quant research is
disciplined rejection — testing a plausible thesis, finding it doesn't hold
up statistically, and having the judgment not to trade it anyway. A
report that only shows strategies that "backtest well" (often because the
hedge ratio was fit in-sample on the same data used to evaluate it) is the
actual red flag.

## A real next step

Scan a broader universe of crypto pairs for ones that *do* pass an
out-of-sample cointegration test, and only proceed to strategy design from
there — rather than assuming any two correlated assets are a tradeable pair.

## Data

- BTC/USD daily (2010–2026):
  [Habrador/Bitcoin-price-visualization](https://github.com/Habrador/Bitcoin-price-visualization)
- ETH/USD daily (2015–2019):
  [blockchain-unica/ethereum-ponzi](https://github.com/blockchain-unica/ethereum-ponzi)
