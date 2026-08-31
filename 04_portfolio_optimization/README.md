# Multi-Asset Portfolio Optimization: Equities + Crypto + FX

**Notebook:** [`portfolio_optimization.ipynb`](./portfolio_optimization.ipynb)

Builds and out-of-sample tests four portfolio construction approaches
across an 8-asset universe spanning three asset classes. Two of the four
are now solved via **exact convex optimization**
(`quantlib.optimize`) rather than the approximations a first pass at this
problem typically reaches for:

- **Max-Sharpe (tangency)** — solved exactly via the Charnes-Cooper convex
  QP transformation, not by grid-searching the efficient frontier and
  picking the best point.
- **Risk-Parity** — solved via Spinu's (2013) convex reformulation, not
  iterative gradient descent toward approximately-equal risk
  contributions.

Both are unit-tested for actual numerical correctness (`tests/test_optimize.py`):
the exact tangency solution is checked against every point on a 60-point
frontier grid, and the risk-parity weights are checked to produce risk
contributions within 0.1% of exactly equal, not just "close enough."

## Method

The sample (2010-07-19 to 2016-12-30, bounded by the equities data) is
split in half. Expected returns and the covariance matrix are estimated
**only** on the training half; four portfolios are constructed from that
in-sample information, then evaluated **out of sample** on the untouched
second half. A bootstrap confidence interval is added around the two most
interesting out-of-sample Sharpe ratios.

## Results

**In-sample weights:**

| Asset | Min-Var | Max-Sharpe | Equal-Wt | Risk-Parity |
|---|---|---|---|---|
| AAPL | 1.7% | 29.2% | 12.5% | 6.6% |
| MSFT | 3.5% | 0.0% | 12.5% | 7.5% |
| XOM | 11.4% | 0.0% | 12.5% | 8.8% |
| JPM | 0.0% | 0.0% | 12.5% | 5.2% |
| KO | 10.3% | 40.6% | 12.5% | 11.0% |
| BTC | 0.1% | 20.3% | 12.5% | 1.9% |
| EUR/USD | 47.9% | 0.0% | 12.5% | 38.7% |
| JPY/USD | 25.1% | 9.9% | 12.5% | 20.3% |

**Risk contribution check (Risk-Parity portfolio):** every asset
contributes **exactly 12.5%** to total portfolio risk — verified directly,
not assumed.

**In-sample vs. out-of-sample Sharpe:**

| Portfolio | In-sample Sharpe | OOS Sharpe | OOS 90% bootstrap CI |
|---|---|---|---|
| Min-Variance | 0.28 | 0.63 | — |
| Max-Sharpe (tangency) | 1.90 | 0.89 | [-0.21, 2.26] |
| Equal-Weight | 1.74 | 1.06 | — |
| Risk-Parity | 1.00 | 0.90 | [-0.03, 1.92] |

## Interpretation

**In-sample optimization looks great; out-of-sample it degrades — as it
should.** The tangency portfolio's in-sample Sharpe (1.90, driven heavily
by BTC's extreme training-period return — it traded for cents in mid-2010
and crossed into the hundreds of dollars by late 2013, a genuine feature
of this window, not a data error) nearly halves out of sample (0.89).
Mean-variance optimization is highly sensitive to estimated expected
returns, and historical means are noisy, especially with an outlier asset
like early-history BTC in the mix.

**Risk-parity is more stable across samples, and now it's exactly
risk-parity.** Its Sharpe declines from 1.00 in-sample to 0.90
out-of-sample — a much smaller relative drop than the tangency
portfolio's. That stability is a well-documented, practical reason risk
parity is popular despite being "sub-optimal" on paper — and here it's
backed by an exact solve and a passing unit test confirming equal risk
contribution, not eyeballed.

**The bootstrap CIs add an important caveat to that comparison.** Both OOS
Sharpe ratios individually have wide, overlapping confidence intervals —
on ~3 years of daily data, neither is estimated precisely enough to call
the difference between them statistically decisive on its own. The more
robust claim is the in-sample-to-out-of-sample *degradation* comparison
above (1.90→0.89 vs. 1.00→0.90), which is about each strategy's
consistency with itself and much less sensitive to the small-sample noise
that makes a single OOS Sharpe hard to pin down precisely.

## Honest limitations

- No rebalancing costs between observations.
- Expected returns are simple historical means; a real desk would blend in
  forward-looking views (e.g., Black-Litterman) rather than extrapolating
  raw historical averages.
- BTC's extreme early volatility is a genuine outlier that a live risk
  process would likely cap or treat as a separate risk bucket.

## Data

- Equities: [kyi3081/stock-analysis](https://github.com/kyi3081/stock-analysis)
- BTC/USD: [Habrador/Bitcoin-price-visualization](https://github.com/Habrador/Bitcoin-price-visualization)
- FX: [datasets/exchange-rates](https://github.com/datasets/exchange-rates) (U.S. Federal Reserve H.10)
