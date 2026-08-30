# Multi-Asset Portfolio Optimization: Equities + Crypto + FX

**Notebook:** [`portfolio_optimization.ipynb`](./portfolio_optimization.ipynb)

Builds and out-of-sample tests four portfolio construction approaches across
an 8-asset universe spanning three asset classes: AAPL, MSFT, XOM, JPM, KO
(equities), BTC (crypto), and EUR/USD, JPY/USD (FX).

## Method

The sample (2010-07-19 to 2016-12-30, bounded by the equities data) is split
in half. Expected returns and the covariance matrix are estimated **only** on
the training half; four portfolios are constructed from that in-sample
information alone, then evaluated **out of sample** on the untouched second
half:

- **Minimum-variance** — long-only, minimizes portfolio variance
- **Maximum-Sharpe (tangency)** — long-only, maximizes (return − rf) / vol
- **Equal-weight** — naive 1/N benchmark
- **Risk-parity** — each asset contributes equally to portfolio risk

## Results

**In-sample weights:**

| Asset | Min-Var | Max-Sharpe | Equal-Wt | Risk-Parity |
|---|---|---|---|---|
| AAPL | 1.7% | 28.9% | 12.5% | 10.9% |
| MSFT | 3.5% | 0.0% | 12.5% | 11.4% |
| XOM | 11.4% | 0.0% | 12.5% | 12.2% |
| JPM | 0.0% | 0.0% | 12.5% | 8.8% |
| KO | 10.3% | 40.3% | 12.5% | 14.1% |
| BTC | 0.1% | 20.0% | 12.5% | 2.8% |
| EUR/USD | 47.9% | 0.0% | 12.5% | 21.6% |
| JPY/USD | 25.1% | 10.9% | 12.5% | 18.3% |

**Out-of-sample performance:**

| Portfolio | Ann. return | Ann. vol | Sharpe | Max drawdown |
|---|---|---|---|---|
| Min-Variance | 6.6% | 7.4% | 0.63 | −8.0% |
| Max-Sharpe (tangency) | 18.9% | 19.1% | 0.89 | −26.4% |
| Equal-Weight | 17.0% | 14.2% | 1.06 | −17.5% |
| Risk-Parity | 10.6% | 9.8% | 0.88 | −11.0% |

## Interpretation

**In-sample optimization looks great; out-of-sample it degrades — as it
should.** The tangency portfolio's in-sample Sharpe (~1.90, driven heavily by
BTC's extreme training-period return — it traded for cents in mid-2010 and
crossed into the hundreds of dollars by late 2013, a genuine feature of this
window, not a data error) nearly halves out of sample (0.89). This is the
textbook failure mode of mean-variance optimization: it's highly sensitive to
estimated expected returns, and historical means are noisy — especially with
an outlier asset like early-history BTC in the mix.

**Risk-parity is the most stable across samples** (in-sample Sharpe 1.15 →
out-of-sample 0.88), because it only uses the covariance matrix — not
expected returns — to size positions. That stability is a well-documented,
practical reason risk parity is popular despite being "sub-optimal" on paper.

**The correlation matrix explains the min-variance weights.** FX and
equities are weakly correlated in this sample, so the minimum-variance
solution leans heavily on EUR/USD and JPY/USD — genuine diversification, not
just an artifact of low FX volatility.

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
