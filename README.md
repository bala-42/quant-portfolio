# Quant Research & Trading Portfolio

Four projects spanning systematic trading, statistical arbitrage,
derivatives pricing/risk, and portfolio construction — built on real,
publicly available market data (equities, crypto, and FX), backed by a
shared, unit-tested library rather than four separate one-off scripts.

| # | Project | Area | Asset class | Key result |
|---|---|---|---|---|
| 1 | [Cross-Sectional Momentum](./01_momentum_equities) | Systematic trading | Equities | Monotonic decile spread confirms the signal; long-short Sharpe (0.28) has a 90% bootstrap CI of **[-0.41, 1.06]** — not statistically distinguishable from zero with 71 monthly observations |
| 2 | [Crypto Pairs Trading](./02_crypto_stat_arb) | Statistical arbitrage | Crypto | BTC/ETH fails cointegration (p=0.41) and is correctly not traded. A systematic scan of 21 pairs finds XRP/DOGE genuinely cointegrates (p=0.0013); diagnosing the mean-reversion half-life (49 days) turns a losing naive backtest into a modestly positive one (Sharpe 0.33), though its CI still spans zero |
| 3 | [Derivatives Pricing & Risk](./03_derivatives_risk) | Derivatives / risk | Equities | Monte Carlo converges to the Black-Scholes price to within 0.03% at 1mm paths; Kupiec backtest **rejects** normal-VaR calibration; the VaR estimate itself gets a bootstrap CI, not just a bare number |
| 4 | [Multi-Asset Portfolio Optimization](./04_portfolio_optimization) | Portfolio construction | Equities + crypto + FX | Tangency and risk-parity portfolios solved via **exact convex optimization** (not grid search / gradient descent); tangency's Sharpe nearly halves out-of-sample (1.90 → 0.89) while risk-parity is far more stable (1.00 → 0.90) |

## What changed from a first pass

This repo went through a real refactor, not just a rewrite. The engineering
choices below were made specifically to fix things a first pass at this
project got away with:

- **A shared, tested library instead of four copies of the same math.**
  [`quantlib/`](./quantlib) holds the decile-backtest engine, cointegration
  testing, pairs-trading backtest, performance statistics, bootstrap
  confidence intervals, and portfolio optimizers — used by every notebook,
  covered by [18 unit tests](./tests) that check actual numerical
  correctness (e.g., that the exact max-Sharpe solution beats every point
  on a 60-point frontier grid; that risk-parity weights produce genuinely
  equal risk contributions, not approximately equal ones).
- **Confidence intervals, not just point estimates.** Every headline Sharpe
  ratio and the VaR estimate in Project 3 now carry a moving-block
  bootstrap confidence interval. Several of the "positive" results turn
  out to have confidence intervals that span zero — which is reported
  plainly rather than rounded up to "it works."
- **Exact convex optimization, not approximations.** The tangency portfolio
  is solved via the Charnes-Cooper QP transformation instead of
  grid-searching the efficient frontier; risk parity is solved via Spinu's
  (2013) convex reformulation instead of hand-rolled gradient descent.
  Both are unit-tested for correctness, not just "runs without crashing."
- **A second, real cointegrated pair, found systematically.** Rather than
  BTC/ETH being the only pair tested (which makes a failed cointegration
  test look like a foregone conclusion), Project 2 now scans 21 pairs
  across 7 major coins and finds one — XRP/DOGE — that genuinely
  cointegrates, then diagnoses (via a measured mean-reversion half-life)
  why the initial naive backtest on it still failed.
- **A pinned `requirements.txt`** so this doesn't silently break in a year.

## Why the honesty is still the point

None of the above changes the underlying philosophy: report what actually
happened, including the failures and the uncertainty, and explain why. A
losing strategy, a rejected risk model, and a confidence interval that
spans zero are not embarrassing — presenting only results that look clean
would be the actual red flag in real quant work.

## Repository structure

```
quant_portfolio/
├── quantlib/                       # shared, tested library
│   ├── stats.py                    # perf_stats, bootstrap_ci, sharpe_ci
│   ├── backtest.py                 # decile_backtest, cointegration_test, pairs_backtest
│   └── optimize.py                 # min_variance, max_sharpe, risk_parity (all convex/cvxpy)
├── tests/                          # 18 unit tests covering quantlib
├── requirements.txt                # pinned dependency versions
├── download_data.py                # fetches all source datasets into data/
├── 01_momentum_equities/
├── 02_crypto_stat_arb/
├── 03_derivatives_risk/
└── 04_portfolio_optimization/
```

## Reproducing locally

```bash
git clone <this-repo>
cd quant_portfolio
pip install -r requirements.txt
python download_data.py
python -m pytest tests/ -v        # confirm the library itself is correct
jupyter notebook                  # open any project's .ipynb and run all cells
```

## Data sources (all public)

- **NYSE equities (2010–2016, ~500 tickers, split-adjusted OHLCV):**
  [kyi3081/stock-analysis](https://github.com/kyi3081/stock-analysis)
  (originally the Kaggle "New York Stock Exchange" dataset, dgawlik)
- **BTC/USD daily (2010–2026), ETH/USD daily (2015–2019):**
  [Habrador/Bitcoin-price-visualization](https://github.com/Habrador/Bitcoin-price-visualization),
  [blockchain-unica/ethereum-ponzi](https://github.com/blockchain-unica/ethereum-ponzi)
- **7-coin crypto universe (BTC, ETH, LTC, XRP, DOGE, LINK, ADA) for the pair scan:**
  [MainakRepositor/Datasets](https://github.com/MainakRepositor/Datasets) (Cryptocurrency folder)
- **FX daily rates (1971–2026, U.S. Federal Reserve H.10 release):**
  [datasets/exchange-rates](https://github.com/datasets/exchange-rates)

## Stack

Python, pandas, NumPy, SciPy, statsmodels, cvxpy, matplotlib, Jupyter, pytest.
