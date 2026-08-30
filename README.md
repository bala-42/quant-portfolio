# Quant Research & Trading Portfolio

Four self-contained projects spanning systematic trading, statistical
arbitrage, derivatives pricing/risk, and portfolio construction — built on
real, publicly available market data (equities, crypto, and FX), with every
notebook fully executed so results, statistics, and charts render directly
on GitHub without needing to run anything locally.

| # | Project | Area | Asset class | Key result |
|---|---|---|---|---|
| 1 | [Cross-Sectional Momentum](./01_momentum_equities) | Systematic trading | Equities | Monotonic decile spread confirms the signal; long-short Sharpe (0.28) trails long-only top decile (1.15) — a real, documented momentum-crash effect in this window |
| 2 | [BTC/ETH Pairs Trading](./02_crypto_stat_arb) | Statistical arbitrage | Crypto | Cointegration test fails to reject the null (p=0.41); a naive mean-reversion strategy loses money, as it should given the failed test |
| 3 | [Derivatives Pricing & Risk](./03_derivatives_risk) | Derivatives / risk | Equities | Monte Carlo converges to the Black-Scholes price to within 0.03% at 1mm paths; Kupiec backtest **rejects** normal-VaR calibration (fat tails) |
| 4 | [Multi-Asset Portfolio Optimization](./04_portfolio_optimization) | Portfolio construction | Equities + crypto + FX | Tangency portfolio's in-sample Sharpe (1.90) nearly halves out-of-sample (0.89); risk-parity is far more stable (1.15 → 0.88) |

## Why these four

Each project targets a different core discipline in quant research/trading —
systematic signal construction and backtesting, statistical testing before
committing capital, pricing/risk-model validation, and portfolio
construction under estimation uncertainty — and each is built on a
different asset class to show the methodology generalizes rather than being
tuned to one dataset.

## A note on the results

None of these backtests are cherry-picked to look good. Project 2 reports a
losing strategy because the underlying statistical test says it should lose;
Project 3 reports a risk model that fails its own backtest; Project 4 shows
mean-variance optimization degrading out of sample, which is the expected,
well-documented behavior of the method. Presenting only "clean" results would
be a red flag in real quant work — this portfolio is meant to show the
testing discipline, not just the wins.

## Repository structure

```
quant_portfolio/
├── download_data.py          # fetches all source datasets into data/
├── 01_momentum_equities/
│   ├── momentum_strategy.ipynb
│   └── README.md
├── 02_crypto_stat_arb/
│   ├── pairs_trading.ipynb
│   └── README.md
├── 03_derivatives_risk/
│   ├── options_pricing_risk.ipynb
│   └── README.md
└── 04_portfolio_optimization/
    ├── portfolio_optimization.ipynb
    └── README.md
```

## Reproducing locally

```bash
git clone <this-repo>
cd quant_portfolio
pip install pandas numpy scipy matplotlib statsmodels arch cvxpy scikit-learn jupyter
python download_data.py
jupyter notebook   # open any project's .ipynb and run all cells
```

## Data sources (all public)

- **NYSE equities (2010–2016, ~500 tickers, split-adjusted OHLCV):** originally
  the Kaggle "New York Stock Exchange" dataset (dgawlik), mirrored at
  [kyi3081/stock-analysis](https://github.com/kyi3081/stock-analysis)
- **BTC/USD daily (2010–2026):**
  [Habrador/Bitcoin-price-visualization](https://github.com/Habrador/Bitcoin-price-visualization)
- **ETH/USD daily (2015–2019):**
  [blockchain-unica/ethereum-ponzi](https://github.com/blockchain-unica/ethereum-ponzi)
- **FX daily rates (1971–2026, U.S. Federal Reserve H.10 release):**
  [datasets/exchange-rates](https://github.com/datasets/exchange-rates)

## Stack

Python, pandas, NumPy, SciPy, statsmodels, `arch`, cvxpy, scikit-learn,
matplotlib, Jupyter.
