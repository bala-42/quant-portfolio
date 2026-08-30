# Derivatives Pricing & Portfolio Risk

**Notebook:** [`options_pricing_risk.ipynb`](./options_pricing_risk.ipynb)

Four connected pieces: Black-Scholes pricing with closed-form Greeks
calibrated to real realized volatility, a Monte Carlo pricer validated
against it, portfolio Value-at-Risk under three methodologies, and a formal
backtest of the VaR model's calibration.

## Method

1. Estimate trailing annualized realized volatility for 5 large-cap names
   (AAPL, MSFT, XOM, JPM, KO) from real daily returns.
2. Price an AAPL ATM 6-month call via closed-form Black-Scholes; compute
   Delta, Gamma, Vega, Theta, Rho.
3. Validate a Monte Carlo pricer's convergence to the closed-form price as
   path count scales from 100 to 1,000,000.
4. Compute 1-day 99% VaR on a $10mm 5-stock book three ways — parametric
   (variance-covariance), historical simulation, and Monte Carlo — plus
   historical Expected Shortfall (CVaR).
5. Backtest the parametric VaR model with a rolling 250-day window using the
   Kupiec proportion-of-failures test.

## Results

**Monte Carlo convergence to Black-Scholes ($9.30 closed-form price):**

| Paths | MC price | Abs. error |
|---|---|---|
| 100 | 6.53 | 2.767 |
| 1,000 | 8.74 | 0.565 |
| 10,000 | 9.20 | 0.099 |
| 100,000 | 9.28 | 0.023 |
| 1,000,000 | 9.305 | 0.003 |

**1-day 99% VaR on a $10mm book:**

| Method | VaR |
|---|---|
| Parametric | $256,153 |
| Historical simulation | $338,831 |
| Monte Carlo | $258,271 |
| Historical Expected Shortfall (CVaR) | $398,860 |

**Kupiec backtest of rolling parametric VaR:** 24 breaches in 1,511
observations (1.59% observed vs. 1.00% expected) → LR statistic 4.48,
p-value 0.034 — **rejects** model calibration at the 5% level.

## Interpretation

Monte Carlo converges cleanly to the closed-form price (0.03% error at 1mm
paths) — exactly the sanity check needed before trusting an MC pricer on
path-dependent products with no closed form.

More importantly: **the parametric VaR model fails its own backtest.** The
observed breach rate is meaningfully above the 1% the model targets, a
well-documented real phenomenon — daily equity returns have fatter tails
than a normal distribution, so variance-covariance VaR systematically
understates true tail risk. It's also exactly why regulators require
backtesting of internal VaR models rather than trusting the parametric
number at face value. The historical-simulation VaR and CVaR (which don't
assume normality) are meaningfully higher here, and are the more defensible
risk numbers for this book given the backtest failure.

## Data

Same NYSE daily price panel as Project 1:
[kyi3081/stock-analysis](https://github.com/kyi3081/stock-analysis)
