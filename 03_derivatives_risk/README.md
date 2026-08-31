# Derivatives Pricing & Portfolio Risk

**Notebook:** [`options_pricing_risk.ipynb`](./options_pricing_risk.ipynb)

Black-Scholes pricing with closed-form Greeks, a Monte Carlo pricer
validated against it, portfolio Value-at-Risk under three methodologies —
now including a **bootstrap confidence interval on the VaR estimate
itself**, via the same generic `quantlib.stats.bootstrap_ci` machinery
used for Sharpe ratios elsewhere in this repo — and a formal backtest of
the VaR model's calibration.

## Method

1. Estimate trailing annualized realized volatility for 5 large-cap names
   (AAPL, MSFT, XOM, JPM, KO) from real daily returns.
2. Price an AAPL ATM 6-month call via closed-form Black-Scholes; compute
   Delta, Gamma, Vega, Theta, Rho.
3. Validate a Monte Carlo pricer's convergence to the closed-form price.
4. Compute 1-day 99% VaR on a $10mm 5-stock book three ways, plus
   historical Expected Shortfall (CVaR).
5. **Bootstrap a confidence interval around the historical VaR estimate** —
   VaR is normally reported as a bare dollar figure, but it's estimated
   from a finite sample and has sampling uncertainty just like any other
   statistic.
6. Backtest the parametric VaR model with a rolling 250-day window using
   the Kupiec proportion-of-failures test.

## Results

**Monte Carlo convergence to Black-Scholes ($9.30 closed-form price):** error
falls to 0.03% at 1,000,000 simulated paths.

**1-day 99% VaR on a $10mm book:**

| Method | VaR |
|---|---|
| Parametric | $256,153 |
| Historical simulation | $338,831 |
| Monte Carlo | $258,271 |
| Historical Expected Shortfall (CVaR) | $398,860 |

**Bootstrap CI on the historical VaR estimate:** point estimate $338,831,
**90% CI [$258,142, $371,994]** (block size 12, 3,000 resamples).

**Kupiec backtest of rolling parametric VaR:** 24 breaches in 1,511
observations (1.59% observed vs. 1.00% expected) → LR statistic 4.48,
p-value 0.034 — **rejects** model calibration at the 5% level.

## Interpretation

Monte Carlo converges cleanly to the closed-form price — exactly the
sanity check needed before trusting an MC pricer on path-dependent
products with no closed form.

**The parametric VaR model fails its own backtest.** The observed breach
rate is meaningfully above the 1% the model targets — a well-documented
real phenomenon: daily equity returns have fatter tails than a normal
distribution, so variance-covariance VaR systematically understates true
tail risk.

**The bootstrap CI is a genuinely different check than the Kupiec test,
and both matter.** Kupiec asks "is the model's target breach rate
correct?" The CI asks "how much would this specific dollar figure move if
sampled from a slightly different history?" A VaR estimate can be
precisely estimated (tight CI) while still being the wrong number (fails
Kupiec) — which is exactly what's shown here: the CI is reasonably tight
relative to the point estimate, but the parametric model it's meant to
validate fails its calibration test regardless.

**Practical takeaway:** the historical-simulation VaR and CVaR (which
don't assume normality) are meaningfully higher than the parametric
estimate, and are the more defensible risk numbers for this book given
the parametric model's backtest failure.

## Data

Same NYSE daily price panel as Project 1:
[kyi3081/stock-analysis](https://github.com/kyi3081/stock-analysis)
