# Quant Research & Trading Portfolio

Four projects spanning systematic trading, statistical arbitrage,
derivatives pricing/risk, and portfolio construction — built on real,
publicly available market data (equities, crypto, and FX), backed by a
shared, unit-tested library rather than four separate one-off scripts.

| # | Project | Area | Asset class | Key result |
|---|---|---|---|---|
| 1 | [Cross-Sectional Momentum](./01_momentum_equities) | Systematic trading | Equities | Monotonic decile spread confirms the signal; long-short Sharpe (0.35) has a 90% bootstrap CI of **[-0.35, 1.06]** — not statistically distinguishable from zero with 71 monthly observations |
| 2 | [Crypto Pairs Trading](./02_crypto_stat_arb) | Statistical arbitrage | Crypto | BTC/ETH fails cointegration (p=0.41) and is correctly not traded. A systematic scan of 21 pairs finds XRP/DOGE genuinely cointegrates (p=0.0013); diagnosing the mean-reversion half-life (49 days) turns a losing naive backtest (Sharpe -0.33) into a positive one (Sharpe 0.65), though its 90% CI **[-0.07, 1.41]** still spans zero |
| 3 | [Derivatives Pricing & Risk](./03_derivatives_risk) | Derivatives / risk | Equities | Monte Carlo converges to the Black-Scholes price to within 0.03% at 1mm paths; Kupiec backtest **rejects** normal-VaR calibration; the VaR estimate itself gets a bootstrap CI, not just a bare number |
| 4 | [Multi-Asset Portfolio Optimization](./04_portfolio_optimization) | Portfolio construction | Equities + crypto + FX | Tangency and risk-parity portfolios solved via **exact convex optimization** (not grid search / gradient descent); tangency's Sharpe more than halves out-of-sample (1.90 → 0.90) while risk-parity is far more stable (1.00 → 0.89) |

## What changed from a first pass

This repo went through a real refactor, not just a rewrite. The engineering
choices below were made specifically to fix things a first pass at this
project got away with:

- **A shared, tested library instead of four copies of the same math.**
  [`quantlib/`](./quantlib) holds the decile-backtest engine, cointegration
  testing, pairs-trading backtest, performance statistics, bootstrap
  confidence intervals, and portfolio optimizers — used by every notebook,
  covered by [35 unit tests](./tests) that check actual numerical
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

## What the second pass fixed

The first refactor moved four copies of the same math into one library. This
pass went back through that library looking for the specific failure the whole
repo is about: **a number that could not really be measured, reported as though
it had been.** Five instances, each now covered by a regression test that fails
if the fix is reverted.

**1. The Sharpe ratio had an undocumented convention.** `perf_stats` divided a
*geometric* annualized return by an *arithmetic* annualized volatility, and said
so nowhere. Over a long sample the two conventions land within about 8% of each
other; over a short window with large moves they do not, because the geometric
numerator raises the window's cumulative return to the power `freq / len(r)` and
reports a year that did not happen. `sharpe_convention` is now an explicit
argument defaulting to `arithmetic`, and **every result records which convention
produced it**. `sharpe_ci` takes the same argument, so an interval and its own
point estimate can no longer be computed under different rules.

**2. A return of -100% or worse produced a silent NaN.** `(1 + r).prod()` is
zero or negative in that case — and with *two* such returns it is positive
again through sign cancellation, yielding a plausible-looking annualized return
from a series that cannot exist. This now raises.

**3. The bootstrap reported intervals built from the survivors.** Resamples on
which the statistic failed were dropped by `nanpercentile` without comment, so
an interval still printed no matter how many draws had failed. Failures are now
counted and surfaced as `n_failed`; more than 1% of them raises rather than
quietly answering a different question. The full draw distribution is returned
as well, which is what makes BCa intervals and one-sided p-values possible
later, and the index is carried into each resample so date-aware statistics
still work.

**4. The efficient frontier could not handle negative expected returns.** The
target grid was built by scaling `mu`'s endpoints multiplicatively, which only
moves them in the intended direction when they are positive: with a negative
maximum, `mu.max() * 0.98` sits *above* the highest attainable return and those
QPs are infeasible, so the sweep silently returned fewer points than requested.
The grid is now padded by a fraction of the span of `mu`.

**5. The pairs backtest measured the wrong quantity three ways.** It differenced
log prices and then handed the result to a function that compounds with
`(1 + r).prod()`, compounding log returns as if they were simple ones. It
charged `cost * 2` for two legs whose notionals are 1 and `|beta|`, which is
only correct when `beta` happens to be 1. And it never normalised by the capital
actually committed, so `ann_return` and `max_drawdown` were returns on an
unstated base. Leg returns are now simple, costs scale with `1 + |beta|`, and
everything is expressed per unit of gross notional.

### Two things found while re-running the notebooks

**The out-of-sample Sharpe was computed twice, two different ways.** Project 4
re-derived annualized return, volatility, Sharpe and drawdown by hand in one
cell, then printed a bootstrap CI from `quantlib` in the next. Before fix 1 the
two happened to share a convention and agreed, so the duplication was invisible;
afterwards the same portfolio printed 0.885 in one cell and 0.898 in the next.
That cell now calls `perf_stats` — which is the point of having the library.

**The published summary CSVs were maintained by hand and had drifted.**
`oos_performance.csv` reported a risk-parity annualized return of 10.6% where
the notebook that supposedly produced it printed 9.3%. Every notebook now
*writes* its own summary file, so the repository and the code behind it cannot
disagree again.

### What moved, and what did not

| | before | after |
|---|---|---|
| Momentum L/S Sharpe | 0.275 | **0.346** |
| Momentum L/S 90% CI | [-0.41, 1.06] | **[-0.35, 1.06]** |
| XRP/DOGE calibrated Sharpe | 0.33 | **0.65** |
| XRP/DOGE 90% CI | spans zero | **[-0.07, 1.41]**, still spans zero |
| BTC/ETH stress-test Sharpe | -0.650 | **-0.676** |
| Tangency OOS Sharpe | 0.887 published / 0.885 in-notebook | **0.898** |
| Risk-parity OOS Sharpe | 0.881 published / 0.897 in-notebook | **0.888** |
| VaR, Greeks, Kupiec, cointegration p-values | unchanged | unchanged |

Project 4 has two "before" figures because the published CSV and the notebook
that was supposed to have produced it disagreed — which is the drift described
above, and the reason the notebooks now write these files themselves.

**Every verdict in this README survives.** The momentum long-short interval
still spans zero, the calibrated pair's interval still spans zero, BTC/ETH still
fails cointegration and still loses money when traded anyway, Kupiec still
rejects the normal-VaR calibration, and tangency still gives up far more
out-of-sample than risk parity. The numbers moved; none of the conclusions did.

## Known limitations

Named here rather than left for a reader to find.

- **The pairs work selects on the full sample, twice over.** The hedge ratio is
  fitted on the entire history and then traded from the first day, and the pair
  itself is chosen as the best of 21 candidates scanned over that same history.
  Both are look-ahead at the *parameter* layer: every bar-level shift in the
  backtest is causal, but the parameters those bars are traded against were
  chosen with knowledge of the test period. A clean version fits the hedge ratio
  on a rolling or expanding window and selects the pair out-of-sample. This is
  documented in `pairs_backtest`'s docstring rather than silently fixed, because
  fixing it properly is a rewrite and the honest reading of the current result
  is "not yet evidence".
- **`decile_backtest` trusts its caller to pre-shift.** It cannot tell a forward
  return from a contemporaneous one. The contract is stated in the docstring and
  is not enforced.
- **Half-life calibration was chosen after seeing the naive result.** The 120-day
  window is justified by a measured half-life rather than a sweep, which is
  better than fitting it — but it was still selected after the 30-day version
  failed, on the same data.

## Repository structure

```
quant_portfolio/
├── quantlib/                       # shared, tested library
│   ├── stats.py                    # perf_stats, bootstrap_ci, sharpe_ci
│   ├── backtest.py                 # decile_backtest, cointegration_test, pairs_backtest
│   └── optimize.py                 # min_variance, max_sharpe, risk_parity (all convex/cvxpy)
├── tests/                          # 35 unit tests covering quantlib
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
