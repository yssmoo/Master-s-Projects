# SPY — Statistical Analysis, Prediction & GARCH Volatility Modelling

**M1 ESA (Applied Econometrics & Statistics) — Advanced Python Programming, Université d'Orléans**

Authors: Abdellah Nait Akli, Yassir Motia

---

## Overview

This project studies the **SPY ETF**, which tracks the S&P 500 index, using daily prices since January 2000. It covers the three stages of a data project:

1. **Data preparation**: download from the Yahoo Finance API and feature construction
2. **Statistical analysis and visualisation**: stationarity, return distribution, rolling volatility
3. **Prediction and classification** with scikit-learn: next-day price regression and up/down direction classification

As an **extension**, it models conditional volatility with a **GARCH(2,2)** model (`arch` package). This includes an ARCH-effect test, residual diagnostics and a 7-day volatility forecast.

Everything is driven from an interactive terminal menu.

## Data

- **Source:** Yahoo Finance, through the [`yfinance`](https://github.com/ranaroussi/yfinance) package (no authentication required)
- **Asset:** SPY (SPDR S&P 500 ETF Trust), daily close, from 2000-01-01 to the current date
- **Derived variables:**

| Column | Definition |
|---|---|
| `Close` | Daily closing price (USD) |
| `returns` | Daily simple return, in % |
| `volatility` | 21-day rolling standard deviation of returns (about one trading month) |
| `ma200` | 200-day moving average of the price, a long-term trend indicator widely followed by institutional investors |

The first 200 rows are dropped, since the 200-day moving average is not yet defined there.

## Analyses

Each analysis has a **compute** function, which returns a DataFrame or a value, and a **display** function, which prints or plots the result.

| # | Analysis | Compute / Display |
|---|---|---|
| 1 | **ADF stationarity test** on prices and returns | `compute_adf` / `display_adf` |
| 2 | **Return distribution**: mean, std, skewness, kurtosis, min, max + histogram | `compute_distribution` / `display_distribution` |
| 3 | **Rolling volatility**: average, peak and its date + time-series plot | `compute_volatility` / `display_volatility` |

## Prediction & classification

| Task | Model | Features | Benchmark |
|---|---|---|---|
| Next-day **price** (regression) | `LinearRegression` | 5 lagged prices | Naive forecast: tomorrow's price = today's price |
| Next-day **direction** (up = 1 / down = 0) | `RandomForestClassifier` and `LogisticRegression` | 5 lagged returns, 21-day volatility, distance to MA200 | Majority-class rule |

Both use a **chronological 80/20 split** (`shuffle=False`), so the models are never trained on data from the future.

## Extension — GARCH volatility model

1. **ARCH LM test** (`het_arch`), to check for conditional heteroskedasticity in returns
2. **GARCH(2,2)** estimation with the `arch` package, reporting AIC and BIC
3. **Validation** on standardised residuals:
   - Ljung-Box on residuals: no remaining autocorrelation
   - Ljung-Box on squared residuals: no remaining ARCH effect
   - Jarque-Bera: normality
4. **7-trading-day volatility forecast**
5. **PACF of squared returns**, to visualise volatility clustering

## Requirements

Python 3.9+ and:

```bash
pip install pandas numpy matplotlib yfinance statsmodels scikit-learn arch
```

An internet connection is required to download the data.

## Usage

```bash
python RUN_ALL.py
```

```
=======================================================
  SPY ANALYSIS
=======================================================
  1. Show the asset price
  2. Stationarity test - ADF
  3. Return distribution
  4. Rolling volatility
  5. Linear regression on prices
  6. Direction classification
  7. Extension - GARCH model
  0. Quit
=======================================================
```

## Project structure

```
.
├── RUN_ALL.py   # Data loading, analyses, models, GARCH extension, menu
└── README.md
```

## Notes and limitations

- **Price regression:** prices are non-stationary (close to a random walk), so a linear model on lagged prices essentially learns "tomorrow ≈ today". A low MSE is therefore not evidence of predictive power. This is why the naive benchmark is printed next to it: the model should be judged by whether it beats that benchmark. Predicting returns rather than price levels would be the more meaningful exercise.
- **Direction classification:** the S&P 500 rises on slightly more than half of trading days. For that reason the relevant benchmark is the majority-class rule, not 50%. Accuracies close to that benchmark are consistent with the weak-form efficient market hypothesis.
- **Reproducibility:** the data runs up to the current date, so the results change every time the script is run.
- **GARCH specification:** returns are modelled with a constant mean and normal innovations (the `arch` defaults). The Jarque-Bera test usually rejects normality on daily equity returns. A Student-t distribution (`dist="t"`) would be a natural refinement.
