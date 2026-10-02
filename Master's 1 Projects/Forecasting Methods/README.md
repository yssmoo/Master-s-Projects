# Tourism Flow Forecasting — June 2025 to June 2027

**M1 ESA (Applied Econometrics & Statistics) — Forecasting Methods, Université d'Orléans**

Authors: Abdellah Nait Akli, Yassir Motia, Fabian Zetu, Loïc Wagale

---

## Overview

This project forecasts monthly tourist arrivals (in millions) over a 25-month horizon, from **June 2025 to June 2027**. It compares several classical and modern time-series forecasting methods on a hold-out test set, then combines the best performers into a weighted ensemble used for the final forecast.

The whole pipeline (data preparation, statistical tests, model estimation, evaluation, figures and result export) runs from a single script: `RUN_ALL.py`.

## Data

- **File:** `Data - Projet MP - 202605.xlsx` (must be placed in the same folder as the script)
- **Structure:** two columns — a monthly date and the number of tourist arrivals (millions)
- **Frequency:** monthly (month start), 112 observations, ending May 2025

### COVID-19 treatment

Observations from **March 2020 to March 2022** are treated as a structural disruption. They are removed and replaced by a **linear (time-based) interpolation** between February 2020 and April 2022, so that the models learn the "normal" dynamics of the series rather than the shock.

## Methodology

1. **Exploratory analysis** — raw series plot, ACF/PACF, seasonal plot and monthly boxplots (excluding the COVID period).
2. **Stationarity tests** — ADF and KPSS on the series in levels and after first differencing.
3. **Decompositions** — classical additive and multiplicative decompositions on both the raw and interpolated series, plus an STL (LOESS) decomposition.
4. **Train / test split** — first 88 observations for estimation, last 24 for out-of-sample evaluation.
5. **Models compared:**

| Model | Description |
|---|---|
| Polynomial decomposition | Naive benchmark: degree-2 polynomial trend + additive seasonal coefficients |
| Holt-Winters | Additive and multiplicative seasonality; the one with the lower test MAPE is kept |
| SARIMA | Grid search over 36 specifications SARIMA(p,1,q)(P,1,Q)₁₂ with p, q ∈ {0,1,2} and P, Q ∈ {0,1}; Ljung-Box test on residuals |
| Prophet | Additive yearly seasonality, `changepoint_prior_scale = 0.05` |
| Bootstrap Bagging | Residual bootstrap on an STL decomposition (Bergmeir et al., 2016), 100 replications, additive HW fitted on each series and forecasts averaged |
| Weighted ensemble | Combination of Holt-Winters and SARIMA, with weights inversely proportional to each model's test MAPE |

6. **Evaluation metrics** — RMSE and MAPE on the 24-month test set.
7. **Final forecasts** — all retained models are re-estimated on the full sample (112 observations) and projected 25 months ahead, with 90% intervals from SARIMA and from the bootstrap.

## Requirements

Python 3.9+ and the following packages:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn prophet scipy openpyxl
```

`openpyxl` is required to read the Excel data file. `prophet` installs `cmdstanpy` as a dependency.

## Usage

```bash
python RUN_ALL.py
```

The script prints the test results, the SARIMA grid search summary and the final forecasts to the console. Full execution takes a few minutes, mainly because of the SARIMA grid search and the 2 × 100 bootstrap replications.

## Outputs

### `figures/`

| File | Content |
|---|---|
| `fig1_raw_series.png` | Raw series with the COVID-19 period highlighted |
| `fig2_acf_pacf_raw.png` | ACF and PACF of the raw series (48 lags) |
| `fig3_seasonal_boxplots.png` | Seasonal plot and monthly boxplots (excluding COVID) |
| `fig4_decomp_raw.png` | Additive vs multiplicative decomposition — raw series |
| `fig5_decomp_interpolated.png` | Additive vs multiplicative decomposition — interpolated series |
| `fig6_hw_comparison.png` | Additive vs multiplicative Holt-Winters on the test period |
| `fig_acf_pacf_diff.png` | ACF/PACF after first and seasonal differencing (Box-Jenkins identification) |
| `fig7_sarima_diagnostics.png` | Residual diagnostics of the selected SARIMA (time plot, histogram, QQ-plot, ACF) |
| `fig8_stl.png` | STL decomposition of the interpolated series |
| `fig9_bootstrap.png` | Bootstrapped series and bagging forecasts with 90% interval |
| `fig10_model_comparison.png` | All models compared over the test period |
| `fig11_final_forecasts.png` | Final forecasts June 2025 – June 2027 with 90% intervals |

### `results.json`

Key numerical results used in the report: ADF/KPSS statistics and p-values, Holt-Winters smoothing parameters (α, β, γ), the selected SARIMA specification and its AIC, test MAPE for each model, and the ensemble weights.

## Project structure

```
.
├── RUN_ALL.py                         # Full pipeline
├── Data - Projet MP - 202605.xlsx     # Input data
├── README.md
├── results.json                       # Generated
└── figures/                           # Generated
```

## Notes and limitations

- The SARIMA specification is selected on **test-set MAPE**, not on an information criterion. This makes the test set part of model selection, so the reported test MAPE of the selected SARIMA (and of the ensemble, whose weights also depend on test MAPE) is optimistic. A cleaner approach would use AIC/BIC or a separate validation window / rolling-origin cross-validation.
- The COVID interpolation assumes the pre-2020 dynamics would have continued unchanged; forecasts inherit that assumption.
- Random seeds are fixed (`42` for the bagging, `99` for the illustrative bootstrap figure) for reproducibility.

## Reference

Bergmeir, C., Hyndman, R. J., & Benítez, J. M. (2016). Bagging exponential smoothing methods using STL decomposition and Box-Cox transformation. *International Journal of Forecasting*, 32(2), 303–312.
