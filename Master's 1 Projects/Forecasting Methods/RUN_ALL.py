"""
Tourism Flow Forecasting / June 2025 to June 2027
M1 ESA — Forecasting Methods
Abdellah Nait Akli, Yassir Motia, Fabian Zetu, Loïc Wagale
"""

#  Imports and configuration
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from scipy import stats
from itertools import product
import warnings, os, logging, json

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# Consistent plotting style across the whole report
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

# Report colour palette
NAVY   = "#1a2e4a"
TEAL   = "#0e7490"
AMBER  = "#d97706"
CORAL  = "#dc2626"
SAGE   = "#059669"
LBLUE  = "#bae6fd"
PURPLE = "#7c3aed"

# Colours for the decompositions (Figure 4 / Figure 5 style)
C_OBS   = "#4040E0"   # blue / violet
C_TREND = "#2A9D8F"   # teal green
C_SEAS  = "#E88C30"   # orange
C_RES   = "#D45B5B"   # red / salmon

os.makedirs("figures", exist_ok=True)

def save(name):
    """Save the current figure and close the plot."""
    plt.savefig(f"figures/{name}.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  -> {name}.png")


# 1. DATA LOADING

DATA_FILE = "Data - Projet MP - 202605.xlsx"

df = pd.read_excel(DATA_FILE)
df.columns = ["Date", "Tourists"]
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
df.index = pd.DatetimeIndex(df.index, freq="MS")

print(f"Data: {len(df)} obs, {df.index[0]:%b %Y} → {df.index[-1]:%b %Y}")


# 2. LINEAR INTERPOLATION — COVID TREATMENT

# Observations from March 2020 to March 2022 are replaced by a
# linear interpolation between February 2020 and April 2022
df_interp = df.copy()
covid_mask = df_interp.index.to_series().between("2020-03-01", "2022-03-31")
df_interp.loc[covid_mask, "Tourists"] = np.nan
df_interp["Tourists"] = df_interp["Tourists"].interpolate(method="time")

print(f"COVID interpolated: {covid_mask.sum()} observations")


# 3. EXPLORATORY STATISTICS

# For the seasonal plot and boxplots, we work on the non-COVID sample
df_ex_covid = df[~df.index.to_series().between("2020-03-01", "2022-03-31")].copy()
df_ex_covid["Month"] = df_ex_covid.index.month
df_ex_covid["Year"] = df_ex_covid.index.year

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# 4. STATIONARITY TESTS

# ADF and KPSS on the series in levels
adf_lvl  = adfuller(df_interp["Tourists"], autolag="AIC")
kpss_lvl = kpss(df_interp["Tourists"], regression="c", nlags="auto")

# After first differencing
diff1     = df_interp["Tourists"].diff().dropna()
adf_d1    = adfuller(diff1, autolag="AIC")
kpss_d1   = kpss(diff1, regression="c", nlags="auto")

print(f"\nStationarity:")
print(f"  ADF  Yt  : stat={adf_lvl[0]:.3f}  p={adf_lvl[1]:.3f}")
print(f"  KPSS Yt  : stat={kpss_lvl[0]:.3f}  p={kpss_lvl[1]:.3f}")
print(f"  ADF  dYt : stat={adf_d1[0]:.3f}  p={adf_d1[1]:.3f}")
print(f"  KPSS dYt : stat={kpss_d1[0]:.3f}  p={kpss_d1[1]:.3f}")


# 5. DECOMPOSITIONS

# On the raw series
decomp_raw_add = seasonal_decompose(df["Tourists"], model="additive", period=12)
decomp_raw_mul = seasonal_decompose(df["Tourists"], model="multiplicative", period=12)

# On the interpolated series
decomp_interp_add = seasonal_decompose(df_interp["Tourists"], model="additive", period=12)
decomp_interp_mul = seasonal_decompose(df_interp["Tourists"], model="multiplicative", period=12)

# Seasonal coefficients (additive, interpolated)
seasonal_coefs = decomp_interp_add.seasonal.iloc[:12].values


# 6. TRAIN / TEST SPLIT

# First 88 obs for estimation, last 24 for testing
train  = df_interp.iloc[:-24]
test   = df_interp.iloc[-24:]
n_test = len(test)

print(f"\nSplit: Train n={len(train)} | Test n={n_test}")


# 7a. POLYNOMIAL DECOMPOSITION (naive benchmark)

decomp_train = seasonal_decompose(train["Tourists"], model="additive", period=12)
trend_tr = decomp_train.trend.dropna()
seasonal_coefs_tr = decomp_train.seasonal.iloc[:12].values

# Degree-2 polynomial fitted on the trend
poly_coeffs = np.polyfit(np.arange(len(trend_tr)), trend_tr, 2)
x_fut = np.arange(len(trend_tr), len(trend_tr) + n_test)
pred_decomp = np.polyval(poly_coeffs, x_fut) + \
    [seasonal_coefs_tr[(test.index[i].month - 1) % 12] for i in range(n_test)]

rmse_decomp = np.sqrt(mean_squared_error(test["Tourists"], pred_decomp))
mape_decomp = mean_absolute_percentage_error(test["Tourists"], pred_decomp) * 100
print(f"\nPolynomial decomp.: MAPE = {mape_decomp:.1f}%")



# 7b. HOLT-WINTERS: ADDITIVE vs MULTIPLICATIVE

hw_add = ExponentialSmoothing(
    train["Tourists"], trend="add", seasonal="add", seasonal_periods=12
).fit(optimized=True)
pred_hw_add = hw_add.forecast(n_test)
rmse_hw_add = np.sqrt(mean_squared_error(test["Tourists"], pred_hw_add))
mape_hw_add = mean_absolute_percentage_error(test["Tourists"], pred_hw_add) * 100

hw_mul = ExponentialSmoothing(
    train["Tourists"], trend="add", seasonal="mul", seasonal_periods=12
).fit(optimized=True)
pred_hw_mul = hw_mul.forecast(n_test)
rmse_hw_mul = np.sqrt(mean_squared_error(test["Tourists"], pred_hw_mul))
mape_hw_mul = mean_absolute_percentage_error(test["Tourists"], pred_hw_mul) * 100

print(f"HW Additive       : α={hw_add.params['smoothing_level']:.4f}  "
      f"β={hw_add.params['smoothing_trend']:.4f}  "
      f"γ={hw_add.params['smoothing_seasonal']:.4f}  MAPE={mape_hw_add:.1f}%")
print(f"HW Multiplicative : α={hw_mul.params['smoothing_level']:.4f}  "
      f"β={hw_mul.params['smoothing_trend']:.4f}  "
      f"γ={hw_mul.params['smoothing_seasonal']:.4f}  MAPE={mape_hw_mul:.1f}%")

# Keep the best one
if mape_hw_add <= mape_hw_mul:
    pred_hw_test = pred_hw_add
    rmse_hw, mape_hw, hw_type = rmse_hw_add, mape_hw_add, "additive"
else:
    pred_hw_test = pred_hw_mul
    rmse_hw, mape_hw, hw_type = rmse_hw_mul, mape_hw_mul, "multiplicative"
print(f"  => Selected: HW {hw_type}")


# 7c. SARIMA — GRID SEARCH

print("\nSARIMA grid search (36 specifications)...")
sarima_results = []

for p, q, P, Q in product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
    try:
        m = SARIMAX(
            train["Tourists"],
            order=(p, 1, q),
            seasonal_order=(P, 1, Q, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=200)

        pred = m.forecast(n_test)
        mv = mean_absolute_percentage_error(test["Tourists"], pred) * 100
        rv = np.sqrt(mean_squared_error(test["Tourists"], pred))
        lb = acorr_ljungbox(m.resid.dropna(), lags=[12], return_df=True)

        sarima_results.append({
            "order": (p, 1, q),
            "seasonal": (P, 1, Q, 12),
            "aic": m.aic,
            "bic": m.bic,
            "mape": mv,
            "rmse": rv,
            "lb12_p": lb["lb_pvalue"].values[0],
            "model": m
        })
    except Exception:
        pass

# Sorted by test MAPE (not by AIC!)
sarima_results.sort(key=lambda x: x["mape"])
best_sar = sarima_results[0]
sarima_train = best_sar["model"]
pred_sarima_test = sarima_train.forecast(n_test)
rmse_sarima = best_sar["rmse"]
mape_sarima = best_sar["mape"]

print(f"Best SARIMA: {best_sar['order']}{best_sar['seasonal'][:3]}  "
      f"AIC={best_sar['aic']:.2f}  MAPE={mape_sarima:.1f}%")

# Display the 8 best by AIC for the report table
top8 = sorted(sarima_results, key=lambda x: x["aic"])[:8]
print("  Top 8 by AIC:")
for r in top8:
    label = f"  SARIMA{r['order']}{r['seasonal'][:3]}"
    print(f"  {label:<30} AIC={r['aic']:>8.2f}  MAPE={r['mape']:.1f}%")


# 7d. PROPHET

print("\nProphet...")
df_pr = train.reset_index()
df_pr.columns = ["ds", "y"]

prophet_m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="additive",
    changepoint_prior_scale=0.05
)
prophet_m.fit(df_pr)
fc_pr = prophet_m.predict(prophet_m.make_future_dataframe(periods=n_test, freq="MS"))
pred_prophet_test = fc_pr.iloc[-n_test:]["yhat"].values

rmse_prophet = np.sqrt(mean_squared_error(test["Tourists"], pred_prophet_test))
mape_prophet = mean_absolute_percentage_error(test["Tourists"], pred_prophet_test) * 100
print(f"Prophet: MAPE = {mape_prophet:.1f}%")


# 7e. RESIDUAL BOOTSTRAP & BAGGING

# See course on Bootstrap for Time Series — Bergmeir et al. (2016)
# Step 1: STL decomposition on the training set
# Step 2: extract the residuals
# Step 3: resample with replacement (classic bootstrap)
# Step 4: rebuild series = trend + seasonality + bootstrapped residuals
# Step 5: fit an additive HW model on each series
# Step 6: average the forecasts

print("\nResidual Bootstrap (Bagging)...")
N_BOOT = 100
np.random.seed(42)

stl_train = STL(train["Tourists"], period=12, robust=True).fit()

boot_preds = []
for b in range(N_BOOT):
    resid_b = np.random.choice(stl_train.resid.values, size=len(train), replace=True)
    y_b = pd.Series(
        stl_train.trend.values + stl_train.seasonal.values + resid_b,
        index=train.index
    )
    try:
        hw_b = ExponentialSmoothing(
            y_b, trend="add", seasonal="add", seasonal_periods=12
        ).fit(optimized=True)
        boot_preds.append(hw_b.forecast(n_test).values)
    except Exception:
        pass

boot_preds = np.array(boot_preds)
pred_bag_test = boot_preds.mean(axis=0)
pred_bag_lo   = np.percentile(boot_preds, 5, axis=0)
pred_bag_hi   = np.percentile(boot_preds, 95, axis=0)

rmse_bag = np.sqrt(mean_squared_error(test["Tourists"], pred_bag_test))
mape_bag = mean_absolute_percentage_error(test["Tourists"], pred_bag_test) * 100
print(f"Bagging ({len(boot_preds)} replications): MAPE = {mape_bag:.1f}%")


# 7f. WEIGHTED ENSEMBLE (HW + SARIMA)

# Weights inversely proportional to MAPE
w_hw  = mape_sarima / (mape_hw + mape_sarima)
w_sar = mape_hw / (mape_hw + mape_sarima)

pred_ens_test = w_hw * pred_hw_test.values + w_sar * pred_sarima_test.values
rmse_ens = np.sqrt(mean_squared_error(test["Tourists"], pred_ens_test))
mape_ens = mean_absolute_percentage_error(test["Tourists"], pred_ens_test) * 100

print(f"\nEnsemble (w_HW={w_hw:.2f}, w_SAR={w_sar:.2f}): MAPE = {mape_ens:.1f}%")


# PERFORMANCE SUMMARY TABLE

print("\n" + "=" * 60)
print("  OUT-OF-SAMPLE PERFORMANCE")
print("=" * 60)
for name, rm, ma in [
    ("Polynomial decomposition", rmse_decomp, mape_decomp),
    ("HW additive", rmse_hw_add, mape_hw_add),
    ("HW multiplicative", rmse_hw_mul, mape_hw_mul),
    (f"SARIMA{best_sar['order']}{best_sar['seasonal'][:3]}", rmse_sarima, mape_sarima),
    ("Prophet", rmse_prophet, mape_prophet),
    ("Bootstrap Bagging", rmse_bag, mape_bag),
    ("Weighted ensemble", rmse_ens, mape_ens),
]:
    print(f"  {name:<35} RMSE={rm:.3f}  MAPE={ma:.1f}%")


# 8. FINAL FORECASTS — JUNE 2025 TO JUNE 2027

# Re-estimation on the full sample of 112 observations
N_FC = 25
fi = pd.date_range("2025-06-01", periods=N_FC, freq="MS")

# Additive HW on the full series
hw_full = ExponentialSmoothing(
    df_interp["Tourists"], trend="add", seasonal="add", seasonal_periods=12
).fit(optimized=True)
pred_hw_f = hw_full.forecast(N_FC)

# SARIMA on the full series
sarima_full = SARIMAX(
    df_interp["Tourists"],
    order=best_sar["order"],
    seasonal_order=best_sar["seasonal"],
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
fc_obj = sarima_full.get_forecast(steps=N_FC)
pred_sar_f = fc_obj.predicted_mean
ci90 = fc_obj.conf_int(alpha=0.10)

# Bootstrap on the full series
stl_full = STL(df_interp["Tourists"], period=12, robust=True).fit()
boot_f = []
np.random.seed(42)
for b in range(N_BOOT):
    rb = np.random.choice(stl_full.resid.values, size=len(df_interp), replace=True)
    yb = pd.Series(
        stl_full.trend.values + stl_full.seasonal.values + rb,
        index=df_interp.index
    )
    try:
        hb = ExponentialSmoothing(
            yb, trend="add", seasonal="add", seasonal_periods=12
        ).fit(optimized=True)
        boot_f.append(hb.forecast(N_FC).values)
    except Exception:
        pass

boot_f = np.array(boot_f)
pred_bag_f    = boot_f.mean(axis=0)
pred_bag_f_lo = np.percentile(boot_f, 5, axis=0)
pred_bag_f_hi = np.percentile(boot_f, 95, axis=0)

# Final ensemble
pred_ens_f = w_hw * pred_hw_f.values + w_sar * pred_sar_f.values

# Print the forecasts
print("\n=== Final forecasts ===")
for d, s, h, bg, e, lo, hi in zip(
    fi, pred_sar_f, pred_hw_f, pred_bag_f, pred_ens_f,
    ci90.iloc[:, 0], ci90.iloc[:, 1]
):
    print(f"  {d.year} {d:%b}  SAR={s:.3f}  HW={h:.3f}  "
          f"Bag={bg:.3f}  Ens={e:.3f}  CI90=[{lo:.3f} ; {hi:.3f}]")



#                       FIGURE GENERATION

print("\n=== Generating figures ===")


# ------ FIGURE 1: Raw series with COVID period ------
fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(df.index, df["Tourists"], alpha=0.12, color=TEAL)
ax.plot(df.index, df["Tourists"], color=TEAL, lw=1.8, label="Monthly arrivals")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2022-03-31"),
           alpha=0.10, color=CORAL, label="COVID-19 disruption")
ax.axvline(pd.Timestamp("2020-03-01"), color=CORAL, lw=1, ls="--")
ax.axvline(pd.Timestamp("2022-03-31"), color=CORAL, lw=1, ls="--")
ax.set_ylabel("Arrivals (millions)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=10)
plt.tight_layout()
save("fig1_raw_series")


# ------ FIGURE 2: ACF / PACF of the raw series ------
nlags_raw = 48
acf_raw  = acf(df["Tourists"].dropna(), nlags=nlags_raw, fft=True)
pacf_raw = pacf(df["Tourists"].dropna(), nlags=nlags_raw, method="ywm")
cb_raw = 1.96 / np.sqrt(len(df))

fig, axes = plt.subplots(2, 1, figsize=(11, 7))

# ACF
axes[0].bar(range(nlags_raw + 1), acf_raw, color=TEAL, width=0.4)
axes[0].fill_between(range(nlags_raw + 1), -cb_raw, cb_raw, alpha=0.15, color=LBLUE)
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_title("ACF — Autocorrelation function", fontweight="bold", color=NAVY)
axes[0].set_ylabel("Autocorrelation")

# PACF
axes[1].bar(range(nlags_raw + 1), pacf_raw, color=NAVY, width=0.4)
axes[1].fill_between(range(nlags_raw + 1), -cb_raw, cb_raw, alpha=0.15, color=LBLUE)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_title("PACF — Partial autocorrelation function", fontweight="bold", color=NAVY)
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("Partial autocorrelation")

plt.tight_layout()
save("fig2_acf_pacf_raw")


# ------ FIGURE 3: Seasonal plot + monthly boxplots ------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Seasonal plot: overlay each year
for year in sorted(df_ex_covid["Year"].unique()):
    sub = df_ex_covid[df_ex_covid["Year"] == year]
    axes[0].plot(sub["Month"], sub["Tourists"], marker="o", ms=3,
                 label=str(year), alpha=0.7)

# Also add the COVID years to show the break
for year in [2020, 2021]:
    sub = df[df.index.year == year]
    axes[0].plot(sub.index.month, sub["Tourists"], marker="o", ms=3,
                 label=str(year), ls="--", alpha=0.5)

axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_labels)
axes[0].set_title("Seasonal Plot", fontweight="bold", color=NAVY)
axes[0].set_ylabel("Arrivals (millions)")
axes[0].legend(fontsize=7, ncol=2)

# Boxplots excluding COVID
data_bp = [df_ex_covid[df_ex_covid["Month"] == m]["Tourists"].values
           for m in range(1, 13)]
bp = axes[1].boxplot(data_bp, labels=month_labels, patch_artist=True,
                     medianprops=dict(color=AMBER, lw=2))
for p in bp["boxes"]:
    p.set_facecolor(LBLUE)
    p.set_edgecolor(TEAL)
axes[1].set_title("Monthly boxplots (excluding COVID)", fontweight="bold", color=NAVY)
axes[1].set_ylabel("Arrivals (millions)")

plt.tight_layout()
save("fig3_seasonal_boxplots")


# ------ FIGURE 4: Additive + multiplicative decomposition (RAW series) ------
fig, axes = plt.subplots(4, 2, figsize=(16, 10), sharex=True)

decomp_titles = ["ADDITIVE", "MULTIPLICATIVE"]
decomps_raw = [decomp_raw_add, decomp_raw_mul]

components = [
    ("Observed",    lambda d: d.observed, C_OBS),
    ("Trend",       lambda d: d.trend,    C_TREND),
    ("Seasonality", lambda d: d.seasonal, C_SEAS),
    ("Residuals",   lambda d: d.resid,    C_RES),
]

for col, (title, decomp) in enumerate(zip(decomp_titles, decomps_raw)):
    axes[0, col].set_title(title, fontsize=14, fontweight="bold", color="#2A2A8A")
    for row, (label, getter, color) in enumerate(components):
        data = getter(decomp)
        ax = axes[row, col]
        if label == "Residuals":
            baseline = 0 if col == 0 else 1
            ax.fill_between(data.index, baseline, data.values, alpha=0.25, color=color)
            ax.plot(data.index, data.values, color=color, linewidth=0.8)
            ax.axhline(y=baseline, color="grey", linewidth=0.5)
        else:
            ax.plot(data.index, data.values, color=color, linewidth=1.2)
        if col == 0:
            ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

plt.tight_layout()
save("fig4_decomp_raw")


# ------ FIGURE 5: Additive + multiplicative decomposition (INTERPOLATED series) ------
fig, axes = plt.subplots(4, 2, figsize=(16, 10), sharex=True)

decomps_interp = [decomp_interp_add, decomp_interp_mul]

for col, (title, decomp) in enumerate(zip(decomp_titles, decomps_interp)):
    axes[0, col].set_title(title, fontsize=14, fontweight="bold", color="#2A2A8A")
    for row, (label, getter, color) in enumerate(components):
        data = getter(decomp)
        ax = axes[row, col]
        if label == "Residuals":
            baseline = 0 if col == 0 else 1
            ax.fill_between(data.index, baseline, data.values, alpha=0.25, color=color)
            ax.plot(data.index, data.values, color=color, linewidth=0.8)
            ax.axhline(y=baseline, color="grey", linewidth=0.5)
        else:
            ax.plot(data.index, data.values, color=color, linewidth=1.2)
        if col == 0:
            ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

plt.tight_layout()
save("fig5_decomp_interpolated")


# ------ FIGURE 6: Additive vs multiplicative HW (test period) ------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(test.index, test["Tourists"], color=NAVY, lw=2,
        label="Observed", marker="o", ms=4)
ax.plot(test.index, pred_hw_add, color=TEAL, lw=1.5, ls="--",
        label=f"HW Additive (MAPE={mape_hw_add:.1f}%)")
ax.plot(test.index, pred_hw_mul, color=CORAL, lw=1.5, ls="-.",
        label=f"HW Multiplicative (MAPE={mape_hw_mul:.1f}%)")
ax.set_ylabel("Arrivals (millions)")
ax.set_ylim(bottom=0)
ax.set_title("Holt-Winters: Additive vs Multiplicative (test period)",
             fontweight="bold", color=NAVY, fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
save("fig6_hw_comparison")


# ------ ACF/PACF after double differencing (for the Box-Jenkins section) ------
dd = df_interp["Tourists"].diff(12).diff(1).dropna()
td = dd[dd.index <= train.index[-1]]
nl = 36
av = acf(td, nlags=nl, fft=True)
pv = pacf(td, nlags=nl, method="ywm")
cb = 1.96 / np.sqrt(len(td))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Correlograms after first and seasonal differencing",
             fontweight="bold", color=NAVY, fontsize=12)

axes[0].bar(range(nl + 1), av, color=TEAL, width=0.3)
axes[0].axhline(cb, color=CORAL, ls="--")
axes[0].axhline(-cb, color=CORAL, ls="--")
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_title("ACF", color=NAVY)
axes[0].set_xlabel("Lag (months)")

axes[1].bar(range(nl + 1), pv, color=NAVY, width=0.3)
axes[1].axhline(cb, color=CORAL, ls="--")
axes[1].axhline(-cb, color=CORAL, ls="--")
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_title("PACF", color=NAVY)
axes[1].set_xlabel("Lag (months)")

plt.tight_layout()
save("fig_acf_pacf_diff")


# ------ FIGURE 7: SARIMA residual diagnostics ------
rs = sarima_train.resid.dropna()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    f"Residual diagnostics — SARIMA{best_sar['order']}{best_sar['seasonal'][:3]}",
    fontweight="bold", color=NAVY, fontsize=12
)

# Residuals over time
axes[0, 0].plot(rs.index, rs, color=TEAL, lw=1)
axes[0, 0].axhline(0, color=CORAL, ls="--")
axes[0, 0].set_title("Residuals over time", color=NAVY)

# Distribution
axes[0, 1].hist(rs, bins=20, color=LBLUE, edgecolor=TEAL, density=True)
xn = np.linspace(rs.min(), rs.max(), 100)
axes[0, 1].plot(xn, stats.norm.pdf(xn, rs.mean(), rs.std()),
                color=CORAL, lw=2, label="N(0,σ²)")
axes[0, 1].set_title("Distribution", color=NAVY)
axes[0, 1].legend(fontsize=9)

# QQ-Plot
stats.probplot(rs, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("QQ-Plot", color=NAVY)
axes[1, 0].get_lines()[0].set_color(TEAL)
axes[1, 0].get_lines()[1].set_color(CORAL)

# Residual ACF
racf = acf(rs, nlags=24, fft=True)
cr = 1.96 / np.sqrt(len(rs))
axes[1, 1].bar(range(25), racf, color=TEAL, width=0.3)
axes[1, 1].axhline(cr, color=CORAL, ls="--")
axes[1, 1].axhline(-cr, color=CORAL, ls="--")
axes[1, 1].axhline(0, color="black", lw=0.5)
axes[1, 1].set_title("Residual ACF", color=NAVY)

plt.tight_layout()
save("fig7_sarima_diagnostics")


# ------ FIGURE 8: STL decomposition ------
fig, axes = plt.subplots(4, 1, figsize=(11, 9))
stl_data = [df_interp["Tourists"], stl_full.trend, stl_full.seasonal, stl_full.resid]
stl_labels = ["Observed series", "Trend (LOESS)", "Seasonality", "Residuals"]
stl_colors = [TEAL, NAVY, AMBER, CORAL]

for ax, data, label, co in zip(axes, stl_data, stl_labels, stl_colors):
    ax.plot(df_interp.index, data, color=co, lw=1.5)
    if label == "Residuals":
        ax.fill_between(df_interp.index, data, alpha=0.3, color=co)
        ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel(label, fontsize=10, color=NAVY)
    if label != "Residuals":
        ax.set_xticklabels([])

axes[0].set_title("STL Decomposition (Seasonal-Trend using LOESS)",
                   fontweight="bold", color=NAVY, fontsize=12)
axes[-1].set_xlabel("Date")
plt.tight_layout()
save("fig8_stl")


# ------ FIGURE 9: Bootstrap (series + forecasts) ------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: bootstrapped series
axes[0].plot(train.index, train["Tourists"], color=NAVY, lw=1.5,
             label="Original series")
np.random.seed(99)
for i in range(10):
    rb = np.random.choice(stl_train.resid.values, size=len(train), replace=True)
    y_boot = stl_train.trend.values + stl_train.seasonal.values + rb
    axes[0].plot(train.index, y_boot, color=TEAL, alpha=0.15, lw=0.8)
axes[0].set_title("Bootstrapped series (10 samples)",
                   fontweight="bold", color=NAVY)
axes[0].set_ylabel("Arrivals (millions)")

# Right: bagging forecasts with interval
axes[1].plot(test.index, test["Tourists"], color=NAVY, lw=2,
             label="Observed", marker="o", ms=4)
axes[1].fill_between(test.index, pred_bag_lo, pred_bag_hi,
                     alpha=0.2, color=TEAL, label="90% Bootstrap interval")
axes[1].plot(test.index, pred_bag_test, color=TEAL, lw=2,
             label=f"Bagging (MAPE={mape_bag:.1f}%)")
axes[1].set_title("Bagging forecasts", fontweight="bold", color=NAVY)
axes[1].set_ylabel("Arrivals (millions)")
axes[1].legend(fontsize=9)

plt.tight_layout()
save("fig9_bootstrap")


# ------ FIGURE 10: Comparison of all models (test period) ------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df_interp.index, df_interp["Tourists"], color=NAVY, lw=1.5,
        label="Observed")
ax.plot(test.index, pred_sarima_test, color=CORAL, lw=1.5, ls="--",
        label=f"SARIMA (MAPE={mape_sarima:.1f}%)")
ax.plot(test.index, pred_hw_test, color=SAGE, lw=1.5, ls="-.",
        label=f"HW {hw_type} (MAPE={mape_hw:.1f}%)")
ax.plot(test.index, pred_bag_test, color=PURPLE, lw=1.5, ls=":",
        label=f"Bagging (MAPE={mape_bag:.1f}%)")
ax.plot(test.index, pred_ens_test, color=AMBER, lw=2,
        label=f"Ensemble (MAPE={mape_ens:.1f}%)")
ax.axvline(test.index[0], color="gray", ls=":", lw=1.2)
ax.set_ylabel("Arrivals (millions)")
ax.set_ylim(bottom=0)
ax.set_title("Model comparison over the test period",
             fontweight="bold", color=NAVY, fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
save("fig10_model_comparison")


# ------ FIGURE 11: Final forecasts June 2025 – June 2027 ------
fig, ax = plt.subplots(figsize=(12, 5))
hist = df_interp.iloc[-30:]
ax.plot(hist.index, hist["Tourists"], color=NAVY, lw=1.8, label="History")
ax.fill_between(fi, ci90.iloc[:, 0], ci90.iloc[:, 1],
                alpha=0.12, color=TEAL, label="90% CI SARIMA")
ax.fill_between(fi, pred_bag_f_lo, pred_bag_f_hi,
                alpha=0.12, color=PURPLE, label="90% CI Bootstrap")
ax.plot(fi, pred_sar_f, color=TEAL, lw=1.5, ls="--", label="SARIMA")
ax.plot(fi, pred_hw_f, color=SAGE, lw=1.5, ls="-.", label="Holt-Winters")
ax.plot(fi, pred_bag_f, color=PURPLE, lw=1.5, ls=":", label="Bagging")
ax.plot(fi, pred_ens_f, color=AMBER, lw=2.5,
        label="Weighted ensemble (selected)")
ax.axvline(pd.Timestamp("2025-06-01"), color="gray", ls=":", lw=1.2)
ax.set_ylabel("Arrivals (millions)")
ax.set_ylim(bottom=0)
ax.set_title("Final forecasts: June 2025 – June 2027",
             fontweight="bold", color=NAVY, fontsize=12)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
save("fig11_final_forecasts")



# SAVE RESULTS

res = {
    "adf_lvl_p": round(adf_lvl[1], 4),
    "adf_lvl_s": round(adf_lvl[0], 4),
    "kpss_lvl_p": round(kpss_lvl[1], 4),
    "kpss_lvl_s": round(kpss_lvl[0], 4),
    "adf_d1_p": round(adf_d1[1], 4),
    "kpss_d1_p": round(kpss_d1[1], 4),
    "hw_add_a": round(hw_add.params["smoothing_level"], 4),
    "hw_add_b": round(hw_add.params["smoothing_trend"], 4),
    "hw_add_g": round(hw_add.params["smoothing_seasonal"], 4),
    "hw_mul_a": round(hw_mul.params["smoothing_level"], 4),
    "hw_mul_b": round(hw_mul.params["smoothing_trend"], 4),
    "hw_mul_g": round(hw_mul.params["smoothing_seasonal"], 4),
    "mape_hw_add": round(mape_hw_add, 1),
    "mape_hw_mul": round(mape_hw_mul, 1),
    "best_sarima": f"{best_sar['order']}{best_sar['seasonal'][:3]}",
    "sarima_aic": round(best_sar["aic"], 2),
    "mape_sarima": round(mape_sarima, 1),
    "mape_prophet": round(mape_prophet, 1),
    "mape_bag": round(mape_bag, 1),
    "mape_ens": round(mape_ens, 1),
    "w_hw": round(w_hw, 2),
    "w_sar": round(w_sar, 2),
    "hw_type": hw_type,
}
with open("results.json", "w") as f:
    json.dump(res, f, indent=2)

print("\nDone — all figures are in ./figures/")
