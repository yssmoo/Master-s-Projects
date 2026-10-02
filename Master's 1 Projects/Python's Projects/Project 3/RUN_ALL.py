#                        Project 3: Preparation, Analysis and Prediction


# Abdellah Nait Akli
# Yassir Motia
# M1 ESA

################# Package imports ###########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

from arch import arch_model


# 1. Data import and preparation


# An ETF (Exchange Traded Fund) is an asset that gives exposure to a diversified
# basket of stocks. We chose SPY, the ETF tracking the S&P 500 index.

def load_data(symbol="SPY", start="2000-01-01"):

    """
    Download SPY data through the yfinance API.
    Return a DataFrame with the columns Close, returns, volatility, ma200.
    The MA200 (200-day moving average) gives an idea of the current market
    trend. It is widely followed by institutional investors, hence this choice.
    """

    df = yf.download(symbol, start=start, end=datetime.today(), progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.loc[:, df.columns.get_level_values(0) == "Close"].copy()
        df.columns = ["Close"]
    else:
        df = df[["Close"]].copy()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()

    # Returns in percent
    df["returns"] = 100 * df["Close"].pct_change()

    # 21-day rolling volatility
    df["volatility"] = df["returns"].rolling(21).std()

    # 200-day moving average
    df["ma200"] = df["Close"].rolling(200).mean()

    df = df.dropna()
    return df


# 2. Statistical analyses

## Analysis 1: ADF test — the Augmented Dickey-Fuller test checks whether a time series is stationary

def compute_adf(series):

    """Run the ADF test. Return the statistic, the p-value and the conclusion."""

    result = adfuller(series.dropna())
    return {"stat": result[0],
            "p_value": result[1],
            "stationary": result[1] < 0.05}


def display_adf(adf_price, adf_ret):

    """Print the ADF results for prices and returns."""

    print("\n ADF test")
    print(f"  Prices  -> stat = {adf_price['stat']:.4f}, p-value = {adf_price['p_value']:.5f}")
    if adf_price["stationary"]:
        print("            => Stationary")
    else:
        print("            => Non-stationary")

    print(f"  Returns -> stat = {adf_ret['stat']:.4f}, p-value = {adf_ret['p_value']:.5f}")
    if adf_ret["stationary"]:
        print("            => Stationary")
    else:
        print("            => Non-stationary")


## Analysis 2: Return distribution

def compute_distribution(df):

    """Compute descriptive statistics of returns."""

    r = df["returns"]
    stats = {"mean": r.mean(),
             "std": r.std(),
             "skewness": r.skew(),
             "kurtosis": r.kurtosis(),
             "min": r.min(),
             "max": r.max()}
    return pd.DataFrame(stats, index=["returns"])


def display_distribution(df, stats_df):

    """Print the statistics and plot the histogram."""

    print("\n Return distribution")
    print(stats_df.to_string())

    plt.figure(figsize=(10, 5))
    plt.hist(df["returns"], bins=100, color="#0000FF", edgecolor="white", density=True)
    plt.xlabel("Returns (%)")
    plt.ylabel("Density")
    plt.title("SPY return distribution")
    plt.show()


## Analysis 3: Rolling volatility

def compute_volatility(df):
    """Compute rolling volatility and return its summary statistics."""
    vol = df[["volatility"]].copy()
    stats = {"avg_vol": vol["volatility"].mean(),
             "max_vol": vol["volatility"].max(),
             "max_vol_date": vol["volatility"].idxmax()}
    return vol, stats


def display_volatility(vol, stats):

    """Plot the volatility and print its statistics."""

    print("\n 21-day rolling volatility")
    print(f"  Average : {stats['avg_vol']:.4f} %")
    print(f"  Maximum : {stats['max_vol']:.4f} % (on {stats['max_vol_date'].date()})")

    plt.figure(figsize=(10, 5))
    plt.plot(vol.index, vol["volatility"], color="#0000FF", linewidth=0.8)
    plt.ylabel("Volatility (%)")
    plt.title("SPY rolling volatility (21-day window)")
    plt.show()


# 3. Prediction with scikit-learn

## 3a: Linear regression on prices

def price_regression(df):

    """
    Linear regression: predict the next day's price from the last 5 prices.
    """

    dfp = df[["Close"]].copy()

    for i in range(1, 6):
        dfp[f"lag_{i}"] = dfp["Close"].shift(i)
    dfp["target"] = dfp["Close"].shift(-1)
    dfp = dfp.dropna()

    X = dfp[[f"lag_{i}" for i in range(1, 6)]]
    y = dfp["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)

    # Naive benchmark: tomorrow's price = today's price (random walk)
    naive_mse = mean_squared_error(y_test, dfp.loc[y_test.index, "Close"])

    return mse, naive_mse, y_test, pred, model.coef_


def display_regression(mse, naive_mse, y_test, pred, coefs):
    """Print the regression results."""
    print("\n Linear regression on prices")
    print(f"  MSE                         = {mse:.2f}")
    print(f"  Naive MSE (price unchanged) = {naive_mse:.2f}")
    for i, c in enumerate(coefs):
        print(f"  Coef lag_{i+1} = {c:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test.values, label="Actual", linewidth=0.8)
    plt.plot(y_test.index, pred, label="Predicted", linewidth=0.8, color="red")
    plt.legend()
    plt.title("Price prediction (linear regression)")
    plt.show()


## 3b: Return direction classification

def direction_classification(df):

    """
    Supervised classification: predict whether the next day's return
    will be positive (1) or negative (0).
    Features: past returns, volatility, distance to the MA200.
    """

    dfc = df.copy()

    for i in range(1, 6):
        dfc[f"ret_lag_{i}"] = dfc["returns"].shift(i)
    dfc["dist_ma200"] = (dfc["Close"] - dfc["ma200"]) / dfc["ma200"]

    # Target: up or down the next day
    dfc["target"] = (dfc["returns"].shift(-1) > 0).astype(int)
    dfc = dfc.iloc[:-1]  # last row has no next-day return
    dfc = dfc.dropna()

    features = [f"ret_lag_{i}" for i in range(1, 6)] + ["volatility", "dist_ma200"]
    X = dfc[features]
    y = dfc["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)

    # Logistic regression for comparison
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, pred_lr)

    # Majority-class benchmark: always predict the most frequent class of the training set
    majority_class = int(y_train.mean() >= 0.5)
    acc_majority = (y_test == majority_class).mean()

    return {"rf_acc": acc_rf,
            "lr_acc": acc_lr,
            "majority_acc": acc_majority,
            "lr_pred": pred_lr,
            "rf_pred": pred_rf,
            "y_test": y_test,
            "features": features,
            "importances": rf.feature_importances_}


def display_classification(results):

    """Print the classification results."""

    print("\n Classification: return direction")
    print(f"  Random Forest        -> accuracy = {results['rf_acc']:.4f}")
    print(f"  Logistic regression  -> accuracy = {results['lr_acc']:.4f}")
    print(f"  Majority-class rule  -> accuracy = {results['majority_acc']:.4f}  (benchmark)")

    print("\n  Feature importance (Random Forest):")
    order = np.argsort(results["importances"])[::-1]
    for i in order:
        print(f"    {results['features'][i]:12s} : {results['importances'][i]:.4f}")

    plt.figure(figsize=(8, 4))
    idx = np.argsort(results["importances"])
    plt.barh(
        [results["features"][i] for i in idx],
        results["importances"][idx],
        color="#0000FF"
    )
    plt.title("Feature importance (Random Forest)")
    plt.tight_layout()
    plt.show()


# 4. Extension: ARCH / GARCH modelling

def compute_arch_test(returns):

    """ARCH test to check for conditional heteroskedasticity.
    This part is also detailed in the report."""

    stat, pvalue, _, _ = het_arch(returns.dropna())
    return {"stat": stat, "p_value": pvalue, "arch_present": pvalue < 0.05}


def fit_garch(returns, p=2, q=2):

    """Fit a GARCH(p,q) model and return the fitted result."""

    model = arch_model(returns.dropna(), vol="GARCH", p=p, q=q)
    result = model.fit(disp="off")
    return result


def validate_garch(result):

    """Ljung-Box and Jarque-Bera tests on standardised residuals to validate the model."""

    residuals = (result.resid / result.conditional_volatility).dropna()

    lb = acorr_ljungbox(residuals, lags=10, return_df=True)
    lb2 = acorr_ljungbox(residuals ** 2, lags=10, return_df=True)
    jb_stat, jb_pvalue, _, _ = jarque_bera(residuals)

    return {"lb_pvalue": lb["lb_pvalue"].iloc[-1],
            "lb2_pvalue": lb2["lb_pvalue"].iloc[-1],
            "jb_pvalue": jb_pvalue,
            "autocorr_ok": lb["lb_pvalue"].iloc[-1] > 0.05,
            "arch_resid_ok": lb2["lb_pvalue"].iloc[-1] > 0.05,
            "normality_ok": jb_pvalue > 0.05}


def predict_volatility_garch(result, last_date, horizon=7):

    """Forecast volatility over the next trading days from a fitted GARCH model."""

    pred = result.forecast(horizon=horizon)

    # Business days only (markets are closed on weekends)
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    vol_pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)
    return vol_pred


def display_garch(returns, arch_test, garch_result, validation):

    """Print and plot all GARCH results."""

    print("\n Extension: GARCH model")

    print(f"\n  ARCH test: stat = {arch_test['stat']:.4f}, p = {arch_test['p_value']:.4f}")
    if arch_test["arch_present"]:
        print("  => ARCH effect present, GARCH is justified")
    else:
        print("  => No ARCH effect")

    print(f"\n  AIC = {garch_result.aic:.2f}")
    print(f"  BIC = {garch_result.bic:.2f}")

    print(f"\n  Ljung-Box residuals    : p = {validation['lb_pvalue']:.4f}", end="")
    print(f"  -> {'OK' if validation['autocorr_ok'] else 'Issue'}")
    print(f"  Ljung-Box residuals²   : p = {validation['lb2_pvalue']:.4f}", end="")
    print(f"  -> {'OK' if validation['arch_resid_ok'] else 'Issue'}")
    print(f"  Jarque-Bera            : p = {validation['jb_pvalue']:.4f}", end="")
    print(f"  -> {'OK' if validation['normality_ok'] else 'Normality rejected'}")

    vol_pred = predict_volatility_garch(garch_result, returns.index[-1])
    print("\n  Predicted volatility (next 7 trading days):")
    for date, val in vol_pred.items():
        print(f"    {date.date()} : {val:.4f} %")

    plt.figure(figsize=(10, 5))
    plt.plot(vol_pred, marker="o", color="red")
    plt.title("SPY volatility forecast (7 trading days)")
    plt.ylabel("Volatility (%)")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(returns.dropna() ** 2, ax=ax, color="#0000FF")
    ax.set_title("PACF of squared returns")
    plt.show()


# 5. General chart

def display_price(df):

    """Plot the SPY price with its MA200."""

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="Price", color="#0000FF", linewidth=0.8)
    plt.plot(df.index, df["ma200"], label="MA 200", color="red", linewidth=0.5)
    plt.legend()
    plt.ylabel("Price ($)")
    plt.title("SPY price")
    plt.show()


# 6. Menu

def menu():
    print("\n" + "=" * 55)
    print("  SPY ANALYSIS")
    print("=" * 55)
    print("  1. Show the asset price")
    print("  2. Stationarity test - ADF")
    print("  3. Return distribution")
    print("  4. Rolling volatility")
    print("  5. Linear regression on prices")
    print("  6. Direction classification")
    print("  7. Extension - GARCH model")
    print("  0. Quit")
    print("=" * 55)
    return input("  Choice: ").strip()


def main():
    print("Loading SPY data...")
    df = load_data()
    print(f"Data loaded: {len(df)} rows")
    print(f"Period: {df.index[0].date()} -> {df.index[-1].date()}")

    # Pre-compute the analyses
    adf_price = compute_adf(df["Close"])
    adf_ret = compute_adf(df["returns"])
    dist_stats = compute_distribution(df)
    vol, vol_stats = compute_volatility(df)

    while True:
        choice = menu()

        if choice == "1":
            display_price(df)

        elif choice == "2":
            display_adf(adf_price, adf_ret)
            input("\n  Press Enter to continue...")

        elif choice == "3":
            display_distribution(df, dist_stats)

        elif choice == "4":
            display_volatility(vol, vol_stats)

        elif choice == "5":
            mse, naive_mse, y_test, pred, coefs = price_regression(df)
            display_regression(mse, naive_mse, y_test, pred, coefs)

        elif choice == "6":
            results = direction_classification(df)
            display_classification(results)

        elif choice == "7":
            print("Fitting the GARCH model...")
            arch_test = compute_arch_test(df["returns"])
            garch_result = fit_garch(df["returns"])
            validation = validate_garch(garch_result)
            display_garch(df["returns"], arch_test, garch_result, validation)

        elif choice == "0":
            print("Goodbye.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
