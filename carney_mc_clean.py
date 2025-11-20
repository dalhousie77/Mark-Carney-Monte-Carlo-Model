import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- Regime and simulation settings ---
CARNEY_START = "2013-07-01"
CARNEY_END   = "2020-03-15"
TICKER = "^FTSE"
N_SIM = 1000
TRADING_DAYS = 252


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily prices for a ticker and return a dataframe with:
      - price
      - log_ret
    Handles both normal and MultiIndex columns from yfinance.
    """
    data = yf.download(ticker, start=start, end=end, progress=False)

    # Debug: see what columns we actually have
    print("Downloaded columns:", data.columns)

    # If MultiIndex (e.g. ('Close', '^FTSE')), collapse to first level
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # yfinance (with auto_adjust=True default) gives adjusted close in 'Close'
    if "Close" not in data.columns:
        raise ValueError(f"'Close' column not found. Got columns: {data.columns}")

    price = data["Close"].dropna()

    df = pd.DataFrame({"price": price})
    df["log_ret"] = np.log(df["price"] / df["price"].shift(1))
    df = df.dropna()

    return df


def split_period(data: pd.DataFrame):
    mid_idx = len(data) // 2
    return data.iloc[:mid_idx].copy(), data.iloc[mid_idx:].copy()


def calibrate_gbm(log_returns: pd.Series, trading_days: int = 252) -> dict:
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()

    mu_annual = mu_daily * trading_days
    sigma_annual = sigma_daily * np.sqrt(trading_days)

    return {
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "mu_annual": mu_annual,
        "sigma_annual": sigma_annual,
    }


def simulate_gbm_paths(S0, mu_daily, sigma_daily, n_days, n_sim):
    dt = 1.0
    paths = np.zeros((n_days + 1, n_sim))
    paths[0] = S0

    Z = np.random.normal(size=(n_days, n_sim))
    drift_term = (mu_daily - 0.5 * sigma_daily**2) * dt
    diffusion_term = sigma_daily * np.sqrt(dt)

    for t in range(1, n_days + 1):
        paths[t] = paths[t - 1] * np.exp(drift_term + diffusion_term * Z[t - 1])

    return paths


if __name__ == "__main__":
    # 1) Load all data
    all_data = load_data(TICKER, "2000-01-01", "2025-01-01")

    # 2) Restrict to Carney regime
    carney_data = all_data.loc[CARNEY_START:CARNEY_END].copy()

    if len(carney_data) < 10:
        raise ValueError("Not enough data in Carney period. Check ticker or dates.")

    # 3) Train/test split
    carney_train, carney_test = split_period(carney_data)

    # 4) Calibrate GBM on train half
    params = calibrate_gbm(carney_train["log_ret"])
    print("\nCalibrated parameters (Carney regime):")
    for k, v in params.items():
        print(f"{k}: {v:.6f}")

    # 5) Backtest on test half
    n_days_backtest = len(carney_test)
    S0_backtest = carney_test["price"].iloc[0]
    realized_prices = carney_test["price"].values
    time_axis = np.arange(n_days_backtest + 1)

    paths_backtest = simulate_gbm_paths(
        S0=S0_backtest,
        mu_daily=params["mu_daily"],
        sigma_daily=params["sigma_daily"],
        n_days=n_days_backtest,
        n_sim=N_SIM,
    )

    # Percentiles
    p5 = np.percentile(paths_backtest, 5, axis=1)
    p50 = np.percentile(paths_backtest, 50, axis=1)
    p95 = np.percentile(paths_backtest, 95, axis=1)

    # Plot backtest
    plt.figure(figsize=(10, 6))
    plt.plot(
        time_axis,
        np.concatenate([[S0_backtest], realized_prices]),
        label="Realized FTSE",
        linewidth=2,
    )
    plt.plot(time_axis, p50, linestyle="--", label="Sim median")
    plt.fill_between(time_axis, p5, p95, alpha=0.2, label="5–95% band")
    plt.title("Backtest: Carney regime Monte Carlo vs realized FTSE")
    plt.xlabel("Days")
    plt.ylabel("Index level")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6) 1-year forward simulation from last price
    S0_now = all_data["price"].iloc[-1]
    n_days_forward = TRADING_DAYS

    paths_forward = simulate_gbm_paths(
        S0=S0_now,
        mu_daily=params["mu_daily"],
        sigma_daily=params["sigma_daily"],
        n_days=n_days_forward,
        n_sim=N_SIM,
    )

    final_prices = paths_forward[-1]
    print("\nForward 1-year distribution (Carney-style params):")
    print(f"Mean:   {final_prices.mean():.2f}")
    print(f"Median: {np.median(final_prices):.2f}")
    print(f"5%:     {np.percentile(final_prices, 5):.2f}")
    print(f"95%:    {np.percentile(final_prices, 95):.2f}")

    plt.figure(figsize=(8, 5))
    plt.hist(final_prices, bins=40, alpha=0.7)
    plt.title("Distribution of FTSE level in 1 year (Carney-style GBM)")
    plt.xlabel("Index level")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
