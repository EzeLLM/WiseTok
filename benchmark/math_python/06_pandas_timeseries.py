"""
Pandas Time-Series Analysis

DateTimeIndex, resampling, rolling windows, groupby aggregations,
merge_asof, multi-index, and financial data manipulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


def create_timeseries_data() -> pd.DataFrame:
    """Generate synthetic multi-variate time-series data."""
    dates = pd.date_range("2023-01-01", periods=252, freq="D")  # Trading days

    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(252) * 2.0)
    volumes = np.random.randint(1e6, 5e6, 252)
    volatility = 0.1 + 0.05 * np.random.randn(252)

    df = pd.DataFrame({
        "timestamp": dates,
        "close_price": prices,
        "volume": volumes,
        "volatility": volatility,
    })

    df.set_index("timestamp", inplace=True)
    return df


def resampling_example(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Resample time-series to different frequencies."""
    # Daily to weekly (last value)
    weekly_close = df["close_price"].resample("W").last()

    # Daily to monthly with aggregation
    monthly_stats = df["close_price"].resample("ME").agg(["min", "max", "mean", "std"])

    # Hourly intraday data (synthetic)
    intraday = pd.date_range("2023-01-01", periods=252*24, freq="h")
    intraday_prices = 100 + np.cumsum(np.random.randn(252*24) * 0.1)
    intraday_df = pd.DataFrame({"price": intraday_prices}, index=intraday)

    # Resample to 4-hour bars (OHLCV)
    bars = intraday_df["price"].resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })

    return {
        "weekly_close": weekly_close,
        "monthly_stats": monthly_stats,
        "intraday_bars": bars,
    }


def rolling_window_analysis(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Compute rolling statistics for trend analysis."""
    # 20-day simple moving average
    sma_20 = df["close_price"].rolling(window=20).mean()

    # 20-day standard deviation
    rolling_std = df["close_price"].rolling(window=20).std()

    # Bollinger Bands: SMA ± 2*std
    upper_band = sma_20 + 2 * rolling_std
    lower_band = sma_20 - 2 * rolling_std

    # Exponential moving average with span=20
    ema_20 = df["close_price"].ewm(span=20, adjust=False).mean()

    # Rolling Sharpe ratio (assuming daily data, risk-free rate=2%)
    returns = df["close_price"].pct_change()
    rolling_sharpe = returns.rolling(window=252).mean() / returns.rolling(window=252).std() * np.sqrt(252)
    rolling_sharpe -= 0.02  # Risk-free rate

    return {
        "sma_20": sma_20,
        "upper_band": upper_band,
        "lower_band": lower_band,
        "ema_20": ema_20,
        "rolling_sharpe": rolling_sharpe,
    }


def groupby_aggregation(df: pd.DataFrame) -> Dict[str, any]:
    """Groupby operations on time-series data."""
    df_with_year_month = df.copy()
    df_with_year_month["year"] = df_with_year_month.index.year
    df_with_year_month["month"] = df_with_year_month.index.month
    df_with_year_month["quarter"] = df_with_year_month.index.quarter

    # Group by month
    monthly_returns = df["close_price"].pct_change().groupby(
        [df.index.year, df.index.month]
    ).sum()

    # Group by day of week
    by_day_of_week = df["volume"].groupby(df.index.dayofweek).agg(
        ["mean", "median", "std"]
    )
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_day_of_week.index = day_names[:5]  # Only weekdays

    # Multi-level groupby
    df_with_year_month["price_range"] = pd.cut(
        df["close_price"], bins=3, labels=["Low", "Mid", "High"]
    )
    grouped = df_with_year_month.groupby(
        ["year", "month", "price_range"]
    )["volume"].agg(["count", "sum", "mean"])

    return {
        "monthly_returns": monthly_returns,
        "by_day_of_week": by_day_of_week,
        "multi_index_agg": grouped,
    }


def merge_asof_example() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-series matching with merge_asof."""
    # Trades: high-frequency order execution
    trades = pd.DataFrame({
        "time": pd.date_range("2023-01-01 09:30:00", periods=100, freq="1min"),
        "price": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "size": np.random.randint(100, 1000, 100),
    })

    # Quotes: lower frequency bid-ask
    quotes = pd.DataFrame({
        "time": pd.date_range("2023-01-01 09:30:00", periods=20, freq="5min"),
        "bid": 99.5 + np.cumsum(np.random.randn(20) * 0.3),
        "ask": 100.5 + np.cumsum(np.random.randn(20) * 0.3),
    })

    # Match each trade to the most recent quote
    merged = pd.merge_asof(
        trades,
        quotes,
        on="time",
        direction="backward",  # Last quote before/at trade time
    )

    return trades, quotes, merged


def multi_index_operations() -> Dict[str, any]:
    """Multi-level indexing (dates × ticker)."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL"]

    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    prices = np.random.uniform(100, 150, len(index))

    df = pd.DataFrame({"close": prices}, index=index)

    # Unstack: dates rows × tickers columns
    unstacked = df.unstack(level="ticker")

    # Stack back
    stacked = unstacked.stack()

    # Select by level
    aapl_data = df.loc[(slice(None), "AAPL"), :]

    return {
        "multi_index_df": df,
        "unstacked": unstacked,
        "aapl_prices": aapl_data,
    }


def forward_fill_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing data in time-series."""
    # Introduce some missing values
    df_with_gaps = df.copy()
    df_with_gaps.loc[df_with_gaps.index[::20], "close_price"] = np.nan

    # Forward fill: last observation carried forward
    df_ffill = df_with_gaps.fillna(method="ffill")

    # Backward fill
    df_bfill = df_with_gaps.fillna(method="bfill")

    # Linear interpolation
    df_interp = df_with_gaps.interpolate(method="linear")

    return pd.concat({
        "original": df,
        "with_gaps": df_with_gaps,
        "ffill": df_ffill,
        "interp": df_interp,
    }, keys=["scenario"])


def time_window_operations(df: pd.DataFrame) -> Dict[str, any]:
    """Operations within time windows."""
    # Trades within last 30 days
    last_30_days = df.last("30D")

    # Data for specific date range
    date_range = df["2023-02-01":"2023-02-28"]

    # Business day offset
    from pandas.tseries.offsets import BDay
    df_with_bday = df.copy()
    df_with_bday["bday_offset"] = df_with_bday.index + BDay(5)

    return {
        "last_30_days": last_30_days,
        "feb_2023": date_range,
        "business_day_offset": df_with_bday["bday_offset"].head(),
    }


def financial_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute financial metrics from time-series."""
    returns = df["close_price"].pct_change().dropna()

    # Cumulative return
    cum_return = (1 + returns).prod() - 1

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming 2% risk-free rate)
    excess_return = returns.mean() - 0.02 / 252
    sharpe = excess_return / returns.std() * np.sqrt(252)

    # Max drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    return {
        "cumulative_return": cum_return,
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Pandas Time-Series Analysis")
    print("=" * 70)

    # Create data
    df = create_timeseries_data()
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print("\nFirst few rows:")
    print(df.head())

    print("\n" + "=" * 70)
    print("Resampling")
    print("=" * 70)
    resampled = resampling_example(df)
    print(f"Weekly close shape: {resampled['weekly_close'].shape}")
    print(f"Monthly stats shape: {resampled['monthly_stats'].shape}")

    print("\n" + "=" * 70)
    print("Rolling Window Analysis")
    print("=" * 70)
    rolling = rolling_window_analysis(df)
    print(f"SMA-20 shape: {rolling['sma_20'].shape}")
    print(f"Last 5 SMA-20 values:\n{rolling['sma_20'].tail()}")

    print("\n" + "=" * 70)
    print("Groupby Aggregations")
    print("=" * 70)
    grouped = groupby_aggregation(df)
    print(f"Monthly returns shape: {grouped['monthly_returns'].shape}")
    print(f"By day of week:\n{grouped['by_day_of_week']}")

    print("\n" + "=" * 70)
    print("Merge AsOf (Trade-Quote Matching)")
    print("=" * 70)
    trades, quotes, merged = merge_asof_example()
    print(f"Trades: {len(trades)}, Quotes: {len(quotes)}")
    print(f"Merged shape: {merged.shape}")

    print("\n" + "=" * 70)
    print("Financial Metrics")
    print("=" * 70)
    metrics = financial_metrics(df)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
