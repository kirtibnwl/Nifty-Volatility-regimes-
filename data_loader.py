"""
data_loader.py
--------------
Downloads and prepares Nifty 50 daily price data from Yahoo Finance.

Ticker : ^NSEI  (Yahoo Finance symbol for Nifty 50)
Output  : cleaned DataFrame with log returns and demeaned returns
"""

import yfinance as yf
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1.  Download & clean
# ─────────────────────────────────────────────

def load_nifty(start: str = "2010-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Download Nifty 50 daily OHLCV data and compute log returns.

    Parameters
    ----------
    start : str   e.g. "2010-01-01"
    end   : str   e.g. "2024-12-31"

    Returns
    -------
    pd.DataFrame with columns:
        Close           – adjusted closing price
        log_return      – r_t = log(P_t / P_{t-1})
        return_demeaned – log_return minus its mean  (fed to GARCH)
    """
    print(f"  Downloading Nifty 50 (^NSEI) from {start} to {end} …")
    raw = yf.download("^NSEI", start=start, end=end,
                      auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(
            "No data returned from Yahoo Finance. "
            "Check your internet connection or try a different date range."
        )

    # Keep only closing price; flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    df.dropna(inplace=True)

    # ── Log returns: r_t = log(P_t / P_{t-1}) ──────────────────────────────
    # Why log returns?
    #   • Additive across time  (r_1 + r_2 + r_3 = total compounded return)
    #   • Approximately stationary – required for GARCH
    #   • Symmetric around 0 for rises and falls of equal magnitude
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)

    # ── Demean ──────────────────────────────────────────────────────────────
    # GARCH models the variance of the *residual* (deviation from mean),
    # so we subtract the sample mean first.
    df["return_demeaned"] = df["log_return"] - df["log_return"].mean()

    print(f"  Loaded {len(df):,} trading days "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─────────────────────────────────────────────
# 2.  Descriptive statistics
# ─────────────────────────────────────────────

def get_summary_stats(df: pd.DataFrame) -> pd.Series:
    """Print and return summary statistics for the log return series."""
    r = df["log_return"]
    stats = {
        "Observations"          : len(r),
        "Mean return"           : round(r.mean(), 8),
        "Std deviation"         : round(r.std(),  6),
        "Min return"            : round(r.min(),  6),
        "Max return"            : round(r.max(),  6),
        "Skewness"              : round(r.skew(), 4),
        "Excess kurtosis"       : round(r.kurtosis(), 4),
        "Ann. mean return (%)"  : round(r.mean() * 252 * 100, 4),
        "Ann. volatility (%)"   : round(r.std()  * np.sqrt(252) * 100, 4),
    }
    print("\n── Summary statistics (log returns) ──────────────────")
    for k, v in stats.items():
        print(f"  {k:<30} {v}")
    print()

    # Interpretation hints
    if abs(stats["Skewness"]) > 0.3:
        print("  Note: |skewness| > 0.3 → asymmetric return distribution")
    if stats["Excess kurtosis"] > 1:
        print("  Note: excess kurtosis > 1 → fat tails (non-Normal) → "
              "Student-t GARCH is appropriate")
    print("─" * 54)
    return pd.Series(stats)
