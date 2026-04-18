"""
backtesting.py
--------------
Simple regime-based trading strategy.

Strategy logic:
    • Long Nifty 50 when in Low volatility regime  (regime = 0)
    • Exit to cash when in High volatility regime  (regime = 1)
    • No short selling; no transaction costs

This is intentionally simple — it demonstrates that regimes have
predictive power and makes for a strong talking point in interviews.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1.  Run backtest
# ─────────────────────────────────────────────

def run_backtest(
    df            : pd.DataFrame,
    regime_series : pd.Series,
) -> pd.DataFrame:
    """
    Run a simple regime-filtered long strategy vs buy-and-hold.

    Parameters
    ----------
    df            : DataFrame with 'log_return' column
    regime_series : Series of 0/1 regime labels

    Returns
    -------
    results : pd.DataFrame with daily cumulative returns for both strategies
    """
    bt = df[["log_return"]].join(regime_series, how="inner").dropna()

    # Strategy: invest only in low-vol regime (regime = 0)
    # We use the PREVIOUS day's regime label to avoid look-ahead bias:
    # "We know yesterday's regime, so we decide today's position."
    bt["position"]       = bt["regime"].shift(1).fillna(0)
    bt["position"]       = np.where(bt["position"] == 0, 1, 0)   # 1=long, 0=cash

    bt["strat_return"]   = bt["log_return"] * bt["position"]
    bt["buyhold_return"] = bt["log_return"]

    # Cumulative log returns
    bt["strat_cumret"]   = bt["strat_return"].cumsum().apply(np.exp) - 1
    bt["buyhold_cumret"] = bt["buyhold_return"].cumsum().apply(np.exp) - 1

    return bt


# ─────────────────────────────────────────────
# 2.  Performance metrics
# ─────────────────────────────────────────────

def performance_metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """Compute annualised return, volatility, Sharpe and max drawdown."""

    rows = []
    for col, label in [
        ("strat_return",   "Regime strategy"),
        ("buyhold_return", "Buy-and-hold"),
    ]:
        r = bt[col].dropna()

        ann_ret  = r.mean() * 252
        ann_vol  = r.std()  * np.sqrt(252)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan

        # Maximum drawdown
        cum = (1 + r).cumprod()
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        max_dd   = drawdown.min()

        rows.append({
            "Strategy"           : label,
            "Ann. return (%)"    : round(ann_ret * 100, 2),
            "Ann. volatility (%)": round(ann_vol * 100, 2),
            "Sharpe ratio"       : round(sharpe, 3),
            "Max drawdown (%)"   : round(max_dd * 100, 2),
            "Days invested"      : int(bt["position"].sum()) if col == "strat_return" else len(r),
        })

    metrics = pd.DataFrame(rows).set_index("Strategy")
    print("\n── Backtest results ──────────────────────────────────")
    print(metrics.to_string())
    print("\n  Note: no transaction costs assumed; "
          "position set using previous-day regime (no look-ahead bias)")
    print("─" * 54)
    return metrics


# ─────────────────────────────────────────────
# 3.  Plot equity curves
# ─────────────────────────────────────────────

def plot_equity_curves(
    bt        : pd.DataFrame,
    save_path : str = "outputs/backtest_equity.png",
) -> None:
    """Plot cumulative return curves for strategy vs buy-and-hold."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Regime-filtered strategy vs Buy-and-hold  (Nifty 50)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(bt.index, (bt["strat_cumret"]   + 1) * 100,
            color="#1a6ca8", lw=1.4, label="Regime strategy (long in low-vol only)")
    ax.plot(bt.index, (bt["buyhold_cumret"] + 1) * 100,
            color="#c0392b", lw=1.0, alpha=0.8, linestyle="--",
            label="Buy-and-hold")
    ax.set_ylabel("Portfolio value (base = 100)", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    # Regime overlay
    for date, reg in bt["regime"].items():
        if reg == 1:
            ax.axvspan(date, date + pd.Timedelta(days=1),
                       alpha=0.12, color="#e67e22", linewidth=0)

    # Panel 2: position (invested / cash)
    ax2 = axes[1]
    ax2.fill_between(bt.index, bt["position"],
                     step="post", color="#1a6ca8", alpha=0.6)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Cash", "Invested"])
    ax2.set_title("Position: long or cash", fontsize=9, pad=2)
    ax2.set_xlabel("Year", fontsize=10)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.set_xlim(bt.index[0], bt.index[-1])
        ax.grid(True, alpha=0.4, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.show()
