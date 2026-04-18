"""
visualise.py
------------
All plotting functions for the Nifty 50 volatility regime project.

Figures produced:
    1. regime_dashboard.png   – 4-panel master figure (main README image)
    2. vol_distributions.png  – KDE of volatility per regime
    3. transition_heatmap.png – HMM transition matrix heatmap
    4. regime_calendar.png    – annual regime breakdown bar chart
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# ── Colour palette ──────────────────────────────────────────────────────────
COLORS = {
    "price"    : "#1a6ca8",
    "vol"      : "#c0392b",
    "low_vol"  : "#27ae60",   # green  – calm
    "high_vol" : "#e67e22",   # orange – stressed
    "low_fill" : "#d5f5e3",
    "high_fill": "#fdebd0",
    "grid"     : "#e8e8e8",
}

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : COLORS["grid"],
    "grid.linewidth"   : 0.5,
    "figure.dpi"       : 150,
})

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Helper: shade regime periods on an axis
# ─────────────────────────────────────────────

def _shade_regimes(ax, regime_series: pd.Series) -> None:
    """Draw green / orange vertical band for each regime period."""
    dates   = regime_series.index
    regimes = regime_series.values
    in_r, start = regimes[0], dates[0]

    for i in range(1, len(dates)):
        if regimes[i] != in_r or i == len(dates) - 1:
            end   = dates[i]
            color = COLORS["high_fill"] if in_r == 1 else COLORS["low_fill"]
            ax.axvspan(start, end, alpha=0.55, color=color,
                       linewidth=0, zorder=0)
            in_r, start = regimes[i], dates[i]


# ─────────────────────────────────────────────
# Figure 1 – 4-panel dashboard
# ─────────────────────────────────────────────

def plot_full_dashboard(
    df            : pd.DataFrame,
    cond_vol      : pd.Series,
    regime_series : pd.Series,
    save_path     : str = "outputs/regime_dashboard.png",
) -> None:
    """
    Master 4-panel figure:
        Panel 1 – Nifty 50 price level + regime shading
        Panel 2 – Daily log returns
        Panel 3 – GARCH conditional volatility + regime shading
        Panel 4 – Binary regime indicator
    """
    fig, axes = plt.subplots(
        4, 1, figsize=(16, 14),
        gridspec_kw={"height_ratios": [3, 1.5, 2.5, 1]},
        sharex=True,
    )
    fig.suptitle(
        "Nifty 50 — Volatility Regime Detection\n"
        "GARCH(1,1) conditional volatility  +  Hidden Markov Model",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # ── Panel 1: Price ───────────────────────────────────────────────────────
    ax = axes[0]
    _shade_regimes(ax, regime_series)
    ax.plot(df.index, df["Close"], color=COLORS["price"],
            lw=0.9, zorder=2, label="Nifty 50 Close")
    ax.set_ylabel("Index level", fontsize=10)
    ax.set_title("Price level  (shading = detected volatility regime)",
                 fontsize=10, pad=4)

    # Legend patches
    low_patch  = mpatches.Patch(color=COLORS["low_fill"],  label="Low vol regime",
                                alpha=0.9)
    high_patch = mpatches.Patch(color=COLORS["high_fill"], label="High vol regime",
                                alpha=0.9)
    ax.legend(handles=[low_patch, high_patch], fontsize=8,
              loc="upper left", framealpha=0.8)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    # ── Panel 2: Log returns ─────────────────────────────────────────────────
    ax = axes[1]
    ret_pct = df["log_return"] * 100
    ax.fill_between(df.index, ret_pct, 0,
                    where=(ret_pct >= 0), color=COLORS["low_vol"],
                    alpha=0.55, linewidth=0)
    ax.fill_between(df.index, ret_pct, 0,
                    where=(ret_pct < 0),  color=COLORS["high_vol"],
                    alpha=0.55, linewidth=0)
    ax.plot(df.index, ret_pct, color="#555", lw=0.35, alpha=0.6)
    ax.axhline(0, color="#333", lw=0.6, linestyle="--")
    ax.set_ylabel("Log return (%)", fontsize=10)
    ax.set_title("Daily log returns", fontsize=10, pad=4)

    # ── Panel 3: GARCH conditional volatility ───────────────────────────────
    ax = axes[2]
    _shade_regimes(ax, regime_series)
    ax.plot(cond_vol.index, cond_vol * 100,
            color=COLORS["vol"], lw=1.1, zorder=2,
            label="GARCH conditional vol (ann. %)")
    # Add 20 % and 30 % threshold lines for reference
    for thresh, ls in [(20, "--"), (30, ":")]:
        ax.axhline(thresh, color="#888", lw=0.7, linestyle=ls, alpha=0.7)
        ax.text(cond_vol.index[-1], thresh + 0.5, f"{thresh}%",
                fontsize=7, color="#888", va="bottom", ha="right")
    ax.set_ylabel("Annualised vol (%)", fontsize=10)
    ax.set_title("GARCH conditional volatility", fontsize=10, pad=4)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    # ── Panel 4: Regime binary ───────────────────────────────────────────────
    ax = axes[3]
    ax.fill_between(regime_series.index, regime_series.values,
                    step="post", color=COLORS["high_vol"],
                    alpha=0.75, linewidth=0, label="High vol")
    ax.fill_between(regime_series.index,
                    1 - regime_series.values,
                    step="post", color=COLORS["low_vol"],
                    alpha=0.75, linewidth=0, label="Low vol")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Low vol", "High vol"], fontsize=8)
    ax.set_title("Detected regime", fontsize=10, pad=4)
    ax.set_xlabel("Year", fontsize=10)

    # ── Shared x-axis formatting ─────────────────────────────────────────────
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

    for ax in axes:
        ax.set_xlim(df.index[0], df.index[-1])

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Figure 2 – Volatility KDE by regime
# ─────────────────────────────────────────────

def plot_vol_distributions(
    cond_vol      : pd.Series,
    regime_series : pd.Series,
    save_path     : str = "outputs/vol_distributions.png",
) -> None:
    """
    Overlapping KDE plots of conditional volatility for each regime.
    Visually demonstrates how well HMM separated the two distributions.
    """
    combined = cond_vol.rename("vol").to_frame().join(regime_series)

    fig, ax = plt.subplots(figsize=(9, 5))

    for code, label, color in [
        (0, "Low volatility regime",  COLORS["low_vol"]),
        (1, "High volatility regime", COLORS["high_vol"]),
    ]:
        sub = combined[combined["regime"] == code]["vol"] * 100
        sub.plot.kde(ax=ax, label=f"{label}  (n = {len(sub):,})",
                     color=color, lw=2.2)
        ax.axvline(sub.mean(), color=color, lw=1.2,
                   linestyle="--", alpha=0.8,
                   label=f"  Mean = {sub.mean():.1f}%")
        # no fill under KDE – keep clean

    ax.set_xlabel("Annualised conditional volatility (%)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Volatility distribution by detected regime", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Figure 3 – Transition matrix heatmap
# ─────────────────────────────────────────────

def plot_transition_matrix(
    hmm_model,
    low_state     : int,
    save_path     : str = "outputs/transition_heatmap.png",
) -> None:
    """
    Heatmap of the HMM transition matrix.
    Diagonal values = regime stickiness / persistence.
    """
    A = hmm_model.transmat_
    hi_state = 1 - low_state
    A_ord = np.array([
        [A[low_state, low_state], A[low_state, hi_state]],
        [A[hi_state,  low_state], A[hi_state,  hi_state]],
    ])

    labels = ["Low vol", "High vol"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        A_ord * 100,
        annot       = True,
        fmt         = ".2f",
        cmap        = "YlOrRd",
        xticklabels = labels,
        yticklabels = labels,
        linewidths  = 0.5,
        ax          = ax,
        vmin        = 0, vmax = 100,
        annot_kws   = {"size": 12},
        cbar_kws    = {"label": "Probability (%)"},
    )
    ax.set_title("HMM transition matrix  (%)\n"
                 "Row = current state, Column = next state",
                 fontsize=11)
    ax.set_xlabel("Next state", fontsize=10)
    ax.set_ylabel("Current state", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Figure 4 – Annual regime breakdown
# ─────────────────────────────────────────────

def plot_annual_regime_breakdown(
    regime_series : pd.Series,
    save_path     : str = "outputs/regime_calendar.png",
) -> None:
    """
    Stacked horizontal bar chart: for each calendar year, what
    percentage of trading days were in each regime?
    Makes it easy to identify crisis years (high % orange).
    """
    df_r = regime_series.to_frame()
    df_r["year"] = df_r.index.year
    df_r["low"]  = (df_r["regime"] == 0).astype(int)
    df_r["high"] = (df_r["regime"] == 1).astype(int)

    by_year = df_r.groupby("year")[["low", "high"]].sum()
    by_year["pct_low"]  = by_year["low"]  / (by_year["low"] + by_year["high"]) * 100
    by_year["pct_high"] = by_year["high"] / (by_year["low"] + by_year["high"]) * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    years = by_year.index.astype(str)
    ax.barh(years, by_year["pct_low"],
            color=COLORS["low_vol"],  alpha=0.8, label="Low vol")
    ax.barh(years, by_year["pct_high"],
            left=by_year["pct_low"],
            color=COLORS["high_vol"], alpha=0.8, label="High vol")

    ax.set_xlabel("% of trading days", fontsize=11)
    ax.set_title("Annual regime breakdown — percentage of days in each regime",
                 fontsize=12)
    ax.axvline(50, color="#333", lw=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.show()
