"""
main.py
-------
Full pipeline: Download → GARCH → HMM → Visualise → Backtest

Run:
    python main.py

Optional arguments (edit CONFIG below):
    START_DATE  : beginning of historical data
    END_DATE    : end of historical data
    N_STATES    : number of HMM hidden states (default 2)
    RUN_BACKTEST: whether to run the trading strategy comparison
"""

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    "START_DATE"   : "2010-01-01",
    "END_DATE"     : "2024-12-31",
    "N_STATES"     : 2,           # 2 = Low / High vol
    "RUN_BACKTEST" : True,        # set False to skip trading strategy
    "SAVE_OUTPUTS" : True,        # save all plots to outputs/
}
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Allow running from project root without installing as a package
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader  import load_nifty, get_summary_stats
from src.garch_model  import select_garch_order, fit_garch, garch_diagnostics
from src.hmm_model    import select_hmm_states, fit_hmm, regime_statistics
from src.visualise    import (
    plot_full_dashboard,
    plot_vol_distributions,
    plot_transition_matrix,
    plot_annual_regime_breakdown,
)
from src.backtesting  import run_backtest, performance_metrics, plot_equity_curves

import numpy as np


def main():
    print("=" * 60)
    print("  Nifty 50 — Volatility Regime Detection")
    print("  GARCH(1,1)  +  Hidden Markov Model")
    print("=" * 60)

    # ── 1. Load & inspect data ────────────────────────────────────────────────
    print("\n[Step 1/5]  Loading data …")
    df = load_nifty(
        start = CONFIG["START_DATE"],
        end   = CONFIG["END_DATE"],
    )
    get_summary_stats(df)

    # ── 2. Fit GARCH ──────────────────────────────────────────────────────────
    print("\n[Step 2/5]  Fitting GARCH model …")
    best_p, best_q = select_garch_order(df["return_demeaned"],
                                        max_p=2, max_q=2)
    garch_result, cond_vol = fit_garch(
        df["return_demeaned"], p=best_p, q=best_q
    )
    garch_diagnostics(garch_result, df["return_demeaned"])

    # ── 3. Fit HMM ────────────────────────────────────────────────────────────
    print("\n[Step 3/5]  Fitting HMM for regime detection …")

    # Optional: confirm 2-state choice via AIC
    best_n = select_hmm_states(cond_vol, max_states=4)
    n_states = CONFIG["N_STATES"]
    if best_n != n_states:
        print(f"\n  AIC prefers {best_n} states but config uses {n_states}. "
              f"Proceeding with {n_states} for interpretability.")

    hmm_model, regime_series, scaler = fit_hmm(
        cond_vol,
        n_states     = n_states,
        random_state = 42,
    )

    stats_df = regime_statistics(df, regime_series)

    # Identify which internal HMM state index is the low-vol state
    low_state = int(np.argmin(hmm_model.means_.flatten()))

    # ── 4. Visualise ──────────────────────────────────────────────────────────
    print("\n[Step 4/5]  Generating plots …")
    plot_full_dashboard(df, cond_vol, regime_series)
    plot_vol_distributions(cond_vol, regime_series)
    plot_transition_matrix(hmm_model, low_state)
    plot_annual_regime_breakdown(regime_series)

    # ── 5. Backtest ───────────────────────────────────────────────────────────
    if CONFIG["RUN_BACKTEST"]:
        print("\n[Step 5/5]  Running backtest …")
        bt      = run_backtest(df, regime_series)
        metrics = performance_metrics(bt)
        plot_equity_curves(bt)
    else:
        print("\n[Step 5/5]  Backtest skipped (RUN_BACKTEST = False)")

    print("\n" + "=" * 60)
    print("  Pipeline complete.  All outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
