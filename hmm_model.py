"""
hmm_model.py
------------
Fits a Gaussian Hidden Markov Model (HMM) to the GARCH conditional
volatility series to detect market regimes.

HMM components:
    π   – initial state distribution
    A   – transition matrix  P(S_t | S_{t-1})
    B   – Gaussian emission  N(μ_k, σ²_k)  per state k

Decoding:
    Viterbi algorithm finds the most likely hidden state sequence
    given the observed volatility series.

States:
    0 = Low  volatility regime  (calm / bull market)
    1 = High volatility regime  (stressed / crisis market)
"""

from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 1.  Fit HMM and decode regimes
# ─────────────────────────────────────────────

def fit_hmm(
    cond_vol: pd.Series,
    n_states: int = 2,
    n_iter: int   = 2000,
    random_state: int = 42,
) -> tuple:
    """
    Fit a Gaussian HMM to the conditional volatility series and
    return the decoded regime label for every day.

    Parameters
    ----------
    cond_vol     : pd.Series   annualised conditional volatility from GARCH
    n_states     : int         number of hidden states (2 = Low/High vol)
    n_iter       : int         max EM iterations
    random_state : int         reproducibility seed

    Returns
    -------
    model         : GaussianHMM   fitted model
    regime_series : pd.Series     0 = Low vol, 1 = High vol (per day)
    scaler        : StandardScaler for transforming new data
    """
    print(f"\n── Fitting Gaussian HMM  ({n_states} states) ────────────────")

    # ── Prepare input ───────────────────────────────────────────────────────
    X = cond_vol.values.reshape(-1, 1)   # shape (T, 1)

    # Standardise: HMM EM converges much better on N(0,1) scale
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Fit ─────────────────────────────────────────────────────────────────
    model = GaussianHMM(
        n_components   = n_states,
        covariance_type= "full",
        n_iter         = n_iter,
        random_state   = random_state,
        tol            = 1e-6,
    )
    model.fit(X_scaled)

    # ── Viterbi decoding ─────────────────────────────────────────────────────
    # Finds the single most likely sequence of hidden states.
    # Algorithm: dynamic programming O(T · K²) where K = n_states.
    hidden_states = model.predict(X_scaled)

    # ── Normalise state labels ───────────────────────────────────────────────
    # hmmlearn assigns state indices arbitrarily; we relabel so that:
    #   0 = the state with the LOWER mean volatility  (calm)
    #   1 = the state with the HIGHER mean volatility (stressed)
    means     = model.means_.flatten()
    low_state = int(np.argmin(means))

    regime_labels = np.where(hidden_states == low_state, 0, 1)

    regime_series = pd.Series(
        regime_labels,
        index = cond_vol.index,
        name  = "regime",
    )

    # ── Print transition matrix ──────────────────────────────────────────────
    A = model.transmat_
    # Re-order rows/cols so row 0 = low-vol state
    hi_state = 1 - low_state
    A_ordered = np.array([
        [A[low_state, low_state], A[low_state, hi_state]],
        [A[hi_state,  low_state], A[hi_state,  hi_state]],
    ])

    print("\n  Transition matrix A  (rows = current, cols = next state):")
    print("                   → Low vol   → High vol")
    print(f"  From Low vol  :    {A_ordered[0,0]:.4f}      {A_ordered[0,1]:.4f}")
    print(f"  From High vol :    {A_ordered[1,0]:.4f}      {A_ordered[1,1]:.4f}")

    # ── Regime persistence ───────────────────────────────────────────────────
    # Expected days in each regime = 1 / (1 - P(stay))
    stay_low  = A_ordered[0, 0]
    stay_high = A_ordered[1, 1]
    avg_low   = 1.0 / (1.0 - stay_low)
    avg_high  = 1.0 / (1.0 - stay_high)

    print(f"\n  Avg consecutive days in Low  vol regime: {avg_low:.1f}")
    print(f"  Avg consecutive days in High vol regime: {avg_high:.1f}")

    # ── Emission parameters (in original vol scale) ─────────────────────────
    mean_low  = scaler.inverse_transform([[model.means_[low_state, 0]]])[0, 0]
    mean_high = scaler.inverse_transform([[model.means_[hi_state,  0]]])[0, 0]

    print(f"\n  Mean volatility in Low  vol regime: {mean_low*100:.1f}%  p.a.")
    print(f"  Mean volatility in High vol regime: {mean_high*100:.1f}%  p.a.")
    print("─" * 54)

    return model, regime_series, scaler


# ─────────────────────────────────────────────
# 2.  Regime statistics
# ─────────────────────────────────────────────

def regime_statistics(
    df: pd.DataFrame,
    regime_series: pd.Series,
) -> pd.DataFrame:
    """
    Compute return and volatility statistics broken down by regime.
    This table is the key output for the README and interviews.

    Returns pd.DataFrame with one row per regime.
    """
    combined = df[["log_return"]].join(regime_series, how="inner")
    combined.dropna(inplace=True)

    rows = []
    for code, label in [(0, "Low volatility"), (1, "High volatility")]:
        sub = combined[combined["regime"] == code]["log_return"]
        if len(sub) == 0:
            continue
        ann_ret  = sub.mean() * 252
        ann_vol  = sub.std()  * np.sqrt(252)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
        rows.append({
            "Regime"               : label,
            "Days"                 : len(sub),
            "% of total"          : f"{len(sub)/len(combined)*100:.1f}%",
            "Ann. return (%)"      : round(ann_ret * 100, 2),
            "Ann. volatility (%)"  : round(ann_vol * 100, 2),
            "Sharpe ratio"         : round(sharpe, 3),
            "Max daily loss (%)"   : round(sub.min() * 100, 2),
            "Max daily gain (%)"   : round(sub.max() * 100, 2),
        })

    stats_df = pd.DataFrame(rows).set_index("Regime")

    print("\n── Regime statistics ─────────────────────────────────")
    print(stats_df.to_string())
    print("─" * 54)
    return stats_df


# ─────────────────────────────────────────────
# 3.  Model selection: compare n_states by AIC
# ─────────────────────────────────────────────

def select_hmm_states(
    cond_vol: pd.Series,
    max_states: int = 4,
    n_iter: int     = 1000,
    random_state: int = 42,
) -> int:
    """
    Fit HMMs with 2, 3, … max_states and compare AIC.
    Useful as a robustness check to justify the 2-state choice.

    AIC = -2·log(L) + 2·k
        k = n_states² + 2·n_states  (transition + emission params)
    """
    print("\n── HMM state selection (AIC comparison) ─────────────")
    X = cond_vol.values.reshape(-1, 1)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    best_aic   = np.inf
    best_n     = 2
    T          = len(X)

    for n in range(2, max_states + 1):
        try:
            m = GaussianHMM(n_components=n, covariance_type="full",
                            n_iter=n_iter, random_state=random_state,
                            tol=1e-6)
            m.fit(X_sc)
            log_lik = m.score(X_sc) * T
            k       = n * n + 2 * n    # parameters
            aic     = -2 * log_lik + 2 * k
            if aic < best_aic:
                best_aic = aic
                best_n   = n
            marker = "  ◄ best" if n == best_n else ""
            print(f"  {n} states:  AIC = {aic:,.2f}{marker}")
        except Exception as e:
            print(f"  {n} states:  failed ({e})")

    print(f"\n  Recommended number of states: {best_n}")
    print("─" * 54)
    return best_n
