"""
garch_model.py
--------------
Fits a GARCH(p,q) model to demeaned Nifty 50 log returns and returns
the conditional (day-by-day) volatility series.

Model:
    r_t = ε_t,           ε_t = σ_t · z_t,   z_t ~ Student-t(ν)
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Where:
    ω         – long-run variance floor  (must be > 0)
    α         – ARCH term: yesterday's shock weight
    β         – GARCH term: yesterday's variance weight
    α + β     – persistence  (≈ 0.97–0.99 for Nifty 50)
    ν         – degrees of freedom of Student-t  (controls tail fatness)
"""

from arch import arch_model
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1.  Grid-search best (p,q) order by AIC
# ─────────────────────────────────────────────

def select_garch_order(
    returns_demeaned: pd.Series,
    max_p: int = 2,
    max_q: int = 2,
) -> tuple[int, int]:
    """
    Grid search over GARCH(p, q) orders and select the one with lowest AIC.

    AIC = Akaike Information Criterion = 2k - 2·ln(L)
    Lower AIC is better; it penalises extra parameters.

    Returns
    -------
    (p, q) tuple for the best order
    """
    print("\n── GARCH order selection (AIC grid search) ───────────")
    best_aic   = np.inf
    best_order = (1, 1)
    results    = {}

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                m = arch_model(
                    returns_demeaned * 100,   # scale ×100 for numerical stability
                    vol="Garch", p=p, q=q,
                    mean="Zero", dist="t"
                )
                r = m.fit(disp="off")
                results[(p, q)] = r.aic
                if r.aic < best_aic:
                    best_aic   = r.aic
                    best_order = (p, q)
            except Exception:
                pass   # skip non-convergent orders silently

    for order, aic in sorted(results.items()):
        marker = "  ◄ best" if order == best_order else ""
        print(f"  GARCH{order}  AIC = {aic:,.2f}{marker}")

    print(f"\n  Selected order: GARCH{best_order}")
    print("─" * 54)
    return best_order


# ─────────────────────────────────────────────
# 2.  Fit GARCH and extract conditional volatility
# ─────────────────────────────────────────────

def fit_garch(
    returns_demeaned: pd.Series,
    p: int = 1,
    q: int = 1,
) -> tuple:
    """
    Fit GARCH(p,q) with Student-t innovations and return fitted result
    plus the annualised conditional volatility series.

    Parameters
    ----------
    returns_demeaned : pd.Series   demeaned log returns
    p                : int         ARCH order
    q                : int         GARCH order

    Returns
    -------
    result   : arch ARCHModelResult   full fitted model object
    cond_vol : pd.Series              annualised conditional volatility
                                      (e.g. 0.18 = 18% p.a.)
    """
    print(f"\n── Fitting GARCH({p},{q}) with Student-t innovations ──")

    # Scale by 100 so the variance ≈ 1–3 instead of 0.0001–0.0003
    # This helps the numerical optimiser converge reliably.
    scaled = returns_demeaned * 100

    model = arch_model(
        scaled,
        vol  = "Garch",
        p    = p,
        q    = q,
        mean = "Zero",   # already demeaned outside
        dist = "t",      # Student-t handles fat tails better than Normal
    )

    result = model.fit(
        disp         = "off",    # suppress iteration output
        options      = {"maxiter": 500},
        show_warning = False,
    )

    # ── Print key parameters ────────────────────────────────────────────────
    omega  = result.params["omega"]
    alpha  = result.params["alpha[1]"]
    beta   = result.params["beta[1]"]
    nu     = result.params["nu"]
    persist = alpha + beta

    print(f"\n  ω  (long-run variance floor)  = {omega:.6f}")
    print(f"  α  (ARCH term)               = {alpha:.4f}")
    print(f"  β  (GARCH term)              = {beta:.4f}")
    print(f"  ν  (degrees of freedom)      = {nu:.2f}  "
          f"{'(fat tails)' if nu < 10 else '(near-Normal tails)'}")
    print(f"\n  Persistence  α + β           = {persist:.4f}")

    if persist > 0.97:
        print("  → High persistence: volatility shocks decay very slowly "
              "(typical for Nifty 50)")
    elif persist > 0.90:
        print("  → Moderate persistence: shocks decay within a few weeks")
    else:
        print("  → Low persistence: shocks decay quickly")

    long_run_vol = np.sqrt(omega / (1 - persist)) / 100 * np.sqrt(252) * 100
    print(f"\n  Long-run annualised volatility = {long_run_vol:.1f}%")

    # Log-likelihood and information criteria
    print(f"\n  Log-likelihood  = {result.loglikelihood:,.2f}")
    print(f"  AIC             = {result.aic:,.2f}")
    print(f"  BIC             = {result.bic:,.2f}")
    print("─" * 54)

    # ── Extract and annualise conditional volatility ────────────────────────
    # arch returns DAILY conditional std dev in ×100 scaled units.
    # To get annual decimal vol:
    #   divide by 100  →  unscale
    #   multiply by sqrt(252)  →  annualise  (252 trading days/year)
    cond_vol = (result.conditional_volatility / 100) * np.sqrt(252)

    return result, cond_vol


# ─────────────────────────────────────────────
# 3.  Diagnostic helper
# ─────────────────────────────────────────────

def garch_diagnostics(result, returns_demeaned: pd.Series) -> None:
    """
    Print residual diagnostics to validate GARCH fit quality.
    Standardised residuals should be approximately iid Student-t(ν).
    """
    from scipy import stats as scipy_stats

    std_resid = result.resid / result.conditional_volatility
    std_resid = std_resid.dropna()

    # Ljung-Box test on squared standardised residuals
    # H0: no autocorrelation in squared residuals
    # p > 0.05 → GARCH has adequately captured the volatility structure
    from scipy import stats as _sp_stats

    sq = (std_resid ** 2).values
    n  = len(sq)

    def _ljung_box_pval(x, lag):
        acf_vals = [np.corrcoef(x[:-k], x[k:])[0, 1] for k in range(1, lag + 1)]
        q = n * (n + 2) * sum(r**2 / (n - k) for k, r in enumerate(acf_vals, 1))
        return q, float(1 - _sp_stats.chi2.cdf(q, df=lag))

    print("\n── GARCH residual diagnostics ────────────────────────")
    print("  Ljung-Box test on squared standardised residuals")
    print("  H0: no remaining autocorrelation (p > 0.05 = good fit)")
    for lag in [10, 20]:
        _, pval = _ljung_box_pval(sq, lag)
        verdict = "PASS" if pval > 0.05 else "FAIL"
        print(f"    Lag {lag:2d}  p-value = {pval:.4f}  [{verdict}]")

    # Jarque-Bera normality test (expect FAIL → confirms fat tails)
    jb_stat, jb_p = scipy_stats.jarque_bera(std_resid)
    print(f"\n  Jarque-Bera normality test  p = {jb_p:.4f}")
    print("  (Expect p < 0.05 for financial returns → justifies Student-t)")
    print("─" * 54)
