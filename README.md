# Nifty 50 — Volatility Regime Detection
### GARCH(1,1) + Hidden Markov Model

> **Detecting bull and crisis market regimes in Indian equities using a two-layer statistical model.**

---

## Overview

This project builds a two-layer time series model to automatically detect **market regimes** (calm vs stressed) in the Nifty 50 index (NSE India) from 2010 to 2024.

**Layer 1 — GARCH(1,1):** Extracts the day-by-day conditional volatility from Nifty 50 log returns, capturing the well-known phenomenon of *volatility clustering* — large moves follow large moves.

**Layer 2 — Hidden Markov Model:** Segments the GARCH volatility series into discrete regimes using the Viterbi algorithm. The model learns the transition probabilities between regimes entirely from data.

**Why both?** GARCH gives a smooth, continuous volatility estimate. HMM converts it into an actionable discrete label. Neither alone achieves both.

---

## Mathematical model

**GARCH(1,1) variance equation:**

```
σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
```

| Parameter | Meaning | Typical Nifty value |
|-----------|---------|---------------------|
| ω | Long-run variance floor | small positive |
| α | Weight on yesterday's shock | ~0.08–0.15 |
| β | Weight on yesterday's variance | ~0.82–0.90 |
| α + β | Persistence | **0.97–0.99** |

High persistence means volatility shocks take weeks to decay — characteristic of emerging markets.

**HMM components:**

| Symbol | Meaning |
|--------|---------|
| π | Initial state distribution |
| A | Transition matrix — P(S_t \| S_{t-1}) |
| B | Gaussian emission — N(μ_k, σ²_k) per state |

The Viterbi algorithm decodes the most likely hidden state sequence.

---

## Results

### Regime statistics (sample output)

| Regime | Days | Ann. return | Ann. volatility | Sharpe |
|--------|------|-------------|-----------------|--------|
| Low volatility | ~2,400 | +14–16% | ~12% | ~1.1 |
| High volatility | ~1,000 | −2–5% | ~28% | ~−0.1 |

### Backtest: regime-filtered strategy vs buy-and-hold

Strategy: long Nifty in low-vol regime; cash in high-vol regime.

| Strategy | Ann. return | Ann. vol | Sharpe | Max drawdown |
|----------|-------------|----------|--------|--------------|
| Regime strategy | higher | lower | better | smaller |
| Buy-and-hold | baseline | higher | lower | larger |

*Note: No transaction costs. Position determined by previous day's regime (no look-ahead bias).*

---

## Project structure

```
nifty-volatility-regimes/
│
├── src/
│   ├── data_loader.py    # download & clean Nifty 50 data (yfinance)
│   ├── garch_model.py    # GARCH order selection, fitting, diagnostics
│   ├── hmm_model.py      # HMM fitting, Viterbi decoding, statistics
│   ├── visualise.py      # all 4 plot functions
│   └── backtesting.py    # regime-filtered trading strategy
│
├── notebooks/
│   └── analysis.ipynb    # step-by-step walkthrough with plots
│
├── outputs/              # generated plots (created on first run)
├── main.py               # run the full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/nifty-volatility-regimes.git
cd nifty-volatility-regimes
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
- Download Nifty 50 data automatically (internet required)
- Fit GARCH and print parameter estimates
- Fit HMM and print transition matrix + regime statistics
- Save 4 plots to `outputs/`
- Print backtest performance metrics

### 3. Or explore the notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Key outputs (plots saved to `outputs/`)

| File | Description |
|------|-------------|
| `regime_dashboard.png` | 4-panel: price, returns, volatility, regime — all with shading |
| `vol_distributions.png` | KDE of volatility in each regime |
| `transition_heatmap.png` | HMM transition matrix heatmap |
| `regime_calendar.png` | Annual % of days in each regime |
| `backtest_equity.png` | Equity curve: strategy vs buy-and-hold |

---

## Dependencies

```
yfinance    — download historical market data
arch        — GARCH model estimation
hmmlearn    — Hidden Markov Model
scikit-learn — StandardScaler
pandas, numpy, matplotlib, seaborn
```

All installable via `pip install -r requirements.txt`.

---

## Extensions (ideas for further work)

- **3-state HMM**: Low / Medium / High volatility with AIC comparison
- **Online regime detection**: update regime label in real time as new data arrives
- **Multivariate extension**: model regime across Nifty 50 + USD/INR + VIX India jointly
- **Regime-conditional options pricing**: use regime-specific volatility in Black-Scholes

---

## Author

**Kirti Beniwal**  
PhD, Applied Mathematics — Delhi Technological University  
Research areas: Machine Learning, Numerical Methods, Time Series, Cyclone Modelling  
CSIR-UGC NET (Mathematical Sciences) — AIR 22 (2020)

---

## References

1. Engle, R.F. (1982). *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation.* Econometrica.
2. Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity.* Journal of Econometrics.
3. Baum, L. et al. (1970). *A Maximization Technique Occurring in the Statistical Analysis of Probabilistic Functions of Markov Chains.* Annals of Mathematical Statistics.
4. Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica.
