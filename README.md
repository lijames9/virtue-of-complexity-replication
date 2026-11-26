# The Virtue of Complexity in Return Prediction: Replication

Replication of the empirical analysis (Section V) of Kelly, Malamud & Zhou (2024),
*The Virtue of Complexity in Return Prediction*, Journal of Finance 79(1).

Monthly market-timing forecasts of the vol-standardized CRSP value-weighted excess return
from the 15 Goyal–Welch predictors, using random Fourier features (RFF) and rolling-window
ridge regression across model complexity c = P/T.

## Files

| File | Purpose |
|---|---|
| `data.py` | Loads CRSP + Goyal–Welch data, builds the 15 predictors, standardizes, aligns `G_t` with `R_{t+1}` (1930-01 to 2020-11, 1,091 months) |
| `backtest.py` | Engine: `rff()` random Fourier features (eq. 20), `rolling_ridge()` rolling ridge for a grid of P and z, `performance()` out-of-sample stats |
| `run_experiment.py` | Driver: VoC curves (Figs 7–9), Table I, timing positions (Fig 10), variable importance (Fig 11) |
| `PredictorData2025.xlsx` | Goyal–Welch predictor data (public) |

## Data

`CRSP_Stock_Index_Data.csv` is **not** in the repo because CRSP data is licensed through WRDS.
Get it from the shared group folder (or download monthly `vwretd`, `vwretx`, `vwtotval` for the
CRSP value-weighted index from WRDS) and place it in the project root.

## Setup

Python 3.11+:

```bash
pip install -r requirements.txt
```

## Running

```bash
python run_experiment.py                      # 100 seeds, T = 12, 60, 120 (~20 min on 8 cores)
python run_experiment.py --T 12 --seeds 20    # quick run
python run_experiment.py --seeds 1000 --vi    # paper setting (~3 h on 8 cores)
```

Outputs go to `results/`: `voc_curves.csv`, `fig7_T*.png`, `fig8_T*.png`, `table1.csv`,
`positions.csv`, `fig10_positions.png`, and with `--vi`, `variable_importance.csv` and `fig11_vi.png`.

## Implementation notes

- RFFs: `S = [sin(γ ω'G), cos(γ ω'G)]`, `ω ~ N(0, I)`, γ = 2, P from 2 to 12,000.
- Ridge: `β(z) = (zI + S'S/T)^{-1} S'R/T`, no intercept. It is solved in kernel form so that P ≫ T is cheap.
  Training signals are standardized by their training-window std (footnote 39). `z = 0` is the ridgeless (min-norm) limit.
- Out-of-sample R² = 1 − Σ(R − F)² / ΣR² (benchmarked against a zero forecast).
- Sharpe ratio and information ratio are annualized by √12. Alpha is from a regression of the strategy return on the vol-standardized market.
- VoC curves average each statistic across seeds. Table I and Fig 10 use the seed-averaged forecast.
