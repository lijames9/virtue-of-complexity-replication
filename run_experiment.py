"""
Replication of the empirical analysis (Section V) of
Kelly, Malamud & Zhou (2024), "The Virtue of Complexity in Return Prediction".

Produces (in --out, default ./results):
  voc_curves.csv          seed-averaged OOS stats for every (T, P, z)   -> Figs 7, 8, 9
  fig7_T{T}.png           R^2, ||beta||^2, E[R], volatility vs c = P/T
  fig8_T{T}.png           Sharpe, alpha, IR, alpha t-stat vs c
  table1.csv              Linear (Goyal-Welch kitchen sink) vs nonlinear  -> Table I
  positions.csv / fig10_positions.png   timing positions, P=12000, z=1e3 -> Fig 10
  variable_importance.csv / fig11_vi.png  (with --vi)                   -> Fig 11

Usage
  python run_experiment.py                      # 100 seeds, T = 12, 60, 120
  python run_experiment.py --seeds 1000         # full paper setting (slow)
  python run_experiment.py --T 12 --seeds 20 --vi
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # small matrices: 1 BLAS thread per process is much faster

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_data
from backtest import Z_GRID, rff, rolling_ridge, performance, alpha_vs

P_GRID = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 36, 48, 60, 90, 120, 150, 180,
          240, 360, 480, 600, 1200, 2400, 3600, 6000, 9000, 12000]
METRICS = ["R2", "beta_norm2", "ER", "Vol", "SR", "SR_t", "alpha", "alpha_t",
           "IR", "max_loss", "skew"]

NBER = [("1929-08", "1933-03"), ("1937-05", "1938-06"), ("1945-02", "1945-10"),
        ("1948-11", "1949-10"), ("1953-07", "1954-05"), ("1957-08", "1958-04"),
        ("1960-04", "1961-02"), ("1969-12", "1970-11"), ("1973-11", "1975-03"),
        ("1980-01", "1980-07"), ("1981-07", "1982-11"), ("1990-07", "1991-03"),
        ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04")]


# --------------------------------------------------------------------------- #
# Workers (top level so they can be pickled on Windows)
# --------------------------------------------------------------------------- #
def run_seed(seed, G, R, Ts, P_grid, z_grid, gamma):
    """One repetition of steps (i)-(iv): draw RFFs, run every T, P, z."""
    S = rff(G, max(P_grid), gamma, seed)
    out = {}
    for T in Ts:
        fc, bn, r = rolling_ridge(S, R, T, z_grid, P_grid)
        m = performance(fc, r)
        m["beta_norm2"] = bn.mean(-1)
        out[T] = (m, fc)
    return out


def run_vi_seed(seed, G, R, T, P, z, gamma):
    """Full model and the 15 leave-one-predictor-out models, one seed."""
    res = []
    for drop in [None] + list(range(G.shape[1])):
        Gi = G if drop is None else np.delete(G, drop, axis=1)
        fc, _, r = rolling_ridge(rff(Gi, P, gamma, seed), R, T, [z])
        m = performance(fc[0, 0], r)
        res.append((m["R2"], m["SR"]))
    return np.array(res)          # (16, 2)


# --------------------------------------------------------------------------- #
def run_main(G, R, args):
    Ts, nP, nz = args.T, len(args.P_grid), len(Z_GRID)
    sums = {T: {k: np.zeros((nP, nz)) for k in METRICS} for T in Ts}
    fc_sum = {T: np.zeros((nP, nz, len(R) - T)) for T in Ts}

    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_seed, s, G, R, Ts, args.P_grid, Z_GRID, args.gamma)
                for s in range(args.seeds)]
        for f in as_completed(futs):
            for T, (m, fc) in f.result().items():
                for k in METRICS:
                    sums[T][k] += m[k]
                fc_sum[T] += fc
            done += 1
            if done % max(1, args.seeds // 20) == 0 or done == args.seeds:
                print(f"  seeds {done}/{args.seeds}  ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for T in Ts:
        for i, P in enumerate(args.P_grid):
            for j, z in enumerate(Z_GRID):
                row = {"T": T, "P": P, "c": P / T, "log10z": int(np.log10(z))}
                row.update({k: sums[T][k][i, j] / args.seeds for k in METRICS})
                rows.append(row)
    voc = pd.DataFrame(rows)
    fc_avg = {T: fc_sum[T] / args.seeds for T in Ts}
    return voc, fc_avg


def make_table1(G, R, fc_avg, args):
    """Table I: linear kitchen-sink ridge (z=0+, 1e3) vs nonlinear (P=Pmax, z=1e3)."""
    rows = []
    jz = list(Z_GRID).index(1e3)
    for T in args.T:
        r = R[T:]
        lin_fc, _, _ = rolling_ridge(G, R, T, [0.0, 1e3], standardize=False)
        lin_strat = lin_fc[0, 1] * r
        models = [("Linear", "0+", lin_fc[0, 0]),
                  ("Linear", "1e3", lin_fc[0, 1]),
                  ("Nonlinear", "1e3", fc_avg[T][-1, jz])]
        for name, z, f in models:
            m = performance(f, r)
            row = {"T": T, "Model": name, "z": z, "R2": m["R2"], "SR": m["SR"],
                   "SR_t": m["SR_t"], "IR_v_Mkt": m["IR"], "t_v_Mkt": m["alpha_t"],
                   "IR_v_Lin": np.nan, "t_v_Lin": np.nan,
                   "MaxLoss": m["max_loss"], "Skew": m["skew"]}
            if name == "Nonlinear":
                row["IR_v_Lin"], row["t_v_Lin"] = alpha_vs(f * r, lin_strat)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_voc(voc, T, out):
    d = voc[voc["T"] == T]
    specs = {
        "fig7": [("R2", r"Panel A: $R^2$", (-3, 0.1)),
                 ("beta_norm2", r"Panel B: $\|\hat\beta\|^2$", (0, 3)),
                 ("ER", "Panel C: Expected Return", None),
                 ("Vol", "Panel D: Volatility", (0, 5))],
        "fig8": [("SR", "Panel A: Sharpe Ratio", None),
                 ("alpha", "Panel B: Alpha", None),
                 ("IR", "Panel C: Information Ratio", None),
                 ("alpha_t", r"Panel D: Alpha $t$-statistic", None)],
    }
    cmap = plt.get_cmap("viridis")
    for name, panels in specs.items():
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (col, title, ylim) in zip(axes.flat, panels):
            for k, lz in enumerate(sorted(d["log10z"].unique())):
                s = d[d["log10z"] == lz].sort_values("c")
                ax.plot(s["c"], s[col], color=cmap(k / 6), lw=1.4,
                        label=rf"$\log_{{10}}(z)={lz}$")
            ax.axvline(1, color="grey", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("c = P / T (log scale)")
            ax.set_title(title)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Out-of-sample market timing performance (T = {T})")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}_T{T}.png"), dpi=150)
        plt.close(fig)


def plot_positions(pos, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    for s, e in NBER:
        ax.axvspan(pd.Period(s, "M").to_timestamp(), pd.Period(e, "M").to_timestamp(),
                   color="grey", alpha=0.25, lw=0)
    for col in pos.columns:
        ser = pos[col].rolling(6).mean()
        ax.plot(ser.index.to_timestamp(), ser.values, lw=1, label=col)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Market timing positions (6-month MA), P = 12000, z = 1e3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig10_positions.png"), dpi=150)
    plt.close(fig)


def plot_vi(vi, out):
    vi = vi.sort_values("VI_R2", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(vi.index, vi["VI_R2"] * 100, color="tab:blue")
    ax.set_ylabel(r"VI ($R^2$, %)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(vi.index, vi["VI_SR"], color="tab:red", marker="o")
    ax2.set_ylabel("VI (Sharpe ratio)", color="tab:red")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Variable importance (T = 12, P = 12000, z = 1e3)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig11_vi.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=100, help="RFF repetitions (paper: 1000)")
    p.add_argument("--T", type=int, nargs="+", default=[12, 60, 120], help="training windows")
    p.add_argument("--gamma", type=float, default=2.0, help="RFF bandwidth")
    p.add_argument("--pmax", type=int, default=12000, help="largest number of RFFs")
    p.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel processes")
    p.add_argument("--vi", action="store_true", help="also run variable importance (Fig 11)")
    p.add_argument("--vi-seeds", type=int, default=50)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    args.P_grid = [P for P in P_GRID if P < args.pmax] + [args.pmax]
    os.makedirs(args.out, exist_ok=True)

    G_df, R_ser = get_data()
    G, R = G_df.values, R_ser.values
    print(f"Data: {G.shape[0]} months {G_df.index[0]}..{G_df.index[-1]}, {G.shape[1]} predictors")

    # ---- VoC curves (Figs 7-9) -------------------------------------------- #
    print(f"Running {args.seeds} seeds, T={args.T}, P up to {args.pmax}, {args.jobs} jobs")
    voc, fc_avg = run_main(G, R, args)
    voc.to_csv(os.path.join(args.out, "voc_curves.csv"), index=False)
    for T in args.T:
        plot_voc(voc, T, args.out)

    # ---- Table I ----------------------------------------------------------- #
    tab = make_table1(G, R, fc_avg, args)
    tab.to_csv(os.path.join(args.out, "table1.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nTable I  (nonlinear = seed-averaged forecasts, P = Pmax, z = 1e3)")
    print(tab.round(3).to_string(index=False))

    # ---- Fig 10 positions --------------------------------------------------- #
    jz = list(Z_GRID).index(1e3)
    pos = pd.DataFrame({f"T={T}": pd.Series(fc_avg[T][-1, jz], index=G_df.index[T:])
                        for T in args.T})
    pos.to_csv(os.path.join(args.out, "positions.csv"))
    plot_positions(pos, args.out)
    if len(args.T) > 1:
        print("\nPosition correlations across T:\n", pos.corr().round(2))

    # ---- High-complexity summary -------------------------------------------- #
    hi = voc[(voc.P == args.pmax)].set_index(["T", "log10z"])[
        ["R2", "ER", "Vol", "SR", "IR", "alpha_t"]]
    print(f"\nSeed-averaged stats at P = {args.pmax}:\n", hi.round(3).to_string())

    # ---- Fig 11 variable importance --------------------------------------- #
    if args.vi:
        print(f"\nVariable importance: {args.vi_seeds} seeds")
        acc = np.zeros((G.shape[1] + 1, 2))
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_vi_seed, s, G, R, 12, args.pmax, 1e3, args.gamma)
                    for s in range(args.vi_seeds)]
            for f in as_completed(futs):
                acc += f.result()
        acc /= args.vi_seeds
        vi = pd.DataFrame({"VI_R2": acc[0, 0] - acc[1:, 0], "VI_SR": acc[0, 1] - acc[1:, 1]},
                          index=G_df.columns)
        vi.to_csv(os.path.join(args.out, "variable_importance.csv"))
        plot_vi(vi, args.out)
        print(vi.sort_values("VI_R2", ascending=False).round(4).to_string())

    print(f"\nOutputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
