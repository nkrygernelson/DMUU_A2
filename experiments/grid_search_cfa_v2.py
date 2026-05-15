"""Finer grid search for the threshold-form CFA penalties.

Threshold form:
  cold-buffer: alpha * sum_leaves wp * (max(0, T_LOW_PEN - T_leaf) per room)
  humid-buffer: beta * sum_leaves wp * max(0, H_leaf - H_HIGH_PEN)

Defaults: T_LOW_PEN=20 (override at 18), H_HIGH_PEN=60 (override at 70).

Grid: alpha x beta. Screen on 50 days; re-eval top configs on 100.

Outputs:
  plots/cfa_grid_v2_heatmap.png  — mean cost as alpha x beta heatmap
  plots/cfa_grid_v2_results.csv  — raw numbers
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.simulator import load_experiments, evaluate
from policies import sp_cfa_policy

ALPHAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
BETAS  = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

SCREEN_DAYS = 50
FINAL_DAYS = 100
TOP_K_RECHECK = 3


def run_one(alpha, beta, exp):
    sp_cfa_policy.ALPHA = alpha
    sp_cfa_policy.BETA = beta
    t0 = time.time()
    avg, costs = evaluate(sp_cfa_policy, exp)
    return float(avg), float(np.std(costs)), time.time() - t0


def plot_heatmap(df, out_path, title="CFA mean cost"):
    pivot = df.pivot(index="beta", columns="alpha", values="mean_cost")
    fig, ax = plt.subplots(figsize=(7, 5))
    data = pivot.to_numpy().astype(float)
    # Center colorbar around the median for contrast
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    im = ax.imshow(data, aspect="auto", cmap="viridis_r", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{v:g}" for v in pivot.index])
    ax.set_xlabel("alpha (cold-buffer)")
    ax.set_ylabel("beta (humid-buffer)")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            color = "white" if v > (vmin + 0.7 * (vmax - vmin)) else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="mean cost")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    experiments = load_experiments()
    print(f"Loaded {len(experiments)} days. Screening on {SCREEN_DAYS}, "
          f"final on {FINAL_DAYS}.")
    print(f"Thresholds: T_LOW_PEN={sp_cfa_policy.T_LOW_PEN}, "
          f"H_HIGH_PEN={sp_cfa_policy.H_HIGH_PEN}")

    print(f"\nGrid: alpha in {ALPHAS}, beta in {BETAS}  "
          f"({len(ALPHAS) * len(BETAS)} configs)")

    rows = []
    print(f"\n{'alpha':>6} {'beta':>6}  {'mean':>8}  {'std':>7}  {'wall':>6}")
    for alpha in ALPHAS:
        for beta in BETAS:
            mean, std, wall = run_one(alpha, beta, experiments[:SCREEN_DAYS])
            rows.append({"alpha": alpha, "beta": beta,
                         "mean_cost": mean, "std_cost": std, "wall_s": wall,
                         "n_days": SCREEN_DAYS})
            print(f"{alpha:6.2f} {beta:6.2f}  {mean:8.2f}  {std:7.2f}  {wall:6.1f}")

    df = pd.DataFrame(rows)
    out_dir = ROOT / "plots"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "cfa_grid_v2_results.csv", index=False)
    plot_heatmap(df, out_dir / "cfa_grid_v2_heatmap_screen.png",
                 title=f"CFA mean cost (screen on {SCREEN_DAYS} days)")

    # Top-K recheck on 100 days
    df_sorted = df.sort_values("mean_cost")
    print(f"\nTop {TOP_K_RECHECK} on screening, re-evaluating on {FINAL_DAYS} days:")
    final_rows = []
    for _, row in df_sorted.head(TOP_K_RECHECK).iterrows():
        alpha, beta = row["alpha"], row["beta"]
        mean, std, wall = run_one(alpha, beta, experiments)
        final_rows.append({"alpha": alpha, "beta": beta,
                           "mean_cost": mean, "std_cost": std, "wall_s": wall,
                           "n_days": FINAL_DAYS})
        print(f"  alpha={alpha:5.2f}  beta={beta:5.2f}  "
              f"100-day mean={mean:.2f}  std={std:.2f}  wall={wall:.1f}s")

    # Plain-SP baseline at top of final list for comparison
    plain_mean, plain_std, plain_wall = run_one(0.0, 0.0, experiments)
    print(f"  PLAIN SP (alpha=0, beta=0): 100-day mean={plain_mean:.2f}  "
          f"std={plain_std:.2f}  wall={plain_wall:.1f}s")
    final_rows.append({"alpha": 0.0, "beta": 0.0,
                       "mean_cost": plain_mean, "std_cost": plain_std,
                       "wall_s": plain_wall, "n_days": FINAL_DAYS})

    pd.DataFrame(final_rows).to_csv(out_dir / "cfa_grid_v2_final.csv", index=False)
    print("\nDelta vs plain SP:")
    for row in final_rows:
        print(f"  alpha={row['alpha']:5.2f} beta={row['beta']:5.2f}: "
              f"{row['mean_cost'] - plain_mean:+.2f}")
