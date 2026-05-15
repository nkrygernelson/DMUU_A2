"""Probe what actually correlates with per-day SP regret.

For each day, compute many candidate "surprise" / "difficulty" measures and
report correlations with the regret. The goal is to falsify or confirm
several intuitions:
  - Are high-regret days statistical outliers from the AR1 process? → AR1 surprise
  - Are they expensive overall? → sum/mean/max of prices
  - Do they have volatile prices? → price std within day
  - Do they have unusual occupancy? → occupancy mean/std
  - Does the timing of cheap hours matter? → e.g., late-day price drops that SP couldn't see at t=0
  - Is regret simply scaled by the realized cost level? → corr with hindsight cost itself
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ar1_expected(cur, prev):
    return 1.48 * cur - 0.6 * prev + 0.48


def main():
    prices_df = pd.read_csv(ROOT / "data" / "v2_PriceData.csv")
    occ1_df = pd.read_csv(ROOT / "data" / "OccupancyRoom1.csv")
    occ2_df = pd.read_csv(ROOT / "data" / "OccupancyRoom2.csv")
    sp = np.load(ROOT / "plots" / "sp_gurobi_per_day.npz")["costs"]
    hind = np.load(ROOT / "plots" / "hindsight_optimal_per_day.npz")["costs"]
    regret = sp - hind

    measures = {}
    paths_p = []
    paths_o1 = []
    paths_o2 = []
    for d in range(len(sp)):
        row = prices_df.iloc[d].values
        prev = float(row[0])
        prices = np.array([float(v) for v in row[1:]])
        o1 = occ1_df.iloc[d].values.astype(float)
        o2 = occ2_df.iloc[d].values.astype(float)
        paths_p.append(prices); paths_o1.append(o1); paths_o2.append(o2)

        # AR1 surprise
        res = [prices[0] - ar1_expected(prev, prev),
               prices[1] - ar1_expected(prices[0], prev)]
        for t in range(2, len(prices)):
            res.append(prices[t] - ar1_expected(prices[t - 1], prices[t - 2]))
        measures.setdefault("ar1_surprise", []).append(float(sum(r ** 2 for r in res)))
        measures.setdefault("price_mean", []).append(float(prices.mean()))
        measures.setdefault("price_std", []).append(float(prices.std()))
        measures.setdefault("price_max", []).append(float(prices.max()))
        measures.setdefault("price_min", []).append(float(prices.min()))
        measures.setdefault("price_range", []).append(float(prices.max() - prices.min()))
        # Late-day cheap stretch SP can't see early
        measures.setdefault("late_min_price", []).append(float(prices[5:].min()))
        measures.setdefault("early_max_price", []).append(float(prices[:5].max()))
        # Trajectory shape descriptors
        measures.setdefault("price_trend", []).append(float(prices[-1] - prices[0]))
        # Occupancy stats
        measures.setdefault("occ_total_mean", []).append(float((o1 + o2).mean()))
        measures.setdefault("occ_total_max", []).append(float((o1 + o2).max()))
        measures.setdefault("occ_total_std", []).append(float((o1 + o2).std()))
        # Day "cost level" descriptors
        measures.setdefault("hindsight_cost", []).append(float(hind[d]))
        measures.setdefault("sp_cost", []).append(float(sp[d]))

    print("Correlations with regret (SP - hindsight):")
    cors = []
    for k, v in measures.items():
        arr = np.array(v, dtype=float)
        c = np.corrcoef(arr, regret)[0, 1]
        cors.append((k, c))
    cors.sort(key=lambda kv: -abs(kv[1]))
    for k, c in cors:
        print(f"  {k:20s}  r = {c:+.3f}")

    # Also try the same with RELATIVE regret = regret / hindsight
    rel_regret = regret / np.maximum(hind, 1.0)
    print("\nCorrelations with relative regret ((SP - hindsight)/hindsight):")
    cors_rel = []
    for k, v in measures.items():
        arr = np.array(v, dtype=float)
        c = np.corrcoef(arr, rel_regret)[0, 1]
        cors_rel.append((k, c))
    cors_rel.sort(key=lambda kv: -abs(kv[1]))
    for k, c in cors_rel:
        print(f"  {k:20s}  r = {c:+.3f}")

    # Bar chart of correlations: absolute regret vs relative regret
    fig, ax = plt.subplots(figsize=(11, 5))
    keys = [k for k, _ in cors]
    abs_r = [c for _, c in cors]
    rel_r = [dict(cors_rel)[k] for k in keys]
    x = np.arange(len(keys))
    width = 0.4
    ax.bar(x - width/2, abs_r, width, label="vs. absolute regret", color="C0")
    ax.bar(x + width/2, rel_r, width, label="vs. relative regret", color="C3")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha="right")
    ax.set_ylabel("Pearson correlation with regret")
    ax.set_title("Per-day measures vs. SP regret over hindsight (n=100)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ROOT / "plots" / "regret_correlates.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
