"""Plot SP cost vs hindsight cost per day.

Two panels:
  (a) Scatter of (hindsight_d, sp_d) for d = 1..100, with the y=x line.
      Points above the line are days where SP underperformed hindsight; the
      vertical gap is the per-day regret. No point can sit below y=x in
      expectation — hindsight is the absolute lower bound.
  (b) Sorted gap (sp_d − hindsight_d) per day, ranked. Shows how the
      total regret distributes across days.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

sp = np.load(ROOT / "plots" / "sp_gurobi_per_day.npz")["costs"]
hind = np.load(ROOT / "plots" / "hindsight_optimal_per_day.npz")["costs"]

mean_sp = float(sp.mean())
mean_hind = float(hind.mean())
gap = sp - hind

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(hind, sp, s=24, alpha=0.7, color="C0")
lo = min(sp.min(), hind.min()) - 5
hi = max(sp.max(), hind.max()) + 5
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x (perfect)")
ax.set_xlabel("Hindsight-optimal cost (day)")
ax.set_ylabel("SP-Gurobi cost (day)")
ax.set_title(f"Per-day cost: SP vs hindsight\n"
             f"SP mean = {mean_sp:.2f}, hindsight mean = {mean_hind:.2f}")
ax.legend()
ax.text(0.04, 0.96,
        f"mean gap = {(mean_sp - mean_hind):.2f}\n"
        f"days where SP < hindsight = {int((gap < 0).sum())} "
        f"(numerical noise)\n"
        f"days where SP > hindsight + 10 = {int((gap > 10).sum())}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

ax = axes[1]
sorted_gap = np.sort(gap)[::-1]
ax.bar(range(len(sorted_gap)), sorted_gap, color="C3", alpha=0.8)
ax.axhline(0, color="k", lw=0.5)
ax.axhline(gap.mean(), color="k", ls="--", lw=0.8,
           label=f"mean = {gap.mean():.2f}")
ax.set_xlabel("Day (sorted by gap, descending)")
ax.set_ylabel("SP − hindsight (cost units)")
ax.set_title("Per-day SP regret over hindsight, sorted")
ax.legend()

fig.tight_layout()
out = ROOT / "plots" / "sp_vs_hindsight_per_day.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"wrote {out}")
print(f"SP mean: {mean_sp:.2f}  hindsight mean: {mean_hind:.2f}  "
      f"gap mean: {gap.mean():.2f}  gap std: {gap.std():.2f}")
print(f"median per-day gap: {np.median(gap):.2f}")
print(f"days with gap > 20: {int((gap > 20).sum())}")
print(f"days with gap < 2:  {int((gap < 2).sum())}")
