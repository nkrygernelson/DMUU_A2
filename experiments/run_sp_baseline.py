"""Run plain SP on the 100-day eval and save per-day costs.

Output: plots/sp_gurobi_per_day.npz (npz with a single array `costs`).

Used downstream by:
  - experiments/plot_sp_vs_hindsight.py
  - experiments/regret_correlates.py
  - experiments/plot_regret_vs_trend.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.simulator import load_experiments, evaluate
from policies import sp_policy


if __name__ == "__main__":
    exp = load_experiments()
    avg, costs = evaluate(sp_policy, exp)
    out = ROOT / "plots" / "sp_gurobi_per_day.npz"
    np.savez(out, costs=np.array(costs))
    print(f"SP 100-day mean cost: {avg:.2f}")
    print(f"  std: {np.std(costs):.2f}  min: {min(costs):.1f}  max: {max(costs):.1f}")
    print(f"  saved per-day costs to {out}")
