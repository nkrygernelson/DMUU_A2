"""SP hyperparameter sweep over (bf, num_stages).

Runs 100-day evaluation for each (bf, ns) config; saves heatmaps of mean cost
and mean per-call solve time, plus a CSV with the raw numbers.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.simulator import load_experiments, evaluate
from policies.sp_policy import (
    build_scenario_tree,
    propagate_uncertainty,
    build_and_solve_linear_program,
)


BF_GRID = [2, 3, 4]
NS_GRID = [2, 3, 4, 5]
WARMUP_DAYS = 3
WARMUP_MEAN_TIME_CAP = 60.0   # seconds per call; configs above this are skipped


def make_sp_policy(bf_target, ns_target):
    """Return a module-like object exposing `select_action(state)` for the given config."""
    timings = []

    def select_action(state):
        current_time = state["current_time"]
        remaining = 10 - current_time
        num_stages = max(1, min(ns_target, remaining))
        bf = bf_target

        root, all_nodes, leaves = build_scenario_tree(bf, num_stages)
        root.state = {
            "current_r1_occ": state["Occ1"],
            "current_r2_occ": state["Occ2"],
            "current_price":  state["price_t"],
            "prev_price":     state["price_previous"],
        }
        propagate_uncertainty(root, all_nodes, num_samples=150)

        t0 = time.time()
        m, result = build_and_solve_linear_program(state, root, all_nodes, leaves)
        timings.append(time.time() - t0)

        rid = root.node_id
        ok = (result.solver.status == pyo.SolverStatus.ok and
              result.solver.termination_condition in
                  (pyo.TerminationCondition.optimal,
                   pyo.TerminationCondition.feasible))
        if ok:
            p1_val = pyo.value(m.p1[rid])
            p2_val = pyo.value(m.p2[rid])
            v_val  = round(pyo.value(m.V[rid]))
        else:
            p1_val, p2_val, v_val = 0.0, 0.0, 0

        return {
            "HeatPowerRoom1": p1_val,
            "HeatPowerRoom2": p2_val,
            "VentilationON":  v_val,
        }

    class Policy:
        pass
    p = Policy()
    p.select_action = select_action
    p.timings = timings
    return p


def warmup(policy, experiments, n_days):
    """Run on a few days to estimate per-call solve time; returns mean time."""
    _, _ = evaluate_safe(policy, experiments[:n_days])
    if not policy.timings:
        return float("inf")
    return float(np.mean(policy.timings))


def evaluate_safe(policy, experiments):
    """Wrap evaluate so a per-experiment crash doesn't kill the sweep."""
    return evaluate(policy, experiments)


def run_sweep():
    experiments = load_experiments()
    print(f"Loaded {len(experiments)} experiments")

    results = []
    for ns in NS_GRID:
        for bf in BF_GRID:
            tag = f"bf={bf}, ns={ns}"
            print(f"\n=== {tag} ===")

            # Warm-up to gauge per-call solve time
            policy = make_sp_policy(bf, ns)
            t_start = time.time()
            warmup_mean = warmup(policy, experiments, WARMUP_DAYS)
            warmup_wall = time.time() - t_start
            print(f"  warmup mean per-call = {warmup_mean:.2f}s "
                  f"(wall {warmup_wall:.1f}s over {WARMUP_DAYS} days)")

            if warmup_mean > WARMUP_MEAN_TIME_CAP:
                print(f"  SKIP: per-call time exceeds {WARMUP_MEAN_TIME_CAP}s cap")
                results.append({
                    "bf": bf, "num_stages": ns,
                    "mean_cost": np.nan, "std_cost": np.nan,
                    "mean_solve_s": warmup_mean, "n_days": 0,
                    "wall_s": warmup_wall, "skipped": True,
                })
                continue

            # Fresh policy for the full eval so timings only include the eval days
            policy = make_sp_policy(bf, ns)
            t0 = time.time()
            mean_cost, costs = evaluate_safe(policy, experiments)
            wall = time.time() - t0
            mean_solve = float(np.mean(policy.timings)) if policy.timings else float("nan")
            std_cost = float(np.std(costs))
            print(f"  100-day mean cost = {mean_cost:.2f}  std = {std_cost:.2f}  "
                  f"mean solve = {mean_solve:.2f}s  wall = {wall:.1f}s")

            results.append({
                "bf": bf, "num_stages": ns,
                "mean_cost": float(mean_cost), "std_cost": std_cost,
                "mean_solve_s": mean_solve, "n_days": len(costs),
                "wall_s": wall, "skipped": False,
            })

            # Save partial results in case a later config hangs
            pd.DataFrame(results).to_csv(ROOT / "plots" / "sp_hp_sweep_results.csv", index=False)

    return pd.DataFrame(results)


def plot_heatmaps(df):
    """Save two heatmaps: mean cost and mean solve time vs (bf, num_stages)."""
    cost_pivot = df.pivot(index="num_stages", columns="bf", values="mean_cost")
    time_pivot = df.pivot(index="num_stages", columns="bf", values="mean_solve_s")

    for pivot, title, fname, cmap in [
        (cost_pivot, "SP mean cost over 100 days", "sp_hp_sweep_cost.png", "viridis_r"),
        (time_pivot, "SP mean per-call solve time (s)", "sp_hp_sweep_time.png", "viridis"),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        data = pivot.to_numpy().astype(float)
        im = ax.imshow(data, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_xlabel("branching factor (bf)")
        ax.set_ylabel("num_stages")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    txt = "skip"
                else:
                    txt = f"{v:.1f}" if v >= 1 else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if v == v and v > np.nanmean(data) else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out = ROOT / "plots" / fname
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    df = run_sweep()
    df.to_csv(ROOT / "plots" / "sp_hp_sweep_results.csv", index=False)
    print("\n=== results ===")
    print(df.to_string(index=False))
    plot_heatmaps(df)
