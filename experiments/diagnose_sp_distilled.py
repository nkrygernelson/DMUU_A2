"""Diagnose the SP-distilled single-eta V_theta on the cached SP-rollout records.

Plots:
  plots/sp_distilled_per_stage_scatter.png — predicted vs actual scatter per stage,
                                             y=x line, per-stage fit, R^2, bias.
  plots/sp_distilled_per_stage_error.png   — bar charts: bias, RMSE, slope per stage.

This is a held-out-ish diagnostic: we trained on the same records, so this measures
*how well the ridge fits its training set*, not generalization. Still useful to see
whether the model has any signal at every stage.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "policies" / "adp_single_sp_rollouts.npz"
ETAS = ROOT / "policies" / "adp_etas_single_sp.npy"


def per_stage_scatter(records, etas, out_path):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    n_stages = etas.shape[0]
    for t in range(min(10, n_stages - 1)):
        ax = axes[t // 5, t % 5]
        recs = [r for r in records if r["t"] == t]
        if not recs:
            ax.set_title(f"t={t}\nno data"); continue
        phis = np.array([r["phi"] for r in recs])
        actuals = np.array([r["rtg"] for r in recs])
        preds = phis @ etas[t]
        ax.scatter(actuals, preds, s=14, alpha=0.6, color="C0")
        lo = min(actuals.min(), preds.min())
        hi = max(actuals.max(), preds.max())
        if hi <= lo:
            hi = lo + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        if len(actuals) > 1 and actuals.std() > 1e-9:
            slope, intercept = np.polyfit(actuals, preds, 1)
            xs = np.linspace(lo, hi, 50)
            ax.plot(xs, slope * xs + intercept, "r-", lw=1.2)
            ss_res = ((preds - actuals) ** 2).sum()
            ss_tot = ((actuals - actuals.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            bias = (preds - actuals).mean()
            ax.set_title(f"t={t}  n={len(actuals)}\n"
                         f"slope={slope:.2f}  bias={bias:+.1f}  R²={r2:.2f}",
                         fontsize=9)
        else:
            ax.set_title(f"t={t}  n={len(actuals)}", fontsize=9)
        if t // 5 == 1:
            ax.set_xlabel("actual SP cost-to-go")
        if t % 5 == 0:
            ax.set_ylabel("predicted V_θ")
    fig.suptitle("SP-distilled V_θ per stage (training-set scatter)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def per_stage_error(records, etas, out_path):
    n_stages = etas.shape[0]
    stages = list(range(min(10, n_stages - 1)))
    bias_t, rmse_t, slope_t, r2_t = [], [], [], []
    for t in stages:
        recs = [r for r in records if r["t"] == t]
        if not recs:
            bias_t.append(np.nan); rmse_t.append(np.nan)
            slope_t.append(np.nan); r2_t.append(np.nan); continue
        phis = np.array([r["phi"] for r in recs])
        actuals = np.array([r["rtg"] for r in recs])
        preds = phis @ etas[t]
        bias_t.append(float((preds - actuals).mean()))
        rmse_t.append(float(np.sqrt(((preds - actuals) ** 2).mean())))
        if len(actuals) > 1 and actuals.std() > 1e-9:
            slope, _ = np.polyfit(actuals, preds, 1)
            ss_res = ((preds - actuals) ** 2).sum()
            ss_tot = ((actuals - actuals.mean()) ** 2).sum()
            slope_t.append(slope)
            r2_t.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        else:
            slope_t.append(float("nan")); r2_t.append(float("nan"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(stages, bias_t, color="steelblue")
    axes[0, 0].axhline(0, color="k", lw=0.5)
    axes[0, 0].set_title("Bias (predicted − actual) per stage"); axes[0, 0].set_xlabel("stage t")
    axes[0, 1].bar(stages, rmse_t, color="indianred")
    axes[0, 1].set_title("RMSE per stage"); axes[0, 1].set_xlabel("stage t")
    axes[1, 0].bar(stages, slope_t, color="seagreen")
    axes[1, 0].axhline(1, color="k", ls="--", lw=0.7, label="ideal slope=1")
    axes[1, 0].set_title("Slope per stage"); axes[1, 0].set_xlabel("stage t"); axes[1, 0].legend()
    axes[1, 1].bar(stages, r2_t, color="goldenrod")
    axes[1, 1].axhline(1, color="k", ls="--", lw=0.7)
    axes[1, 1].set_title("R² per stage"); axes[1, 1].set_xlabel("stage t")
    fig.suptitle("SP-distilled V_θ — per-stage fit metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    print(f"Loading records from {CACHE}")
    cache = np.load(CACHE)
    records = [{"t": int(cache["t"][i]), "phi": cache["phi"][i], "rtg": float(cache["rtg"][i])}
               for i in range(len(cache["t"]))]
    etas = np.load(ETAS)
    print(f"Loaded {len(records)} records and etas {etas.shape}")
    out_dir = ROOT / "plots"
    per_stage_scatter(records, etas, out_dir / "sp_distilled_per_stage_scatter.png")
    per_stage_error(records, etas, out_dir / "sp_distilled_per_stage_error.png")
