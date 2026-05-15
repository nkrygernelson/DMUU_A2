"""SP distillation: train single ridge V_theta using SP rollouts as training data.

Idea: train V_theta to predict the realized cost-to-go *of the SP policy* on
states *the SP policy visits*. This is what the hybrid's leaf evaluation
actually needs — predict what SP would cost continuing from each leaf.

Differs from `train_single_eta.py` in only one place: the rollout policy is
SP instead of the region-ADP. Saves to a separate etas file so we can
compare side-by-side.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policies import sp_policy
from policies.adp_policy import (
    features, advance_state, sample_exogenous, NUM_SLOTS, FEATURE_DIM,
)


SP_SINGLE_ETAS_PATH = ROOT / "policies" / "adp_etas_single_sp.npy"
SP_RECORDS_CACHE = ROOT / "policies" / "adp_single_sp_rollouts.npz"

# Wider initial state distribution so the t=0 regression has input variance
# (same reasoning as the wide-init experiment in adp_policy.sample_init_state).
def sample_init_state():
    return {
        "T1": float(np.random.uniform(12, 24)),
        "T2": float(np.random.uniform(12, 24)),
        "H": float(np.random.uniform(40, 90)),
        "Occ1": float(np.random.uniform(25, 35)),
        "Occ2": float(np.random.uniform(15, 25)),
        "price_t": float(np.random.uniform(2, 8)),
        "price_previous": float(np.random.uniform(2, 8)),
        "vent_counter": 0,
        "low_override_r1": 0,
        "low_override_r2": 0,
        "current_time": 0,
    }


def collect_sp_rollouts(n_days=200):
    """Roll out SP from random init states; return (t, phi, return_to_go) records."""
    rng = np.random.default_rng(11)
    records = []
    t_start = time.time()
    for d in range(n_days):
        np.random.seed(int(rng.integers(0, 2**31)))
        state = sample_init_state()
        traj = []
        costs = []
        for t in range(NUM_SLOTS):
            u = sp_policy.select_action(state)
            c_t = state["price_t"] * (u["HeatPowerRoom1"] + u["HeatPowerRoom2"]
                                       + 0.5 * u["VentilationON"])
            traj.append(state)
            costs.append(c_t)
            state = advance_state(state, u, sample_exogenous(state))
        for t in range(NUM_SLOTS):
            records.append({
                "t": t, "phi": features(traj[t]),
                "rtg": float(sum(costs[t:])),
            })
        if (d + 1) % 25 == 0 or d == 0:
            elapsed = time.time() - t_start
            print(f"  SP rollout {d+1}/{n_days} (elapsed {elapsed:.0f}s)")
    return records


def fit_ridge_cv(records, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    etas = np.zeros((NUM_SLOTS + 1, FEATURE_DIM))
    for t in range(NUM_SLOTS):
        recs = [r for r in records if r["t"] == t]
        if not recs:
            continue
        X = np.array([r["phi"] for r in recs])
        y = np.array([r["rtg"] for r in recs])
        Xf = X[:, 1:]
        ridge = RidgeCV(alphas=alphas, fit_intercept=True)
        ridge.fit(Xf, y)
        preds = ridge.predict(Xf)
        rmse = float(np.sqrt(((preds - y) ** 2).mean()))
        bias = float((preds - y).mean())
        full = np.zeros(FEATURE_DIM)
        full[0] = ridge.intercept_
        full[1:] = ridge.coef_
        etas[t] = full
        print(f"  t={t:2d} n={len(y):3d} y_mean={y.mean():7.2f} "
              f"y_std={y.std():6.2f} rmse={rmse:6.2f} bias={bias:+5.2f} "
              f"alpha*={ridge.alpha_:7.1f} intercept={ridge.intercept_:7.2f}")
    return etas


if __name__ == "__main__":
    if os.path.exists(SP_RECORDS_CACHE) and os.environ.get("RECOLLECT") != "1":
        print(f"Loading cached SP rollouts from {SP_RECORDS_CACHE}")
        cache = np.load(SP_RECORDS_CACHE)
        ts, phis, rtgs = cache["t"], cache["phi"], cache["rtg"]
        records = [{"t": int(ts[i]), "phi": phis[i], "rtg": float(rtgs[i])}
                   for i in range(len(ts))]
    else:
        print("Collecting SP rollouts (this is the slow part)...")
        records = collect_sp_rollouts(n_days=200)
        np.savez(SP_RECORDS_CACHE,
                 t=np.array([r["t"] for r in records], dtype=int),
                 phi=np.array([r["phi"] for r in records]),
                 rtg=np.array([r["rtg"] for r in records]))
        print(f"Cached SP rollouts to {SP_RECORDS_CACHE}")

    print(f"Total records: {len(records)}")
    print("Fitting RidgeCV per stage:")
    etas = fit_ridge_cv(records)
    np.save(SP_SINGLE_ETAS_PATH, etas)
    print(f"Saved SP-distilled etas to {SP_SINGLE_ETAS_PATH}")
    print("Per-stage eta norms:", [round(float(np.linalg.norm(etas[t])), 2)
                                    for t in range(etas.shape[0])])
