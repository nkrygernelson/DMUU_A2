"""SP MILP with per-time-step V_θ trained via fitted backwards induction.

Inference structure is identical to `deep_adp_milp_policy` (Attempt 2): the
bf=3, num_stages=2 scenario tree built by `sp_policy`'s tree builder, with a
trained MLP V_θ embedded at the stage-2 leaves via big-M ReLU.

What's new here is the **training procedure** — strict fitted backwards induction:

  1. Roll out the original `sp_policy` for N days to bucket visited states
     by time index, giving {x_t^n : n=1..N} for each t ∈ {0..T-1}.
  2. Set V_θ_T ≡ 0 (terminal).
  3. For t = T-1, T-2, ..., 0:
       a. For every x in the t-th bucket compute the Bellman target
              y(x) = min_u { c(x, u) + (1/M) Σ_m V_θ_{t+1}(f(x, u, ω_m)) }
          by grid search over the constrained action set (overrules respected).
       b. Fit a fresh per-time-step MLP V_θ_t on those (x, y) pairs.

Per-t MLPs are tiny (1 hidden layer × 16 units) and fit with scikit-learn's
L-BFGS solver — that's much better than Adam for small datasets, where Adam's
fixed step size makes the output bias take many epochs to reach y_mean.

Train offline:
    uv run python -m policies.sp_backward_induction_policy
"""

import os

import numpy as np
import torch
import pyomo.environ as pyo
from sklearn.neural_network import MLPRegressor

from policies import sp_policy as _sp
from policies.sp_policy import build_scenario_tree, propagate_uncertainty
from policies.deep_adp_policy import (
    encode_batch,
    sample_init_state, sample_exog, advance,
    feasible_action_grid,
    NUM_SLOTS, P_VENT,
)
from policies.deep_adp_milp_policy import build_and_solve


MODELS_DIR = os.path.join(os.path.dirname(__file__), "sp_backward_induction_models")
HIDDEN = 16  # 1 hidden layer, 16 units → 225 parameters per t


# ---- numpy-only forward pass for V_θ (used in the backward sweep) --------

def vnet_predict(W1, b1, W2, b2, X):
    """Manual forward pass of the per-t MLP: h = ReLU(X W1ᵀ + b1); y = h W2 + b2."""
    h = np.maximum(0.0, X @ W1.T + b1)
    return h @ W2 + b2


def fit_per_t(X, y, hidden=HIDDEN, alpha=1e-2, seed=0):
    """Fit a 1-hidden-layer MLP using L-BFGS. Returns (W1, b1, W2, b2)."""
    reg = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="lbfgs",
        alpha=alpha,
        max_iter=2000,
        tol=1e-6,
        random_state=seed,
    )
    reg.fit(X.astype(np.float64), y.astype(np.float64))
    W1 = reg.coefs_[0].T.astype(np.float64)        # (hidden, FEAT_DIM)
    b1 = reg.intercepts_[0].astype(np.float64)     # (hidden,)
    W2 = reg.coefs_[1].squeeze().astype(np.float64)  # (hidden,)
    b2_arr = reg.intercepts_[1]
    b2 = float(b2_arr.item() if hasattr(b2_arr, "item") else b2_arr)
    return W1, b1, W2, b2


# ---- State collection under the original SP policy -----------------------

def collect_sp_states_by_time(N=100, verbose=False):
    """Roll out sp_policy for N random init states. Returns a list of T lists,
    one per time index, of visited states."""
    states_by_t = [[] for _ in range(NUM_SLOTS)]
    totals = []
    for n in range(N):
        s = sample_init_state()
        traj_cost = 0.0
        for t in range(NUM_SLOTS):
            states_by_t[t].append(s)
            u = _sp.select_action(s)
            traj_cost += s["price_t"] * (
                u["HeatPowerRoom1"] + u["HeatPowerRoom2"] + P_VENT * u["VentilationON"]
            )
            w = sample_exog(s, 1)[0]
            s = advance(
                s,
                (u["HeatPowerRoom1"], u["HeatPowerRoom2"], u["VentilationON"]),
                w,
            )
        totals.append(traj_cost)
        if verbose and (n + 1) % 10 == 0:
            a = np.array(totals)
            print(f"  SP rollout {n + 1}/{N}  running mean={a.mean():.2f}")
    if verbose:
        a = np.array(totals)
        print(f"  collected {N} rollouts  mean={a.mean():.2f}  sd={a.std():.2f}")
    return states_by_t


# ---- Bellman target via grid search + MC ----------------------------------

def bellman_targets(states, V_next_weights, M=20, n_levels=7):
    """For each x in `states`, compute the one-step Bellman target

        y(x) = min_u  c(x, u) + (1/M) Σ_m V_next(f(x, u, ω_m))

    using grid search over the constrained action set. V_next_weights may be
    None (terminal V ≡ 0). Returns a numpy array of length len(states)."""
    targets = np.empty(len(states), dtype=np.float64)
    for i, x in enumerate(states):
        actions = feasible_action_grid(x, n_levels=n_levels)
        if V_next_weights is None:
            best = min(x["price_t"] * (u[0] + u[1] + P_VENT * u[2]) for u in actions)
            targets[i] = best
            continue
        W1, b1, W2, b2 = V_next_weights
        exogs = sample_exog(x, M)
        immediate = np.empty(len(actions), dtype=np.float64)
        next_all = []
        for j, u in enumerate(actions):
            immediate[j] = x["price_t"] * (u[0] + u[1] + P_VENT * u[2])
            for w in exogs:
                next_all.append(advance(x, u, w))
        X = encode_batch(next_all).astype(np.float64)
        yp = vnet_predict(W1, b1, W2, b2, X)
        yp = yp.reshape(len(actions), M).mean(axis=1)
        targets[i] = float((immediate + yp).min())
    return targets


# ---- Backwards-induction training ----------------------------------------

def train_backward_induction(N=100, M=20, hidden=HIDDEN, alpha=1e-2,
                              save_dir=MODELS_DIR, verbose=True, seed=0):
    """Train V_θ_t for t = T-1 down to 0 via fitted backwards induction."""
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print(f"=== rolling out SP for {N} days to gather visited states ===")
    states_by_t = collect_sp_states_by_time(N=N, verbose=verbose)

    weights_per_t = [None] * NUM_SLOTS  # weights_per_t[T] would be None ⇒ V_T ≡ 0
    V_next = None  # V_θ_{T} ≡ 0
    for t in range(NUM_SLOTS - 1, -1, -1):
        states_t = states_by_t[t]
        X = encode_batch(states_t).astype(np.float64)
        y = bellman_targets(states_t, V_next, M=M)

        W1, b1, W2, b2 = fit_per_t(X, y, hidden=hidden, alpha=alpha, seed=seed + t)
        weights_per_t[t] = (W1, b1, W2, b2)
        V_next = weights_per_t[t]

        # Save
        path = os.path.join(save_dir, f"V_t{t}.npz")
        np.savez(path, W1=W1, b1=b1, W2=W2, b2=np.float64(b2))

        if verbose:
            yhat = vnet_predict(W1, b1, W2, b2, X)
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
            r2 = 1.0 - ss_res / ss_tot
            print(f"  V_θ_t{t}: n={len(y)}  y mean={y.mean():.2f}  sd={y.std():.2f}  R²={r2:.3f}  → {path}")
    return weights_per_t


# ---- Online policy --------------------------------------------------------

_CACHED_WEIGHTS = None


def _load_weights():
    """Returns list[NUM_SLOTS] of weight tuples (W1, b1, W2, b2) or None."""
    global _CACHED_WEIGHTS
    if _CACHED_WEIGHTS is None:
        weights = []
        for t in range(NUM_SLOTS):
            path = os.path.join(MODELS_DIR, f"V_t{t}.npz")
            if os.path.exists(path):
                d = np.load(path)
                weights.append((d["W1"], d["b1"], d["W2"], float(d["b2"])))
            else:
                weights.append(None)
        _CACHED_WEIGHTS = weights
    return _CACHED_WEIGHTS


def _dummy_weights():
    """Shape-valid placeholder; used when build_and_solve will skip V_θ anyway."""
    return (np.zeros((HIDDEN, 12)), np.zeros(HIDDEN), np.zeros(HIDDEN), 0.0)


def select_action(state, time_limit=10.0, mip_gap=0.05):
    weights_per_t = _load_weights()
    t = int(state["current_time"])
    remaining = NUM_SLOTS - t
    num_stages = max(1, min(2, remaining))
    bf = 3
    t_leaf = t + num_stages

    root, all_nodes, leaves = build_scenario_tree(bf, num_stages)
    root.state = {
        "current_r1_occ": state["Occ1"],
        "current_r2_occ": state["Occ2"],
        "current_price": state["price_t"],
        "prev_price": state["price_previous"],
    }
    propagate_uncertainty(root, all_nodes, num_samples=150)

    if t_leaf >= NUM_SLOTS or weights_per_t[t_leaf] is None:
        W1, b1, W2, b2 = _dummy_weights()
    else:
        W1, b1, W2, b2 = weights_per_t[t_leaf]

    m, result = build_and_solve(state, root, all_nodes, leaves, W1, b1, W2, b2,
                                time_limit=time_limit, mip_gap=mip_gap)
    rid = root.node_id
    ok = (result.solver.status == pyo.SolverStatus.ok and
          result.solver.termination_condition in (
              pyo.TerminationCondition.optimal,
              pyo.TerminationCondition.feasible))
    if ok:
        return {
            "HeatPowerRoom1": float(pyo.value(m.p1[rid])),
            "HeatPowerRoom2": float(pyo.value(m.p2[rid])),
            "VentilationON": int(round(pyo.value(m.V[rid]))),
        }
    return {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}


if __name__ == "__main__":
    train_backward_induction(N=100, M=20)
