"""MILP rollout v3 — 3-stage tree with the v1 fitted-VI V_θ.

Same big-M ReLU + V_θ-at-leaves structure as `deep_adp_milp_policy`, but with
`num_stages=3` so the MILP plans three full stages of decisions exactly (just
like `sp_policy`) before deferring to V_θ at the leaves. `bf=2` (rather than
SP's `bf=3`) keeps the MIP within the size-limited Gurobi license.

V_θ is loaded from the v1 training run (`deep_adp_model_small.pt`). The
SP-distillation experiment (training V_θ on Monte Carlo return-to-go from
SP rollouts) is preserved as `train_distill_attempt` for the writeup — it
underperformed v1's fitted-VI targets because the MC variance dominated.
"""

import os

import numpy as np
import torch
import pyomo.environ as pyo

from policies import sp_policy as _sp
from policies.sp_policy import build_scenario_tree, propagate_uncertainty
from policies.deep_adp_policy import (
    encode_batch, fit_V, load_vnet,
    sample_init_state, sample_exog, advance,
    NUM_SLOTS, P_VENT,
)
from policies.deep_adp_milp_policy import (
    extract_small_weights, build_and_solve,
)


V1_MODEL_PATH = os.path.join(os.path.dirname(__file__), "deep_adp_model_small.pt")
V3_DISTILL_PATH = os.path.join(os.path.dirname(__file__), "deep_adp_model_small_v3.pt")


# ---- Online policy: 3-stage MILP rollout with v1 fitted-VI V_θ -----------

_CACHED_WEIGHTS = None


def _load_weights():
    global _CACHED_WEIGHTS
    if _CACHED_WEIGHTS is None:
        model = load_vnet(V1_MODEL_PATH, hidden_sizes=(32,))
        _CACHED_WEIGHTS = extract_small_weights(model)
    return _CACHED_WEIGHTS


def select_action(state, time_limit=12.0, mip_gap=0.05):
    W1, b1, W2, b2 = _load_weights()
    current_time = int(state["current_time"])
    remaining = NUM_SLOTS - current_time
    num_stages = max(1, min(3, remaining))  # deeper tactical lookahead than v1
    bf = 2  # keep MIP under the Gurobi size-limited license

    root, all_nodes, leaves = build_scenario_tree(bf, num_stages)
    root.state = {
        "current_r1_occ": state["Occ1"],
        "current_r2_occ": state["Occ2"],
        "current_price": state["price_t"],
        "prev_price": state["price_previous"],
    }
    propagate_uncertainty(root, all_nodes, num_samples=150)

    m, result = build_and_solve(state, root, all_nodes, leaves, W1, b1, W2, b2,
                                time_limit=time_limit, mip_gap=mip_gap)
    rid = root.node_id
    ok = (result.solver.status == pyo.SolverStatus.ok and
          result.solver.termination_condition in (
              pyo.TerminationCondition.optimal,
              pyo.TerminationCondition.feasible))
    if ok:
        p1 = float(pyo.value(m.p1[rid]))
        p2 = float(pyo.value(m.p2[rid]))
        v = int(round(pyo.value(m.V[rid])))
    else:
        p1 = p2 = 0.0
        v = 0
    return {
        "HeatPowerRoom1": p1,
        "HeatPowerRoom2": p2,
        "VentilationON": v,
    }


# ---- SP distillation (kept for the writeup; did not improve V_θ) ---------

def collect_sp_rollouts(N=100, verbose=False):
    """Roll out sp_policy for N random init states. Returns parallel lists
    (states, MC return-to-go)."""
    states_all, returns_all = [], []
    totals = []
    for n in range(N):
        s = sample_init_state()
        traj_states = []
        traj_costs = []
        for _ in range(NUM_SLOTS):
            u = _sp.select_action(s)
            cost = s["price_t"] * (
                u["HeatPowerRoom1"] + u["HeatPowerRoom2"] + P_VENT * u["VentilationON"]
            )
            traj_states.append(s)
            traj_costs.append(cost)
            w = sample_exog(s, 1)[0]
            s = advance(
                s,
                (u["HeatPowerRoom1"], u["HeatPowerRoom2"], u["VentilationON"]),
                w,
            )
        traj_returns = [0.0] * NUM_SLOTS
        cum = 0.0
        for t in range(NUM_SLOTS - 1, -1, -1):
            cum += traj_costs[t]
            traj_returns[t] = cum
        states_all.extend(traj_states)
        returns_all.extend(traj_returns)
        totals.append(sum(traj_costs))
        if verbose and (n + 1) % 5 == 0:
            a = np.array(totals)
            print(f"  SP rollouts {n + 1}/{N}  mean={a.mean():.2f}  sd={a.std():.2f}")
    return states_all, returns_all


def train_distill_attempt(N=100, save_path=V3_DISTILL_PATH, hidden_sizes=(32,),
                          lr=1e-3, epochs=800, verbose=True, seed=0):
    """Distill SP into V_θ via MC cost-to-go regression. Kept for the writeup.

    Empirically this V_θ underperformed v1's fitted-VI V_θ inside the same MILP
    rollout (R²≈0.78 vs 0.92; eval cost 161 vs 143). The select_action above
    uses the v1 model instead.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if verbose:
        print(f"=== collecting {N} SP rollouts ===")
    states, returns = collect_sp_rollouts(N=N, verbose=verbose)
    X = encode_batch(states)
    y = np.asarray(returns, dtype=np.float32)
    V_theta = fit_V(X, y, hidden_sizes=hidden_sizes, prior=None,
                    lr=lr, epochs=epochs, verbose=verbose)
    yhat = V_theta(torch.from_numpy(X)).detach().numpy()
    r2 = 1.0 - float(((y - yhat) ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1.0)
    if verbose:
        print(f"=== fit: n={len(y)}  y mean={y.mean():.2f}  sd={y.std():.2f}  R²={r2:.3f}")
    torch.save(V_theta.state_dict(), save_path)
    return V_theta


if __name__ == "__main__":
    # Reproduce the underwhelming distillation run for the record:
    # train_distill_attempt(N=100, epochs=800)
    print("v3 select_action uses the v1 fitted-VI V_θ inside a 3-stage tree.")
    print("Run train_distill_attempt() to reproduce the SP-distillation experiment.")
