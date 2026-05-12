"""MILP rollout policy with on-policy Monte-Carlo retraining of V_θ.

Same MILP structure as `deep_adp_milp_policy` (2-stage tree, V_θ embedded at
leaves via big-M ReLU). The difference is purely in how V_θ is trained:

  1. Bootstrap V_θ from the v1 model (trained via fitted VI under grid search).
  2. Repeatedly:
       a. Roll out the MILP-rollout policy under the current V_θ for N days.
       b. Compute Monte Carlo return-to-go at every visited (state, t).
       c. Refit V_θ on (state, MC return) — warm-started from the current model.

The training data thus comes from the same policy that uses V_θ at inference,
and the targets are unbiased estimates of cost-to-go (no Bellman bootstrap).

Train offline:
    uv run python -m policies.deep_adp_milp_v2_policy
"""

import os

import numpy as np
import torch
import pyomo.environ as pyo

from policies.sp_policy import build_scenario_tree, propagate_uncertainty
from policies.deep_adp_policy import (
    encode_batch, load_vnet, fit_V,
    sample_init_state, sample_exog, advance,
    NUM_SLOTS, P_VENT,
)
from policies.deep_adp_milp_policy import (
    extract_small_weights, build_and_solve,
)


V2_MODEL_PATH = os.path.join(os.path.dirname(__file__), "deep_adp_model_small_v2.pt")
INIT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "deep_adp_model_small.pt")


def _select_action_with_weights(state, weights, time_limit=10.0, mip_gap=0.05):
    W1, b1, W2, b2 = weights
    current_time = int(state["current_time"])
    remaining = NUM_SLOTS - current_time
    num_stages = max(1, min(2, remaining))
    bf = 3

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
        return (
            float(pyo.value(m.p1[rid])),
            float(pyo.value(m.p2[rid])),
            int(round(pyo.value(m.V[rid]))),
        )
    return (0.0, 0.0, 0)


def collect_rollouts(V_theta, N=25, verbose=False):
    """Run N MILP-rollout trajectories under the frozen V_theta. Returns
    (states, returns) parallel lists; returns[i] is the Monte Carlo cost-to-go
    from state[i] under the same policy."""
    weights = extract_small_weights(V_theta)
    states_all, returns_all = [], []
    total_costs = []
    for n in range(N):
        s = sample_init_state()
        traj_states = []
        traj_costs = []
        for _ in range(NUM_SLOTS):
            u = _select_action_with_weights(s, weights)
            cost = s["price_t"] * (u[0] + u[1] + P_VENT * u[2])
            traj_states.append(s)
            traj_costs.append(cost)
            w = sample_exog(s, 1)[0]
            s = advance(s, u, w)
        # Return-to-go (running tail sum)
        traj_returns = [0.0] * NUM_SLOTS
        cum = 0.0
        for t in range(NUM_SLOTS - 1, -1, -1):
            cum += traj_costs[t]
            traj_returns[t] = cum
        states_all.extend(traj_states)
        returns_all.extend(traj_returns)
        total_costs.append(sum(traj_costs))
        if verbose:
            print(f"    rollout {n+1}/{N}  total cost = {total_costs[-1]:.2f}")
    if verbose:
        a = np.array(total_costs)
        print(f"  collected {N} rollouts  mean={a.mean():.2f}  sd={a.std():.2f}  min={a.min():.2f}  max={a.max():.2f}")
    return states_all, returns_all


def train_on_policy_mc(K_outer=6, N=25, save_path=V2_MODEL_PATH,
                       init_path=INIT_MODEL_PATH, hidden_sizes=(32,),
                       lr=5e-4, epochs=400, seed=0, verbose=True):
    """Policy iteration: roll out MILP-rollout policy under current V_θ,
    refit V_θ on MC returns. On-policy each iter (no replay buffer)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    V_theta = load_vnet(init_path, hidden_sizes=hidden_sizes)
    for k in range(K_outer):
        if verbose:
            print(f"=== outer iter {k + 1}/{K_outer} ===")
        states, returns = collect_rollouts(V_theta, N=N, verbose=verbose)
        X = encode_batch(states)
        y = np.asarray(returns, dtype=np.float32)
        V_theta = fit_V(X, y, hidden_sizes=hidden_sizes, prior=V_theta,
                        lr=lr, epochs=epochs, verbose=verbose)
        if verbose:
            yhat = V_theta(torch.from_numpy(X)).detach().numpy()
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
            print(f"  fit: n={len(y)}  y mean={y.mean():.2f}  sd={y.std():.2f}  R²={1 - ss_res/ss_tot:.3f}")
        torch.save(V_theta.state_dict(), save_path)
        if verbose:
            print(f"  saved V_θ → {save_path}")
    return V_theta


_CACHED_WEIGHTS = None


def _load_weights():
    global _CACHED_WEIGHTS
    if _CACHED_WEIGHTS is None:
        path = V2_MODEL_PATH if os.path.exists(V2_MODEL_PATH) else INIT_MODEL_PATH
        if path == INIT_MODEL_PATH:
            print(f"No v2 model at {V2_MODEL_PATH}; falling back to v1 weights.")
        model = load_vnet(path, hidden_sizes=(32,))
        _CACHED_WEIGHTS = extract_small_weights(model)
    return _CACHED_WEIGHTS


def select_action(state):
    weights = _load_weights()
    u = _select_action_with_weights(state, weights)
    return {
        "HeatPowerRoom1": u[0],
        "HeatPowerRoom2": u[1],
        "VentilationON": u[2],
    }


if __name__ == "__main__":
    train_on_policy_mc(K_outer=6, N=25, epochs=400)
