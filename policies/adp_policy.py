"""Approximate Dynamic Programming policy with linear value-function approximation.

Offline: forward-backward fitted value iteration produces one eta vector per stage.
Online: select_action loads eta and solves a one-step Bellman subproblem.

Feature vector phi(x) (length 11):
    [1, T1, T2, H, Occ1, Occ2, price_t, price_previous,
     vent_counter, low_override_r1, low_override_r2]
"""

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import pyomo.environ as pyo

from processes.PriceProcessRestaurant import price_model
from processes.OccupancyProcessRestaurant import next_occupancy_levels
from SystemCharacteristics import get_fixed_data


FEATURE_DIM = 11
NUM_SLOTS = int(get_fixed_data()["num_timeslots"])
ETAS_PATH = os.path.join(os.path.dirname(__file__), "adp_etas.npy")


def features(state):
    """Map a state dict to phi(x) as a length-11 numpy array."""
    return np.array([
        1.0,
        float(state["T1"]),
        float(state["T2"]),
        float(state["H"]),
        float(state["Occ1"]),
        float(state["Occ2"]),
        float(state["price_t"]),
        float(state["price_previous"]),
        float(state["vent_counter"]),
        float(bool(state["low_override_r1"])),
        float(bool(state["low_override_r2"])),
    ], dtype=float)


def gen_samples(current_r1_occ, current_r2_occ, current_price, prev_price, num_samples=100):
    occ_samples = []
    price_samples = []
    for _ in range(num_samples):
        occ_samples.append(next_occupancy_levels(r1_current=current_r1_occ, r2_current=current_r2_occ))
        price_samples.append(price_model(current_price=current_price, previous_price=prev_price))
    price_samples = np.array(price_samples)[:, None]
    occ_samples = np.array(occ_samples)
    return np.hstack([occ_samples, price_samples])


def gen_scenarios(state, K, num_samples=150):
    """Return K cluster centers (Occ1, Occ2, price) and their probabilities."""
    joint = gen_samples(
        current_r1_occ=state["Occ1"],
        current_r2_occ=state["Occ2"],
        current_price=state["price_t"],
        prev_price=state["price_previous"],
        num_samples=num_samples,
    )
    km = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(joint)
    counts = np.bincount(km.labels_, minlength=K)
    probs = counts / counts.sum()
    return km.cluster_centers_, probs


def solve_bellman(state, eta_next, K=10, time_limit=5.0, mip_gap=0.05):
    """Solve the one-step Bellman problem: min cost_t + E[eta_next^T phi(x_{t+1})].

    Returns (decision_dict, V_star).
    """
    fixed_data = get_fixed_data()

    xi_exh = fixed_data["heat_exchange_coeff"]
    xi_loss = fixed_data["thermal_loss_coeff"]
    xi_conv = fixed_data["heating_efficiency_coeff"]
    xi_cool = fixed_data["heat_vent_coeff"]
    xi_occ = fixed_data["heat_occupancy_coeff"]
    eta_occ = fixed_data["humidity_occupancy_coeff"]
    eta_vent = fixed_data["humidity_vent_coeff"]

    p_vent = fixed_data["ventilation_power"]
    T_low = fixed_data["temp_min_comfort_threshold"]
    T_high = fixed_data["temp_max_comfort_threshold"]
    T_ok = fixed_data["temp_OK_threshold"]
    T_out = fixed_data["outdoor_temperature"]
    H_high = fixed_data["humidity_threshold"]
    P_overline = fixed_data["heating_max_power"]
    T_circ = -3
    M_low = T_low - T_circ
    M_high = T_ok - T_circ
    M_hum = 100 - H_high

    # Root state (constants from the state dict)
    t_root = int(state["current_time"])
    T1_r = float(state["T1"])
    T2_r = float(state["T2"])
    H_r = float(state["H"])
    Occ1_r = float(state["Occ1"])
    Occ2_r = float(state["Occ2"])
    price_r = float(state["price_t"])
    z1c_r = int(bool(state["low_override_r1"]))
    z2c_r = int(bool(state["low_override_r2"]))
    vc_r = int(state["vent_counter"])
    T_out_r = T_out[min(t_root, len(T_out) - 1)]

    # Exogenous scenarios for t+1
    centers, probs = gen_scenarios(state, K)

    m = pyo.ConcreteModel()
    m.K = pyo.RangeSet(0, K - 1)

    # Root decisions
    m.p1 = pyo.Var(bounds=(0, P_overline))
    m.p2 = pyo.Var(bounds=(0, P_overline))
    m.V = pyo.Var(within=pyo.Binary)
    m.ON = pyo.Var(within=pyo.Binary)
    m.OFF = pyo.Var(within=pyo.Binary)
    m.z1_hot = pyo.Var(within=pyo.Binary)
    m.z2_hot = pyo.Var(within=pyo.Binary)

    m.cons = pyo.ConstraintList()

    # High-temperature overrule at root
    m.cons.add(T1_r - T_high <= M_high * m.z1_hot)
    m.cons.add(m.p1 <= P_overline * (1 - m.z1_hot))
    m.cons.add(T2_r - T_high <= M_high * m.z2_hot)
    m.cons.add(m.p2 <= P_overline * (1 - m.z2_hot))

    # Low-temperature overrule at root (z_cold is a known constant from state)
    if z1c_r == 1:
        m.cons.add(m.p1 >= P_overline)
    if z2c_r == 1:
        m.cons.add(m.p2 >= P_overline)

    # Humidity overrule at root
    m.cons.add(H_r - H_high <= M_hum * m.V)

    # Ventilation inertia at root (same logic as sp_policy)
    if 0 < vc_r < 3:
        m.V.fix(1)
        m.ON.fix(0)
        m.OFF.fix(0)
    elif vc_r == 0:
        m.cons.add(m.V == m.ON)
        m.OFF.fix(0)
    else:
        m.ON.fix(0)
        m.cons.add(m.V == 1 - m.OFF)
    m.cons.add(m.ON + m.OFF <= 1)

    # Energy cost at root
    root_cost = price_r * (m.p1 + m.p2 + p_vent * m.V)

    # Child state vars (one per scenario)
    m.T1c = pyo.Var(m.K, bounds=(T_circ, 2 * T_high))
    m.T2c = pyo.Var(m.K, bounds=(T_circ, 2 * T_high))
    m.Hc = pyo.Var(m.K, bounds=(0, 100))
    m.z1c = pyo.Var(m.K, within=pyo.Binary)
    m.z2c = pyo.Var(m.K, within=pyo.Binary)

    for k in m.K:
        m.cons.add(
            m.T1c[k] == T1_r
            - xi_exh * (T1_r - T2_r)
            - xi_loss * (T1_r - T_out_r)
            + xi_conv * m.p1
            - xi_cool * m.V
            + xi_occ * Occ1_r
        )
        m.cons.add(
            m.T2c[k] == T2_r
            - xi_exh * (T2_r - T1_r)
            - xi_loss * (T2_r - T_out_r)
            + xi_conv * m.p2
            - xi_cool * m.V
            + xi_occ * Occ2_r
        )
        m.cons.add(
            m.Hc[k] == H_r
            + eta_occ * (Occ1_r + Occ2_r)
            - eta_vent * m.V
        )

        # Low-override transitions at child
        m.cons.add(T_low - m.T1c[k] <= M_low * m.z1c[k])
        m.cons.add(T_low - m.T2c[k] <= M_low * m.z2c[k])
        m.cons.add(T_ok - m.T1c[k] <= M_high * (1 - z1c_r + m.z1c[k]))
        m.cons.add(T_ok - m.T2c[k] <= M_high * (1 - z2c_r + m.z2c[k]))

    # vent_counter at t+1 as a linear expression in root decisions
    if vc_r == 0:
        vc_next_expr = m.ON
    elif vc_r < 3:
        vc_next_expr = vc_r + 1
    else:
        vc_next_expr = (vc_r + 1) * m.V

    # Continuation term: piecewise linear in (z1c[k], z2c[k]).
    # eta_next is shape (4, FEATURE_DIM), indexed by region = 2*z1c + z2c.
    # For each scenario we introduce region-indicator binaries delta_{k,r} and a
    # single per-scenario continuation var c_k = sum_r delta_{k,r} * f_{k,r}.
    # We further clip the per-scenario continuation at 0 since cost-to-go is
    # non-negative — without this clip the linear approximator extrapolates to
    # large negative values outside the training distribution and the solver
    # exploits it, driving V* targets negative and breaking convergence.
    continuation = 0.0
    if eta_next is not None and np.any(np.asarray(eta_next) != 0):
        eta_next = np.asarray(eta_next, dtype=float)
        assert eta_next.shape == (4, FEATURE_DIM), \
            f"expected eta_next shape (4, {FEATURE_DIM}), got {eta_next.shape}"

        REGIONS = [(0, 0), (0, 1), (1, 0), (1, 1)]
        M_big = 1.0e5  # big-M for product linearization

        m.R = pyo.RangeSet(0, 3)
        m.delta = pyo.Var(m.K, m.R, within=pyo.Binary)
        m.c = pyo.Var(m.K, m.R)  # c_{k,r} = delta_{k,r} * f_{k,r}
        m.V_next = pyo.Var(m.K, within=pyo.NonNegativeReals)  # max(0, sum_r c_{k,r})

        for k in m.K:
            Occ1_k = float(centers[k, 0])
            Occ2_k = float(centers[k, 1])
            price_k = float(centers[k, 2])

            # Region-indicator logic: delta_{k,r} = 1 iff (z1c[k], z2c[k]) == r.
            # Use standard AND-of-binaries-or-their-complements linearization.
            for r_idx, (a, b) in enumerate(REGIONS):
                z1_lit = m.z1c[k] if a == 1 else (1 - m.z1c[k])
                z2_lit = m.z2c[k] if b == 1 else (1 - m.z2c[k])
                m.cons.add(m.delta[k, r_idx] <= z1_lit)
                m.cons.add(m.delta[k, r_idx] <= z2_lit)
                m.cons.add(m.delta[k, r_idx] >= z1_lit + z2_lit - 1)
            m.cons.add(sum(m.delta[k, r_idx] for r_idx in range(4)) == 1)

            # Per-region affine value f_{k,r} = eta_r^T phi(x_{k,t+1}).
            for r_idx, (a, b) in enumerate(REGIONS):
                er = eta_next[r_idx]
                f_kr = (
                    er[0] * 1.0
                    + er[1] * m.T1c[k]
                    + er[2] * m.T2c[k]
                    + er[3] * m.Hc[k]
                    + er[4] * Occ1_k
                    + er[5] * Occ2_k
                    + er[6] * price_k
                    + er[7] * price_r
                    + er[8] * vc_next_expr
                    + er[9] * a  # z1c fixed inside this region
                    + er[10] * b
                )
                # c_{k,r} = delta * f_kr via big-M product linearization.
                d = m.delta[k, r_idx]
                m.cons.add(m.c[k, r_idx] <= M_big * d)
                m.cons.add(m.c[k, r_idx] >= -M_big * d)
                m.cons.add(m.c[k, r_idx] <= f_kr + M_big * (1 - d))
                m.cons.add(m.c[k, r_idx] >= f_kr - M_big * (1 - d))

            # V_next[k] = max(0, sum_r c[k,r]); we minimize, so the lower bounds
            # implied by V_next >= 0 (declared on the var) and V_next >= sum c
            # are active and tight.
            m.cons.add(m.V_next[k] >= sum(m.c[k, r_idx] for r_idx in range(4)))
            continuation = continuation + float(probs[k]) * m.V_next[k]

    m.objective = pyo.Objective(expr=root_cost + continuation, sense=pyo.minimize)

    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = mip_gap
    solver.options["OutputFlag"] = 0
    result = solver.solve(m, tee=False)

    ok = (result.solver.status == pyo.SolverStatus.ok and
          result.solver.termination_condition in (
              pyo.TerminationCondition.optimal,
              pyo.TerminationCondition.feasible))
    if ok:
        p1_val = float(pyo.value(m.p1))
        p2_val = float(pyo.value(m.p2))
        v_val = int(round(pyo.value(m.V)))
        V_star = float(pyo.value(m.objective))
    else:
        p1_val, p2_val, v_val = 0.0, 0.0, 0
        V_star = 0.0

    decision = {
        "HeatPowerRoom1": p1_val,
        "HeatPowerRoom2": p2_val,
        "VentilationON": v_val,
    }
    return decision, V_star


def advance_state(state, decision, exog):
    """Apply system dynamics to produce the next state dict.

    Must mirror the Pyomo constraints in sp_policy exactly.
    """
    fixed_data = get_fixed_data()
    xi_exh = fixed_data["heat_exchange_coeff"]
    xi_loss = fixed_data["thermal_loss_coeff"]
    xi_conv = fixed_data["heating_efficiency_coeff"]
    xi_cool = fixed_data["heat_vent_coeff"]
    xi_occ = fixed_data["heat_occupancy_coeff"]
    eta_occ = fixed_data["humidity_occupancy_coeff"]
    eta_vent = fixed_data["humidity_vent_coeff"]
    T_low = fixed_data["temp_min_comfort_threshold"]
    T_ok = fixed_data["temp_OK_threshold"]
    T_out = fixed_data["outdoor_temperature"]

    t = int(state["current_time"])
    T_out_t = T_out[min(t, len(T_out) - 1)]

    T1 = state["T1"]; T2 = state["T2"]; H = state["H"]
    Occ1 = state["Occ1"]; Occ2 = state["Occ2"]
    p1 = decision["HeatPowerRoom1"]
    p2 = decision["HeatPowerRoom2"]
    V = decision["VentilationON"]

    T1_new = T1 - xi_exh*(T1-T2) - xi_loss*(T1-T_out_t) + xi_conv*p1 - xi_cool*V + xi_occ*Occ1
    T2_new = T2 - xi_exh*(T2-T1) - xi_loss*(T2-T_out_t) + xi_conv*p2 - xi_cool*V + xi_occ*Occ2
    H_new = H + eta_occ*(Occ1+Occ2) - eta_vent*V
    H_new = float(np.clip(H_new, 0, 100))

    vc = int(state["vent_counter"])
    if V == 1:
        vc_new = vc + 1 if vc > 0 else 1
    else:
        vc_new = 0

    def update_override(old, new_temp):
        if new_temp < T_low:
            return 1
        if old == 1 and new_temp < T_ok:
            return 1
        return 0

    lo1_new = update_override(int(bool(state["low_override_r1"])), T1_new)
    lo2_new = update_override(int(bool(state["low_override_r2"])), T2_new)

    return {
        "T1": float(T1_new),
        "T2": float(T2_new),
        "H": float(H_new),
        "Occ1": float(exog["Occ1"]),
        "Occ2": float(exog["Occ2"]),
        "price_t": float(exog["price"]),
        "price_previous": float(state["price_t"]),
        "vent_counter": vc_new,
        "low_override_r1": lo1_new,
        "low_override_r2": lo2_new,
        "current_time": t + 1,
    }


def sample_init_state():
    """Random initial state at t=0, widened to span SP's full operating range.

    The simulator resets every day to T1=T2=20, H=50, vc=lo1=lo2=0; we
    deliberately widen this here so the ridge regression at t=0 sees input
    variance (otherwise it collapses to slope ≈ 0 — see
    pdfs/ridge_diagnostic_log.md, Experiment A).
    """
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


def sample_exogenous(state):
    r1_next, r2_next = next_occupancy_levels(state["Occ1"], state["Occ2"])
    price_next = price_model(state["price_t"], state["price_previous"])
    return {"Occ1": r1_next, "Occ2": r2_next, "price": price_next}


def _region_idx(state):
    """Map state's (low_override_r1, low_override_r2) to region index 0..3."""
    a = int(bool(state["low_override_r1"]))
    b = int(bool(state["low_override_r2"]))
    return 2 * a + b


def _zero_etas(T):
    """Shape (T+1, 4, FEATURE_DIM) — one eta per stage per region."""
    return np.zeros((T + 1, 4, FEATURE_DIM))


def forward_pass(etas, N=30, K=10, T=NUM_SLOTS, time_limit=10.0, mip_gap=0.01):
    """Roll out N trajectories using the current etas. Returns list of trajectories."""
    trajectories = []
    for n in range(N):
        state = sample_init_state()
        traj = [state]
        for t in range(T):
            eta_next = etas[t + 1] if (t + 1) < len(etas) else np.zeros((4, FEATURE_DIM))
            u, _ = solve_bellman(state, eta_next, K=K, time_limit=time_limit, mip_gap=mip_gap)
            exog = sample_exogenous(state)
            state = advance_state(state, u, exog)
            traj.append(state)
        trajectories.append(traj)
        print(f"  forward trajectory {n+1}/{N} done")
    return trajectories


def backward_pass(trajectories, etas, K=10, T=NUM_SLOTS, ridge_alpha=1.0,
                  time_limit=10.0, mip_gap=0.01):
    """Refit etas[t, r] via Ridge on targets V*(x_{n,t}), grouped by root region r.

    Returns (new_etas, diagnostics) where diagnostics is a list of dicts, one per
    (t, r) bucket with at least one sample, containing:
      n           sample count
      mse         in-sample Ridge MSE  ((Xr@coef - yr)**2).mean()
      rel_mse     mse / (mean(|yr|)**2 + eps) — scale-free fit quality
      eta_delta   ||new_eta[t,r] - old_eta[t,r]||_2
      y_mean,y_std,y_min,y_max  target stats
    """
    old_etas = np.asarray(etas, dtype=float).copy()
    new_etas = old_etas.copy()
    new_etas[T] = 0.0

    diagnostics = []

    for t in range(T - 1, -1, -1):
        # Collect (region, phi, V*) per root state at stage t.
        buckets = {r: ([], []) for r in range(4)}
        for traj in trajectories:
            state_t = traj[t]
            _, V_star = solve_bellman(state_t, new_etas[t + 1], K=K,
                                      time_limit=time_limit, mip_gap=mip_gap)
            r = _region_idx(state_t)
            buckets[r][0].append(features(state_t))
            buckets[r][1].append(V_star)

        for r in range(4):
            Xs, ys = buckets[r]
            if len(ys) == 0:
                # No samples in this region at this stage — carry over previous fit.
                continue
            Xr = np.asarray(Xs)
            yr = np.asarray(ys)
            ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
            ridge.fit(Xr, yr)
            new_etas[t, r] = ridge.coef_

            preds = Xr @ ridge.coef_
            mse = float(((preds - yr) ** 2).mean())
            scale = float(np.mean(np.abs(yr))) ** 2 + 1e-9
            eta_delta = float(np.linalg.norm(new_etas[t, r] - old_etas[t, r]))
            diagnostics.append({
                "t": t, "r": r, "n": int(len(yr)),
                "mse": mse, "rel_mse": mse / scale,
                "eta_delta": eta_delta,
                "y_mean": float(yr.mean()), "y_std": float(yr.std()),
                "y_min": float(yr.min()), "y_max": float(yr.max()),
            })
            print(
                f"  backward t={t} r={r} n={len(yr):3d} "
                f"y_mean={yr.mean():8.2f} y_std={yr.std():7.2f} "
                f"mse={mse:9.3f} rel_mse={mse/scale:7.4f} "
                f"|Δeta|={eta_delta:8.3f}"
            )

        counts = [len(buckets[r][1]) for r in range(4)]
        print(f"  backward t={t}: region sample counts {counts}")
    return new_etas, diagnostics


def train(I=15, N=30, K=10, T=NUM_SLOTS, ridge_alpha=1.0, save_path=ETAS_PATH):
    """Run I outer iterations of forward-backward fitted value iteration."""
    etas = _zero_etas(T)
    history = []  # one diagnostics list per iteration
    for i in range(I):
        print(f"=== Iteration {i+1}/{I} ===")
        trajectories = forward_pass(etas, N=N, K=K, T=T)
        etas, diag = backward_pass(trajectories, etas, K=K, T=T, ridge_alpha=ridge_alpha)
        history.append(diag)
        np.save(save_path, etas)
        print(f"  saved etas to {save_path}")
        # Per-iteration summary across all (t, r) buckets that had samples
        if diag:
            mean_rel = float(np.mean([d["rel_mse"] for d in diag]))
            mean_delta = float(np.mean([d["eta_delta"] for d in diag]))
            print(f"  iter summary: mean rel_mse={mean_rel:.4f}  mean |Δeta|={mean_delta:.3f}")
    return etas, history


_CACHED_ETAS = None


def _load_etas():
    global _CACHED_ETAS
    if _CACHED_ETAS is None:
        if os.path.exists(ETAS_PATH):
            arr = np.load(ETAS_PATH)
            if arr.ndim == 2:  # legacy single-piece fit
                print("Legacy etas shape detected; broadcasting to 4 regions.")
                arr = np.broadcast_to(arr[:, None, :], (arr.shape[0], 4, arr.shape[1])).copy()
            _CACHED_ETAS = arr
        else:
            print(f"No trained etas at {ETAS_PATH}; using zeros. Run train() first.")
            _CACHED_ETAS = _zero_etas(NUM_SLOTS)
    return _CACHED_ETAS


def select_action(state):
    etas = _load_etas()
    t = int(state["current_time"])
    eta_next = etas[t + 1] if (t + 1) < len(etas) else np.zeros((4, FEATURE_DIM))
    decision, _ = solve_bellman(state, eta_next, K=10, time_limit=10.0, mip_gap=0.02)
    return decision


if __name__ == "__main__":
    train()
