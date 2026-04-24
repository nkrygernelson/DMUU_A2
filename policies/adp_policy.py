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


def solve_bellman(state, eta_next, K=5, time_limit=5.0, mip_gap=0.05):
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

    # Continuation term: sum_k p_k * eta_next^T phi(x_{k,t+1})
    continuation = 0.0
    if eta_next is not None and np.any(eta_next != 0):
        for k in m.K:
            Occ1_k = float(centers[k, 0])
            Occ2_k = float(centers[k, 1])
            price_k = float(centers[k, 2])
            feat_expr = (
                eta_next[0] * 1.0
                + eta_next[1] * m.T1c[k]
                + eta_next[2] * m.T2c[k]
                + eta_next[3] * m.Hc[k]
                + eta_next[4] * Occ1_k
                + eta_next[5] * Occ2_k
                + eta_next[6] * price_k
                + eta_next[7] * price_r
                + eta_next[8] * vc_next_expr
                + eta_next[9] * m.z1c[k]
                + eta_next[10] * m.z2c[k]
            )
            continuation = continuation + float(probs[k]) * feat_expr

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
    """Random initial state at t=0, consistent with SystemCharacteristics ranges."""
    return {
        "T1": float(np.random.uniform(19, 23)),
        "T2": float(np.random.uniform(19, 23)),
        "H": float(np.random.uniform(40, 60)),
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


def forward_pass(etas, N=30, K=5, T=NUM_SLOTS):
    """Roll out N trajectories using the current etas. Returns list of trajectories."""
    trajectories = []
    for n in range(N):
        state = sample_init_state()
        traj = [state]
        for t in range(T):
            eta_next = etas[t + 1] if (t + 1) < len(etas) else np.zeros(FEATURE_DIM)
            u, _ = solve_bellman(state, eta_next, K=K)
            exog = sample_exogenous(state)
            state = advance_state(state, u, exog)
            traj.append(state)
        trajectories.append(traj)
        print(f"  forward trajectory {n+1}/{N} done")
    return trajectories


def backward_pass(trajectories, etas, K=5, T=NUM_SLOTS):
    """Refit etas[t] via least squares on targets V*(x_{n,t}) using updated etas[t+1]."""
    new_etas = [np.asarray(e, dtype=float).copy() for e in etas]
    new_etas[T] = np.zeros(FEATURE_DIM)

    for t in range(T - 1, -1, -1):
        X = []
        y = []
        for traj in trajectories:
            state_t = traj[t]
            _, V_star = solve_bellman(state_t, new_etas[t + 1], K=K)
            X.append(features(state_t))
            y.append(V_star)
        X = np.asarray(X)
        y = np.asarray(y)
        eta_t, *_ = np.linalg.lstsq(X, y, rcond=None)
        new_etas[t] = eta_t
        print(f"  backward t={t}: target mean={y.mean():.3f}, std={y.std():.3f}")
    return new_etas


def train(I=15, N=30, K=5, T=NUM_SLOTS, save_path=ETAS_PATH):
    """Run I outer iterations of forward-backward fitted value iteration."""
    etas = [np.zeros(FEATURE_DIM) for _ in range(T + 1)]
    for i in range(I):
        print(f"=== Iteration {i+1}/{I} ===")
        trajectories = forward_pass(etas, N=N, K=K, T=T)
        etas = backward_pass(trajectories, etas, K=K, T=T)
        np.save(save_path, np.array(etas))
        print(f"  saved etas to {save_path}")
    return etas


_CACHED_ETAS = None


def _load_etas():
    global _CACHED_ETAS
    if _CACHED_ETAS is None:
        if os.path.exists(ETAS_PATH):
            _CACHED_ETAS = np.load(ETAS_PATH)
        else:
            print(f"No trained etas at {ETAS_PATH}; using zeros. Run train() first.")
            _CACHED_ETAS = np.zeros((NUM_SLOTS + 1, FEATURE_DIM))
    return _CACHED_ETAS


def select_action(state):
    etas = _load_etas()
    t = int(state["current_time"])
    eta_next = etas[t + 1] if (t + 1) < len(etas) else np.zeros(FEATURE_DIM)
    decision, _ = solve_bellman(state, eta_next, K=5, time_limit=10.0, mip_gap=0.02)
    return decision


if __name__ == "__main__":
    train()
