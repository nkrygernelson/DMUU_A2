"""Approximate Dynamic Programming policy — single linear VFA.

Architecture:
  - Single value-function approximator per stage (no piecewise / region split).
  - Original 11-feature phi:
        [1, T1, T2, H, Occ1, Occ2, price_t, price_previous,
         vent_counter, low_override_r1, low_override_r2]
  - Forward-backward fitted value iteration with unregularised least squares.
  - Polyak averaging (tau=0.5) on the eta updates: each iteration's etas are a
    50/50 blend of the previous iterate and the freshly-fitted Ridge/lstsq
    solution. Damps iteration-to-iteration noise.
  - Continuation in the Bellman MILP is clipped at zero (V_next >= 0) so the
    solver cannot exploit negative off-sample extrapolations of the linear VFA.

Etas file format: shape (T+1, FEATURE_DIM) = (11, 11), persisted to
policies/adp_etas.npy after every outer iteration.

Inference: select_action(state) loads the cached etas and solves a one-step
Bellman subproblem (Gurobi MILP, K=10 scenario clusters).
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
    occ_samples, price_samples = [], []
    for _ in range(num_samples):
        occ_samples.append(next_occupancy_levels(r1_current=current_r1_occ, r2_current=current_r2_occ))
        price_samples.append(price_model(current_price=current_price, previous_price=prev_price))
    price_samples = np.array(price_samples)[:, None]
    occ_samples = np.array(occ_samples)
    return np.hstack([occ_samples, price_samples])


def gen_scenarios(state, K, num_samples=150):
    """Return K cluster centers (Occ1, Occ2, price) and their probabilities."""
    joint = gen_samples(
        current_r1_occ=state["Occ1"], current_r2_occ=state["Occ2"],
        current_price=state["price_t"], prev_price=state["price_previous"],
        num_samples=num_samples,
    )
    km = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(joint)
    counts = np.bincount(km.labels_, minlength=K)
    probs = counts / counts.sum()
    return km.cluster_centers_, probs


def solve_bellman(state, eta_next, K=10, time_limit=5.0, mip_gap=0.05):
    """Solve the one-step Bellman problem: min cost_t + E[max(0, eta_next^T phi(x_{t+1}))].

    eta_next has shape (FEATURE_DIM,).
    Returns (decision_dict, V_star).
    """
    fd = get_fixed_data()
    xi_exh = fd["heat_exchange_coeff"]; xi_loss = fd["thermal_loss_coeff"]
    xi_conv = fd["heating_efficiency_coeff"]; xi_cool = fd["heat_vent_coeff"]
    xi_occ = fd["heat_occupancy_coeff"]
    eta_occ = fd["humidity_occupancy_coeff"]; eta_vent = fd["humidity_vent_coeff"]
    p_vent = fd["ventilation_power"]
    T_low = fd["temp_min_comfort_threshold"]
    T_high = fd["temp_max_comfort_threshold"]
    T_ok = fd["temp_OK_threshold"]
    T_out = fd["outdoor_temperature"]
    H_high = fd["humidity_threshold"]
    P_overline = fd["heating_max_power"]
    T_circ = -3
    M_low = T_low - T_circ
    M_high = T_ok - T_circ
    M_hum = 100 - H_high

    t_root = int(state["current_time"])
    T1_r = float(state["T1"]); T2_r = float(state["T2"]); H_r = float(state["H"])
    Occ1_r = float(state["Occ1"]); Occ2_r = float(state["Occ2"])
    price_r = float(state["price_t"])
    z1c_r = int(bool(state["low_override_r1"]))
    z2c_r = int(bool(state["low_override_r2"]))
    vc_r = int(state["vent_counter"])
    T_out_r = T_out[min(t_root, len(T_out) - 1)]

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

    # Low-temperature overrule at root
    if z1c_r == 1: m.cons.add(m.p1 >= P_overline)
    if z2c_r == 1: m.cons.add(m.p2 >= P_overline)

    # Humidity overrule at root
    m.cons.add(H_r - H_high <= M_hum * m.V)

    # Ventilation inertia at root
    if 0 < vc_r < 3:
        m.V.fix(1); m.ON.fix(0); m.OFF.fix(0)
    elif vc_r == 0:
        m.cons.add(m.V == m.ON); m.OFF.fix(0)
    else:
        m.ON.fix(0); m.cons.add(m.V == 1 - m.OFF)
    m.cons.add(m.ON + m.OFF <= 1)

    root_cost = price_r * (m.p1 + m.p2 + p_vent * m.V)

    # Child state vars (one per scenario)
    m.T1c = pyo.Var(m.K, bounds=(T_circ, 2 * T_high))
    m.T2c = pyo.Var(m.K, bounds=(T_circ, 2 * T_high))
    m.Hc = pyo.Var(m.K, bounds=(0, 100))
    m.z1c = pyo.Var(m.K, within=pyo.Binary)
    m.z2c = pyo.Var(m.K, within=pyo.Binary)

    for k in m.K:
        m.cons.add(m.T1c[k] == T1_r - xi_exh*(T1_r-T2_r) - xi_loss*(T1_r-T_out_r)
                   + xi_conv*m.p1 - xi_cool*m.V + xi_occ*Occ1_r)
        m.cons.add(m.T2c[k] == T2_r - xi_exh*(T2_r-T1_r) - xi_loss*(T2_r-T_out_r)
                   + xi_conv*m.p2 - xi_cool*m.V + xi_occ*Occ2_r)
        m.cons.add(m.Hc[k] == H_r + eta_occ*(Occ1_r+Occ2_r) - eta_vent*m.V)
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

    # Single-VFA continuation, clipped at 0 to block negative off-sample extrapolation.
    continuation = 0.0
    if eta_next is not None and np.any(np.asarray(eta_next) != 0):
        e = np.asarray(eta_next, dtype=float)
        assert e.shape == (FEATURE_DIM,), \
            f"expected eta_next shape ({FEATURE_DIM},), got {e.shape}"

        m.V_next = pyo.Var(m.K, within=pyo.NonNegativeReals)
        for k in m.K:
            Occ1_k = float(centers[k, 0])
            Occ2_k = float(centers[k, 1])
            price_k = float(centers[k, 2])
            f_k = (
                e[0] * 1.0
                + e[1] * m.T1c[k]
                + e[2] * m.T2c[k]
                + e[3] * m.Hc[k]
                + e[4] * Occ1_k
                + e[5] * Occ2_k
                + e[6] * price_k
                + e[7] * price_r
                + e[8] * vc_next_expr
                + e[9] * m.z1c[k]
                + e[10] * m.z2c[k]
            )
            m.cons.add(m.V_next[k] >= f_k)
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

    return ({"HeatPowerRoom1": p1_val, "HeatPowerRoom2": p2_val,
             "VentilationON": v_val}, V_star)


def advance_state(state, decision, exog):
    """Apply system dynamics to produce the next state dict."""
    fd = get_fixed_data()
    xi_exh = fd["heat_exchange_coeff"]; xi_loss = fd["thermal_loss_coeff"]
    xi_conv = fd["heating_efficiency_coeff"]; xi_cool = fd["heat_vent_coeff"]
    xi_occ = fd["heat_occupancy_coeff"]
    eta_occ = fd["humidity_occupancy_coeff"]; eta_vent = fd["humidity_vent_coeff"]
    T_low = fd["temp_min_comfort_threshold"]; T_ok = fd["temp_OK_threshold"]
    T_out = fd["outdoor_temperature"]

    t = int(state["current_time"])
    T_out_t = T_out[min(t, len(T_out) - 1)]
    T1 = state["T1"]; T2 = state["T2"]; H = state["H"]
    Occ1 = state["Occ1"]; Occ2 = state["Occ2"]
    p1 = decision["HeatPowerRoom1"]
    p2 = decision["HeatPowerRoom2"]
    V = decision["VentilationON"]

    T1_new = T1 - xi_exh*(T1-T2) - xi_loss*(T1-T_out_t) + xi_conv*p1 - xi_cool*V + xi_occ*Occ1
    T2_new = T2 - xi_exh*(T2-T1) - xi_loss*(T2-T_out_t) + xi_conv*p2 - xi_cool*V + xi_occ*Occ2
    H_new = float(np.clip(H + eta_occ*(Occ1+Occ2) - eta_vent*V, 0, 100))

    vc = int(state["vent_counter"])
    vc_new = (vc + 1 if vc > 0 else 1) if V == 1 else 0

    def upd(old, nt):
        if nt < T_low: return 1
        if old == 1 and nt < T_ok: return 1
        return 0

    lo1_new = upd(int(bool(state["low_override_r1"])), T1_new)
    lo2_new = upd(int(bool(state["low_override_r2"])), T2_new)

    return {
        "T1": float(T1_new), "T2": float(T2_new), "H": float(H_new),
        "Occ1": float(exog["Occ1"]), "Occ2": float(exog["Occ2"]),
        "price_t": float(exog["price"]), "price_previous": float(state["price_t"]),
        "vent_counter": vc_new,
        "low_override_r1": lo1_new, "low_override_r2": lo2_new,
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
        "low_override_r1": 0, "low_override_r2": 0,
        "current_time": 0,
    }


def sample_exogenous(state):
    r1_next, r2_next = next_occupancy_levels(state["Occ1"], state["Occ2"])
    price_next = price_model(state["price_t"], state["price_previous"])
    return {"Occ1": r1_next, "Occ2": r2_next, "price": price_next}


def _zero_etas(T):
    """Shape (T+1, FEATURE_DIM)."""
    return np.zeros((T + 1, FEATURE_DIM))


def forward_pass(etas, N=50, K=10, T=NUM_SLOTS, time_limit=10.0, mip_gap=0.01):
    """Roll out N trajectories using the current etas. Returns list of trajectories."""
    trajectories = []
    for n in range(N):
        state = sample_init_state()
        traj = [state]
        for t in range(T):
            eta_next = etas[t + 1] if (t + 1) < len(etas) else np.zeros(FEATURE_DIM)
            u, _ = solve_bellman(state, eta_next, K=K, time_limit=time_limit, mip_gap=mip_gap)
            exog = sample_exogenous(state)
            state = advance_state(state, u, exog)
            traj.append(state)
        trajectories.append(traj)
        print(f"  forward trajectory {n+1}/{N} done")
    return trajectories


def backward_pass(trajectories, etas, K=10, T=NUM_SLOTS,
                  time_limit=10.0, mip_gap=0.01):
    """Refit etas[t] via unregularised least squares on V* targets."""
    new_etas = np.asarray(etas, dtype=float).copy()
    new_etas[T] = 0.0
    for t in range(T - 1, -1, -1):
        Xs, ys = [], []
        for traj in trajectories:
            state_t = traj[t]
            _, V_star = solve_bellman(state_t, new_etas[t + 1], K=K,
                                      time_limit=time_limit, mip_gap=mip_gap)
            Xs.append(features(state_t))
            ys.append(V_star)
        Xr = np.asarray(Xs); yr = np.asarray(ys)
        coef, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
        new_etas[t] = coef
        print(f"  backward t={t}  n={len(yr):3d}  y_mean={yr.mean():.2f}  "
              f"|eta|_inf={np.abs(coef).max():.2f}")
    return new_etas


def train(I=15, N=50, K=10, T=NUM_SLOTS, tau=0.5, save_path=ETAS_PATH):
    """Run I outer iterations of forward-backward fitted value iteration.

    tau : Polyak averaging factor (default 0.5). tau=1.0 disables averaging.
    """
    etas = _zero_etas(T)
    for i in range(I):
        print(f"=== Iteration {i+1}/{I}  (tau={tau}) ===")
        trajectories = forward_pass(etas, N=N, K=K, T=T)
        fitted_etas = backward_pass(trajectories, etas, K=K, T=T)
        if tau < 1.0:
            etas = (1.0 - tau) * etas + tau * fitted_etas
        else:
            etas = fitted_etas
        np.save(save_path, etas)
        print(f"  saved etas to {save_path}")
    return etas


_CACHED_ETAS = None


def _load_etas():
    global _CACHED_ETAS
    if _CACHED_ETAS is None:
        if os.path.exists(ETAS_PATH):
            arr = np.load(ETAS_PATH)
            assert arr.shape == (NUM_SLOTS + 1, FEATURE_DIM), \
                f"expected etas shape ({NUM_SLOTS + 1}, {FEATURE_DIM}), got {arr.shape}"
            _CACHED_ETAS = arr
        else:
            print(f"No trained etas at {ETAS_PATH}; using zeros. Run train() first.")
            _CACHED_ETAS = _zero_etas(NUM_SLOTS)
    return _CACHED_ETAS


def select_action(state):
    etas = _load_etas()
    t = int(state["current_time"])
    eta_next = etas[t + 1] if (t + 1) < len(etas) else np.zeros(FEATURE_DIM)
    decision, _ = solve_bellman(state, eta_next, K=10, time_limit=10.0, mip_gap=0.02)
    return decision


if __name__ == "__main__":
    train()
