"""SP policy via backwards induction with a linear value function.

The original `sp_policy` solves a multistage stochastic program at every call
(scenario tree + one big MILP, terminal V=0 at the leaves). This variant
replaces that with classical fitted backwards induction:

  Offline (one backward sweep, no outer loop):
    V_T(x) = 0
    for t = T-1, T-2, ..., 0:
        # State sample at time t — visited states under the original SP policy
        states_t = roll out sp_policy for N days, bucket by t
        for x in states_t:
            y(x) = min_u { c(x, u) + E[V_{t+1}(f(x, u, ω))] }    # 1-step Bellman MILP
        V_t = LinearRegression(states_t, y)                       # ordinary least squares

  Online:
    select_action(state, t):
        return argmin_u { c(state, u) + E[V_{t+1}(state')] }    # the same MILP

No neural networks, no big-M trickery, no forward-backward outer loop, no
scenario tree built at inference time. Fitted backwards induction with linear
function approximation — the textbook formulation. Self-contained file.

Train offline:
    uv run python -m policies.sp_policy_bi
"""

import os

import numpy as np
import pyomo.environ as pyo
from sklearn.cluster import KMeans

from processes.PriceProcessRestaurant import price_model
from processes.OccupancyProcessRestaurant import next_occupancy_levels
from SystemCharacteristics import get_fixed_data
from policies import sp_policy as _sp


# --------------------------------------------------------------------------
# Constants from SystemCharacteristics
# --------------------------------------------------------------------------

FIXED = get_fixed_data()
NUM_SLOTS = int(FIXED["num_timeslots"])
P_MAX = FIXED["heating_max_power"]
P_VENT = FIXED["ventilation_power"]
T_LOW = FIXED["temp_min_comfort_threshold"]
T_OK = FIXED["temp_OK_threshold"]
T_HIGH = FIXED["temp_max_comfort_threshold"]
H_HIGH = FIXED["humidity_threshold"]
VENT_MIN = FIXED["vent_min_up_time"]
T_OUT = FIXED["outdoor_temperature"]
XI_EXH = FIXED["heat_exchange_coeff"]
XI_LOSS = FIXED["thermal_loss_coeff"]
XI_CONV = FIXED["heating_efficiency_coeff"]
XI_COOL = FIXED["heat_vent_coeff"]
XI_OCC = FIXED["heat_occupancy_coeff"]
ETA_OCC = FIXED["humidity_occupancy_coeff"]
ETA_VENT = FIXED["humidity_vent_coeff"]
T_CIRC = -3
M_LOW = T_LOW - T_CIRC
M_HIGH = T_OK - T_CIRC
M_HUM = 100.0 - H_HIGH

FEATURE_DIM = 11  # [1, T1, T2, H, Occ1, Occ2, price_t, price_prev, vc, lo1, lo2]
ETAS_PATH = os.path.join(os.path.dirname(__file__), "sp_bi_etas.npy")


# --------------------------------------------------------------------------
# Features φ(x) — must match the one-step Bellman MILP's continuation cost
# --------------------------------------------------------------------------

def features(state):
    """Map a state dict to φ(x) as a length-11 numpy array."""
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


# --------------------------------------------------------------------------
# Exogenous sampling / state advance (mirrors environment/simulator.py)
# --------------------------------------------------------------------------

def sample_init_state():
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


def sample_exog(state):
    r1, r2 = next_occupancy_levels(state["Occ1"], state["Occ2"])
    p = price_model(state["price_t"], state["price_previous"])
    return {"Occ1": r1, "Occ2": r2, "price": p}


def advance(state, action, exog):
    """Apply one step of the simulator's dynamics. `action` is (p1, p2, V).
    Assumes the action already respects overrules (so no override is applied)."""
    t = int(state["current_time"])
    T_out_t = T_OUT[min(t, len(T_OUT) - 1)]
    T1 = state["T1"]; T2 = state["T2"]; H = state["H"]
    Occ1 = state["Occ1"]; Occ2 = state["Occ2"]
    p1, p2, V = action

    T1n = T1 - XI_EXH * (T1 - T2) - XI_LOSS * (T1 - T_out_t) + XI_CONV * p1 - XI_COOL * V + XI_OCC * Occ1
    T2n = T2 - XI_EXH * (T2 - T1) - XI_LOSS * (T2 - T_out_t) + XI_CONV * p2 - XI_COOL * V + XI_OCC * Occ2
    Hn = float(np.clip(H + ETA_OCC * (Occ1 + Occ2) - ETA_VENT * V, 0.0, 100.0))

    vc = int(state["vent_counter"])
    vc_n = vc + 1 if V == 1 else 0

    lo1_n = int(bool(state["low_override_r1"]))
    if T1n < T_LOW:
        lo1_n = 1
    elif T1n >= T_OK:
        lo1_n = 0
    lo2_n = int(bool(state["low_override_r2"]))
    if T2n < T_LOW:
        lo2_n = 1
    elif T2n >= T_OK:
        lo2_n = 0

    return {
        "T1": float(T1n), "T2": float(T2n), "H": Hn,
        "Occ1": float(exog["Occ1"]), "Occ2": float(exog["Occ2"]),
        "price_t": float(exog["price"]), "price_previous": float(state["price_t"]),
        "vent_counter": int(vc_n),
        "low_override_r1": int(lo1_n), "low_override_r2": int(lo2_n),
        "current_time": t + 1,
    }


# --------------------------------------------------------------------------
# One-step Bellman MILP with linear continuation cost η_next · φ(x')
# --------------------------------------------------------------------------

def _gen_scenarios(state, K, num_samples=150):
    """KMeans-cluster M samples of (Occ1, Occ2, price) into K centers."""
    occ_samples = []
    price_samples = []
    for _ in range(num_samples):
        r1, r2 = next_occupancy_levels(state["Occ1"], state["Occ2"])
        p = price_model(state["price_t"], state["price_previous"])
        occ_samples.append([r1, r2])
        price_samples.append([p])
    joint = np.hstack([np.asarray(occ_samples), np.asarray(price_samples)])
    km = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(joint)
    counts = np.bincount(km.labels_, minlength=K)
    probs = counts / counts.sum()
    return km.cluster_centers_, probs


def solve_bellman(state, eta_next, K=5, time_limit=5.0, mip_gap=0.05):
    """Solve  min_u  c(x, u) + Σ_k p_k · η_next · φ(x_{k,t+1}).

    Returns (decision_dict, V_star)."""
    t_root = int(state["current_time"])
    T1_r = float(state["T1"]); T2_r = float(state["T2"]); H_r = float(state["H"])
    Occ1_r = float(state["Occ1"]); Occ2_r = float(state["Occ2"])
    price_r = float(state["price_t"])
    z1c_r = int(bool(state["low_override_r1"]))
    z2c_r = int(bool(state["low_override_r2"]))
    vc_r = int(state["vent_counter"])
    T_out_r = T_OUT[min(t_root, len(T_OUT) - 1)]

    centers, probs = _gen_scenarios(state, K)

    m = pyo.ConcreteModel()
    m.K = pyo.RangeSet(0, K - 1)

    # Root decisions
    m.p1 = pyo.Var(bounds=(0, P_MAX))
    m.p2 = pyo.Var(bounds=(0, P_MAX))
    m.V = pyo.Var(within=pyo.Binary)
    m.ON = pyo.Var(within=pyo.Binary)
    m.OFF = pyo.Var(within=pyo.Binary)
    m.z1_hot = pyo.Var(within=pyo.Binary)
    m.z2_hot = pyo.Var(within=pyo.Binary)
    m.cons = pyo.ConstraintList()

    # Root overrules
    m.cons.add(T1_r - T_HIGH <= M_HIGH * m.z1_hot)
    m.cons.add(m.p1 <= P_MAX * (1 - m.z1_hot))
    m.cons.add(T2_r - T_HIGH <= M_HIGH * m.z2_hot)
    m.cons.add(m.p2 <= P_MAX * (1 - m.z2_hot))
    if z1c_r == 1:
        m.cons.add(m.p1 >= P_MAX)
    if z2c_r == 1:
        m.cons.add(m.p2 >= P_MAX)
    m.cons.add(H_r - H_HIGH <= M_HUM * m.V)

    # Ventilation inertia at root
    if 0 < vc_r < VENT_MIN:
        m.V.fix(1); m.ON.fix(0); m.OFF.fix(0)
    elif vc_r == 0:
        m.cons.add(m.V == m.ON); m.OFF.fix(0)
    else:
        m.ON.fix(0); m.cons.add(m.V == 1 - m.OFF)
    m.cons.add(m.ON + m.OFF <= 1)

    root_cost = price_r * (m.p1 + m.p2 + P_VENT * m.V)

    # Child state variables (one per KMeans scenario)
    m.T1c = pyo.Var(m.K, bounds=(T_CIRC, 2 * T_HIGH))
    m.T2c = pyo.Var(m.K, bounds=(T_CIRC, 2 * T_HIGH))
    m.Hc = pyo.Var(m.K, bounds=(0, 100))
    m.z1c = pyo.Var(m.K, within=pyo.Binary)
    m.z2c = pyo.Var(m.K, within=pyo.Binary)

    for k in m.K:
        m.cons.add(m.T1c[k] == T1_r - XI_EXH * (T1_r - T2_r) - XI_LOSS * (T1_r - T_out_r)
                   + XI_CONV * m.p1 - XI_COOL * m.V + XI_OCC * Occ1_r)
        m.cons.add(m.T2c[k] == T2_r - XI_EXH * (T2_r - T1_r) - XI_LOSS * (T2_r - T_out_r)
                   + XI_CONV * m.p2 - XI_COOL * m.V + XI_OCC * Occ2_r)
        m.cons.add(m.Hc[k] == H_r + ETA_OCC * (Occ1_r + Occ2_r) - ETA_VENT * m.V)

        m.cons.add(T_LOW - m.T1c[k] <= M_LOW * m.z1c[k])
        m.cons.add(T_LOW - m.T2c[k] <= M_LOW * m.z2c[k])
        m.cons.add(T_OK - m.T1c[k] <= M_HIGH * (1 - z1c_r + m.z1c[k]))
        m.cons.add(T_OK - m.T2c[k] <= M_HIGH * (1 - z2c_r + m.z2c[k]))

    # vent_counter at t+1 (linear in root decisions)
    if vc_r == 0:
        vc_next_expr = m.ON
    elif vc_r < VENT_MIN:
        vc_next_expr = vc_r + 1
    else:
        vc_next_expr = (vc_r + 1) * m.V

    # Linear continuation Σ_k p_k · η_next · φ(x'_k)
    continuation = 0.0
    if eta_next is not None and np.any(eta_next != 0):
        for k in m.K:
            occ1_k = float(centers[k, 0]); occ2_k = float(centers[k, 1])
            price_k = float(centers[k, 2])
            feat_expr = (
                eta_next[0] * 1.0
                + eta_next[1] * m.T1c[k]
                + eta_next[2] * m.T2c[k]
                + eta_next[3] * m.Hc[k]
                + eta_next[4] * occ1_k
                + eta_next[5] * occ2_k
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
        decision = {
            "HeatPowerRoom1": float(pyo.value(m.p1)),
            "HeatPowerRoom2": float(pyo.value(m.p2)),
            "VentilationON": int(round(pyo.value(m.V))),
        }
        V_star = float(pyo.value(m.objective))
    else:
        decision = {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}
        V_star = 0.0
    return decision, V_star


# --------------------------------------------------------------------------
# State collection under the original SP policy + backwards induction training
# --------------------------------------------------------------------------

def collect_sp_states_by_time(N=100, verbose=False):
    """Roll out the original sp_policy for N random init states; bucket every
    visited state by its time index. Returns a list of T lists."""
    states_by_t = [[] for _ in range(NUM_SLOTS)]
    totals = []
    for n in range(N):
        s = sample_init_state()
        cost = 0.0
        for t in range(NUM_SLOTS):
            states_by_t[t].append(s)
            u = _sp.select_action(s)
            cost += s["price_t"] * (
                u["HeatPowerRoom1"] + u["HeatPowerRoom2"] + P_VENT * u["VentilationON"]
            )
            w = sample_exog(s)
            s = advance(
                s,
                (u["HeatPowerRoom1"], u["HeatPowerRoom2"], u["VentilationON"]),
                w,
            )
        totals.append(cost)
        if verbose and (n + 1) % 10 == 0:
            a = np.array(totals)
            print(f"  SP rollout {n + 1}/{N}  running mean={a.mean():.2f}")
    if verbose:
        a = np.array(totals)
        print(f"  collected {N} rollouts  mean={a.mean():.2f}  sd={a.std():.2f}")
    return states_by_t


def train_backward_induction(N=100, K=5, save_path=ETAS_PATH, verbose=True, seed=0,
                              time_limit=5.0, mip_gap=0.05):
    """Single backwards sweep with linear function approximation."""
    np.random.seed(seed)

    if verbose:
        print(f"=== rolling out the original sp_policy for {N} days ===")
    states_by_t = collect_sp_states_by_time(N=N, verbose=verbose)

    etas = [np.zeros(FEATURE_DIM) for _ in range(NUM_SLOTS + 1)]  # etas[T] = 0
    for t in range(NUM_SLOTS - 1, -1, -1):
        if verbose:
            print(f"\n=== backwards induction at t = {t} ===")
        states_t = states_by_t[t]
        X = np.stack([features(x) for x in states_t])
        y = np.empty(len(states_t), dtype=np.float64)
        for i, x in enumerate(states_t):
            _, V_star = solve_bellman(x, etas[t + 1], K=K,
                                      time_limit=time_limit, mip_gap=mip_gap)
            y[i] = V_star
        eta_t, *_ = np.linalg.lstsq(X, y, rcond=None)
        etas[t] = eta_t
        if verbose:
            yhat = X @ eta_t
            r2 = 1.0 - float(((y - yhat) ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1.0)
            print(f"  fit η_{t}: n={len(y)}  y mean={y.mean():.2f}  sd={y.std():.2f}  R²={r2:.3f}")

    np.save(save_path, np.array(etas))
    if verbose:
        print(f"\nsaved etas → {save_path}")
    return etas


# --------------------------------------------------------------------------
# Online policy
# --------------------------------------------------------------------------

_CACHED_ETAS = None


def _load_etas():
    global _CACHED_ETAS
    if _CACHED_ETAS is None:
        if os.path.exists(ETAS_PATH):
            _CACHED_ETAS = np.load(ETAS_PATH)
        else:
            print(f"No trained etas at {ETAS_PATH}; using zeros. Run train_backward_induction() first.")
            _CACHED_ETAS = np.zeros((NUM_SLOTS + 1, FEATURE_DIM))
    return _CACHED_ETAS


def select_action(state):
    etas = _load_etas()
    t = int(state["current_time"])
    eta_next = etas[t + 1] if t + 1 < len(etas) else np.zeros(FEATURE_DIM)
    decision, _ = solve_bellman(state, eta_next, K=5, time_limit=10.0, mip_gap=0.02)
    return decision


if __name__ == "__main__":
    train_backward_induction(N=100)
