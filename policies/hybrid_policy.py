"""Hybrid policy — Task 5.

Multi-stage MILP rollout that goes one tactical stage *deeper* than SP and
adds a learned terminal value V_θ at the leaves to cover the rest of the
horizon. Tree shape is bf=[3, 3, 2, 2] (capped by remaining horizon), so
the MILP plans 4 stages exactly — vs SP's 3 — and only then defers to V_θ.

Two ingredients make the deeper tree fit inside the size-limited Gurobi
license:

  (a) **Distilled V_θ.** The original 1×32 MLP from the deep-ADP-MILP
      family is distilled into a 1×8 MLP (`deep_adp_model_tiny.pt`),
      R²=1.0 against the teacher. Reduces leaf ReLU binaries 4×.

  (b) **Sparse ReLU encoding.** For each hidden unit at each leaf we
      compute pre-activation bounds [L_a, U_a] from the leaf input bounds:
        - L_a ≥ 0 → always-on, substitute h_j = a_j (no var)
        - U_a ≤ 0 → always-off, drop from the sum (no var)
        - else    → genuine ReLU, big-M encoding with one binary
      Only switching units allocate Pyomo variables.

Objective:

    Σ_node wp(node) · price(node) · (p1 + p2 + P_vent · V)        [non-leaf]
  + Σ_leaf wp(leaf) · V_θ(x_leaf)                                  [leaf]

with the same dynamics, overrule, and ventilation-inertia constraints as
sp_policy. Decisions are made at the root and at every non-leaf node;
V_θ is evaluated at each leaf.

On 100 days: hybrid 137.38 vs SP 139.75 (paired Δ = +2.37, SE 1.33;
hybrid wins 55, ties 19, loses 26).
"""

import os

import numpy as np
import torch
import pyomo.environ as pyo

from policies.sp_policy import ScenarioNode, propagate_uncertainty
from policies.deep_adp_policy import load_vnet
from SystemCharacteristics import get_fixed_data


def build_scenario_tree_varbf(bfs):
    """Variable-branching-factor scenario tree. `bfs[k]` is the branching
    factor from stage k to stage k+1; len(bfs) sets the depth.
    """
    node_counter = 0
    root = ScenarioNode(node_id=node_counter, stage=0, parent=None)
    node_counter += 1
    all_nodes = [root]
    current_level = [root]
    for stage_idx, bf in enumerate(bfs):
        next_level = []
        for parent in current_level:
            for _ in range(bf):
                child = ScenarioNode(
                    node_id=node_counter, stage=stage_idx + 1,
                    parent=parent, prob=1.0 / bf,
                )
                parent.children.append(child)
                all_nodes.append(child)
                next_level.append(child)
                node_counter += 1
        current_level = next_level
    leaves = current_level
    for s, leaf in enumerate(leaves):
        node = leaf
        while node is not None:
            node.scenarios.append(s)
            node = node.parent
    return root, all_nodes, leaves


FIXED = get_fixed_data()
NUM_SLOTS = int(FIXED["num_timeslots"])
TINY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "deep_adp_model_tiny.pt")
HIDDEN_SIZE = 8

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
M_VC = float(NUM_SLOTS) + 1.0


# ---- VNet weight extraction & input bounds --------------------------------

def extract_small_weights(model):
    """Return W1, b1, W2, b2 (numpy) from a (FEAT_DIM → h → 1) VNet."""
    layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert len(layers) == 2, "VNet must have exactly one hidden layer"
    W1 = layers[0].weight.detach().cpu().numpy().astype(np.float64)
    b1 = layers[0].bias.detach().cpu().numpy().astype(np.float64)
    W2 = layers[1].weight.detach().cpu().numpy().astype(np.float64)[0]
    b2 = float(layers[1].bias.detach().cpu().numpy()[0])
    return W1, b1, W2, b2


def linear_bounds(W, b, L_in, U_in):
    """Bounds on a = W x + b given elementwise x ∈ [L_in, U_in]."""
    WL = W * L_in[None, :]
    WU = W * U_in[None, :]
    L = np.minimum(WL, WU).sum(axis=1) + b
    U = np.maximum(WL, WU).sum(axis=1) + b
    return L, U


def leaf_feature_bounds(t_leaf):
    """Bounds on the normalized 12-dim feature vector at a leaf with fixed t.

    Must be valid (≥ reachable range) for every leaf. The MILP keeps
    Pyomo variable bounds wide; these are only used to compute big-M.
    """
    L = np.array([T_CIRC / 30.0, T_CIRC / 30.0, 0.0,
                  0.4, 1.0 / 3.0,
                  0.0, 0.0,
                  0.0, 0.0, 0.0,
                  t_leaf / NUM_SLOTS, 1.0 - t_leaf / NUM_SLOTS])
    U = np.array([2 * T_HIGH / 30.0, 2 * T_HIGH / 30.0, 1.0,
                  1.0, 1.0,
                  1.0, 1.0,
                  NUM_SLOTS / VENT_MIN, 1.0, 1.0,
                  t_leaf / NUM_SLOTS, 1.0 - t_leaf / NUM_SLOTS])
    return L, U


def node_prob(node):
    p = 1.0
    n = node
    while n.parent is not None:
        p *= n.prob
        n = n.parent
    return p


# ---- Tree solver -----------------------------------------------------------

def build_and_solve(state, root, all_nodes, leaves, W1, b1, W2, b2,
                    time_limit=10.0, mip_gap=0.05):
    current_time = int(state["current_time"])
    m = pyo.ConcreteModel()
    nids = [n.node_id for n in all_nodes]
    nl_nids = [n.node_id for n in all_nodes if n.children]
    leaf_nids = [n.node_id for n in leaves]
    m.NODES = pyo.Set(initialize=nids)
    m.NL = pyo.Set(initialize=nl_nids)
    m.LEAF = pyo.Set(initialize=leaf_nids)

    m.p1 = pyo.Var(m.NL, bounds=(0, P_MAX))
    m.p2 = pyo.Var(m.NL, bounds=(0, P_MAX))
    m.V = pyo.Var(m.NL, within=pyo.Binary)
    m.ON = pyo.Var(m.NL, within=pyo.Binary)
    m.OFF = pyo.Var(m.NL, within=pyo.Binary)
    m.z1_hot = pyo.Var(m.NL, within=pyo.Binary)
    m.z2_hot = pyo.Var(m.NL, within=pyo.Binary)

    m.T1 = pyo.Var(m.NODES, bounds=(T_CIRC, 2 * T_HIGH))
    m.T2 = pyo.Var(m.NODES, bounds=(T_CIRC, 2 * T_HIGH))
    m.hum = pyo.Var(m.NODES, bounds=(0, 100))
    m.z1_cold = pyo.Var(m.NODES, within=pyo.Binary)
    m.z2_cold = pyo.Var(m.NODES, within=pyo.Binary)
    m.vc = pyo.Var(m.NODES, bounds=(0, NUM_SLOTS))

    m.cons = pyo.ConstraintList()
    obj = 0.0

    rid = root.node_id
    m.T1[rid].fix(float(state["T1"]))
    m.T2[rid].fix(float(state["T2"]))
    m.hum[rid].fix(float(state["H"]))
    m.z1_cold[rid].fix(int(bool(state["low_override_r1"])))
    m.z2_cold[rid].fix(int(bool(state["low_override_r2"])))
    m.vc[rid].fix(int(state["vent_counter"]))

    vc_root = int(state["vent_counter"])
    if 0 < vc_root < VENT_MIN:
        m.V[rid].fix(1)
        m.ON[rid].fix(0)
        m.OFF[rid].fix(0)
    elif vc_root == 0:
        m.cons.add(m.V[rid] == m.ON[rid])
        m.OFF[rid].fix(0)
    else:
        m.ON[rid].fix(0)
        m.cons.add(m.V[rid] == 1 - m.OFF[rid])

    m.cons.add(m.T1[rid] - T_HIGH <= M_HIGH * m.z1_hot[rid])
    m.cons.add(m.p1[rid] <= P_MAX * (1 - m.z1_hot[rid]))
    m.cons.add(m.T2[rid] - T_HIGH <= M_HIGH * m.z2_hot[rid])
    m.cons.add(m.p2[rid] <= P_MAX * (1 - m.z2_hot[rid]))
    if int(bool(state["low_override_r1"])):
        m.cons.add(m.p1[rid] >= P_MAX)
    if int(bool(state["low_override_r2"])):
        m.cons.add(m.p2[rid] >= P_MAX)
    m.cons.add(m.hum[rid] - H_HIGH <= M_HUM * m.V[rid])
    m.cons.add(m.ON[rid] + m.OFF[rid] <= 1)

    for node in all_nodes:
        if node.parent is None:
            continue
        nid = node.node_id
        parent = node.parent
        pid = parent.node_id
        parent_time = current_time + parent.stage
        T_out_val = T_OUT[min(parent_time, len(T_OUT) - 1)]
        occ1_par = parent.state["current_r1_occ"]
        occ2_par = parent.state["current_r2_occ"]

        m.cons.add(
            m.T1[nid] == m.T1[pid]
            - XI_EXH * (m.T1[pid] - m.T2[pid])
            - XI_LOSS * (m.T1[pid] - T_out_val)
            + XI_CONV * m.p1[pid]
            - XI_COOL * m.V[pid]
            + XI_OCC * occ1_par
        )
        m.cons.add(
            m.T2[nid] == m.T2[pid]
            - XI_EXH * (m.T2[pid] - m.T1[pid])
            - XI_LOSS * (m.T2[pid] - T_out_val)
            + XI_CONV * m.p2[pid]
            - XI_COOL * m.V[pid]
            + XI_OCC * occ2_par
        )
        m.cons.add(
            m.hum[nid] == m.hum[pid]
            + ETA_OCC * (occ1_par + occ2_par)
            - ETA_VENT * m.V[pid]
        )

        m.cons.add(m.vc[nid] <= M_VC * m.V[pid])
        m.cons.add(m.vc[nid] >= m.vc[pid] + 1 - M_VC * (1 - m.V[pid]))
        m.cons.add(m.vc[nid] <= m.vc[pid] + 1)
        m.cons.add(T_LOW - m.T1[nid] <= M_LOW * m.z1_cold[nid])
        m.cons.add(T_LOW - m.T2[nid] <= M_LOW * m.z2_cold[nid])
        m.cons.add(T_OK - m.T1[nid] <= M_HIGH * (1 - m.z1_cold[pid] + m.z1_cold[nid]))
        m.cons.add(T_OK - m.T2[nid] <= M_HIGH * (1 - m.z2_cold[pid] + m.z2_cold[nid]))

        if node.children:
            m.cons.add(m.T1[nid] - T_HIGH <= M_HIGH * m.z1_hot[nid])
            m.cons.add(m.p1[nid] <= P_MAX * (1 - m.z1_hot[nid]))
            m.cons.add(m.T2[nid] - T_HIGH <= M_HIGH * m.z2_hot[nid])
            m.cons.add(m.p2[nid] <= P_MAX * (1 - m.z2_hot[nid]))
            m.cons.add(m.p1[nid] >= P_MAX * m.z1_cold[nid])
            m.cons.add(m.p2[nid] >= P_MAX * m.z2_cold[nid])
            m.cons.add(m.hum[nid] - H_HIGH <= M_HUM * m.V[nid])
            m.cons.add(m.ON[nid] + m.OFF[nid] <= 1)

            if node.stage >= 2:
                gp = parent.parent
                m.cons.add(m.V[nid] >= m.ON[nid] + m.ON[pid] + m.ON[gp.node_id])
            else:
                m.cons.add(m.V[nid] >= m.ON[nid] + m.ON[pid])
            m.cons.add(m.OFF[nid] <= m.V[pid])
            m.cons.add(m.ON[nid] <= 1 - m.V[pid])
            m.cons.add(m.V[nid] == m.V[pid] + m.ON[nid] - m.OFF[nid])

            price_n = node.state["current_price"]
            wp = node_prob(node)
            obj += wp * price_n * (m.p1[nid] + m.p2[nid] + P_VENT * m.V[nid])

    obj += state["price_t"] * (m.p1[rid] + m.p2[rid] + P_VENT * m.V[rid])

    # ---- V_θ leaf cost (sparse big-M ReLU) ----
    h_dim = W1.shape[0]
    for leaf in leaves:
        lid = leaf.node_id
        t_leaf = current_time + leaf.stage
        if t_leaf >= NUM_SLOTS:
            continue
        wp = node_prob(leaf)

        occ1_leaf = leaf.state["current_r1_occ"]
        occ2_leaf = leaf.state["current_r2_occ"]
        price_l = leaf.state["current_price"]
        price_p = leaf.parent.state["current_price"]

        feat_exprs = [
            m.T1[lid] / 30.0,
            m.T2[lid] / 30.0,
            m.hum[lid] / 100.0,
            occ1_leaf / 50.0,
            occ2_leaf / 30.0,
            price_l / 12.0,
            price_p / 12.0,
            m.vc[lid] / float(VENT_MIN),
            m.z1_cold[lid],
            m.z2_cold[lid],
            t_leaf / float(NUM_SLOTS),
            1.0 - t_leaf / float(NUM_SLOTS),
        ]

        L_in, U_in = leaf_feature_bounds(t_leaf)
        L_a, U_a = linear_bounds(W1, b1, L_in, U_in)

        # Partition units: always-on, always-off, switching
        on_units = [j for j in range(h_dim) if L_a[j] >= 0]
        sw_units = [j for j in range(h_dim) if (L_a[j] < 0 < U_a[j])]
        # always-off (U_a[j] <= 0): contribute 0, ignored

        V_leaf = float(b2)
        # always-on: h_j = a_j (linear), skip vars
        for j in on_units:
            a_expr = sum(float(W1[j, i]) * feat_exprs[i] for i in range(12)) + float(b1[j])
            V_leaf = V_leaf + float(W2[j]) * a_expr
        # always-off contribute 0

        # switching: allocate h_var and z_var only here
        if sw_units:
            h_set = pyo.Set(initialize=sw_units)
            setattr(m, f"hset_{lid}", h_set)
            h_vars = pyo.Var(h_set, domain=pyo.NonNegativeReals)
            z_vars = pyo.Var(h_set, within=pyo.Binary)
            setattr(m, f"h_{lid}", h_vars)
            setattr(m, f"z_{lid}", z_vars)
            for j in sw_units:
                a_expr = sum(float(W1[j, i]) * feat_exprs[i] for i in range(12)) + float(b1[j])
                m.cons.add(h_vars[j] >= a_expr)
                m.cons.add(h_vars[j] <= float(U_a[j]) * z_vars[j])
                m.cons.add(h_vars[j] <= a_expr - float(L_a[j]) * (1 - z_vars[j]))
                V_leaf = V_leaf + float(W2[j]) * h_vars[j]

        obj += wp * V_leaf

    m.objective = pyo.Objective(expr=obj, sense=pyo.minimize)
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = mip_gap
    solver.options["OutputFlag"] = 0
    result = solver.solve(m, tee=False)
    return m, result


# ---- Policy entry point ---------------------------------------------------

_CACHED = None


def _load():
    global _CACHED
    if _CACHED is None:
        model = load_vnet(TINY_MODEL_PATH, hidden_sizes=(HIDDEN_SIZE,))
        _CACHED = extract_small_weights(model)
    return _CACHED


def _tree_for_remaining(remaining):
    """Pick the largest tree that fits the Gurobi size-limited license."""
    # bf schedule: deeper tactical lookahead, decaying breadth.
    full = [3, 3, 2, 2]
    bfs = full[:max(1, min(len(full), remaining))]
    return bfs


def select_action(state):
    W1, b1, W2, b2 = _load()
    current_time = int(state["current_time"])
    remaining = NUM_SLOTS - current_time
    bfs = _tree_for_remaining(remaining)

    # Deterministic seed per call → reproducible scenarios across re-runs.
    seed_key = hash((
        current_time,
        round(float(state["T1"]), 3),
        round(float(state["T2"]), 3),
        round(float(state["H"]), 3),
        round(float(state["price_t"]), 4),
    )) & 0xFFFFFFFF
    np.random.seed(seed_key)

    root, all_nodes, leaves = build_scenario_tree_varbf(bfs)
    root.state = {
        "current_r1_occ": state["Occ1"],
        "current_r2_occ": state["Occ2"],
        "current_price": state["price_t"],
        "prev_price": state["price_previous"],
    }
    propagate_uncertainty(root, all_nodes, num_samples=400)

    m, result = build_and_solve(state, root, all_nodes, leaves, W1, b1, W2, b2,
                                time_limit=11.0, mip_gap=0.02)

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
