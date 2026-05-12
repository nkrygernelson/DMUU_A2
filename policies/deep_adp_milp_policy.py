"""Hybrid policy: 2-stage MILP rollout with neural-network value function at leaves.

Structure (mirrors `sp_policy` for the tree and dynamics):
    stage 0 — current state (root, fixed)
    stage 1 — bf children, here-and-now decision becomes recourse uncertainty
    stage 2 — bf×bf leaves; the small VNet V_θ is evaluated at each leaf

Decisions (continuous heating powers, binary ventilation) are made at the root
and at each stage-1 child. At the leaves, V_θ is embedded as Pyomo constraints
via the standard big-M ReLU encoding. The objective is

    c(x_0, u_0) + Σ_k p_k [c(x_1^k, u_1^k) + Σ_j p_{kj} V_θ(x_2^{kj})].

Reuses `policies.deep_adp_policy.VNet` and the small (1×32) trained model at
`policies/deep_adp_model_small.pt`.
"""

import os

import numpy as np
import torch
import pyomo.environ as pyo

from policies.sp_policy import build_scenario_tree, propagate_uncertainty
from policies.deep_adp_policy import VNet, load_vnet
from SystemCharacteristics import get_fixed_data


FIXED = get_fixed_data()
NUM_SLOTS = int(FIXED["num_timeslots"])
SMALL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "deep_adp_model_small.pt")

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
M_VC = float(NUM_SLOTS) + 1.0  # big-M for vc transitions


# ---- VNet weight extraction & ReLU encoding -------------------------------

def extract_small_weights(model):
    """Return W1, b1, W2, b2 (numpy) from a (FEAT_DIM → 32 → 1) VNet."""
    layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert len(layers) == 2, "small VNet must have exactly one hidden layer"
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
    """Bounds on the normalized 12-dim feature vector at a leaf with fixed t."""
    # Bounds match VNet variable ranges in `policies.deep_adp_policy.encode_batch`.
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


# ---- Tree solver -----------------------------------------------------------

def build_and_solve(state, root, all_nodes, leaves, W1, b1, W2, b2,
                    time_limit=10.0, mip_gap=0.05):
    current_time = int(state["current_time"])
    m = pyo.ConcreteModel()
    nids = [n.node_id for n in all_nodes]
    nl_nids = [n.node_id for n in all_nodes if n.children]  # non-leaf (have decisions)
    leaf_nids = [n.node_id for n in leaves]
    m.NODES = pyo.Set(initialize=nids)
    m.NL = pyo.Set(initialize=nl_nids)
    m.LEAF = pyo.Set(initialize=leaf_nids)

    # Decisions at non-leaf nodes
    m.p1 = pyo.Var(m.NL, bounds=(0, P_MAX))
    m.p2 = pyo.Var(m.NL, bounds=(0, P_MAX))
    m.V = pyo.Var(m.NL, within=pyo.Binary)
    m.ON = pyo.Var(m.NL, within=pyo.Binary)
    m.OFF = pyo.Var(m.NL, within=pyo.Binary)
    m.z1_hot = pyo.Var(m.NL, within=pyo.Binary)
    m.z2_hot = pyo.Var(m.NL, within=pyo.Binary)

    # State variables (defined at every node)
    m.T1 = pyo.Var(m.NODES, bounds=(T_CIRC, 2 * T_HIGH))
    m.T2 = pyo.Var(m.NODES, bounds=(T_CIRC, 2 * T_HIGH))
    m.hum = pyo.Var(m.NODES, bounds=(0, 100))
    m.z1_cold = pyo.Var(m.NODES, within=pyo.Binary)
    m.z2_cold = pyo.Var(m.NODES, within=pyo.Binary)
    m.vc = pyo.Var(m.NODES, bounds=(0, NUM_SLOTS))  # vc-going-into-this-stage

    m.cons = pyo.ConstraintList()
    obj = 0.0

    rid = root.node_id
    # ---- Fix root state ----
    m.T1[rid].fix(float(state["T1"]))
    m.T2[rid].fix(float(state["T2"]))
    m.hum[rid].fix(float(state["H"]))
    m.z1_cold[rid].fix(int(bool(state["low_override_r1"])))
    m.z2_cold[rid].fix(int(bool(state["low_override_r2"])))
    m.vc[rid].fix(int(state["vent_counter"]))

    vc_root = int(state["vent_counter"])

    # Ventilation inertia init at root, mirrored from sp_policy
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

    # Overrules at root (same as sp_policy)
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

    # ---- Walk all non-root nodes ----
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

        # Dynamics — depend on parent's state + parent's decisions (parent is non-leaf)
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

        # vc transition: vc[child] = (vc[parent] + 1) * V[parent], big-M
        m.cons.add(m.vc[nid] <= M_VC * m.V[pid])
        m.cons.add(m.vc[nid] >= m.vc[pid] + 1 - M_VC * (1 - m.V[pid]))
        m.cons.add(m.vc[nid] <= m.vc[pid] + 1)
        # Low-override transition (hysteresis)
        m.cons.add(T_LOW - m.T1[nid] <= M_LOW * m.z1_cold[nid])
        m.cons.add(T_LOW - m.T2[nid] <= M_LOW * m.z2_cold[nid])
        m.cons.add(T_OK - m.T1[nid] <= M_HIGH * (1 - m.z1_cold[pid] + m.z1_cold[nid]))
        m.cons.add(T_OK - m.T2[nid] <= M_HIGH * (1 - m.z2_cold[pid] + m.z2_cold[nid]))

        if node.children:  # non-leaf child (stage 1 in 2-stage tree)
            # Decision-side overrules
            m.cons.add(m.T1[nid] - T_HIGH <= M_HIGH * m.z1_hot[nid])
            m.cons.add(m.p1[nid] <= P_MAX * (1 - m.z1_hot[nid]))
            m.cons.add(m.T2[nid] - T_HIGH <= M_HIGH * m.z2_hot[nid])
            m.cons.add(m.p2[nid] <= P_MAX * (1 - m.z2_hot[nid]))
            m.cons.add(m.p1[nid] >= P_MAX * m.z1_cold[nid])
            m.cons.add(m.p2[nid] >= P_MAX * m.z2_cold[nid])
            m.cons.add(m.hum[nid] - H_HIGH <= M_HUM * m.V[nid])
            m.cons.add(m.ON[nid] + m.OFF[nid] <= 1)

            # Vent inertia: V[nid] >= ON[nid] + ON[parent]  (+ ON[grandparent] if exists)
            if node.stage >= 2:
                gp = parent.parent
                m.cons.add(m.V[nid] >= m.ON[nid] + m.ON[pid] + m.ON[gp.node_id])
            else:
                m.cons.add(m.V[nid] >= m.ON[nid] + m.ON[pid])
            m.cons.add(m.OFF[nid] <= m.V[pid])
            m.cons.add(m.ON[nid] <= 1 - m.V[pid])
            m.cons.add(m.V[nid] == m.V[pid] + m.ON[nid] - m.OFF[nid])

            # Stage cost
            price_n = node.state["current_price"]
            wp = node_prob(node)
            obj += wp * price_n * (m.p1[nid] + m.p2[nid] + P_VENT * m.V[nid])

    # ---- Root cost ----
    obj += state["price_t"] * (m.p1[rid] + m.p2[rid] + P_VENT * m.V[rid])

    # ---- V_θ leaf cost ----
    h_dim = W1.shape[0]
    for leaf in leaves:
        lid = leaf.node_id
        t_leaf = current_time + leaf.stage
        if t_leaf >= NUM_SLOTS:
            continue  # terminal: V = 0
        wp = node_prob(leaf)

        occ1_p = leaf.parent.state["current_r1_occ"]
        occ2_p = leaf.parent.state["current_r2_occ"]
        price_l = leaf.state["current_price"]
        price_p = leaf.parent.state["current_price"]

        # Build the 12-dim feature vector as Pyomo expressions/constants
        feat_exprs = [
            m.T1[lid] / 30.0,
            m.T2[lid] / 30.0,
            m.hum[lid] / 100.0,
            occ1_p / 50.0,            # exogenous Occ at parent influences this leaf — but the
            occ2_p / 30.0,            # encode_batch uses leaf occupancy. Use leaf's exogenous.
            price_l / 12.0,
            price_p / 12.0,
            m.vc[lid] / float(VENT_MIN),
            m.z1_cold[lid],
            m.z2_cold[lid],
            t_leaf / float(NUM_SLOTS),
            1.0 - t_leaf / float(NUM_SLOTS),
        ]
        # Correct: the encode features at a state x_t use that state's own
        # Occ1, Occ2 — which at a stage-2 leaf are the *grandchild* exogenous.
        # We approximate Occ at the leaf with the leaf-scenario's occupancy
        # centers (drawn from the leaf node's state).
        occ1_leaf = leaf.state["current_r1_occ"]
        occ2_leaf = leaf.state["current_r2_occ"]
        feat_exprs[3] = occ1_leaf / 50.0
        feat_exprs[4] = occ2_leaf / 30.0

        # Bounds (numpy) for big-M ReLU
        L_in, U_in = leaf_feature_bounds(t_leaf)
        L_a, U_a = linear_bounds(W1, b1, L_in, U_in)

        h_set = pyo.Set(initialize=range(h_dim))
        setattr(m, f"hset_{lid}", h_set)
        h_vars = pyo.Var(h_set, domain=pyo.NonNegativeReals)
        z_vars = pyo.Var(h_set, within=pyo.Binary)
        setattr(m, f"h_{lid}", h_vars)
        setattr(m, f"z_{lid}", z_vars)

        for j in range(h_dim):
            a_expr = sum(float(W1[j, i]) * feat_exprs[i] for i in range(12)) + float(b1[j])
            if L_a[j] >= 0:
                m.cons.add(h_vars[j] == a_expr)
                m.cons.add(z_vars[j] == 1)
            elif U_a[j] <= 0:
                m.cons.add(h_vars[j] == 0)
                m.cons.add(z_vars[j] == 0)
            else:
                m.cons.add(h_vars[j] >= a_expr)
                m.cons.add(h_vars[j] <= float(U_a[j]) * z_vars[j])
                m.cons.add(h_vars[j] <= a_expr - float(L_a[j]) * (1 - z_vars[j]))

        V_leaf = sum(float(W2[j]) * h_vars[j] for j in range(h_dim)) + float(b2)
        obj += wp * V_leaf

    m.objective = pyo.Objective(expr=obj, sense=pyo.minimize)
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = mip_gap
    solver.options["OutputFlag"] = 0
    result = solver.solve(m, tee=False)
    return m, result


def node_prob(node):
    """Path probability from root to `node` (excluding root which has prob 1)."""
    p = 1.0
    n = node
    while n.parent is not None:
        p *= n.prob
        n = n.parent
    return p


# ---- Model loading & policy entry point -----------------------------------

_CACHED = None


def _load():
    global _CACHED
    if _CACHED is None:
        model = load_vnet(SMALL_MODEL_PATH, hidden_sizes=(32,))
        _CACHED = extract_small_weights(model)
    return _CACHED


def select_action(state):
    W1, b1, W2, b2 = _load()
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
                                time_limit=10.0, mip_gap=0.05)

    rid = root.node_id
    ok = (result.solver.status == pyo.SolverStatus.ok and
          result.solver.termination_condition in (
              pyo.TerminationCondition.optimal,
              pyo.TerminationCondition.feasible))
    if ok:
        p1_val = float(pyo.value(m.p1[rid]))
        p2_val = float(pyo.value(m.p2[rid]))
        v_val = int(round(pyo.value(m.V[rid])))
    else:
        p1_val = p2_val = 0.0
        v_val = 0
    return {
        "HeatPowerRoom1": p1_val,
        "HeatPowerRoom2": p2_val,
        "VentilationON": v_val,
    }
