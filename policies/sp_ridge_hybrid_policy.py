"""SP + ridge V_theta at the leaves (honorable-mention hybrid from the report).

Single linear value function per stage at each leaf: V(x_leaf) = eta_t · phi(x_leaf),
with eta_t fitted by RidgeCV on SP-distilled Monte-Carlo returns-to-go (see
experiments/train_single_eta_sp.py). Pure linear in the MILP — zero extra
binaries beyond what SP already has.

Feature vector phi(x) (length 11):
  [1, T1, T2, H, Occ1, Occ2, price_t, price_previous,
   vent_counter, low_override_r1, low_override_r2]

eta[t, 0] holds the fitted intercept (phi[0] is hardcoded to 1.0).

Approximation: vent_counter at each leaf is treated as the root state's
vc_root (cheap proxy for a feature that's small in magnitude relative to
the others, see pdfs/hybrid_sp.md).

100-day mean cost: 153.21 (vs plain SP 139.74 and the deployed SP+MPC
hybrid_policy at 146.69). Kept for reproducibility of the figure
'sp_distilled_per_stage_scatter.png' and the 153.21 number in the report.
"""

import os
import numpy as np
import pyomo.environ as pyo

from SystemCharacteristics import get_fixed_data
from policies.sp_policy import (
    build_scenario_tree,
    propagate_uncertainty,
)

FEATURE_DIM = 11
SINGLE_ETAS_PATH = os.path.join(os.path.dirname(__file__), "adp_etas_single_sp.npy")
NUM_SLOTS = int(get_fixed_data()["num_timeslots"])

BF = 3
NUM_STAGES = 2

_CACHED_ETAS = None


def _load_etas():
    global _CACHED_ETAS
    if _CACHED_ETAS is None:
        if not os.path.exists(SINGLE_ETAS_PATH):
            raise FileNotFoundError(
                f"No single-eta file at {SINGLE_ETAS_PATH}. Run "
                "experiments/train_single_eta_sp.py first."
            )
        arr = np.load(SINGLE_ETAS_PATH)
        if arr.shape != (NUM_SLOTS + 1, FEATURE_DIM):
            raise ValueError(
                f"Etas shape {arr.shape} doesn't match expected "
                f"({NUM_SLOTS + 1}, {FEATURE_DIM})."
            )
        _CACHED_ETAS = arr
    return _CACHED_ETAS


def _path_prob(node):
    p = 1.0
    while node.parent is not None:
        p *= node.prob
        node = node.parent
    return p


def _build_and_solve(state, root, all_nodes, leaves):
    etas = _load_etas()
    current_time = state["current_time"]

    fixed_data = get_fixed_data()
    xi_exh = fixed_data["heat_exchange_coeff"]
    xi_loss = fixed_data["thermal_loss_coeff"]
    xi_conv = fixed_data["heating_efficiency_coeff"]
    xi_cool = fixed_data["heat_vent_coeff"]
    xi_occ = fixed_data["heat_occupancy_coeff"]
    eta_occ_h = fixed_data["humidity_occupancy_coeff"]
    eta_vent_h = fixed_data["humidity_vent_coeff"]

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

    m = pyo.ConcreteModel()
    nids = [n.node_id for n in all_nodes]
    m.NODES = pyo.Set(initialize=nids)

    m.p1 = pyo.Var(m.NODES, bounds=(0, P_overline))
    m.p2 = pyo.Var(m.NODES, bounds=(0, P_overline))
    m.V = pyo.Var(m.NODES, within=pyo.Binary)
    m.temp1 = pyo.Var(m.NODES, bounds=(T_circ, 2 * T_high))
    m.temp2 = pyo.Var(m.NODES, bounds=(T_circ, 2 * T_high))
    m.z1_cold = pyo.Var(m.NODES, within=pyo.Binary)
    m.z1_hot = pyo.Var(m.NODES, within=pyo.Binary)
    m.z2_cold = pyo.Var(m.NODES, within=pyo.Binary)
    m.z2_hot = pyo.Var(m.NODES, within=pyo.Binary)
    m.ON = pyo.Var(m.NODES, within=pyo.Binary)
    m.OFF = pyo.Var(m.NODES, within=pyo.Binary)
    m.hum = pyo.Var(m.NODES, bounds=(0, 100))
    m.cons = pyo.ConstraintList()
    obj_expr = 0.0

    rid = root.node_id
    m.temp1[rid].fix(state["T1"])
    m.temp2[rid].fix(state["T2"])
    m.hum[rid].fix(state["H"])
    m.z1_cold[rid].fix(state["low_override_r1"])
    m.z2_cold[rid].fix(state["low_override_r2"])

    m.cons.add(T_ok - m.temp1[rid] <= M_high * (1 + m.z1_cold[rid]))
    m.cons.add(T_ok - m.temp2[rid] <= M_high * (1 + m.z2_cold[rid]))

    vc_root = int(state["vent_counter"])
    if 0 < vc_root < 3:
        m.V[rid].fix(1)
        m.ON[rid].fix(0)
        m.OFF[rid].fix(0)
    elif vc_root == 0:
        m.cons.add(m.V[rid] == m.ON[rid])
        m.OFF[rid].fix(0)
    else:
        m.ON[rid].fix(0)
        m.cons.add(m.V[rid] == 1 - m.OFF[rid])

    for node in all_nodes:
        nid = node.node_id

        if node.children:
            m.cons.add(m.temp1[nid] - T_high <= M_high * m.z1_hot[nid])
            m.cons.add(m.p1[nid] <= P_overline * (1 - m.z1_hot[nid]))
            m.cons.add(m.temp2[nid] - T_high <= M_high * m.z2_hot[nid])
            m.cons.add(m.p2[nid] <= P_overline * (1 - m.z2_hot[nid]))

            m.cons.add(T_low - m.temp1[nid] <= M_low * m.z1_cold[nid])
            m.cons.add(m.p1[nid] >= P_overline * m.z1_cold[nid])
            m.cons.add(T_low - m.temp2[nid] <= M_low * m.z2_cold[nid])
            m.cons.add(m.p2[nid] >= P_overline * m.z2_cold[nid])

            m.cons.add(m.hum[nid] - H_high <= M_hum * m.V[nid])
            price = node.state["current_price"]
            wp = _path_prob(node)
            obj_expr += wp * price * (m.p1[nid] + m.p2[nid] + p_vent * m.V[nid])
            m.cons.add(m.ON[nid] + m.OFF[nid] <= 1)

        if node.parent is not None:
            parent = node.parent
            pid = parent.node_id
            parent_time = current_time + parent.stage
            T_out_val = T_out[parent_time]

            m.cons.add(
                m.temp1[nid] == m.temp1[pid]
                - xi_exh * (m.temp1[pid] - m.temp2[pid])
                - xi_loss * (m.temp1[pid] - T_out_val)
                + xi_conv * m.p1[pid]
                - xi_cool * m.V[pid]
                + xi_occ * parent.state["current_r1_occ"]
            )
            m.cons.add(
                m.temp2[nid] == m.temp2[pid]
                - xi_exh * (m.temp2[pid] - m.temp1[pid])
                - xi_loss * (m.temp2[pid] - T_out_val)
                + xi_conv * m.p2[pid]
                - xi_cool * m.V[pid]
                + xi_occ * parent.state["current_r2_occ"]
            )
            m.cons.add(
                m.hum[nid] == m.hum[pid]
                + eta_occ_h * (parent.state["current_r1_occ"] + parent.state["current_r2_occ"])
                - eta_vent_h * m.V[pid]
            )

            if node.stage >= 2:
                grandparent = parent.parent
                m.cons.add(m.V[nid] >= m.ON[nid] + m.ON[pid] + m.ON[grandparent.node_id])
            elif node.stage == 1:
                m.cons.add(m.V[nid] >= m.ON[nid] + m.ON[pid])

            m.cons.add(m.OFF[nid] <= m.V[pid])
            m.cons.add(m.ON[nid] <= 1 - m.V[pid])
            m.cons.add(m.V[nid] == m.V[pid] + m.ON[nid] - m.OFF[nid])

            m.cons.add(T_ok - m.temp1[nid] <= M_high * (1 - m.z1_cold[pid] + m.z1_cold[nid]))
            m.cons.add(T_ok - m.temp2[nid] <= M_high * (1 - m.z2_cold[pid] + m.z2_cold[nid]))

    # Leaf value: single linear term per leaf, no regions, no big-M.
    vc_leaf_const = float(vc_root)
    for leaf in leaves:
        nid = leaf.node_id
        t_leaf = current_time + leaf.stage
        if t_leaf >= etas.shape[0]:
            continue
        eta = etas[t_leaf]  # (FEATURE_DIM,)

        Occ1_leaf = float(leaf.state["current_r1_occ"])
        Occ2_leaf = float(leaf.state["current_r2_occ"])
        price_leaf = float(leaf.state["current_price"])
        price_prev_leaf = float(leaf.parent.state["current_price"])

        V_leaf = (
            eta[0] * 1.0
            + eta[1] * m.temp1[nid]
            + eta[2] * m.temp2[nid]
            + eta[3] * m.hum[nid]
            + eta[4] * Occ1_leaf
            + eta[5] * Occ2_leaf
            + eta[6] * price_leaf
            + eta[7] * price_prev_leaf
            + eta[8] * vc_leaf_const
            + eta[9] * m.z1_cold[nid]
            + eta[10] * m.z2_cold[nid]
        )
        wp = _path_prob(leaf)
        obj_expr += wp * V_leaf

    m.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = 10
    solver.options["MIPGap"] = 0.02
    solver.options["OutputFlag"] = 0
    result = solver.solve(m, tee=False)
    return m, result


def select_action(state):
    current_time = state["current_time"]
    remaining = NUM_SLOTS - current_time
    num_stages = max(1, min(NUM_STAGES, remaining))

    root, all_nodes, leaves = build_scenario_tree(BF, num_stages)
    root.state = {
        "current_r1_occ": state["Occ1"],
        "current_r2_occ": state["Occ2"],
        "current_price":  state["price_t"],
        "prev_price":     state["price_previous"],
    }
    propagate_uncertainty(root, all_nodes, num_samples=150)

    m, result = _build_and_solve(state, root, all_nodes, leaves)

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
        p1_val, p2_val, v_val = 0.0, 0.0, 0

    return {
        "HeatPowerRoom1": p1_val,
        "HeatPowerRoom2": p2_val,
        "VentilationON": v_val,
    }
