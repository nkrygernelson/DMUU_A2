"""SP + Cost Function Approximation (CFA).

A hybrid of SP (Task 3) and CFA (one of the policy classes named in the
assignment's Task 5). Structurally identical to `sp_policy` — same scenario
tree, same constraints, same dynamics — with three additional **linear
penalty terms at the leaves** that shape the policy toward better tail
states without adding any tree depth or training:

  CFA penalty per leaf =
       ALPHA * (slack_cold_r1 + slack_cold_r2)   # max(0, T_target - T_leaf)
     + BETA  * H_leaf                            # discourage high humidity at leaf
     + GAMMA * (z1_cold_leaf + z2_cold_leaf)     # discourage active low-override at leaf

The slacks use the standard `u >= T_target - T_leaf; u >= 0` encoding (no new
binaries, just continuous slack vars). The objective adds the path-probability
weighted sum of these penalties over all leaves.

No training data, no value-function fitting. Three hand-tuned knobs.
"""

from dataclasses import dataclass, field
import numpy as np
import pyomo.environ as pyo

from SystemCharacteristics import get_fixed_data
from policies.sp_policy import (
    build_scenario_tree,
    propagate_uncertainty,
)

NUM_SLOTS = int(get_fixed_data()["num_timeslots"])

# Tree config (same as sp_policy)
BF = 3
NUM_STAGES = 3

# CFA coefficients (tuned by experiments/grid_search_cfa_v2.py).
# Threshold-form penalties: only active near problematic states.
ALPHA = 0.0       # cold-buffer penalty: weight on max(0, T_LOW_PEN - T_leaf)
BETA = 0.0        # humid-buffer penalty: weight on max(0, H_leaf - H_HIGH_PEN)
T_LOW_PEN = 20.0  # penalize T_leaf below this (override fires at T < 18)
H_HIGH_PEN = 60.0 # penalize H_leaf above this (override fires at H > 70)


def _path_prob(node):
    p = 1.0
    while node.parent is not None:
        p *= node.prob
        node = node.parent
    return p


def _build_and_solve(state, root, all_nodes, leaves,
                     alpha=None, beta=None,
                     t_low_pen=None, h_high_pen=None):
    """SP MILP plus CFA leaf penalties."""
    if alpha is None: alpha = ALPHA
    if beta is None:  beta = BETA
    if t_low_pen is None: t_low_pen = T_LOW_PEN
    if h_high_pen is None: h_high_pen = H_HIGH_PEN

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
    P_max = fixed_data["heating_max_power"]
    T_circ = -3
    M_low = T_low - T_circ
    M_high = T_ok - T_circ
    M_hum = 100 - H_high

    m = pyo.ConcreteModel()
    nids = [n.node_id for n in all_nodes]
    m.NODES = pyo.Set(initialize=nids)

    m.p1 = pyo.Var(m.NODES, bounds=(0, P_max))
    m.p2 = pyo.Var(m.NODES, bounds=(0, P_max))
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
            m.cons.add(m.p1[nid] <= P_max * (1 - m.z1_hot[nid]))
            m.cons.add(m.temp2[nid] - T_high <= M_high * m.z2_hot[nid])
            m.cons.add(m.p2[nid] <= P_max * (1 - m.z2_hot[nid]))

            m.cons.add(T_low - m.temp1[nid] <= M_low * m.z1_cold[nid])
            m.cons.add(m.p1[nid] >= P_max * m.z1_cold[nid])
            m.cons.add(T_low - m.temp2[nid] <= M_low * m.z2_cold[nid])
            m.cons.add(m.p2[nid] >= P_max * m.z2_cold[nid])

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

    # ----- CFA penalty terms at leaves (threshold form) -----
    # Penalize only states close to triggering an override:
    #   cold: max(0, T_LOW_PEN - T_leaf)      (override at T < 18)
    #   humid: max(0, H_leaf - H_HIGH_PEN)    (override at H > 70)
    leaf_ids = [leaf.node_id for leaf in leaves]
    m.LEAVES = pyo.Set(initialize=leaf_ids)
    m.slack_cold_r1 = pyo.Var(m.LEAVES, within=pyo.NonNegativeReals)
    m.slack_cold_r2 = pyo.Var(m.LEAVES, within=pyo.NonNegativeReals)
    m.slack_humid = pyo.Var(m.LEAVES, within=pyo.NonNegativeReals)

    for leaf in leaves:
        nid = leaf.node_id
        m.cons.add(m.slack_cold_r1[nid] >= t_low_pen - m.temp1[nid])
        m.cons.add(m.slack_cold_r2[nid] >= t_low_pen - m.temp2[nid])
        m.cons.add(m.slack_humid[nid] >= m.hum[nid] - h_high_pen)

        wp = _path_prob(leaf)
        obj_expr += wp * alpha * (m.slack_cold_r1[nid] + m.slack_cold_r2[nid])
        obj_expr += wp * beta * m.slack_humid[nid]

    m.objective = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = 10
    solver.options["MIPGap"] = 0.02
    solver.options["OutputFlag"] = 0
    result = solver.solve(m, tee=False)
    return m, result


def select_action(state, alpha=None, beta=None, t_low_pen=None, h_high_pen=None):
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

    m, result = _build_and_solve(state, root, all_nodes, leaves,
                                  alpha, beta, t_low_pen, h_high_pen)

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
