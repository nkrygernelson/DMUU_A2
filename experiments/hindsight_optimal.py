"""Hindsight-optimal benchmark: best possible cost per day with full info.

For each day's pre-drawn (prices, occupancy_r1, occupancy_r2) trajectory,
solve a single deterministic 10-stage MILP that mirrors the simulator
dynamics + all overrules + ventilation inertia. The objective value is the
absolute lower bound — no online policy can do better, because no online
policy has access to the realized future.

This bounds the room for improvement over SP (139.75).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.simulator import load_experiments
from SystemCharacteristics import get_fixed_data


def solve_day(prices, occ1, occ2):
    """Deterministic 10-stage MILP with realized prices/occupancies.

    Returns (optimal_cost, actions_list). Stage indices match simulator: t=0..T-1
    decisions, T+1 state vars (state at t propagates to state at t+1).
    """
    fixed = get_fixed_data()
    T = int(fixed["num_timeslots"])
    assert len(prices) == T and len(occ1) == T and len(occ2) == T

    xi_exh = fixed["heat_exchange_coeff"]
    xi_loss = fixed["thermal_loss_coeff"]
    xi_conv = fixed["heating_efficiency_coeff"]
    xi_cool = fixed["heat_vent_coeff"]
    xi_occ = fixed["heat_occupancy_coeff"]
    eta_occ_h = fixed["humidity_occupancy_coeff"]
    eta_vent_h = fixed["humidity_vent_coeff"]
    p_vent = fixed["ventilation_power"]
    T_low = fixed["temp_min_comfort_threshold"]
    T_high = fixed["temp_max_comfort_threshold"]
    T_ok = fixed["temp_OK_threshold"]
    T_out = fixed["outdoor_temperature"]
    H_high = fixed["humidity_threshold"]
    P_max = fixed["heating_max_power"]
    T_circ = -3
    M_low = T_low - T_circ
    M_high = T_ok - T_circ
    M_hum = 100 - H_high

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T)  # states at t=0..T; decisions at t=0..T-1

    # Decision vars (only at t = 0..T-1)
    m.p1 = pyo.Var(range(T), bounds=(0, P_max))
    m.p2 = pyo.Var(range(T), bounds=(0, P_max))
    m.V = pyo.Var(range(T), within=pyo.Binary)
    m.ON = pyo.Var(range(T), within=pyo.Binary)
    m.OFF = pyo.Var(range(T), within=pyo.Binary)
    m.z1_hot = pyo.Var(range(T), within=pyo.Binary)
    m.z2_hot = pyo.Var(range(T), within=pyo.Binary)

    # State vars (t=0..T)
    m.temp1 = pyo.Var(m.T, bounds=(T_circ, 2 * T_high))
    m.temp2 = pyo.Var(m.T, bounds=(T_circ, 2 * T_high))
    m.hum = pyo.Var(m.T, bounds=(0, 100))
    m.z1_cold = pyo.Var(m.T, within=pyo.Binary)
    m.z2_cold = pyo.Var(m.T, within=pyo.Binary)

    m.cons = pyo.ConstraintList()

    # Initial state from SystemCharacteristics.get_fixed_data()
    m.temp1[0].fix(fixed["T1"])  # 21.0
    m.temp2[0].fix(fixed["T2"])  # 21.0
    m.hum[0].fix(fixed["H"])     # 40.0
    m.z1_cold[0].fix(fixed["low_override_r1"])
    m.z2_cold[0].fix(fixed["low_override_r2"])

    for t in range(T):
        T_out_t = T_out[min(t, len(T_out) - 1)]

        # High-temp override
        m.cons.add(m.temp1[t] - T_high <= M_high * m.z1_hot[t])
        m.cons.add(m.p1[t] <= P_max * (1 - m.z1_hot[t]))
        m.cons.add(m.temp2[t] - T_high <= M_high * m.z2_hot[t])
        m.cons.add(m.p2[t] <= P_max * (1 - m.z2_hot[t]))

        # Low-temp override (hysteresis state z_cold updated each step)
        m.cons.add(T_low - m.temp1[t] <= M_low * m.z1_cold[t])
        m.cons.add(m.p1[t] >= P_max * m.z1_cold[t])
        m.cons.add(T_low - m.temp2[t] <= M_low * m.z2_cold[t])
        m.cons.add(m.p2[t] >= P_max * m.z2_cold[t])

        # Humidity override
        m.cons.add(m.hum[t] - H_high <= M_hum * m.V[t])

        # Vent inertia (3-hour min-up)
        m.cons.add(m.ON[t] + m.OFF[t] <= 1)
        if t == 0:
            m.cons.add(m.V[t] == m.ON[t])
            m.OFF[t].fix(0)
        else:
            m.cons.add(m.OFF[t] <= m.V[t - 1])
            m.cons.add(m.ON[t] <= 1 - m.V[t - 1])
            m.cons.add(m.V[t] == m.V[t - 1] + m.ON[t] - m.OFF[t])
            if t == 1:
                m.cons.add(m.V[t] >= m.ON[t] + m.ON[t - 1])
            else:
                m.cons.add(m.V[t] >= m.ON[t] + m.ON[t - 1] + m.ON[t - 2])

        # Dynamics: state at t+1 from state at t and decision at t
        m.cons.add(
            m.temp1[t + 1] == m.temp1[t]
            - xi_exh * (m.temp1[t] - m.temp2[t])
            - xi_loss * (m.temp1[t] - T_out_t)
            + xi_conv * m.p1[t]
            - xi_cool * m.V[t]
            + xi_occ * occ1[t]
        )
        m.cons.add(
            m.temp2[t + 1] == m.temp2[t]
            - xi_exh * (m.temp2[t] - m.temp1[t])
            - xi_loss * (m.temp2[t] - T_out_t)
            + xi_conv * m.p2[t]
            - xi_cool * m.V[t]
            + xi_occ * occ2[t]
        )
        # Relaxed: simulator clips H to [0, 100]. The natural value can go
        # negative when ventilation is on with low occupancy; simulator just
        # clips it to 0. Using >= here (instead of ==) lets the MILP exploit
        # the same clipping. Solver will tighten to equality when natural >= 0
        # because higher humidity → more forced ventilation later → higher cost.
        m.cons.add(
            m.hum[t + 1] >= m.hum[t]
            + eta_occ_h * (occ1[t] + occ2[t])
            - eta_vent_h * m.V[t]
        )

        # Low-temp hysteresis transition for the next stage
        if t + 1 <= T:
            m.cons.add(T_ok - m.temp1[t + 1] <= M_high * (1 - m.z1_cold[t] + m.z1_cold[t + 1]))
            m.cons.add(T_ok - m.temp2[t + 1] <= M_high * (1 - m.z2_cold[t] + m.z2_cold[t + 1]))

    obj = sum(prices[t] * (m.p1[t] + m.p2[t] + p_vent * m.V[t])
              for t in range(T))
    m.objective = pyo.Objective(expr=obj, sense=pyo.minimize)

    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = 30
    solver.options["MIPGap"] = 0.005
    solver.options["OutputFlag"] = 0
    result = solver.solve(m, tee=False)
    ok = (result.solver.status == pyo.SolverStatus.ok and
          result.solver.termination_condition in
              (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible))
    if not ok:
        return float("nan"), None
    cost = float(pyo.value(m.objective))
    actions = [
        {"HeatPowerRoom1": float(pyo.value(m.p1[t])),
         "HeatPowerRoom2": float(pyo.value(m.p2[t])),
         "VentilationON": int(round(pyo.value(m.V[t])))}
        for t in range(T)
    ]
    return cost, actions


if __name__ == "__main__":
    experiments = load_experiments()
    print(f"Loaded {len(experiments)} days")

    costs = []
    t_start = time.time()
    for d, day in enumerate(experiments):
        prices = day["prices"]
        occ1 = day["occupancy_r1"]
        occ2 = day["occupancy_r2"]
        cost, _ = solve_day(prices, occ1, occ2)
        costs.append(cost)
        if (d + 1) % 10 == 0 or d == 0:
            elapsed = time.time() - t_start
            print(f"  day {d+1}/100  cost={cost:.2f}  elapsed={elapsed:.0f}s")

    costs = np.array(costs)
    print()
    print(f"Hindsight-optimal 100-day mean: {costs.mean():.2f}")
    print(f"  std={costs.std():.2f}  min={costs.min():.1f}  max={costs.max():.1f}")
    print(f"  vs SP-Gurobi 139.75:  SP gap = {139.75 - costs.mean():.2f}")

    # Save raw per-day costs for plotting
    out = ROOT / "plots" / "hindsight_optimal_per_day.npz"
    np.savez(out, costs=costs)
    print(f"  saved per-day costs to {out}")
