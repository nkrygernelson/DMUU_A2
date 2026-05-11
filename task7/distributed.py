"""Task 7: Distributed Decision-Making for a 15-store mall.

Each store has 2 rooms and 2 heaters and shares a mall-wide power cap
`sum_n p_{n,t} <= P_mall`. We minimize sum_{n,r,t} w_n (T_{n,r,t} - T_ref)^2
with w_n = n+1 by dual decomposition: each store solves a local QP given a
per-timeslot price `lambda_t`, the master updates lambda by a projected
subgradient on the aggregate power residual.

Public entry points:
- fetch_data()              fixed problem data (do not modify)
- load_occupancies(...)     read Task7Occupancies.csv
- solve_centralized(...)    one big Pyomo QP for the reference optimum
- run_distributed(...)      dual decomposition; returns history dict
"""

import os
import csv
import numpy as np
import pyomo.environ as pyo


N_STORES = 15
ROOMS = (1, 2)


def fetch_data():
    """
    Returns the fixed data for Task 7.
    THIS CODE SHOULD NOT BE CHANGED BY STUDENTS.
    """

    num_timeslots = 10

    return {
        'num_timeslots': num_timeslots,
        'P_mall': 45,
        'Temperature_reference': 21,
        'initial_temperature': 21.0,
        'heating_max_power': 3.0,
        'heat_exchange_coeff': 0.6,
        'heating_efficiency_coeff': 1.0,
        'thermal_loss_coeff': 0.1,
        'heat_vent_coeff': 0.7,
        'heat_occupancy_coeff': 0.02,
        'outdoor_temperature': [
            3 * np.sin(2 * np.pi * t / num_timeslots - np.pi / 2)
            for t in range(num_timeslots)
        ],
    }


def load_occupancies(path=None):
    """Load Task 7 occupancies as a (2, T) array indexed [room-1, t]."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "data", "Task7Occupancies.csv")
    rows = []
    with open(path, "r") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                rows.append([float(x) for x in row if x.strip() != "" and not x.strip().lower().startswith("room")])
            except ValueError:
                # header line "0,1,2,..." — keep only numeric entries
                rows.append([float(x) for x in row if x.strip() != "" and x.strip().lstrip("-").replace(".", "", 1).isdigit()])
    # Drop the integer column-index header row (all values < num_timeslots and
    # exactly num_timeslots of them — the file's first row is 0..T-1).
    numeric_rows = [r for r in rows if r]
    # Assume first row is the header (0..T-1), next two rows are the rooms.
    data_rows = numeric_rows[1:3] if len(numeric_rows) >= 3 else numeric_rows[:2]
    occ = np.array(data_rows, dtype=float)
    assert occ.shape[0] == 2, f"expected 2 rooms, got shape {occ.shape}"
    return occ  # shape (2, T)


def _weights(n_stores=N_STORES):
    return np.array([n + 1 for n in range(1, n_stores + 1)], dtype=float)


def _build_store_model(n_idx, w_n, occ, params):
    """Build a Pyomo model for one store with placeholder dual prices.

    Returns (model, lambda_param) where lambda_param[t] can be re-set each
    iteration via lambda_param[t].set_value(...).
    """
    T = params['num_timeslots']
    xi_exh = params['heat_exchange_coeff']
    xi_loss = params['thermal_loss_coeff']
    xi_conv = params['heating_efficiency_coeff']
    xi_cool = params['heat_vent_coeff']
    xi_occ = params['heat_occupancy_coeff']
    P_max = params['heating_max_power']
    T_init = params['initial_temperature']
    T_ref = params['Temperature_reference']
    T_out = params['outdoor_temperature']

    m = pyo.ConcreteModel(name=f"store_{n_idx}")
    m.T = pyo.RangeSet(0, T - 1)
    m.R = pyo.Set(initialize=[1, 2])

    m.p = pyo.Var(m.R, m.T, bounds=(0.0, P_max))
    m.temp = pyo.Var(m.R, m.T, bounds=(-50.0, 80.0))
    m.lam = pyo.Param(m.T, initialize={t: 0.0 for t in range(T)}, mutable=True)

    m.cons = pyo.ConstraintList()
    # Initial temperatures
    for r in m.R:
        m.cons.add(m.temp[r, 0] == T_init)
    # Dynamics, t >= 1
    for t in range(1, T):
        for r in m.R:
            other = 2 if r == 1 else 1
            m.cons.add(
                m.temp[r, t] == m.temp[r, t - 1]
                - xi_exh * (m.temp[r, t - 1] - m.temp[other, t - 1])
                - xi_loss * (m.temp[r, t - 1] - T_out[t - 1])
                + xi_conv * m.p[r, t - 1]
                - xi_cool
                + xi_occ * occ[r - 1, t - 1]
            )

    def obj_rule(mm):
        tracking = sum(w_n * (mm.temp[r, t] - T_ref) ** 2
                       for r in mm.R for t in mm.T)
        dual = sum(mm.lam[t] * (mm.p[1, t] + mm.p[2, t]) for t in mm.T)
        return tracking + dual

    m.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    return m


def _extract(m, T):
    """Extract per-room (p, temp) arrays."""
    p = np.zeros((2, T))
    tt = np.zeros((2, T))
    for r in (1, 2):
        for t in range(T):
            pv = pyo.value(m.p[r, t], exception=False)
            tv = pyo.value(m.temp[r, t], exception=False)
            p[r - 1, t] = 0.0 if pv is None else float(pv)
            tt[r - 1, t] = 0.0 if tv is None else float(tv)
    return p, tt


def solve_centralized(params=None, occ=None, n_stores=N_STORES, solver_opts=None):
    """One big QP with all stores and the coupling constraint. Returns dict."""
    if params is None:
        params = fetch_data()
    if occ is None:
        occ = load_occupancies()
    T = params['num_timeslots']
    P_mall = params['P_mall']
    P_max = params['heating_max_power']
    T_init = params['initial_temperature']
    T_ref = params['Temperature_reference']
    T_out = params['outdoor_temperature']
    xi_exh = params['heat_exchange_coeff']
    xi_loss = params['thermal_loss_coeff']
    xi_conv = params['heating_efficiency_coeff']
    xi_cool = params['heat_vent_coeff']
    xi_occ = params['heat_occupancy_coeff']
    weights = _weights(n_stores)

    m = pyo.ConcreteModel(name="centralized")
    m.N = pyo.RangeSet(1, n_stores)
    m.R = pyo.Set(initialize=[1, 2])
    m.T = pyo.RangeSet(0, T - 1)

    m.p = pyo.Var(m.N, m.R, m.T, bounds=(0.0, P_max))
    m.temp = pyo.Var(m.N, m.R, m.T, bounds=(-50.0, 80.0))
    m.cons = pyo.ConstraintList()

    for n in m.N:
        for r in m.R:
            m.cons.add(m.temp[n, r, 0] == T_init)
        for t in range(1, T):
            for r in m.R:
                other = 2 if r == 1 else 1
                m.cons.add(
                    m.temp[n, r, t] == m.temp[n, r, t - 1]
                    - xi_exh * (m.temp[n, r, t - 1] - m.temp[n, other, t - 1])
                    - xi_loss * (m.temp[n, r, t - 1] - T_out[t - 1])
                    + xi_conv * m.p[n, r, t - 1]
                    - xi_cool
                    + xi_occ * occ[r - 1, t - 1]
                )
    for t in m.T:
        m.cons.add(sum(m.p[n, r, t] for n in m.N for r in m.R) <= P_mall)

    def obj_rule(mm):
        return sum(weights[n - 1] * (mm.temp[n, r, t] - T_ref) ** 2
                   for n in mm.N for r in mm.R for t in mm.T)

    m.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    solver = pyo.SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    if solver_opts:
        for k, v in solver_opts.items():
            solver.options[k] = v
    solver.solve(m, tee=False)

    p = np.zeros((n_stores, 2, T))
    tt = np.zeros((n_stores, 2, T))
    for n in range(1, n_stores + 1):
        for r in (1, 2):
            for t in range(T):
                p[n - 1, r - 1, t] = pyo.value(m.p[n, r, t])
                tt[n - 1, r - 1, t] = pyo.value(m.temp[n, r, t])

    obj = sum(weights[n] * (tt[n] - T_ref) ** 2 for n in range(n_stores)).sum()
    return {
        "objective": float(obj),
        "p": p,
        "temp": tt,
        "weights": weights,
        "params": params,
        "occ": occ,
    }


def run_distributed(alpha, num_iters=100, adaptive=False, alpha0=5.0,
                    params=None, occ=None, n_stores=N_STORES, verbose=False):
    """Dual decomposition with projected subgradient on the mall power cap.

    Parameters
    ----------
    alpha : float
        Fixed step size (ignored when adaptive=True).
    adaptive : bool
        If True, use alpha_k = alpha0 / (1 + k).
    num_iters : int
        Number of master iterations.

    Returns
    -------
    dict with keys:
        lambdas         (K+1, T)  multipliers after each update
        residuals       (K, T)    sum_n p_{n,t} - P_mall at each iter
        objective       (K,)      primal objective per iteration
        p_final         (N, 2, T)
        temp_final      (N, 2, T)
        weights         (N,)
        step_sizes      (K,)
    """
    if params is None:
        params = fetch_data()
    if occ is None:
        occ = load_occupancies()
    T = params['num_timeslots']
    P_mall = params['P_mall']
    T_ref = params['Temperature_reference']
    weights = _weights(n_stores)

    # Build one local model per store (re-used each iteration).
    models = [_build_store_model(n + 1, weights[n], occ, params) for n in range(n_stores)]

    solver = pyo.SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0

    lambdas = np.zeros((num_iters + 1, T))
    residuals = np.zeros((num_iters, T))
    objective = np.zeros(num_iters)
    step_sizes = np.zeros(num_iters)
    p_iter = np.zeros((n_stores, 2, T))
    temp_iter = np.zeros((n_stores, 2, T))

    for k in range(num_iters):
        step = alpha0 / (1 + k) if adaptive else alpha
        step_sizes[k] = step

        # Solve every local QP with the current multipliers.
        for n in range(n_stores):
            mdl = models[n]
            for t in range(T):
                mdl.lam[t].set_value(float(lambdas[k, t]))
            solver.solve(mdl, tee=False)
            p_iter[n], temp_iter[n] = _extract(mdl, T)

        # Aggregate residual and primal objective.
        residuals[k] = p_iter.sum(axis=(0, 1)) - P_mall
        objective[k] = float(
            sum(weights[n] * (temp_iter[n] - T_ref) ** 2 for n in range(n_stores)).sum()
        )

        # Projected subgradient update.
        lambdas[k + 1] = np.maximum(0.0, lambdas[k] + step * residuals[k])

        if verbose and (k % 10 == 0 or k == num_iters - 1):
            print(f"  iter {k:3d}  step={step:.4f}  obj={objective[k]:9.3f}  "
                  f"max_resid={residuals[k].max():+.3f}  "
                  f"max_lam={lambdas[k+1].max():.3f}")

    return {
        "lambdas": lambdas,
        "residuals": residuals,
        "objective": objective,
        "p_final": p_iter,
        "temp_final": temp_iter,
        "weights": weights,
        "step_sizes": step_sizes,
        "params": params,
        "occ": occ,
        "alpha": alpha,
        "adaptive": adaptive,
        "alpha0": alpha0 if adaptive else None,
    }


def solve_distributed():
    """Wrapper around run_distributed for backwards-compatibility with the
    original stub signature; uses alpha=0.1, 100 iterations."""
    return run_distributed(alpha=0.1, num_iters=100)
