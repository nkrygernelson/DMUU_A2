"""Deep ADP: MLP value-function approximation with forward-backward fitted VI.

Replaces the linear VFA of adp_policy with a single PyTorch MLP V_theta(state).
Time-to-go is part of the input so one network covers the whole horizon.
Action selection at each step is a constrained grid search over (p1, p2, V)
minimizing  c(x, u) + E[V_theta(x_{t+1}) | x, u]  with a Monte Carlo
expectation under common random scenarios.

Train offline:
    uv run python -m policies.deep_adp_policy
"""

import os

import numpy as np
import torch
import torch.nn as nn

from processes.PriceProcessRestaurant import price_model
from processes.OccupancyProcessRestaurant import next_occupancy_levels
from SystemCharacteristics import get_fixed_data


FIXED = get_fixed_data()
NUM_SLOTS = int(FIXED["num_timeslots"])
MODEL_PATH = os.path.join(os.path.dirname(__file__), "deep_adp_model.pt")

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

FEAT_DIM = 12
DEVICE = torch.device("cpu")


class VNet(nn.Module):
    """Configurable MLP. hidden_sizes=(64, 64) for the default deep ADP;
    hidden_sizes=(32,) for the MILP-friendly small variant."""

    def __init__(self, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        prev = FEAT_DIM
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.hidden_sizes = tuple(hidden_sizes)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_batch(states):
    arr = np.empty((len(states), FEAT_DIM), dtype=np.float32)
    for i, s in enumerate(states):
        arr[i, 0] = s["T1"] / 30.0
        arr[i, 1] = s["T2"] / 30.0
        arr[i, 2] = s["H"] / 100.0
        arr[i, 3] = s["Occ1"] / 50.0
        arr[i, 4] = s["Occ2"] / 30.0
        arr[i, 5] = s["price_t"] / 12.0
        arr[i, 6] = s["price_previous"] / 12.0
        arr[i, 7] = float(s["vent_counter"]) / float(VENT_MIN)
        arr[i, 8] = float(bool(s["low_override_r1"]))
        arr[i, 9] = float(bool(s["low_override_r2"]))
        arr[i, 10] = float(s["current_time"]) / float(NUM_SLOTS)
        arr[i, 11] = 1.0 - float(s["current_time"]) / float(NUM_SLOTS)
    return arr


def predict(V_theta, states):
    """Vectorized V_θ over a list of state dicts. Handles terminal (t≥T) as 0."""
    if not states:
        return np.zeros(0, dtype=np.float32)
    if V_theta is None:
        return np.zeros(len(states), dtype=np.float32)
    X = encode_batch(states)
    Xt = torch.from_numpy(X).to(DEVICE)
    V_theta.eval()
    with torch.no_grad():
        y = V_theta(Xt).cpu().numpy()
    return y


def feasible_action_grid(state, n_levels=7):
    """Return list of (p1, p2, V) tuples consistent with the simulator's overrules."""
    T1 = state["T1"]; T2 = state["T2"]; H = state["H"]
    lo1 = int(bool(state["low_override_r1"]))
    lo2 = int(bool(state["low_override_r2"]))
    vc = int(state["vent_counter"])

    if lo1:
        p1_grid = [P_MAX]
    elif T1 >= T_HIGH:
        p1_grid = [0.0]
    else:
        p1_grid = list(np.linspace(0.0, P_MAX, n_levels))

    if lo2:
        p2_grid = [P_MAX]
    elif T2 >= T_HIGH:
        p2_grid = [0.0]
    else:
        p2_grid = list(np.linspace(0.0, P_MAX, n_levels))

    if H > H_HIGH or (0 < vc < VENT_MIN):
        V_grid = [1]
    else:
        V_grid = [0, 1]

    return [(float(p1), float(p2), int(V))
            for p1 in p1_grid for p2 in p2_grid for V in V_grid]


def advance(state, action, exog):
    """Deterministic dynamics matching environment/simulator.py exactly."""
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
        "T1": float(T1n),
        "T2": float(T2n),
        "H": Hn,
        "Occ1": float(exog["Occ1"]),
        "Occ2": float(exog["Occ2"]),
        "price_t": float(exog["price"]),
        "price_previous": float(state["price_t"]),
        "vent_counter": int(vc_n),
        "low_override_r1": int(lo1_n),
        "low_override_r2": int(lo2_n),
        "current_time": t + 1,
    }


def sample_exog(state, n=1):
    out = []
    for _ in range(n):
        r1, r2 = next_occupancy_levels(state["Occ1"], state["Occ2"])
        p = price_model(state["price_t"], state["price_previous"])
        out.append({"Occ1": r1, "Occ2": r2, "price": p})
    return out


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


def one_step_value(state, V_theta, M=20, n_levels=7):
    """min_u  c(x,u) + (1/M) Σ_m V_θ(x_{t+1}(u, ω_m)).

    Common random numbers across u for variance reduction.
    Returns (best_action, V_star).
    """
    t = int(state["current_time"])
    exogs = sample_exog(state, n=M)
    actions = feasible_action_grid(state, n_levels=n_levels)

    immediate = np.empty(len(actions), dtype=np.float32)
    next_states = []
    for i, u in enumerate(actions):
        immediate[i] = state["price_t"] * (u[0] + u[1] + P_VENT * u[2])
        for w in exogs:
            next_states.append(advance(state, u, w))

    if t + 1 >= NUM_SLOTS:
        future = np.zeros(len(actions) * M, dtype=np.float32)
    else:
        future = predict(V_theta, next_states)

    future = future.reshape(len(actions), M).mean(axis=1)
    total = immediate + future
    best = int(np.argmin(total))
    return actions[best], float(total[best])


def fit_V(X, y, hidden_sizes=(64, 64), lr=1e-3, epochs=300, batch_size=256, prior=None, verbose=False):
    """Fit a fresh VNet on (X, y) with MSE. `prior` warm-starts from an existing model."""
    model = VNet(hidden_sizes=hidden_sizes).to(DEVICE)
    if prior is not None and prior.hidden_sizes == tuple(hidden_sizes):
        model.load_state_dict(prior.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y.astype(np.float32)).to(DEVICE)
    n = Xt.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.shape[0]
        if verbose and (ep + 1) % 50 == 0:
            print(f"    ep {ep+1}/{epochs}  mse={total_loss/n:.3f}")
    return model


def train(I=8, N=25, M=20, n_levels=7, save_path=MODEL_PATH, verbose=True, seed=0,
          eps0=0.4, eps_decay=0.07, eps_min=0.05, hidden_sizes=(64, 64), lr=1e-3, epochs=300,
          warm_start=True):
    """Forward-backward fitted value iteration with PyTorch MLP value function.

    For each outer iteration:
      forward — roll out N trajectories with ε-greedy(V_theta_current);
      backward — for every visited (t, state) compute one-step target V_t,
                 refit MLP across all (t, state, target) samples.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    V_theta = None
    for i in range(I):
        eps = max(eps0 - eps_decay * i, eps_min)
        if verbose:
            print(f"=== iter {i + 1}/{I}  (ε={eps:.2f}) ===")

        trajectories = []
        for _ in range(N):
            s = sample_init_state()
            traj = [s]
            for _t in range(NUM_SLOTS):
                if rng.random() < eps:
                    cand = feasible_action_grid(s, n_levels=n_levels)
                    u = cand[int(rng.integers(0, len(cand)))]
                else:
                    u, _ = one_step_value(s, V_theta, M=M, n_levels=n_levels)
                w = sample_exog(s, 1)[0]
                s = advance(s, u, w)
                traj.append(s)
            trajectories.append(traj)
        if verbose:
            print(f"  forward: {N} trajectories collected")

        states_all = []
        targets_all = []
        for t_back in range(NUM_SLOTS - 1, -1, -1):
            for traj in trajectories:
                s_t = traj[t_back]
                _, V_star = one_step_value(s_t, V_theta, M=M, n_levels=n_levels)
                states_all.append(s_t)
                targets_all.append(V_star)

        X = encode_batch(states_all)
        y = np.asarray(targets_all, dtype=np.float32)
        prior = V_theta if warm_start else None
        V_theta = fit_V(X, y, hidden_sizes=hidden_sizes, lr=lr, epochs=epochs, prior=prior, verbose=verbose)

        if verbose:
            yhat = predict(V_theta, states_all)
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
            r2 = 1.0 - ss_res / ss_tot
            print(f"  backward: n={len(y)}, y mean={y.mean():.2f}, sd={y.std():.2f}, R²={r2:.3f}")

        torch.save(V_theta.state_dict(), save_path)
        if verbose:
            print(f"  saved V_theta → {save_path}")

    return V_theta


_CACHED_V = None


def _load_V():
    global _CACHED_V
    if _CACHED_V is None:
        if os.path.exists(MODEL_PATH):
            _CACHED_V = load_vnet(MODEL_PATH)
        else:
            print(f"No trained model at {MODEL_PATH}; using zero V.")
            _CACHED_V = None
    return _CACHED_V


def load_vnet(path, hidden_sizes=(64, 64)):
    model = VNet(hidden_sizes=hidden_sizes).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def select_action(state):
    V_theta = _load_V()
    u, _ = one_step_value(state, V_theta, M=20, n_levels=7)
    return {
        "HeatPowerRoom1": u[0],
        "HeatPowerRoom2": u[1],
        "VentilationON": u[2],
    }


if __name__ == "__main__":
    train(I=8, N=25, M=20, n_levels=7)
