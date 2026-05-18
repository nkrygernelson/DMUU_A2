# DMUU Assignment 2 — Part B: Restaurant Heating & Ventilation Control

## Project overview

Decision-making under uncertainty for a restaurant with 2 rooms, 2 heaters, and shared
ventilation. The agent controls heating power and ventilation each hour over a 10-hour
day to minimise electricity cost while respecting comfort constraints. Stochastic
uncertainty comes from electricity prices and room occupancies.

- **Python**: 3.13 (`.python-version`)
- **Package manager**: `uv` — run `uv sync` to install
- **Solver**: Gurobi (via Pyomo) — must be licensed and on PATH
- **Key deps**: `gurobipy`, `pyomo`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- **Current branch**: `Lukas_adp`

## Repository layout

```
DMUU_A2/
├── SystemCharacteristics.py     # get_fixed_data() (DO NOT MODIFY)
├── Policy_Restaurant.py         # Given template (DO NOT MODIFY)
├── main.py                      # Entry point / scratch runner
│
├── data/
│   ├── v2_PriceData.csv         # 100 days × 10 hours
│   ├── OccupancyRoom1.csv
│   ├── OccupancyRoom2.csv
│   └── Task7Occupancies.csv
│
├── processes/                   # (DO NOT MODIFY)
│   ├── PriceProcessRestaurant.py          # price_model(current, previous)
│   └── OccupancyProcessRestaurant.py      # next_occupancy_levels(r1, r2)
│
├── helper/                      # (DO NOT MODIFY)
│   └── v2_Checks.py             # check_and_sanitize_action(policy, state, PowerMax)
│
├── environment/
│   └── simulator.py             # load_experiments(), run_experiment(), evaluate()
│
├── policies/
│   ├── adp_policy.py            # Task 4: ADP — single VFA, lstsq, Polyak τ=0.5
│   ├── adp_etas.npy             # Trained weights, shape (T+1, FEATURE_DIM) = (11, 11)
│   ├── sp_policy.py             # Task 3: SP via Pyomo+Gurobi
│   ├── hybrid_policy.py         # Task 5: currently a stub
│   └── dummy_policy.py          # Task 6: always returns zero actions
│
├── task7/                       # Task 7: Lagrangian decomposition
├── submissions/                 # Hand-in wrappers
└── pdfs/                        # Assignment spec & lecture notes
```

## State dictionary (passed to every `select_action(state)`)

| key | type | description |
|---|---|---|
| `T1`, `T2` | float | Room temperatures (°C) |
| `H` | float | Humidity (%) |
| `Occ1`, `Occ2` | float | Occupancies (people) |
| `price_t`, `price_previous` | float | Current and previous electricity prices |
| `vent_counter` | int | Consecutive hours ventilation has been ON |
| `low_override_r1`, `low_override_r2` | int | Low-temp override active (0/1) |
| `current_time` | int | Hour of day (0–9) |

## Action dictionary

```python
{
    "HeatPowerRoom1": float,   # kW, clipped to [0, 3.0]
    "HeatPowerRoom2": float,   # kW, clipped to [0, 3.0]
    "VentilationON":  int,     # 0 or 1
}
```

## System parameters (`get_fixed_data()`)

| Param | Value | |
|---|---|---|
| `num_timeslots` | 10 | hours per day |
| `T1, T2, H` (init) | 21°C, 21°C, 40% | eval-time initial state |
| `heating_max_power` | 3.0 kW | per room |
| `heat_exchange_coeff` | 0.6 | °C/hr per °C inter-room diff |
| `heating_efficiency_coeff` | 1.0 | °C/hr per kW |
| `thermal_loss_coeff` | 0.1 | indoor↔outdoor heat loss |
| `heat_vent_coeff` | 0.7 | °C cooling per hour vent ON |
| `heat_occupancy_coeff` | 0.02 | °C/hr per person |
| `humidity_occupancy_coeff` | 0.18 | %/hr per person |
| `humidity_vent_coeff` | 15 | % decrease/hr vent ON |
| `ventilation_power` | 2.0 kW | electrical draw |
| `temp_min_comfort_threshold` | 18°C | low-override activates below |
| `temp_OK_threshold` | 22°C | low-override deactivates above |
| `temp_max_comfort_threshold` | 26°C | heater forced OFF above |
| `humidity_threshold` | 70% | ventilation forced ON above |
| `vent_min_up_time` | 3 hrs | min consecutive ON hours |
| `outdoor_temperature` | sinusoidal | `3*sin(2πt/10 - π/2)` |

## State dynamics

```
T1_new = T1 - 0.6*(T1-T2) - 0.1*(T1-T_out) + p1 - 0.7*V + 0.02*Occ1
T2_new = T2 - 0.6*(T2-T1) - 0.1*(T2-T_out) + p2 - 0.7*V + 0.02*Occ2
H_new  = clip(H + 0.18*(Occ1+Occ2) - 15*V, 0, 100)
```

**Overrules applied by simulator AFTER policy returns:**
- `T ≥ 26` → heater forced OFF
- `low_override = 1` → heater forced to max (3 kW)
- `H > 70` → vent forced ON
- `0 < vent_counter < 3` → vent forced ON (inertia)

**Low-override transitions:** `T_new < 18` → activate; `T_new ≥ 22` → deactivate.

## Stochastic processes

- **Price**: mean-reverting + momentum. `next = current + 0.6*(current−prev) + 0.12*(4−current) + N(0, 0.5)`. Bounds [0, 12], mean 4.
- **Occupancy**: mean-reverting with weak cross-room coupling. Room 1 mean=35 in [20, 50]; Room 2 mean=25 in [10, 30].

---

## Task 4 — ADP (the policy on this branch)

`policies/adp_policy.py` is a **single linear value-function approximator**:

- **One eta vector per stage** (no piecewise / region split). Etas shape: `(T+1, FEATURE_DIM) = (11, 11)`.
- **Original 11-feature phi**: `[1, T1, T2, H, Occ1, Occ2, price_t, price_previous, vent_counter, low_override_r1, low_override_r2]`.
- **Forward-backward fitted value iteration** with unregularised `np.linalg.lstsq` in the backward pass.
- **Polyak averaging** on eta updates: `etas ← (1−τ)·old + τ·fit` with `τ=0.5`. Damps iteration-to-iteration noise from the regression.
- **`V_next ≥ 0` clip** in `solve_bellman` — prevents the solver from exploiting negative off-sample extrapolations of the linear VFA.
- **`K=10` scenario clusters** per Bellman solve, KMeans on joint (Occ1, Occ2, price) samples.

### Training

```python
train(I=15, N=50, K=10, T=NUM_SLOTS, tau=0.5, save_path=ETAS_PATH)
```

- Forward pass rolls out N=50 trajectories from a uniformly-random initial state.
- Backward pass fits `eta_t` via lstsq on V* targets, then blends 50/50 with the previous iter's etas.
- Saves etas to `policies/adp_etas.npy` after every outer iteration.

### Inference

`select_action(state)` loads cached etas and solves a one-step Bellman MILP (`K=10`, `time_limit=10s`, `mip_gap=0.02`).

---

## Task 3 — SP (`sp_policy.py`)

Multi-stage scenario-tree MILP with `bf=3`, `num_stages = max(1, min(3, remaining))`. KMeans on joint exogenous samples. Pyomo+Gurobi, 10s time limit, 2% gap. **SP is the gold-standard reference policy** (~140 cost on 100 days).

## Task 5 — Hybrid (`hybrid_policy.py`)
Stub returning zeros.

## Task 6 — Evaluation (`environment/simulator.py`)

```python
from environment.simulator import load_experiments, evaluate
experiments = load_experiments()   # 100 fixed days from CSV
avg_cost, costs = evaluate(policy, experiments)
```

`check_and_sanitize_action` enforces a 15-second timeout and falls back to dummy `(0,0,0)` on any exception. Note: a silent IndexError inside `select_action` → dummy fallback. Always verify the loaded `etas.shape` matches what `solve_bellman` expects.

## Task 7 — Distributed (`task7/`)
Lagrangian decomposition for N=15 stores sharing P_mall=45 kW.

---

## Submission checklist

1. Verify with `check_and_sanitize_action`:
   ```python
   from helper.v2_Checks import check_and_sanitize_action
   from SystemCharacteristics import get_fixed_data
   params = get_fixed_data()
   PowerMax = {1: params['heating_max_power'], 2: params['heating_max_power']}
   action = check_and_sanitize_action(policy_module, state, PowerMax)
   ```
2. Policy must return in < 15 seconds.
3. Rename submission files: `SP_policy_[number].py`, `ADP_policy_[number].py`, `Hybrid_policy_[number].py`.

## Constraints

- **DO NOT MODIFY**: `SystemCharacteristics.py`, `Policy_Restaurant.py`, `processes/`, `helper/`.
- The simulator applies overrules **after** the policy returns — model them in any MILP to avoid infeasibility / penalties.

---

# Knowledge from extensive ablation work on this assignment

This section preserves the lessons we learned from many ablations — both on this branch
(4-region piecewise → single-VFA simplification) and on the prior `main` branch (single-VFA
+ feature reduction). The principles below are durable even though the experiment scripts
that generated them are not in the repo anymore.

## Performance summary

| policy | mean cost on 100 days |
|---|---|
| Dummy (always 0) | ~183 |
| Pristine ADP (no safeguards) | ~200 (worse than dummy) |
| ADP — best fresh training (single VFA + lstsq + Polyak, seed=2, N=50) | **~147** |
| ADP — best cached (4-region piecewise, our previous deployed file) | 166 |
| SP (multi-stage scenario-tree MPC) | ~140 |

## The seed=2 final architecture, paired comparison at N=50

| config | seed=2 100-day mean |
|---|---|
| 4-region + Ridge, no Polyak | 194.07 |
| 4-region + Ridge + Polyak (τ=0.5) | 179.93 |
| single VFA + reduced 9-feature phi + lstsq + Polyak | 146.82 |
| single VFA + original 11-feature phi + lstsq + Polyak (deployed) | **146.89** |

The deployed etas in `policies/adp_etas.npy` correspond to the last row.

## Key findings

### Single VFA beats 4-region piecewise on this problem

The 4-region piecewise architecture splits etas by `(low_override_r1, low_override_r2)`.
Under any reasonable policy, the asymmetric regions `(0,1)` and `(1,0)` are visited <2% of
the time, so their etas are fit on tiny sample sizes (often ≤5 per stage) → noisy, hurts
inference. Collapsing to a single VFA removed the underfit per-region regressions and was
~33 cost units better on seed=2.

### Polyak averaging (τ=0.5) is the single most reliable safeguard

A one-line EMA `etas ← (1−τ)·old + τ·fit` with τ=0.5 consistently improves mean cost by
7–30+ units across seeds and architectures we tested. It does three things at once:
variance reduction of noisy regression fits, damping of the policy ↔ trajectory feedback
loop, and implicit shrinkage of unidentified eta directions in collinear features. The
isolated effect was ~−28 on main-branch ablations (paired t≈6).

### Ridge vs lstsq

In well-conditioned setups (single VFA + original or reduced phi, N≥30 samples), Ridge
slightly *hurts* mean cost vs unregularised lstsq. Ridge is load-bearing only when
the regression is ill-conditioned (4-region piecewise with very few per-region samples).
For the single-VFA deployment we use lstsq.

### Feature reduction — load-bearing on main branch, no measurable effect here

The 9-feature reduced phi `[1, T_avg, H, Occ1, Occ2, price_t, price_diff, vent_counter, low_count]`
saved ~5 cost units on the main branch where single VFA + Ridge was the baseline. On this
branch's single VFA + lstsq + Polyak setup, the reduced phi gave 146.82 and the original
11-feature phi gave 146.89 on seed=2 — indistinguishable. The reduced phi remains
*defensible* on collinearity-resolution grounds (T1/T2 r=0.96, price_t/price_prev r=0.89,
low_r1/low_r2 r=0.95) and slightly improves tail behaviour, but the mean improvement
disappears when lstsq is paired with Polyak averaging.

### Continuation clipping (`V_next ≥ 0`)

Free tail safety with negligible runtime cost. Keep it on. Without it, the MILP can drive
continuation arbitrarily negative on off-sample futures and the policy occasionally takes
extreme actions (full heating + ventilation simultaneously). Kept enabled in the deployed
`solve_bellman`.

### Why ADP cannot match SP

SP scores ~140; the best ADP we found scores ~147 (one seed) / 177 (N=5 avg). The gap is a
structural limitation of the linear value-function approximator:

1. **Threshold / piecewise dependence**: V* spikes sharply as T approaches the override
   boundary (T<18 triggers forced max heating for hours). A linear approximator can only
   give a constant slope — under-penalises the danger zone, over-penalises the safe zone.
2. **Interaction effects**: optimal "preheat now to avoid override later" depends on
   `price_t × (T_ok − T1)` — a product of features. Linear VFA can't capture it.
3. **Time-varying curvature**: `η_T1` should grow steeper near end of day. With a single
   linear approximator per stage, only the constant changes between stages, not the slopes.
4. **Mean-reverting price dynamics**: optimal "is now a good time to buy" depends on a
   nonlinear expectation of future prices. Linear V can only encode linear momentum.

SP avoids all this by doing explicit multi-stage tree lookahead with no value-function
approximation. The "value function" implied by SP is just the LP/MILP optimum over the
sampled tree, which has no functional-form bias.

### Common pitfalls observed

1. **Cached etas of wrong shape** → silent `IndexError` in `solve_bellman` →
   `check_and_sanitize_action` returns dummy → eval shows the dummy score. The original
   ADP "scoring 185" turned out to be exactly this bug. Always verify
   `np.load(ETAS_PATH).shape` matches what `solve_bellman` expects.
2. **Feature/feat_expr misalignment in the MILP**: if you change `features()` you *must*
   also change the symbolic `feat_expr` in `solve_bellman` to match the new layout. The
   wiring bug between trained eta layout and MILP feature expressions silently degrades
   performance dramatically (responsible for a 40-point illusory benefit attributed to
   feature reduction in one early test).
3. **Random-state contamination across runs**: if you call `np.random.seed(seed)` before
   an `evaluate(...)` call (for reproducible eval), subsequent unseeded training calls
   inherit a fixed state. Pass an explicit `seed` per training, or re-seed after each eval.
4. **Region sample imbalance under any policy**: SP keeps the system in `(low_r1, low_r2) =
   (0, 0)` 100% of the time on 50-day eval. Dummy and ADP visit `(0, 0)` and `(1, 1)`
   ~equally with the asymmetric regions ≤2%. Don't try to fit separate value functions for
   regions you never visit.
5. **Bootstrapping noise in fitted VI**: the backward pass target
   `V*(x_t) = solve_bellman(x_t, eta_{t+1})` depends on the previous iter's eta_{t+1},
   propagating regression noise backward across stages. Polyak averaging dampens this.

### Hyperparameters that matter

| knob | value | effect |
|---|---|---|
| `N` (trajectories per outer iter) | 50 (was 30) | Bigger N → better-fit regression per iter. N=50 → ~7-point improvement over N=30 on the 4-region runs. |
| `K` (scenario clusters per Bellman) | 10 | Doesn't matter much above ~5. |
| `I` (outer iterations) | 15 | Default. Most learning happens in the first 8–10 iters. |
| `τ` (Polyak) | 0.5 | Largest reliable gain. |
| MIP gap | 0.05 (training), 0.02 (eval) | Tighter gap at eval pays for the slight speed loss. |
