# DMUU Assignment 2 — Part B

Restaurant heating & ventilation control under uncertainty.

---

## Project structure

```
DMUU_A2/
├── main.py                        # Entry point / scratch runner
├── pyproject.toml                 # uv project (gurobipy, pyomo, numpy)
├── Policy_Restaurant.py           # Given template — DO NOT MODIFY
│
├── data/                          # Historical time-series (100 days)
│   ├── v2_PriceData.csv           # Electricity prices [hours × days]
│   ├── OccupancyRoom1.csv
│   ├── OccupancyRoom2.csv
│   └── Task7Occupancies.csv       # Occupancy used in Task 7
│
├── pdfs/                          # Assignment spec and notation sheet
│
├── processes/                     # Stochastic process models (given)
│   ├── PriceProcessRestaurant.py  # price_model(...)
│   └── OccupancyProcessRestaurant.py
│
├── helper/                        # Given utilities (do not modify)
│   ├── v2_SystemCharacteristics.py  # get_fixed_data() — all system params
│   └── v2_Checks.py               # check_and_sanitize_action() — run before handing in
│
├── policies/                      # One file per task — implement here
│   ├── sp_policy.py               # Task 3: Stochastic Programming
│   ├── adp_policy.py              # Task 4: Approximate Dynamic Programming
│   ├── hybrid_policy.py           # Task 5: Hybrid policy
│   └── dummy_policy.py            # Task 6: dummy (always returns 0)
│
├── environment/
│   └── simulator.py               # Task 6: simulation environment
│
├── task7/
│   └── distributed.py             # Task 7: distributed optimisation (fetch_data + algorithm)
│
└── submissions/                   # Final hand-in files
    ├── SP_policy_group.py         # ← rename to SP_policy_[number].py
    ├── ADP_policy_group.py
    └── Hybrid_policy_group.py
```

`taskA/` is excluded from git — it contains old Part A reference code.

---

## Setup

```bash
uv sync          # installs dependencies into .venv
```

Gurobi requires a valid licence. The solver is invoked via Pyomo — make sure `gurobi` is on your PATH or the licence is activated before running any policy.

---

## How policies work

Every policy exposes a single function that the environment calls once per hour:

```python
def select_action(state: dict) -> dict:
    ...
    return {
        "HeatPowerRoom1": float,   # kW, clipped to [0, heating_max_power]
        "HeatPowerRoom2": float,
        "VentilationON":  int,     # 0 or 1
    }
```

The `state` dict the environment passes in:

| Key | Description |
|-----|-------------|
| `T1`, `T2` | Current room temperatures (°C) |
| `H` | Current humidity (%) |
| `Occ1`, `Occ2` | Current room occupancies (people) |
| `price_t` | Current electricity price |
| `price_previous` | Price at the previous hour |
| `vent_counter` | Consecutive hours ventilation has been ON |
| `low_override_r1/r2` | Whether the low-temp overrule is active |
| `current_time` | Hour of the day (0–9) |

System parameters (thresholds, coefficients, etc.) come from:

```python
from helper.v2_SystemCharacteristics import get_fixed_data
params = get_fixed_data()
```

Stochastic process models for sampling future prices / occupancies:

```python
from processes.PriceProcessRestaurant import price_model
from processes.OccupancyProcessRestaurant import occupancy_model
```

---

## Workflow per task

### Task 3 — Stochastic Programming (`policies/sp_policy.py`)

Build a multi-stage scenario tree with `build_scenario_tree(bf, num_stages)`, sample realisations from the process models, solve the SP with Pyomo + Gurobi, and return the here-and-now action.

### Task 4 — ADP (`policies/adp_policy.py`)

Train a linear value-function approximation offline, then use it inside `select_action` to make greedy decisions.

### Task 5 — Hybrid (`policies/hybrid_policy.py`)

Any combination of SP, ADP, cost-function approximation, or rule-based logic.

### Task 6 — Evaluation (`environment/simulator.py`)

`run_experiment(policy, day_data)` simulates one day; `evaluate(policy, experiments)` aggregates over 100 days. Compare dummy, hindsight-optimal, SP, ADP, and hybrid policies.

### Task 7 — Distributed (`task7/distributed.py`)

Lagrangian decomposition over N = 15 stores sharing a mall power cap. Data via `fetch_data()`.

---

## Before handing in

Run the assignment's checker on your policy to catch timing, feasibility, and clipping issues:

```python
from helper.v2_Checks import check_and_sanitize_action
from helper.v2_SystemCharacteristics import get_fixed_data

params = get_fixed_data()
PowerMax = {1: params['heating_max_power'], 2: params['heating_max_power']}

import policies.sp_policy as policy
action = check_and_sanitize_action(policy, state, PowerMax)
```

The graders run this same check. If your policy takes > 15 s or crashes, the dummy action is used instead.

Rename the submission wrappers to include your group number before uploading:

```
submissions/SP_policy_group.py      →  SP_policy_[number].py
submissions/ADP_policy_group.py     →  ADP_policy_[number].py
submissions/Hybrid_policy_group.py  →  Hybrid_policy_[number].py
```
