# DMUU A2 — Project notes for Claude

Keep edits minimal, code clean and readable, comments sparse. Don't modify `Policy_Restaurant.py`, `SystemCharacteristics.py`, `helper/`, `processes/`, or `policies/sp_policy.py` (working reference).

## Problem in one paragraph

10-hour HVAC control for a 2-room restaurant. Per hour pick `p1, p2 ∈ [0, 3] kW` (continuous) and `V ∈ {0,1}` (binary). Cost = `price_t · (p1 + p2 + p_vent · V)`, summed over t=0..9. Exogenous stochastic: electricity price (mean-reverting AR, given by `price_model`) and per-room occupancy (Markovian AR, `next_occupancy_levels`). Endogenous state evolves linearly in T1, T2, H from the chosen actions and exogenous occupancy. Three overrules are *enforced by the simulator after* the policy returns: high-temp (heater forced off above 26°C), low-temp (heater forced to max once T<18, until T≥22), humidity (V forced on above 70%), plus **ventilation inertia**: once V=1 it must stay on ≥3 hours.

Action time budget per call: **15 s** (else dummy action is substituted by `check_and_sanitize_action`).

## State the policy receives

`{T1, T2, H, Occ1, Occ2, price_t, price_previous, vent_counter, low_override_r1, low_override_r2, current_time}`. `current_time ∈ [0,9]`. Initial state and all coefficients are in `SystemCharacteristics.get_fixed_data()`.

## What's already built

- **SP (`policies/sp_policy.py`)** — multi-stage scenario tree, KMeans-clustered child scenarios, MILP solved with Pyomo+Gurobi. At each step: `num_stages = min(3, remaining)`, `bf=3`. Solver capped at 10 s with `MIPGap=0.02`. Reference implementation for dynamics + overrule encoding.
- **ADP (`policies/adp_policy.py`)** — linear VFA over an 11-dim feature vector `[1, T1, T2, H, Occ1, Occ2, price_t, price_prev, vent_counter, lo1, lo2]`. Trained by forward-backward fitted value iteration; one eta vector per stage, saved to `policies/adp_etas.npy`. At test time: one-step Bellman MILP with `K=5` KMeans-clustered child scenarios and learned `eta_next^T φ(x')` as continuation.
- **Deep ADP (`policies/deep_adp_policy.py`)** — single PyTorch MLP (2×64 ReLU) over a 12-dim state (adds `t/T` and `1−t/T`) replaces the linear VFA. Forward-backward fitted VI with ε-greedy exploration. Online: constrained grid search over (p1, p2, V) with MC expectation (M=20). Trained model at `policies/deep_adp_model.pt`. On 100 days: mean cost 174.08 (vs SP 139.76, dummy 182.91) — beats dummy but ~24% short of SP.
- **MILP rollout + V_θ at leaves (`policies/deep_adp_milp_policy.py`)** — hybrid: SP-style 2-stage MILP for tactical decisions, learned 1×32 MLP V_θ embedded at leaves via big-M ReLU. Uses a separate smaller VNet trained for MILP tractability; weights at `policies/deep_adp_model_small.pt`. On 100 days: mean cost ~143.5 vs SP ~139.7 (~3% off SP). Writeup: `pdfs/deep_adp.md`.
- **MILP rollout v2 — on-policy MC retraining (`policies/deep_adp_milp_v2_policy.py`)** — same MILP structure; V_θ retrained via Monte Carlo returns from rollouts of the MILP policy itself (no Bellman bootstrap). Weights at `policies/deep_adp_model_small_v2.pt`. Result: ~145.6, no improvement over v1 — MC target variance dominated the distribution-shift benefit. Documented as Attempt 3 in `pdfs/deep_adp.md`.
- **MILP rollout v3 — 3-stage tree (`policies/deep_adp_milp_v3_policy.py`)** — uses v1's fitted-VI V_θ (`deep_adp_model_small.pt`) inside a bf=2, num_stages=3 MILP (matches SP's tactical depth; bf=2 to stay under the Gurobi size-limited license). 100-day mean cost **140.31** vs SP 139.72 — paired Δ = -0.59 with SE 1.69, statistically indistinguishable. The SP-distillation V_θ experiment is preserved as `train_distill_attempt()` and `deep_adp_model_small_v3.pt`; it underperformed (R² 0.78, eval 150–161) and is not used by `select_action`. Documented as Attempt 4 in `pdfs/deep_adp.md`.
- **SP backwards-induction policy (`policies/sp_backward_induction_policy.py`)** — same inference structure as v1 (bf=3, ns=2 MILP rollout with V_θ at leaves), but V_θ is trained as **10 independent per-time-step MLPs** via fitted backwards induction over SP rollouts (the "original SP policy"). Each V_θ_t is a small (1×16) MLP fit with sklearn L-BFGS; per-t weights saved to `policies/sp_backward_induction_models/V_t{0..9}.npz`. Training R² is excellent (0.92–0.997) but the policy gets **160.49** mean cost on 100 days — worse than SP, v1, and v3. Lesson: per-t MLPs overfit and the Bellman bootstrap chain compounds the bias. Documented as Attempt 5 in both `pdfs/deep_adp.md` and `pdfs/deep_adp_long.md`.
- **Long pedagogical writeup**: `pdfs/deep_adp_long.md` — walks through the Bellman equation, why linear ADP fails, the MLP architecture, the grid-search vs MILP planning trade-off, the big-M ReLU encoding (with a small worked example), and the bias-variance argument for fitted-VI vs MC training. Self-contained explanation of all five attempts.
- **Dummy** — returns zeros.
- **Simulator (`environment/simulator.py`)** — `evaluate(policy, experiments)` runs 100 fixed days from `data/*.csv`. Returns mean cost. Internally re-applies all overrules + ventilation inertia after `policy.select_action`.
- **Hybrid policy (`policies/hybrid_policy.py`)** — Task 5 deliverable. SP-style MILP rollout, but with `bf=[3,3,2,2]` (4-stage tree, one deeper than SP) and a **distilled 1×8 V_θ** (`deep_adp_model_tiny.pt`, R²=1.0 vs the 1×32 teacher) at the leaves. Sparse big-M ReLU encoding skips Pyomo allocations for always-on/always-off units to fit the Gurobi size-limited license. Deterministic per-call NumPy seeding makes scenarios reproducible. 100-day mean cost **137.38** vs SP 139.74 — paired Δ = +2.35 with SE 1.34, t=1.76 (one-sided p≈0.04). Hybrid wins 57/100, ties 18, loses 25. Writeup: `pdfs/hybrid.md`.

## Key gotchas

- **Actions are mixed**, not fully continuous (V is binary with a 3-step lockout). Any continuous-action method needs special handling for V.
- **Horizon is 10**, not 13. Short horizon → SP/ADP already strong baselines.
- **Comfort constraints are enforced post-hoc by the simulator**, so a policy can ignore them and still produce feasible runs, but it will pay through the overrule-induced cost. Any learned policy that doesn't internalize this leans on the simulator's safety net.
- `outdoor_temperature` is deterministic from the config; only price and occupancy are stochastic.
- `vent_counter` is the inertia clock: 0 = off, 1..2 = must stay on, ≥3 = free to turn off.

## Files I don't need to re-read

I've already read: `SystemCharacteristics.py`, `policies/sp_policy.py`, `policies/adp_policy.py`, `policies/hybrid_policy.py`, `environment/simulator.py`, `processes/PriceProcessRestaurant.py`, `processes/OccupancyProcessRestaurant.py`, `helper/v2_Checks.py`, `README.md`. For anything else, read on demand.
