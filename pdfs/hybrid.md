# Hybrid policy — beating the SP baseline

**Result on the 100-day eval:**

| Policy | Mean cost | SD   | vs SP |
|--------|-----------|------|-------|
| Dummy  | 182.91    | 76.8 | +43.2 |
| SP     | 139.74    | 58.3 | —     |
| Hybrid | **137.38**| 61.2 | **−2.35** |

Paired stats: hybrid wins 57/100, ties 18, loses 25. `mean(SP − hybrid) =
+2.354`, `SE = 1.340`, `t = 1.76` (one-sided p ≈ 0.04). The improvement is
small in absolute terms but statistically meaningful — the deep-ADP-MILP
v3 baseline was 140.31, statistically tied with SP.

## The idea

SP solves a 3-stage scenario tree (bf=3, ns=3) and stops; everything past
stage 3 is implicitly valued at zero. That terminal truncation hurts early
decisions because it ignores 7 hours of future cost.

The hybrid does two things differently:

1. **Plans one stage deeper than SP.** Tree shape is `bf=[3,3,2,2]` — 4
   stages tactically (vs SP's 3), then a learned terminal value V_θ at the
   36 leaves. The MILP can lay out a 4-step plan exactly, instead of
   handing off to V_θ after only 3 steps as the previous MILP-rollout
   policies did.

2. **Carries a learned tail value V_θ at the leaves.** V_θ is the
   fitted-VI MLP from `deep_adp_milp_policy.py` (v1's well-trained 1×32
   `deep_adp_model_small.pt`), distilled into a 1×8 MLP
   (`deep_adp_model_tiny.pt`) with R² = 1.0 against the teacher. The
   smaller hidden layer is what makes the deeper, wider tree fit inside
   the size-limited Gurobi license.

## Why the deeper tree pays

The previous attempts had to choose between breadth and depth:

- v1 — `bf=3, ns=2` + V_θ → 143.45
- v3 — `bf=2, ns=3` + V_θ → 140.31
- SP — `bf=3, ns=3` + V=0 → 139.74

bf=3, ns=3 with a learned V_θ would dominate all three, but `27 × 32 = 864`
ReLU binaries blew past the Gurobi license. The hybrid resolves that with
**distillation + sparse big-M ReLU encoding** so we can run an even bigger
tree (bf=[3,3,2,2], 36 leaves, ns=4).

## Engineering details

**Distilled V_θ (1×8).** The 1×32 fitted-VI model has 22 hidden units
that fire on every sampled state and 10 that never fire. A 1×8 student
network fits the teacher with R² ≈ 1.0 on ~3k diverse states sampled
from on- and off-policy rollouts. Done in seconds.

**Sparse ReLU encoding.** For each unit `j` at each leaf with input
bounds `[L_in, U_in]`, the pre-activation bound `[L_a, U_a]` is computed
analytically. Then:
- `L_a ≥ 0` → always-on, substitute `h_j = a_j` directly into the V_θ
  output (no Pyomo variables for this unit at this leaf).
- `U_a ≤ 0` → always-off, drop from the V_θ sum.
- otherwise → genuine ReLU, big-M encoding with one binary.

v3 always allocated `h_var` and `z_var` for every (leaf, unit) pair even
when the encoding fixed them to constants. Skipping those allocations is
what frees enough capacity for `bf=[3,3,2,2]`.

**Deterministic seeding.** `select_action` seeds NumPy with a hash of
`(current_time, T1, T2, H, price_t)` so the scenario tree, KMeans
clusters, and resulting decision are reproducible across re-evaluations.
Removes ≈1 cost unit of run-to-run variance.

**Tree schedule.** `_tree_for_remaining(r)` truncates the full
`[3,3,2,2]` schedule to `r` stages when fewer hours remain — at `t = 6`
the policy is 4-stage and exactly covers the remaining horizon (V_θ
is skipped at terminal leaves); at `t = 8` it collapses to a 2-stage
SP-style tree.

## Where the wins and losses concentrate

Hybrid pulls ahead most on cheap-day variants (low prices, mild
comfort) — top wins: 24, 40, 43, 93, 54 (Δ = 20–52 per day) — where the
extra step of explicit planning lets it dodge unnecessary heating that
SP greedily commits to.

Hybrid loses on a handful of hard days (39, 83, 1, 79; Δ = -20 to -30)
where V_θ appears to under-value the cost of being driven into the
low-temp override band. A tighter value function around override
boundaries would close most of the remaining gap, but the current model
is already net-positive.

## Files

- `policies/hybrid_policy.py` — the policy itself.
- `policies/deep_adp_model_tiny.pt` — distilled 1×8 V_θ.
- Reuses `policies.sp_policy.propagate_uncertainty` and the MILP dynamics
  encoding from `policies.deep_adp_milp_policy`.
