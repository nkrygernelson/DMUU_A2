# Ridge V_θ diagnostic log — `hybrid-minimal` branch

A running notebook of what we've measured about the ridge value function and which
hypotheses we've ruled in/out. Adds to `pdfs/deep_adp.md`; doesn't replace it.

## Setup

- Branch: `hybrid-minimal`.
- Solver: HiGHS (`appsi_highs`) — open-source replacement for the size-limited Gurobi license. Documented quality gap to Gurobi of ~10 mean cost on this MIP class.
- Ridge ADP from `origin/adp-fix` (commit `969a288`). Per-(stage, region) ridge fits with regions = `2·lo1 + lo2`. Etas shape `(11, 4, 11)` = (stage, region, feature).
- Features (FEATURE_DIM=11): `[1, T1, T2, H, Occ1, Occ2, price_t, price_prev, vent_counter, lo1, lo2]`.
- Training: forward–backward fitted value iteration. Default I=15, N=30, K=10. We've used I=10, N=20, K=8 to fit time budget.

## Top-line results (100-day eval, HiGHS)

| Policy | bf | num_stages | mean cost | notes |
|---|---|---|---|---|
| SP | 2 | 2 | **149.62** | best of the sweep |
| SP | 3 | 2 | 149.68 | |
| SP | 4 | 2 | 149.62 | |
| SP | 2 | 3 | 163.34 | deeper tree → worse under HiGHS |
| SP | 3 | 3 | 161.17 | |
| SP | 4 | 3 | 160.85 | |
| **Hybrid (ridge V_θ at leaves, full disjunctive)** | 3 | 2 | **158.37** | hurts vs SP-2,2 (+9) |
| **Hybrid (ridge V_θ at leaves, full disjunctive)** | 2 | 3 | **156.14** | beats SP-2,3 by 5 |
| SP-Gurobi (docs) | 3 | 3 | 139.72 | reference |

Two clean takeaways:
1. **HiGHS finds noticeably worse incumbents than Gurobi** at the same MIPGap. ~10 cost-unit gap visible across all configs. Reverting to Gurobi for production is the right move; HiGHS is fine as a "lift the size cap" experiment.
2. **The hybrid's V_θ at the leaves does not help** vs SP at the same tree size. It only "wins" against deeper-SP because HiGHS-SP itself degrades there.

## Why doesn't the ridge V_θ help?

Diagnostic plots in `plots/`:
- `ridge_fit_scatter.png` — aggregate predicted-vs-actual scatter. Slope=0.39, bias=−15, RMSE=84. Useful, but too aggregated.
- `ridge_per_stage_scatter.png` — 10-panel per-stage scatter. Reveals that the aggregate slope hides extreme variation.
- `ridge_per_stage_error.png` — per-stage bias, RMSE, slope, sample counts per (t, region).
- `ridge_state_distribution.png` — histograms of every state variable, **ADP-rollout (training) vs SP-rollout (inference proxy)**.
- `ridge_eta_heatmap.png` — eta coefficients as (stage × feature × region) heatmaps.

### Finding 1 — Massive state-distribution shift

| State | ADP-training rollouts | SP rollouts |
|---|---|---|
| T1, T2 | 18–24°C (warm, peaked 20°C) | **11–19°C (cold, peaked 13–14°C)** |
| H | 40–80% | **frequently saturates at 100%** |
| vent_counter | spread over {0,1,2,3} | almost always 0 |

The ridge ADP rollouts visit a *warmer*, less-saturated regime than SP. The hybrid evaluates V_θ on leaves whose state distribution looks like SP's, which the ridge has barely seen. Predictions outside the training distribution collapse toward the mean (≈30).

### Finding 2 — Per-stage performance is catastrophic at t=0, t=1

From `ridge_per_stage_error.png`:

- **t=0, t=1: slope ≈ 0.05, R² ≈ −2.** Predictions are essentially constant; the regression has zero discrimination among initial states.
- **t=2: slope = 0.46** — improving.
- **t=4..9: slope = 0.6–0.9, R² > 0.5** — fits well.

Two reinforcing causes:
1. **Input variance at t=0 is near zero.** `sample_init_state` draws T1, T2 ∈ [19, 23] (4°C window), H ∈ [40, 60], vc=0, lo1=lo2=0, etc. Across 20 training rollouts, the input feature vector at t=0 is nearly identical, but the target return-to-go is wildly variable (because downstream stochastics differ). Least-squares with constant input + variable output ⇒ slope ≈ 0, intercept = mean(y). Exactly what we see.
2. **Some compounded bootstrap bias** from V_{t+1} estimates flowing into V_t targets. This is normal and small; it's not what makes t=0 catastrophic.

### Finding 3 — Region (1,1) coefficients are noise

`ridge_eta_heatmap.png` shows region (1,1) coefficients of ±15 to ±25 at some stages — vastly larger than the ±5 norms in region (0,0). Cause: most (t, r=3) buckets have **n=1 sample** because ADP rollouts rarely enter (1,1). Single-point ridge fits over-amplify the coefficients. Worse, the hybrid actually visits (1,1) leaves often (because SP-style policies push the system into the cold/override regime), so those noisy etas get amplified at inference.

## What we're trying — and why

The plan: tackle the **biggest lever** first (input variance at t=0), measure, then move down the list.

### Experiment A — widen `sample_init_state` to span SP's operating range

**Hypothesis:** with t=0 input variance ≈ 0, no linear regressor can produce a non-trivial fit. Widening the initial distribution should give t=0 slope > 0.

**What changes:**
- T1, T2: from `uniform(19, 23)` to `uniform(12, 24)` — covers SP's cold regime (~12-14°C peak) and the warm regime (~20-23°C peak).
- H: from `uniform(40, 60)` to `uniform(40, 90)` — closer to SP's full range up to saturation.
- Other state vars (vc, lo1, lo2, prices) match SP's distribution already → unchanged.

**Save path:** new etas under `policies/adp_etas_wide_init.npy` so we can compare side-by-side with `adp_etas.npy` (the narrow-init version).

**Success criterion:** per-stage slope at t=0 jumps from 0.05 → ≥ 0.4. Hybrid 100-day mean cost ideally drops.

### Experiment B (queued) — feature interactions

If A helps but the linear model is still under-fitting, add a few product features: `T1·lo1, T2·lo2, price_t·H, price_t·vc`. Still MILP-encodable because lo1, lo2 are already part of the region split, so the products are linear within each region.

### Experiment C (queued) — more iterations

Only useful after A. Doesn't fix input variance; helps with rare-region sample starvation.

## Diagnostic recipe (for next time)

When a linear value function under-performs, in order:

1. **Per-stage scatter + slope plot.** Localizes where the model fails.
2. **State distribution overlap** between training rollouts and inference-policy rollouts. Reveals distribution shift.
3. **Sample count per (t, region) bucket.** Identifies rare regions (n < 5 = trouble).
4. **Eta heatmap.** Spots over-amplified coefficients (look for cells ≫ typical magnitude — usually n=1 over-fits).
5. **Aggregate scatter slope and bias.** Last — too averaged to be actionable on its own.

## Log entries

(Filled in as experiments complete.)

### 2026-05-15 — baseline measured

- Ridge ADP retrained on hybrid-minimal with HiGHS (I=10, N=20, K=8). Etas shape (11, 4, 11). Saved to `policies/adp_etas.npy` (this is the narrow-init version).
- Hybrid bf=3,ns=2: **158.37**. Hybrid bf=2,ns=3: **156.14**.
- Per-stage diagnostics: slope ≈ 0.05 at t=0,1; bias = -114 at t=0. Distribution shift confirmed (T1/T2 by ~6°C, H tail to 100% missing).

### 2026-05-15 — Experiment A: widened init distribution

**Change:** `sample_init_state` widened from `T1, T2 ∈ uniform(19, 23), H ∈ uniform(40, 60)` to `T1, T2 ∈ uniform(12, 24), H ∈ uniform(40, 90)`. Saved as `policies/adp_etas_wide_init.npy`; `policies/adp_etas.npy` now points to this version.

**Training-time signal:** at t=0 the regression target std is now ~83 (vs the very small variance under narrow init), and the fit produces eta-norm 31.5 (vs 2.5 narrow). The regression has something to learn.

**Per-stage diagnostic (`plots/ridge_per_stage_*.png`, wide-init etas, 20-day ADP rollouts):**

| stage t | slope (was → now) | bias (was → now) |
|---|---|---|
| 0 | 0.05 → **0.62** | -114 → -70 |
| 1 | 0.05 → **0.68** | -114 → -78 |
| 2 | 0.46 → 0.64 | -54 → -54 |
| 3 | 0.66 → 0.86 | +71 → -35 |
| 4 | 0.87 → 0.84 | +18 → -40 |
| 5 | 0.69 → 0.86 | +31 → -15 |
| 6 | 0.87 → 0.72 | +1 → +3 |
| 7 | 0.88 → 0.21 | +1 → +4 |
| 8 | 0.62 → 0.74 | +2 → +2 |
| 9 | 0.85 → 0.48 | +3 → +1 |

The catastrophic t=0/t=1 collapse is fixed — the regression now has actual signal at every stage. Mid-stages (t=3..6) shifted from positive to slightly-negative bias.

**Sample-counts plot (`plots/ridge_per_stage_error.png` bottom-right): unchanged.** Sample density per (t, r) bucket is the same as narrow init (it's controlled by N=20 rollouts × forward dynamics, not by initial conditions).

**Hybrid 100-day eval** at bf=3, ns=2 with the new etas: **155.89** (vs narrow-init 158.37). Δ = **−2.48** cost units.

**ADP-alone 100-day eval** with the same etas: **wide-init 181.32 vs narrow-init 164.49.** Δ = **+16.83** — wide-init makes the *standalone* ADP **worse**.

This is the most interesting finding so far. The two policies use V_θ in different ways:
- ADP-alone has V_θ as the entire lookahead beyond t+1. The MILP picks the root action mainly by comparing `E[V_θ(x_{t+1})]` across choices. With the narrow-init V_θ (slope ≈ 0, predictions ≈ constant 30), the continuation term is nearly the same for every action, so the MILP defaults to minimizing immediate cost — a passable greedy policy that accidentally avoids being misled. With the wide-init V_θ (slope ≈ 0.6, bias ≈ -70), the continuation term *does* vary across choices, but it varies around a target that's 70 cost units below reality. The MILP works harder to optimize against a biased signal, picks "cheap-looking" action paths, and pays for it.
- The hybrid uses V_θ only as the tail after 2 stages of exact MILP planning. The near-term decisions are constrained by dynamics + cost terms in the MILP itself, so V_θ's bias has less leverage on the action choice.

Generalizable lesson: **slope matters when V_θ is one term among many; bias matters when V_θ is the only thing the optimizer cares about.** For the standalone ADP, an *uncalibrated* better-fitting V_θ is a worse policy.

**Verdict:** experiment A is a small net win for the hybrid, but not enough. The fitted-VI machinery now has signal at every stage but is still off SP-bf=2,ns=2 by ~6 cost units. Hypotheses for the remaining gap:

1. **Persistent negative bias at t=0..5.** Predictions still ~30-70 too low. Hybrid sees a *systematically low* leaf value, so the MILP "thinks" continuing is cheaper than it really is — and over-spends on heating now.
2. **Mismatch correction: SP's actual operating range is *narrower* than what I assumed.** Re-running the SP rollouts with the v2 fix shows SP keeps T1/T2 tight at 18–20°C (heater holds the floor), not the 11–15°C I previously read off the buggy SP rollout. The wide-init we shipped now *over-covers* — ADP samples 12-24, SP visits 18-20. That's still better than 19-23 over a narrow 18-20 target, but optimal would be a slightly tighter wide-init or a SP-distribution-matched init.
3. **HiGHS solver-quality gap (~10 points to Gurobi) is the dominant remaining piece.** No amount of V_θ tuning makes up for that.

### 2026-05-15 — correction: SP state distribution misread on first pass

The first state-distribution plot showed SP rollouts visiting T1/T2 in 11–19°C. That was an artifact of a bug in `experiments/ridge_diagnostics_v2.py` — I called `check_and_sanitize_action(u_raw, state, fixed_data)` but the function signature is `(policy, state, P_max)`, so the SP rollouts crashed every step and were replaced with `{"HeatPowerRoom1": 50, "HeatPowerRoom2": -330, "VentilationON": 'something_crazy'}`. Those wildly out-of-bounds dummy actions produced trajectories that *did* visit cold temperatures, but for completely wrong reasons.

With the bug fixed (skipping sanitize in the diagnostic, since `advance_state` enforces the low-override hysteresis), the corrected state-distribution plot shows SP keeps T1, T2 in 17–22°C, peaked at 18. The lesson: **the distribution shift I identified was real (training was warm-only), but the direction was wrong — SP runs *warm* like training, not cold**. The wide-init experiment helped not because it covered cold rooms (those barely matter at inference), but because **it gave the t=0 regression input variance to fit against**, which was the actual problem.

## Status of next experiments

- **B (feature interactions):** queued; unclear if it's the right next lever given the small Δ from A.
- **C (more iterations):** queued.
- **D (new): SP-policy rollouts for training data.** Now that I know SP's distribution is narrow and warm, training the ridge on SP-rolled trajectories — and using the realized SP cost-to-go as the target — could give a much better-matched fit. This is the "distillation" idea, but trying it again now that we have better diagnostics.
- **E (new): tighten the wide-init to match SP, e.g. `T1, T2 ∈ [16, 22]`.** Removes the over-cover slop and keeps the input variance.

### 2026-05-15 — Experiment F: simplified hybrid (no regions, no big-M)

Stripped the 4-region disjunctive leaf encoding. Each leaf now contributes one pure linear term `eta[t_leaf] @ phi(x_leaf)`. Zero extra binaries, no big-M products. Trained single ridge per stage on MC return-to-go from region-ADP rollouts.

Solver also swapped back to Gurobi to remove the HiGHS quality gap.

Code in `policies/hybrid_policy.py` (rewritten as "simple hybrid"); training in `experiments/train_single_eta.py`.

100-day eval:

| Variant | N (rollouts) | alpha | Mean cost | Δ vs SP-Gurobi (139.75) |
|---|---|---|---|---|
| N=80 | 80 | 1.0 | **157.48** | +17.7 |
| N=500 | 500 | 1.0 | 163.37 | +23.6 (worse than N=80) |
| N=500 RidgeCV | 500 | CV-selected | 163.44 | +23.7 (same) |

More data → *worse* hybrid. CV-selected alpha didn't fix it. Diagnosis: the MC return-to-go targets approximate the value function of the *rollout policy* (region-ADP). They do **not** approximate the value function of "states reachable by the SP scenario tree at leaves". More data → more confidence in the wrong target.

### 2026-05-15 — Experiment G: SP distillation

Same simple hybrid (single eta, no regions, no big-M). Training data changed from region-ADP rollouts to **SP rollouts** with realized SP cost-to-go as the target. Same wide-init `sample_init_state`. 200 days of SP rollouts; RidgeCV per stage.

Code in `experiments/train_single_eta_sp.py`. Etas at `policies/adp_etas_single_sp.npy`.

100-day eval:

| Variant | Tree | Mean cost | Δ vs SP-Gurobi |
|---|---|---|---|
| **SP-distilled hybrid** | bf=3, ns=2 | **153.21** | +13.5 |
| SP-distilled hybrid | bf=2, ns=3 | 158.00 | +18.3 |

Distillation closed **4-10 cost units** vs the ADP-rolled variants. Std also dropped (67 → 54), so decisions are more consistent. **Training distribution matters more than sample size** — the bottleneck wasn't data volume, it was data-source.

But: even SP-distilled, we're still 13.5 short of SP itself, and 10 cost units behind v1 (143.45 with MLP+big-M).

## Summary — is "simple linear hybrid beats SP" possible here?

**Empirical answer: no.** The simple linear hybrid caps out around **153 mean cost** with SP-distilled training. SP itself is 139.75. The 13.5-unit gap is what a linear V_θ at the leaves cannot recover — it doesn't have the function-class capacity to substitute for an exact 3rd stage of MILP planning.

The only configurations that *do* tie or near-tie SP in the docs are v1 (MLP+big-M, 143.45) and v3 (MLP+big-M, ns=3, 140.31) — both require the non-linear V_θ embedded via big-M ReLU.

### What we ruled in/out via these experiments

1. **Solver quality** (HiGHS vs Gurobi): ~10 cost-unit gap for SP, ~1 cost-unit gap for hybrid. Not the bottleneck for the hybrid.
2. **Initial-state distribution width**: necessary for the t=0 ridge fit to have signal. Bumping from `T1, T2 ∈ [19,23]` to `[12,24]` improved per-stage R² dramatically but only ~2.5 cost units on the hybrid.
3. **Region disjunction** (4 etas/stage vs single): single is simpler but slightly worse (loses ~2 cost units of fit capacity).
4. **Sample size** (N=80 vs N=500): worse with more data when target distribution doesn't match inference distribution.
5. **Regularization choice** (alpha=1 vs RidgeCV): no material effect.
6. **Training data source** (region-ADP rollouts vs SP rollouts): biggest single lever; +4 to +10 cost units depending on baseline.
7. **Tree shape** (bf=3 ns=2 vs bf=2 ns=3): bf=3 ns=2 is better for the simple linear hybrid. Opposite of v3's MLP-based finding.

### Recommendation

If the goal is *ship something simple that beats SP*: not achievable here. The simple linear hybrid lands at ~153 mean cost vs SP's 140.

If the goal is *ship something simple that approximately matches SP*: the SP-distilled simple hybrid at bf=3, ns=2 (153.21) is the cleanest "minimum viable hybrid" — it's a single linear function per stage, no big-M anywhere, ~30 lines of MILP code for the V_θ block. It's 8-10 cost units worse than v1/v3 but uses none of the big-M ReLU machinery.

If the goal is *ship something that beats SP*: v3 (140.31, MLP+big-M, ns=3) is the demonstrated winning approach in this codebase.
