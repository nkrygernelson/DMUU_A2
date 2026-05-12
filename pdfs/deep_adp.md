# Deep ADP and MILP rollout with neural leaf value

Two value-function-approximation policies for the restaurant HVAC problem, in chronological order. The first replaces the linear VFA of `adp_policy` with a small MLP. The second wraps a learned MLP value function inside a 2-stage MILP scenario tree.

## Setup

State at time $t$:
$$x_t = (T^1_t, T^2_t, H_t, \text{Occ}^1_t, \text{Occ}^2_t, p_t, p_{t-1}, vc_t, \ell^1_t, \ell^2_t)$$

Action: $u_t = (p^1_t, p^2_t, v_t)$ with $p^i_t \in [0, P_{\max}]$, $v_t \in \{0, 1\}$. Immediate cost $c(x_t, u_t) = p_t (p^1_t + p^2_t + P_{\text{vent}} v_t)$. Horizon $T = 10$. Endogenous dynamics are linear; the three overrules and the 3-step ventilation lockout are enforced by the simulator after the policy returns.

Both policies share the same MLP value function $V_\theta$ (different sizes for the two approaches) trained by forward-backward fitted value iteration:

$$V_t(x) = \min_{u \in \mathcal{U}(x)} \Big\{\, c(x, u) + \mathbb{E}_{\omega}\big[V_{t+1}(f(x, u, \omega))\big] \,\Big\}, \quad V_T(\cdot) = 0$$

```
init  V_θ ← None  (returns 0)
for i = 1..I:
    ε ← max(ε₀ − decay·i, ε_min)
    # forward
    for n = 1..N:
        x ← sample_init_state()
        for t = 0..T-1:
            u ← random feasible (with prob ε)  or  argmin_u V̂_t(x; u)
            x ← f(x, u, sample_exog(x))
    # backward — build (state, V_t) regression dataset
    for t = T-1..0, for each visited x_t:
        y_t ← min_u  c(x_t, u) + (1/M) Σ V_θ(f(x_t, u, ω_m))
    V_θ ← fit_MLP(dataset, epochs=300, lr=1e-3)
```

The MC estimator uses $M$ common scenarios per state for variance reduction. Defaults: $I = 8\text{--}10$, $N = 25\text{--}30$, $M = 20$.

State encoding (12-dim): $[T_1/30, T_2/30, H/100, \text{Occ}_1/50, \text{Occ}_2/30, p/12, p_{\text{prev}}/12, vc/3, \ell^1, \ell^2, t/T, 1-t/T]$. Including both $t/T$ and $1-t/T$ lets a single network cover the whole horizon.

---

## Attempt 1 — Deep ADP with action-grid search

`policies/deep_adp_policy.py`. MLP architecture: 2 hidden ReLU layers of 64 units. At each step, action selection enumerates a constrained grid (7 levels for each heater × {0,1} for ventilation, pruned by the overrule states) and picks the argmin of $c(x, u) + (1/M)\sum V_\theta(f(x, u, \omega))$ with $M=20$ Monte Carlo scenarios.

**No MILP at runtime** — because $V_\theta$ is nonlinear in the decisions, the per-step optimization is grid search rather than a Bellman MILP.

### Results (100-day eval)

| Policy        | Mean cost | Std dev |
|---------------|-----------|---------|
| dummy         | 182.91    | 76.77   |
| ADP (linear)  | 197.95    | 78.19   |
| Deep ADP      | 174.08    | 62.41   |
| **SP (multistage)** | **139.76** | **58.26** |

Deep ADP beats dummy and linear ADP, but **falls 24% short of SP**. Diagnosis:

- Action grid is coarse (0.5 kW resolution per heater).
- One-step lookahead: the policy trusts $V_\theta$ for everything past $t+1$.
- Small training set ($N \times T = 250$ samples per outer iter) leaves $V_\theta$ noisy.
- Monte Carlo noise of $E[V_\theta(x')]$ swamps small action-value gaps.

A finer grid or beefier MLP could shave at the first two; the latter two need a different architecture.

---

## Attempt 2 — 2-stage MILP rollout with $V_\theta$ at leaves

`policies/deep_adp_milp_policy.py`. The hybrid: SP's near-term planning + Deep ADP's tail value.

### Tree

Same structure as `sp_policy`: $bf = 3$, $\text{num\_stages} = 2$. The root state is the current observation (fixed). At each of the 3 stage-1 child nodes and 9 stage-2 leaf nodes, exogenous occupancy and price are KMeans-clustered centers (`propagate_uncertainty` from `sp_policy`).

Decisions are continuous-in-$p^i$, binary-in-$v$ at the root and at each stage-1 child. Stage-2 nodes have no decisions — they are evaluation leaves.

### Per-step program

$$\min \;\; c(x_0, u_0) \;+\; \sum_{k} p_k \, c(x_{1,k}, u_{1,k}) \;+\; \sum_{k, j} p_{kj} \, V_\theta(x_{2,kj})$$

subject to (all from `sp_policy`):

- thermal dynamics for $T_1, T_2$; humidity dynamics for $H$;
- low-temperature override hysteresis (heater forced to $P_{\max}$ until $T \ge T_{\text{ok}}$);
- high-temperature override (heater forced to 0 above $T_{\text{high}}$);
- humidity override ($v = 1$ above $H_{\text{high}}$);
- ventilation min-up-time of 3 hours.

Plus, new for this policy:

- $vc$ propagation as a continuous state variable, $vc_{t+1} = (vc_t + 1) v_t$, encoded with big-M;
- $V_\theta$ encoded at every leaf via big-M ReLU. For a 1-hidden-layer MLP $V(x) = W_2 \cdot \mathrm{ReLU}(W_1 x + b_1) + b_2$, each hidden unit $j$ gets pre-activation $a_j$ and post-activation $h_j$ with binary $z_j$:
  $$h_j \ge a_j, \quad h_j \ge 0, \quad h_j \le U_{a_j} \cdot z_j, \quad h_j \le a_j - L_{a_j} (1 - z_j)$$
  where $[L_{a_j}, U_{a_j}]$ are interval-propagated bounds from the feature ranges through $W_1$. Pre-activations that are always positive ($L_{a_j} \ge 0$) or always negative ($U_{a_j} \le 0$) are fixed without binaries.

### Training $V_\theta$

Same forward-backward fitted VI as Attempt 1, but with a **smaller** MLP (1 hidden layer × 32 units) so the big-M encoding stays tractable. With $bf = 3$ this adds $9 \times 32 = 288$ ReLU binaries per MILP — Gurobi solves each call in well under the 15 s policy budget.

Trained model: `policies/deep_adp_model_small.pt`. Backward R² converged to 0.92 after 10 outer iterations.

### Results (100-day eval)

| Policy            | Mean cost | Std dev | Wins vs SP |
|-------------------|-----------|---------|------------|
| dummy             | 182.91    | 76.77   | — |
| ADP (linear)      | 197.95    | 78.19   | — |
| Deep ADP (grid)   | 174.08    | 62.41   | — |
| **MILP rollout + $V_\theta$** | **143.90** | 62.62  | 32/100 |
| SP (multistage)   | 139.76    | 58.26   | — |

The MILP rollout closes 88% of the SP-vs-Deep-ADP gap (from 34 → 4). It beats vanilla Deep ADP on 91/100 days (mean Δ +30), and is within 3% of SP. SP still wins on average (68/100 days), likely because (a) SP uses 3-stage trees while we use 2-stage, and (b) $V_\theta$ has finite capacity and finite training data.

Runtime per 100-day evaluation: SP 33.9 s, MILP rollout 37.7 s — comparable.

### Why this works

Deep ADP-alone forced the value function to absorb *all* of the lookahead, and a small MLP trained on a few hundred samples can't do that precisely. The MILP rollout only asks $V_\theta$ to predict cost-to-go *2 steps ahead*, which is a much easier regression. Near-term planning, where the overrules and the ventilation lockout have to be coordinated against price spikes, is handled by the MILP exactly as in SP.

---

## Attempt 3 — On-policy Monte Carlo retraining of $V_\theta$

`policies/deep_adp_milp_v2_policy.py`. Motivation: in Attempt 2, $V_\theta$ was trained by fitted VI under a grid-search action-selection policy, then served at inference inside a different policy (the MILP). The training distribution and the inference policy do not match. Attempt 3 closes that mismatch.

### Procedure

1. Bootstrap $V_\theta$ from the Attempt-2 small model (1×32, R²=0.92).
2. Repeat $K = 6$ outer iterations:
   - Roll out the MILP-rollout policy under the current $V_\theta$ for $N = 25$ days.
   - For every visited $(t, x_t)$, compute the Monte Carlo cost-to-go $y_t = \sum_{s=t}^{T-1} c(x_s, u_s)$ where $u_s$ is the MILP-rollout action.
   - Refit $V_\theta$ on $(x_t, y_t)$ pairs (MSE, warm-start, 400 epochs).

Targets are unbiased estimates of cost-to-go *under the policy that uses them*; no Bellman bootstrap. Replay-style accumulation across iterations is **not** used — each refit sees only the current iteration's on-policy data.

### Results (100-day eval)

| Policy              | Mean cost | Std dev |
|---------------------|-----------|---------|
| MILP rollout (Attempt 2, fitted-VI $V_\theta$) | 143.45 | 62.17 |
| **MILP rollout (Attempt 3, on-policy MC $V_\theta$)** | **145.57** | 66.96 |
| SP (multistage)     | 139.72    | 58.24   |

Attempt 3 was very slightly **worse** than Attempt 2 on the same 100-day eval (mean Δ −2.12, beats v1 on 30/100 days). The hoped-for win from on-policy training did not materialize.

### Why it didn't help

- **MC variance dominates.** With $N=25$ rollouts per iter and $T=10$ steps, return-to-go targets at early $t$ have standard deviation ~70 over a mean of ~95. The MLP fits noisy targets — backward R² fell from 0.92 (Attempt 2's Bellman targets) to 0.80–0.88.
- **Distribution shift was small.** The state distributions visited by the grid-search policy and the MILP-rollout policy turn out to be similar (same simulator dynamics; both policies obey overrules). The "wrong policy" critique was real but second-order.
- **Already near a local optimum.** Attempt 2's $V_\theta$ was already a competent fit; refitting on higher-variance targets pulled it sideways.

### Notes for next steps

- Variance-reduce the MC targets — e.g. average multiple rollouts per init state (CRN), use a baseline subtraction, or use n-step bootstrap instead of pure MC.
- Try the third proposal — distill SP's cost-to-go into $V_\theta$. SP's realized cost-to-go is a noisier but on-policy target for a known good policy; using it inside our 2-stage rollout gives one more stage of effective lookahead than SP itself.
- Bump tree depth to `num_stages=3` so the MILP captures the same tactical horizon as SP, plus $V_\theta$ at deeper leaves.

---

## Attempt 4 — Deeper tree (3-stage) with fitted-VI $V_\theta$

`policies/deep_adp_milp_v3_policy.py`. Combines the two ideas left on the table:

1. **SP distillation of $V_\theta$** — collect Monte Carlo cost-to-go from SP rollouts and fit $V_\theta$ to them.
2. **3-stage MILP rollout** — `num_stages = 3` so the MILP plans three full stages of decisions before deferring to $V_\theta$ (same tactical depth as `sp_policy`).

To stay inside the size-limited Gurobi license, the branching factor is reduced from $bf = 3$ to $bf = 2$ (8 leaves at stage 3 instead of 27).

### What worked, what didn't

The two ideas pulled in opposite directions and the experiment was informative:

| $V_\theta$ source | Tree | 100-day mean |
|---|---|---|
| SP distillation (MC return-to-go from SP) | $bf=3, ns=2$ | 161.26 |
| SP distillation | $bf=2, ns=3$ | 150.38 |
| Attempt 2 fitted-VI ($V_\theta$ from v1) | $bf=3, ns=2$ | 143.45 (= v1) |
| **Attempt 2 fitted-VI ($V_\theta$ from v1)** | **$bf=2, ns=3$** | **140.31** |

The **distilled** $V_\theta$ underperformed the original fitted-VI $V_\theta$. Even though SP is a stronger source policy than the grid-search rollouts that produced v1's $V_\theta$, the per-state MC return-to-go has $\text{sd}/\text{mean} \approx 0.75$. Fitting a small MLP to such noisy targets gives a noisier value function (R² 0.78 vs 0.92 for v1's Bellman targets). The "good source policy" benefit was second-order to the variance penalty.

The **deeper tree** with the original $V_\theta$, on the other hand, is a clear win: from 143.45 to 140.31, beats v1 on 73/100 days. With one extra stage of exact tactical planning, $V_\theta$ only has to predict cost-to-go three steps out instead of two, and the MILP retains tight constraint handling for the entire near-term horizon.

### Final results (100-day eval)

| Policy | Mean cost | Std dev |
|---|---|---|
| dummy | 182.91 | 76.77 |
| ADP (linear) | 197.95 | 78.19 |
| Deep ADP (grid) | 174.08 | 62.41 |
| MILP rollout v1 (2-stage, fitted-VI $V_\theta$) | 143.45 | 62.17 |
| MILP rollout v2 (on-policy MC $V_\theta$) | 145.57 | 66.96 |
| **MILP rollout v3 (3-stage, fitted-VI $V_\theta$)** | **140.31** | 63.38 |
| SP (multistage, $bf=3, ns=3$) | 139.72 | 58.24 |

Paired statistics vs SP on the same 100 days: $\text{mean}(\text{SP} - \text{v3}) = -0.59$, $\text{SE} = 1.69$, v3 wins on 52/100 days. The two policies are **statistically indistinguishable** at this sample size — v3 ties SP without using SP's larger $bf = 3$ tree.

### Takeaways

- The biggest single lever was tactical lookahead depth, not value-function fidelity. Once $V_\theta$ is "good enough," the MILP wants more stages of exact planning more than it wants a smarter tail.
- The fitted-VI $V_\theta$ trained with Bellman targets and $M = 20$ MC scenarios per state is much lower-variance than per-state MC returns, even though the latter come from a stronger source policy.
- License-bound trade-off: with the full Gurobi license, $bf = 3, ns = 3$ would likely beat SP outright (more branching width *and* the $V_\theta$ tail). Under the size-limited license, $bf = 2, ns = 3$ is the best corner of the trade-off space we can reach.

---

## Attempt 5 — Fitted backwards induction with per-time-step MLPs

`policies/sp_backward_induction_policy.py`. Same inference structure as Attempt 2 (bf=3 scenario tree, 2 stages, $V_\theta$ at leaves via big-M), but trained differently:

- **State distribution**: roll out the original `sp_policy` for $N = 100$ random init states; bucket visited states by time index $t$.
- **Training**: fitted backwards induction with one **independent** MLP per time step. Walk $t$ from $T-1$ down to $0$; at each step compute Bellman targets $y(x) = \min_u\{c(x,u) + (1/M)\sum_m V_{\theta,t+1}(f(x,u,\omega_m))\}$ and fit a fresh small MLP $V_{\theta,t}$ on them.
- **Optimizer**: scikit-learn's L-BFGS. (A first attempt with PyTorch + Adam at lr=1e-3 produced $R^2 < 0$ everywhere because Adam's normalized step size couldn't move the output bias to $y$'s mean in 300 epochs.)

### Result

Each per-t MLP fit very cleanly: training $R^2$ between **0.92** ($t=9$) and **0.997** (intermediate $t$).

100-day mean cost: **160.49**. Worse than every previous Attempt — SP 139.72, v3 140.31, v1 143.45, v2 145.57.

### Why high training $R^2$ failed to translate

1. **Per-t MLPs overfit.** 100 samples per network for 225 parameters is essentially memorization. Out-of-distribution states (those the MILP visits at inference, which differ from SP's visited states) get noisy predictions.
2. **Bellman bootstrap chain compounds bias.** Each $V_{\theta,t}$ inherits noise from $V_{\theta,t+1}$. With independent per-t networks there is no consistency constraint across time, so bias from overfitting at one step propagates and amplifies.

### Takeaway

Independent per-t function approximators are the textbook formulation of fitted backwards induction, but at our sample budget (100 SP rollouts ≈ 5 min) they overfit hard. A shared time-conditioned MLP (as in Attempts 2–4) pools all $(t, x, y)$ tuples — 10× the data per parameter — and regularizes value across time through its shared weights. That trade is decisively in favor of the shared MLP here.
