# Deep ADP — a pedagogical walkthrough

This document explains the four policies we built (`deep_adp_policy`, `deep_adp_milp_policy`, `deep_adp_milp_v2_policy`, `deep_adp_milp_v3_policy`) from first principles. It assumes you are comfortable with the stochastic programming and linear ADP we already use, but it does **not** assume any familiarity with deep RL, the big-M neural-net-in-MILP encoding, or fitted value iteration.

The result we end up with: a policy that ties multistage SP on the 100-day evaluation (140.31 vs 139.72, mean paired difference 0.59 with standard error 1.69), built by wrapping a small neural network value function inside the SP scenario tree.

## 1. The problem, restated

10-hour HVAC control for a two-room restaurant.

**State at time $t$** (the dictionary the simulator passes to `select_action`):

| Symbol | Meaning |
|---|---|
| $T^1_t, T^2_t$ | room temperatures |
| $H_t$ | humidity |
| $\text{Occ}^1_t, \text{Occ}^2_t$ | occupancies |
| $p_t$ | electricity price at $t$ |
| $p_{t-1}$ | electricity price at $t-1$ (needed by the AR price process) |
| $vc_t$ | how many consecutive past hours had ventilation on |
| $\ell^1_t, \ell^2_t$ | low-temperature override flags |

**Action**: $u_t = (p^1_t, p^2_t, v_t)$ — heating powers (continuous, in $[0, 3]$ kW) and ventilation (binary, with a 3-hour minimum up-time).

**Immediate cost**: $c(x_t, u_t) = p_t \cdot (p^1_t + p^2_t + P_{\text{vent}} v_t)$.

**Dynamics**: linear thermal/humidity ODE discretized to hourly steps. Plus the overrules:

- High-temp: if $T^i \ge 26$, $p^i$ is forced to 0.
- Low-temp: if $T^i < 18$, $p^i$ is forced to $P_{\max}$; the override persists until $T^i \ge 22$ (hysteresis).
- Humidity: if $H > 70$, $v$ is forced to 1.
- Ventilation min-up-time: once $v$ flips on it stays on for at least 3 hours.

**Stochastics**: $(p, \text{Occ})$ at the next hour are exogenous random — the simulator draws them from `price_model` and `next_occupancy_levels`.

**Horizon**: $T = 10$. **Objective**: minimize expected total cost.

## 2. The Bellman equation — and why we'll never solve it exactly

The optimal policy is characterized by the **Bellman optimality equation**:

$$V_t(x) = \min_{u \in \mathcal{U}(x)} \Big\{\, c(x, u) + \mathbb{E}_{\omega} \big[ V_{t+1}(f(x, u, \omega)) \big] \,\Big\}, \quad V_T(\cdot) = 0.$$

In plain language: the value of being in state $x$ at time $t$ is the best you can do *right now* (immediate cost) plus the expected value of where you end up *next*. The minimum is over feasible actions; the expectation is over the random next-period exogenous $\omega = (\text{Occ}_{t+1}, p_{t+1})$.

If we knew $V_t$ for every $(t, x)$, we'd be done — every policy decision is just one $\min_u$. We can't compute $V_t$ exactly because:

- The state is continuous (temperatures, humidity, prices), so the table is infinite.
- The expectation is over continuous noise.
- The minimization is a constrained MIP with continuous and binary variables.

The whole subject of **Approximate Dynamic Programming (ADP)** is: pick a parametric approximation $V_\theta \approx V$ and live with it.

The existing `adp_policy.py` uses a **linear** approximation, $V_\theta(x) = \theta^\top \phi(x)$, with $\phi(x)$ an 11-dim feature vector. That fits the Bellman backup nicely because the continuation term $\mathbb{E}[\theta^\top \phi(x_{t+1})]$ stays linear in the decisions, so each per-step planning problem is the same MILP as SP plus a linear continuation cost. It is exactly what `adp_policy.py` does — and it gets 197.95 mean cost on 100 days, **worse than even the dummy policy (182.91)**. That's our starting point.

The reason the linear ADP fails here is that the true value function $V_t$ depends on *interactions* between features — for example, "low-temperature override × current price × remaining horizon" is far more informative than any single feature in isolation, but a linear model can't represent products. We needed a non-linear approximator.

## 3. Attempt 1 — Replace the linear $V_\theta$ with a small MLP

File: `policies/deep_adp_policy.py`.

The idea is mechanical: keep the fitted-value-iteration loop, but make $V_\theta$ a small neural network instead of a linear function.

### 3.1 Architecture

$V_\theta$ is a feed-forward network:

```
input (12 features) → Linear(12 → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → 1)
```

The 12 features are the normalized state plus *two* representations of time (`t/T` and `1 - t/T`). Including time as an input lets one network span the whole horizon instead of training one model per hour.

### 3.2 The price of nonlinearity: no more MILP at inference

The linear ADP's per-step Bellman is a MILP because $\theta^\top \phi(x_{t+1})$ is linear in the decision variables. Once $V_\theta$ is a ReLU network, the continuation term is non-linear in those variables, and the MILP would need to encode the network's piecewise-linear shape (we'll come back to this in Attempt 2).

In Attempt 1 we punt: we discretize the action space and brute-force the min.

```
For each state x:
    candidates = constrained_grid_of_actions(x)            # 7 levels × 7 × {0,1}
    samples    = M scenarios of next-step exogenous ω      # M = 20
    for each candidate u:
        cost(u) = c(x, u) + (1/M) Σ V_θ(f(x, u, ω_m))      # batch all of these through V_θ
    return argmin_u cost(u)
```

`constrained_grid_of_actions` already enforces all overrules: if $\ell^1 = 1$ the only candidate $p^1$ is $P_{\max}$; if $H > 70$ or $0 < vc < 3$ then $v = 1$ is the only candidate. The action returned by the policy therefore passes through the simulator's post-policy overrule logic unchanged.

The Monte Carlo expectation with $M = 20$ samples is **common random numbers** across all candidate actions: we draw $M$ scenarios once per state and re-use them, so the noise cancels out when ranking actions.

### 3.3 Training: forward-backward fitted value iteration

We can't compute $V_t$ analytically, so we generate data and fit a regression.

```
init V_θ ← None (returns 0)
for outer iter i = 1..I:
    # FORWARD — sample trajectories under the current V_θ (ε-greedy)
    for n = 1..N:
        x ← random init state
        for t = 0..T-1:
            with prob ε: pick a random feasible action
            else:        pick argmin from one_step_value(x, V_θ)
            advance x using simulator dynamics + sampled exogenous
            record x in trajectory n

    # BACKWARD — build (state, target) regression data
    targets = []
    for t = T-1, T-2, ..., 0:
        for each visited state x at time t:
            y ← one_step_value(x, V_θ).min        # Bellman target
            targets.append((x, y))

    # FIT — retrain V_θ on all (state, target) pairs, warm-started
    V_θ ← MLP.fit(targets, lr=1e-3, epochs=300)
```

Two things to notice:

1. **Backward in time.** We compute targets from $t = T-1$ first, then $t = T-2$, etc. Targets at $t = T-1$ rely only on the immediate cost (since $V_T \equiv 0$); targets at earlier $t$ rely on $V_\theta$ being already fit for later $t$. But we use the *same* $V_\theta$ everywhere, because we cleverly included time as an input feature: the network "remembers" which $t$ each training sample came from.

2. **ε-greedy exploration.** The first iteration starts with $V_\theta = 0$, so pure greedy would always pick the zero-cost action and never visit any interesting state. We add $\varepsilon$-greedy randomization, decaying from 0.4 to 0.05 over the outer iterations, so the early rollouts cover a broad range of state regions.

### 3.4 Result

100-day mean cost: **174.08**. Beats the dummy (182.91) and the linear ADP (197.95), but loses badly to SP (139.72). The diagnosis:

- The grid is coarse — 0.5 kW resolution per heater, and a one-step-suboptimal heater setting costs real money over a 10-hour horizon.
- The lookahead is one step — everything past $t+1$ is delegated to $V_\theta$, which has noisy approximation error at every state.
- The training set is small — N = 25 trajectories × 10 steps = 250 samples per outer iter.
- The Monte Carlo expectation has only $M = 20$ samples, so small action-value gaps are buried in noise.

We can fight some of these (finer grid, more training), but the structural fix is to give the planning step *real* lookahead instead of immediately deferring to $V_\theta$. That's Attempt 2.

## 4. Attempt 2 — A 2-stage MILP scenario tree with $V_\theta$ embedded at the leaves

File: `policies/deep_adp_milp_policy.py`.

This is the central idea of the project. We **combine SP's tactical near-term planning with deep RL's tail value function** in a single optimization problem.

### 4.1 The picture

At each time $t$ we build the same scenario tree that `sp_policy` does: branching factor $bf = 3$, two stages. The root is the current state (fixed). Stage 1 nodes are three possible realizations of next-period exogenous (occupancy + price), produced by KMeans-clustering a sample from the process models. Stage 2 nodes are the leaves.

```
                              x_0  (current state, fixed)
                              / | \
                          ___/  |  \___
                         /      |      \
                  x_1^1       x_1^2     x_1^3       (stage 1 — 3 children of root)
                 / | \        / | \      / | \
              x_2^1..x_2^3   ...        ...          (stage 2 — 9 leaves)
                                                      V_θ evaluated here
```

The MILP has continuous heating-power decision variables and binary ventilation decision variables at the root and at each of the three stage-1 children — 4 decision-bearing nodes total. Leaves carry only state variables and a $V_\theta$ evaluation.

The objective is the expected cost over the tree:

$$\min \quad c(x_0, u_0) + \sum_{k=1}^{3} p_k \, c(x_{1,k}, u_{1,k}) + \sum_{k=1}^{3}\sum_{j=1}^{3} p_{kj} \, V_\theta(x_{2,kj}),$$

where $p_k$ are the scenario probabilities (from KMeans cluster sizes) and $V_\theta$ is our small neural network. The MILP has the full set of `sp_policy` constraints — thermal dynamics, low- and high-temperature override hysteresis, humidity override, ventilation min-up-time, $vc$ propagation — plus, new for this policy, the encoding of $V_\theta$ at every leaf.

### 4.2 Embedding a ReLU network in a MILP — the big-M trick

This is the technical part of Attempt 2. We want, at each leaf $\ell$, the MILP to know:

$$V_\theta(x_\ell) = W_2 \cdot \mathrm{ReLU}(W_1 x_\ell + b_1) + b_2.$$

If $x_\ell$ is a constant we just plug in. But $x_\ell$ contains decision-dependent state variables ($T^1, T^2, H, vc, \ell^1, \ell^2$), so we need this as a constraint inside the MILP.

The linear parts of the network are easy — they're just linear expressions over decision variables. The non-trivial step is the ReLU, $h_j = \max(0, a_j)$, where $a_j = (W_1 x_\ell + b_1)_j$ is a linear expression we can write down but $\max$ is not natively a MILP construct.

**The encoding.** For each ReLU unit $j$, introduce a binary variable $z_j \in \{0, 1\}$ and a continuous variable $h_j \ge 0$. With pre-activation bounds $L_{a_j} \le a_j \le U_{a_j}$ (computed offline; see below), the four constraints

$$h_j \ge 0, \quad h_j \ge a_j, \quad h_j \le U_{a_j} \cdot z_j, \quad h_j \le a_j - L_{a_j}(1 - z_j)$$

are equivalent to $h_j = \max(0, a_j)$. To see why: $z_j = 1$ means "the unit is active" — the third constraint becomes vacuous and the fourth forces $h_j \le a_j$, which combined with $h_j \ge a_j$ gives $h_j = a_j$. $z_j = 0$ means "the unit is inactive" — the third constraint forces $h_j \le 0$, combined with $h_j \ge 0$ gives $h_j = 0$, and the fourth constraint $h_j \le a_j - L_{a_j}$ is vacuous (since $L_{a_j} \le 0$, $-L_{a_j} \ge 0$).

**A tiny example.** Suppose $a = 2x + 1$ with $x \in [-1, 1]$, so $a \in [-1, 3]$, $L_a = -1$, $U_a = 3$.

- At $x = 0$: $a = 1$, ReLU gives $h = 1$. Solver chooses $z = 1$; constraints reduce to $h \ge 1$ and $h \le 1$, so $h = 1$. ✓
- At $x = -1$: $a = -1$, ReLU gives $h = 0$. Solver chooses $z = 0$; constraints reduce to $h \ge 0$ (trivial), $h \le 0$, and $h \le -1 + 1 = 0$, so $h = 0$. ✓

**Bounds on $a_j$.** Tight bounds matter — they make the MILP solve faster because the LP relaxation is tighter. We compute them once by interval propagation through $W_1$ from the feature ranges $[L_x, U_x]$:

$$L_{a_j} = b_j + \sum_i \min(W_{ji} L_{x_i}, W_{ji} U_{x_i}), \qquad U_{a_j} = b_j + \sum_i \max(W_{ji} L_{x_i}, W_{ji} U_{x_i}).$$

If $L_{a_j} \ge 0$ the unit is always active — we drop the binary entirely and set $h_j = a_j$. If $U_{a_j} \le 0$ it's always inactive — set $h_j = 0$. Both cases save Gurobi binary variables.

For a 1-hidden-layer MLP with 32 ReLU units, each leaf adds 32 binaries × 9 leaves = 288 new binary variables to the MIP. This is why Attempt 2 trains and uses a **smaller** $V_\theta$ (1 hidden layer × 32 units) than Attempt 1's (2 × 64) — the smaller MLP is what makes the embedding tractable.

### 4.3 Training $V_\theta$ for this policy

Same forward-backward fitted-VI loop as Attempt 1, just on the smaller architecture (`hidden_sizes = (32,)`). The training data is generated by the grid-search action selection of Attempt 1 (it does not yet know about the MILP). We'll revisit this choice in Attempt 3.

### 4.4 Result

100-day mean cost: **143.45**. We closed ~88 % of the gap from Attempt 1 (174.08) to SP (139.72). The remaining 3.7-unit gap looks like noise but is consistent across runs.

The pattern is: once the MILP handles the near-term coordination of overrules, ventilation lockout, and price spikes *exactly*, the value function only has to estimate cost-to-go two steps out, which is a much easier regression problem than "the entire remaining horizon".

## 5. Attempt 3 — On-policy retraining with Monte Carlo returns

File: `policies/deep_adp_milp_v2_policy.py`.

A possible objection to Attempt 2 is that $V_\theta$ was trained against a *different policy* (the grid-search policy of Attempt 1) than the one that consumes it (the MILP rollout). Attempt 3 fixes this in the most direct way possible.

### 5.1 The procedure

```
V_θ ← initialized from Attempt 2's trained model
for outer iter k = 1..6:
    # 1. Roll out the *current* MILP-rollout policy for N = 25 days
    rollouts = []
    for n = 1..25:
        x = sample_init_state()
        for t = 0..T-1:
            u = MILP_rollout_select_action(x; V_θ)
            record (x, immediate_cost(x, u))
            x = simulator_advance(x, u, sample_exog())
        rollouts.append(trajectory)

    # 2. For each (t, x_t) we visited, compute the Monte Carlo cost-to-go
    targets = []
    for traj in rollouts:
        for t in 0..T-1:
            y_t = Σ_{s=t..T-1} immediate_cost(s)
            targets.append((traj[t], y_t))

    # 3. Refit V_θ on (state, return) pairs, warm-started
    V_θ ← MLP.fit(targets, lr=5e-4, epochs=400)
```

This is **on-policy Monte Carlo policy evaluation**. The targets $y_t$ are unbiased samples of $V^\pi_t(x_t)$ where $\pi$ is the *current* MILP rollout policy. As we iterate, the policy improves (because $V_\theta$ improves), the state distribution shifts toward the better policy, and the data tracks it. No Bellman bootstrap.

### 5.2 The result and the lesson

100-day mean cost: **145.57** — *slightly worse* than Attempt 2's 143.45.

The reason is a classical bias–variance trade-off. The Attempt 2 fitted-VI targets are

$$y_t = \min_u \Big\{ c(x_t, u) + \frac{1}{M} \sum_m V_\theta^{\text{old}}(f(x_t, u, \omega_m)) \Big\},$$

with $M = 20$ Monte Carlo scenarios. That's a *low-variance* target (20-sample average) but slightly *biased* (uses the old $V_\theta$). The Attempt 3 targets are the realized cost-to-go from a single rollout — *unbiased* but high-variance. With $N = 25$ rollouts and 10 steps, an early-$t$ target has standard deviation roughly equal to its mean (sd ≈ 75, mean ≈ 95). The MLP fits the noise and we lose a little.

R² on the regression dataset went from 0.92 (Attempt 2) to 0.80–0.88 (Attempt 3) — the network is fitting noisier targets, and the resulting value function is noisier.

The deeper lesson is that the "wrong-policy" critique was real but second-order: the grid-search and MILP-rollout policies visit *similar* state distributions because they share the same simulator dynamics and respect the same overrules. The shift mattered less than we hoped.

## 6. Attempt 4 — Deeper tree with the original $V_\theta$

File: `policies/deep_adp_milp_v3_policy.py`.

If $V_\theta$ is already "good enough" (Attempt 2's fitted-VI version), the remaining gap to SP must be in the *planning depth*. SP plans 3 stages of exact decisions before its terminal $V \equiv 0$ takes over; Attempt 2 plans 2 stages and then defers to $V_\theta$. The natural fix is to plan a third stage.

### 6.1 The license trade-off

A full 3-stage tree with $bf = 3$ has $3 + 9 + 27 = 39$ non-root nodes. Each of the 27 leaves requires 32 ReLU binaries from the $V_\theta$ encoding — 864 binaries from the network alone, plus the existing per-node decision/override/inertia binaries from the SP structure. The **size-limited Gurobi license** rejects this MIP outright.

So we make a different trade-off: keep 3 stages of tactical lookahead, but reduce the branching to $bf = 2$. The tree has $2 + 4 + 8 = 14$ non-root nodes; the leaves contribute $8 \times 32 = 256$ ReLU binaries — comparable to Attempt 2's 288 — and the MILP fits in the license.

We are trading SP's wider uncertainty representation ($bf = 3$) for one more stage of exact planning. The bet is that *temporal* lookahead matters more than *width* of scenarios at each stage, given that $V_\theta$ already absorbs the long-run uncertainty for us.

### 6.2 The companion experiment: distilling SP into $V_\theta$

Attempt 4 also tries a different way of training $V_\theta$ — **distillation from SP**:

```
states, returns = []
for n = 1..100:
    x = sample_init_state()
    for t = 0..T-1:
        u = sp_policy.select_action(x)              # roll out SP, not the MILP
        record (x, immediate_cost(x, u))
        x = simulator_advance(x, u, sample_exog())
    Σ_{s=t..T-1} immediate_cost forms MC return at each t
V_θ ← MLP.fit(states, returns)
```

The hypothesis is that SP gives much better targets than grid-search or v2's noisier on-policy rollouts: SP is a strong policy, and the rolled-out cost-to-go *is* its on-policy value.

It didn't work. R² on the regression set was 0.78, far below Attempt 2's 0.92, and the resulting policy scored 150.38 (under the $bf=2, ns=3$ tree) — worse than just using v1's $V_\theta$ under the same tree (140.31). Again, the issue is **variance**: SP's realized cost-to-go has sd / mean ≈ 0.75, and 100 rollouts of 10 steps is only 1000 samples — too few to average out the noise in a 449-parameter MLP.

So Attempt 4's *production* policy keeps v1's fitted-VI $V_\theta$ and only changes the tree depth. The distillation training stays in the file as `train_distill_attempt` for the record.

### 6.3 The full comparison

| Policy | Tree | $V_\theta$ source | 100-day mean | Wins vs SP |
|---|---|---|---|---|
| dummy | — | — | 182.91 | — |
| linear ADP | — | linear features | 197.95 | — |
| Deep ADP (Attempt 1) | — (grid search) | fitted-VI, 2×64 | 174.08 | — |
| MILP rollout v1 (Attempt 2) | $bf=3, ns=2$ | fitted-VI, 1×32 | 143.45 | 27/100 |
| MILP rollout v2 (Attempt 3) | $bf=3, ns=2$ | on-policy MC, 1×32 | 145.57 | 22/100 |
| MILP rollout v3, distilled | $bf=2, ns=3$ | SP-distill MC, 1×32 | 150.38 | 37/100 |
| **MILP rollout v3, fitted-VI** | $bf=2, ns=3$ | fitted-VI, 1×32 | **140.31** | **52/100** |
| SP (multistage) | $bf=3, ns=3$ | none (V=0 terminal) | 139.72 | — |

Paired statistics on the v3 vs SP comparison: mean(SP − v3) = −0.59 with standard error 1.69. They are **statistically indistinguishable** at this sample size, with v3 winning slightly more days than it loses.

## 7. Attempt 5 — Fitted backwards induction with per-time-step MLPs

File: `policies/sp_backward_induction_policy.py`.

A textbook-style variant of Attempt 2 that swaps two pieces:

- **Training procedure**: fitted backwards induction instead of forward-backward fitted VI.
- **Function approximator**: ten *independent* small MLPs $V_{\theta,t}$ — one per time step $t = 0,\dots,T-1$ — instead of one shared time-conditioned network.

State distribution: roll out the **original SP policy** for $N = 100$ random init states and bucket the visited $x$'s by their time index $t$. This treats SP as the reference policy whose value function we are estimating.

Then sweep backwards from $t = T-1$ to $t = 0$:

```
V_θ_T ≡ 0
for t = T-1, T-2, ..., 0:
    for each x in the t-th state bucket:
        y(x) = min_u  c(x, u) + (1/M) Σ_m  V_θ_{t+1}( f(x, u, ω_m) )
    V_θ_t = fit_MLP(states, targets)        # fresh per-t network
```

The min is grid search over the constrained action set (same routine as Attempt 1). Each per-t MLP is tiny — one hidden layer of 16 ReLU units — and is fit with **scikit-learn's L-BFGS** solver, which converges much faster than Adam on small datasets (a lesson we learned the hard way during this attempt: Adam at lr = 1e-3 over 300 epochs can only move a bias by about 0.3, far short of the typical $y_t$ mean of 100 at early $t$, so the first attempt with Adam produced negative $R^2$ everywhere and a useless value function).

### Result

Per-t training $R^2$: between 0.92 (at $t = 9$) and 0.997 (at intermediate $t$). The networks fit their training data essentially perfectly.

100-day evaluation mean cost: **160.49**. Worse than SP (139.72), worse than Attempt 2 (143.45), and worse than the deeper-tree Attempt 4 (140.31).

### Why high-$R^2$ doesn't translate

Two related failures:

1. **The per-t MLPs overfit.** With 100 training samples per $V_{\theta,t}$ and 225 parameters, $R^2 \ge 0.99$ on training data is essentially memorization. At inference, the MILP-rollout's leaf states are *not* drawn from the same distribution as SP's visited states — they come from the continuous-action MILP's recourse decisions, which place mass on slightly different temperatures and humidities. Out-of-distribution, the overfit MLPs predict noise.
2. **The Bellman bootstrap chain compounds bias.** Each $V_{\theta,t}$ inherits the noise of $V_{\theta,t+1}$. Even a 1% bias per step compounds over 10 steps. With a *shared* time-conditioned MLP (Attempts 2/3/4), the network has to satisfy a consistency constraint across $t$ via shared weights, which limits how badly any single time slice can drift.

### Takeaway

Independent per-t function approximators sound clean in theory — the textbook formulation of fitted backwards induction *is* one $\hat V_t$ per $t$. In practice, with the sample budget we can afford (100 SP rollouts ≈ 5 minutes), pooling all $(t, x, y)$ tuples into a single shared MLP with $t$ as an input feature is dramatically more sample-efficient and more robust against bootstrap bias.

---

## 8. Takeaways

1. **The single biggest lever was tactical lookahead depth.** Going from `num_stages = 2` to `num_stages = 3` (Attempt 4) closed three quarters of the remaining gap to SP. By contrast, retraining $V_\theta$ on a strictly better source policy (Attempt 3, on-policy; Attempt 4 distillation) did not help.

2. **Bias is cheaper than variance for value-function regression at this sample size.** The fitted-VI Bellman target, although biased through the bootstrap, averages over $M = 20$ scenarios and is therefore much lower-variance than a single realized return. Until we have an order of magnitude more rollouts, fitted-VI is the right training method for this problem.

3. **Embedding a small neural net inside an SP MILP is genuinely the same problem class.** We pay 256–288 extra binary variables per call and the solver eats them in well under a second. We could not have done this with the original 2×64 architecture of Attempt 1 — the network had to shrink for the embedding to fit in the license, and that's why we maintain two model files (`deep_adp_model.pt` for Attempt 1, `deep_adp_model_small.pt` for everything else).

4. **The simulator's overrule logic is enforced by the MILP constraints, not the simulator's safety net.** Because every feasible action in the MILP already obeys low-temp hysteresis, high-temp cutout, humidity venting, and ventilation min-up-time, the simulator's post-policy override step is a no-op on our outputs. This matters: it means the cost we minimize in the MILP equals the cost the simulator records.

5. **High training $R^2$ is not the same as a good policy** (Attempt 5). Per-t MLPs hit $R^2 \ge 0.99$ on their training sets and still produced a worse policy than a shared, time-conditioned MLP at $R^2 \approx 0.92$. The reason is generalization to the leaf-state distribution induced by the MILP at inference, which is not the same as the SP-visited distribution used in training. A shared MLP regularizes across $t$; per-t MLPs memorize their slice.

## 9. What we'd try next without the license constraint

- **$bf = 3, ns = 3$ with the same fitted-VI $V_\theta$.** Strictly more branching width *and* deeper lookahead. The MIP would have ~955 binaries; modern solvers handle that easily but the size-limited Gurobi license rejects it. This is the most likely "strict win over SP" configuration.
- **Variance-reduced on-policy training.** Average $K \ge 5$ MC rollouts per starting state (common random numbers) before computing the target, or use a control-variate baseline. With enough samples the on-policy idea of Attempt 3 should eventually beat the fitted-VI bootstrap.
- **TD($\lambda$) or n-step bootstrap.** Halfway house between Attempt 2's 1-step Bellman target and Attempt 3's full Monte Carlo return.
