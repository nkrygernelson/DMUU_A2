"""Task 7 driver: run distributed algorithm under several step sizes, plot
convergence, dump summary tables, save figures used in the writeup."""

import os
import numpy as np
import matplotlib.pyplot as plt


try:
    from . import distributed  # python -m task7.run_task7
except ImportError:
    import distributed          # python run_task7.py from inside task7/


FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
NUM_ITERS = 100


def _ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def _label(alpha, adaptive):
    if adaptive:
        return r"adaptive $\alpha_k=5/(1+k)$"
    return rf"$\alpha={alpha:g}$"


def run_all():
    _ensure_fig_dir()
    print("Solving centralized reference…")
    ref = distributed.solve_centralized()
    obj_star = ref["objective"]
    print(f"  centralized objective = {obj_star:.3f}")

    runs = []
    for alpha in ALPHAS:
        print(f"Running fixed α = {alpha}…")
        runs.append(("fixed", alpha, distributed.run_distributed(
            alpha=alpha, num_iters=NUM_ITERS, adaptive=False)))
    print("Running adaptive α_k = 5/(1+k)…")
    runs.append(("adaptive", None, distributed.run_distributed(
        alpha=0.0, num_iters=NUM_ITERS, adaptive=True, alpha0=5.0)))

    # ---- Figure 1: objective vs iteration ----
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for kind, alpha, r in runs:
        ax.plot(np.arange(1, NUM_ITERS + 1), r["objective"],
                label=_label(alpha, kind == "adaptive"), linewidth=1.4)
    ax.axhline(obj_star, color="black", linestyle="--", linewidth=1.0,
               label=f"centralized optimum = {obj_star:.0f}")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"Primal objective $\sum_{n,r,t} w_n (T_{n,r,t} - T^{\mathrm{ref}})^2$")
    ax.set_yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=True, borderaxespad=0.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    obj_path = os.path.join(FIG_DIR, "obj_vs_iter.png")
    fig.savefig(obj_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {obj_path}")

    # ---- Figure 2: λ_t and residual for all step sizes in a 3x2 grid ----
    T = runs[0][2]["lambdas"].shape[1]
    cmap = plt.get_cmap("viridis", T)
    fig = plt.figure(figsize=(16, 9))
    outer = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30,
                             left=0.06, right=0.90, top=0.94, bottom=0.07)
    iters = np.arange(NUM_ITERS + 1)
    iters2 = np.arange(1, NUM_ITERS + 1)
    line_handles = None
    for i, (kind, alpha, r) in enumerate(runs):
        row, col = i // 3, i % 3
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.08)
        ax_lam = fig.add_subplot(inner[0])
        ax_res = fig.add_subplot(inner[1], sharex=ax_lam)

        handles = []
        for t in range(T):
            h, = ax_lam.plot(iters, r["lambdas"][:, t], color=cmap(t), linewidth=1.0)
            ax_res.plot(iters2, r["residuals"][:, t], color=cmap(t), linewidth=1.0)
            handles.append(h)
        if line_handles is None:
            line_handles = handles

        ax_res.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
        ax_lam.set_title(_label(alpha, kind == "adaptive"), fontsize=11)
        ax_lam.set_ylabel(r"$\lambda_t$", fontsize=9)
        ax_res.set_ylabel(r"$\sum_n p_{n,t} - P^{\mathrm{mall}}$", fontsize=9)
        ax_res.set_xlabel("Iteration $k$", fontsize=9)
        ax_lam.grid(True, alpha=0.3)
        ax_res.grid(True, alpha=0.3)
        plt.setp(ax_lam.get_xticklabels(), visible=False)

    fig.legend(line_handles, [f"t={t}" for t in range(T)],
               loc="center right", bbox_to_anchor=(0.99, 0.5),
               fontsize=9, frameon=True, title="Timeslot")
    path = os.path.join(FIG_DIR, "lambda_residual_all.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

    # ---- Summary table ----
    print("\nFinal objective per step size (lower is better):")
    print(f"  centralized            = {obj_star:11.3f}")
    for kind, alpha, r in runs:
        label = _label(alpha, kind == "adaptive").replace("$", "")
        gap = r["objective"][-1] - obj_star
        max_resid = r["residuals"][-1].max()
        print(f"  {label:25s} final obj = {r['objective'][-1]:11.3f}   "
              f"gap = {gap:+9.3f}   max residual = {max_resid:+6.3f}")

    # Per-store consumption at centralized optimum
    print("\nCentralized mean power per store (kW):")
    mp = ref["p"].sum(axis=(1, 2)) / ref["params"]["num_timeslots"]
    for n in range(distributed.N_STORES):
        print(f"  store {n+1:2d}  w={ref['weights'][n]:4.0f}   mean p = {mp[n]:5.3f}")

    return ref, runs


if __name__ == "__main__":
    run_all()
