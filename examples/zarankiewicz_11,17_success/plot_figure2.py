#!/usr/bin/env python3
"""Figure 2: Score (edges in best valid construction) vs. iteration for z(11,17,3,3).

Data: num_edges from best_program_info.json at each checkpoint across all
resumed runs. Each value corresponds to a checkpoint every 10 iterations,
so x = 10, 20, 30, ..., 680.
"""
import os
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
OUT_PATH = os.path.join(ROOT, "figure2_score_vs_iteration.png")
EXACT_VALUE = 96

# Best num_edges recorded at each checkpoint (every 10 iterations, global order).
# 68 checkpoints → iterations 10, 20, ..., 680.
EDGES = [
    67, 78, 78, 89, 89, 89, 89, 89, 91, 91,
    91, 91, 92, 92, 92, 92, 92, 92, 92, 92,
    92, 92, 93, 93, 93, 93, 93, 93, 93, 93,
    93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
    93, 93, 93, 93, 93, 93, 93, 93, 93, 94,
    94, 94, 94, 94, 94, 94, 94, 94, 94, 94,
    94, 94, 94, 94, 94, 95, 96, 96,
]

iters = [10 * (i + 1) for i in range(len(EDGES))]
convergence_iter = next((it for it, e in zip(iters, EDGES) if e >= EXACT_VALUE), None)


def main():
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(iters, EDGES, color="#2563EB", linewidth=2, marker="o",
            markersize=3, label="Best valid construction")

    ax.axhline(EXACT_VALUE, linestyle="--", color="#6B7280", linewidth=1.4,
               alpha=0.7, label=f"Known exact: {EXACT_VALUE}")

    if convergence_iter is not None:
        ax.axvline(convergence_iter, linestyle=":", color="#DC2626",
                   linewidth=1.2, alpha=0.8)
        ax.annotate(
            f"Reaches {EXACT_VALUE}\n(iter {convergence_iter})",
            xy=(convergence_iter, EXACT_VALUE),
            xytext=(convergence_iter + 25, EXACT_VALUE - 9),
            fontsize=9,
            color="#DC2626",
            arrowprops=dict(arrowstyle="-", color="#DC2626", lw=0.8),
        )

    ax.set_xlim(0, max(iters) + 10)
    ax.set_ylim(60, EXACT_VALUE + 6)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Edges in best valid construction", fontsize=11)
    ax.set_title("z(11, 17; 3, 3) — Best Valid Edges vs. Iteration", fontsize=13)
    ax.legend(fontsize=10, framealpha=0.7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
