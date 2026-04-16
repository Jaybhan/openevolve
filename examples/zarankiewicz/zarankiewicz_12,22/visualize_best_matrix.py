#!/usr/bin/env python3
"""
Publication-ready visualization of the best K_{3,3}-free binary matrix
found for the Zarankiewicz problem z(12, 22; 3, 3).

Black cells = 1 (edge present), White cells = 0 (edge absent).

Output: best_matrix_12x22.pdf  (and .png at 300 dpi)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load matrix ───────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX_PATH = os.path.join(HERE, ".best_matrix.npy")

if not os.path.exists(MATRIX_PATH):
    raise FileNotFoundError(
        f"No best matrix found at {MATRIX_PATH}. "
        "Run the evaluator at least once to generate it."
    )

A = np.load(MATRIX_PATH).astype(int)
M, N = A.shape   # rows, cols (should be 12, 22)
num_ones = int(A.sum())

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      8,
    "pdf.fonttype":   42,   # embed fonts as TrueType (journal requirement)
    "ps.fonttype":    42,
})

CELL  = 0.38   # inches per cell
PAD_L = 0.55   # left margin (row labels)
PAD_R = 0.20   # right margin
PAD_B = 0.65   # bottom margin (col labels + legend)
PAD_T = 0.45   # top margin (title)

fig_w = PAD_L + N * CELL + PAD_R
fig_h = PAD_B + M * CELL + PAD_T

fig = plt.figure(figsize=(fig_w, fig_h))

ax_l = PAD_L / fig_w
ax_b = PAD_B / fig_h
ax_w = N * CELL / fig_w
ax_h = M * CELL / fig_h
ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])

# ── Draw grid cells ───────────────────────────────────────────────────────────
for i in range(M):
    for j in range(N):
        fc = "black" if A[i, j] == 1 else "white"
        rect = mpatches.Rectangle(
            (j, M - 1 - i), 1, 1,
            linewidth=0.4,
            edgecolor="#aaaaaa",
            facecolor=fc,
        )
        ax.add_patch(rect)

ax.set_xlim(0, N)
ax.set_ylim(0, M)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_linewidth(0.5)


# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = os.path.join(HERE, "best_matrix_12x22.pdf")
out_png = os.path.join(HERE, "best_matrix_12x22.png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
fig.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Matrix shape : {M} x {N}")
print(f"Edges (ones) : {num_ones}")
print(f"Saved → {out_pdf}")
print(f"Saved → {out_png}")
