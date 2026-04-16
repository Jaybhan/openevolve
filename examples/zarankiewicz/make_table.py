#!/usr/bin/env python3
"""
Produce a publication-ready figure showing Zarankiewicz lower bounds (n_sota)
vs KST upper bounds across all (x, y) problem instances.

Folders whose names start with '*' are excluded.
Output: zarankiewicz_results.pdf  (and .png at 300 dpi)
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# ── Data collection ───────────────────────────────────────────────────────────
SKIP = {"known_bounds", "successes", "to_be_improved", "extensively_tested"}
BASE = os.path.dirname(os.path.abspath(__file__))


def parse_mn(folder_name):
    m = re.search(r"(\d+),(\d+)", folder_name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def read_upper_bound(evaluator_path):
    try:
        with open(evaluator_path) as f:
            for line in f:
                m = re.match(r"\s*KST_UPPER_BOUND\s*=\s*(\d+)", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return None


def read_n_sota(folder_path):
    try:
        with open(os.path.join(folder_path, ".n_sota")) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


rows = []
for name in sorted(os.listdir(BASE)):
    if name in SKIP:
        continue
    if name.startswith("*"):          # exclude starred folders
        continue
    folder = os.path.join(BASE, name)
    if not os.path.isdir(folder):
        continue
    evaluator = os.path.join(folder, "evaluator.py")
    if not os.path.exists(evaluator):
        continue
    x, y = parse_mn(name)
    upper  = read_upper_bound(evaluator)
    n_sota = read_n_sota(folder)
    if x and y and upper is not None and n_sota is not None:
        rows.append((x, y, n_sota, upper))

# Previously established upper bounds for cells without an OpenEvolve run.
# Keys are (m, n); values are the KST / literature upper bounds.
STATIC_BOUNDS = {
    (9, 17): 81,  (9, 18): 85,  (9, 19): 89,  (9, 20): 93,  (9, 21): 96,  (9, 22): 100,
    (10, 17): 90, (10, 18): 94, (10, 19): 98,  (10, 20): 102,
    (11, 17): 96, (11, 18): 101,
}

rows.sort(key=lambda r: (r[0], r[1]))

# Expand all_x / all_y to cover static-bound cells too
all_x = sorted({r[0] for r in rows} | {k[0] for k in STATIC_BOUNDS})
all_y = sorted({r[1] for r in rows} | {k[1] for k in STATIC_BOUNDS})
xi = {v: i for i, v in enumerate(all_x)}
yi = {v: i for i, v in enumerate(all_y)}

nx, ny = len(all_x), len(all_y)

# Build matrices
gap_mat   = np.full((nx, ny), np.nan)   # absolute gap
lower_mat = np.full((nx, ny), np.nan)
upper_mat = np.full((nx, ny), np.nan)

for x, y, n_sota, upper in rows:
    i, j = xi[x], yi[y]
    gap_mat[i, j]   = upper - n_sota
    lower_mat[i, j] = n_sota
    upper_mat[i, j] = upper

# ── Figure layout ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "pdf.fonttype":     42,   # embed fonts as TrueType (required by many journals)
    "ps.fonttype":      42,
})

cell_w, cell_h = 0.52, 0.46          # inches per cell
fig_w = cell_w * ny + 1.4            # + margins / colorbar
fig_h = cell_h * nx + 0.9

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# ── Colour map ────────────────────────────────────────────────────────────────
# Optimal (gap=0) → deep teal; larger gap → pale yellow → orange-red
OPTIMAL_COLOR = "#2a7d6e"
GAP_CMAP_NAME = "YlOrRd"

max_gap = int(np.nanmax(gap_mat)) if not np.all(np.isnan(gap_mat)) else 1
gap_cmap  = plt.get_cmap(GAP_CMAP_NAME)
gap_norm  = mcolors.Normalize(vmin=1, vmax=max_gap)

MISSING_COLOR = "#e8e8e8"

# Draw cells manually so we can mix optimal colour with gradient
for i, x in enumerate(all_x):
    for j, y in enumerate(all_y):
        g = gap_mat[i, j]
        static_ub = STATIC_BOUNDS.get((x, y))

        if not np.isnan(g):
            fc = OPTIMAL_COLOR if g == 0 else gap_cmap(gap_norm(g))
        else:
            fc = MISSING_COLOR

        rect = mpatches.FancyBboxPatch(
            (j - 0.48, i - 0.44), 0.96, 0.88,
            boxstyle="round,pad=0.02",
            linewidth=0.4,
            edgecolor="#999999",
            facecolor=fc,
            zorder=2,
        )
        ax.add_patch(rect)

        if not np.isnan(g):
            # Computed cell: show lower (bold) over upper
            lo = int(lower_mat[i, j])
            up = int(upper_mat[i, j])
            text_color = "white" if (g == 0 or g >= max_gap * 0.55) else "#222222"
            ax.text(
                j, i + 0.13, str(lo),
                ha="center", va="center",
                fontsize=6, fontweight="bold",
                color=text_color, zorder=3,
            )
            ax.text(
                j, i - 0.16, str(up),
                ha="center", va="center",
                fontsize=5.5,
                color=text_color, alpha=0.85, zorder=3,
            )
            ax.plot(
                [j - 0.22, j + 0.22], [i - 0.01, i - 0.01],
                color=text_color, lw=0.4, alpha=0.5, zorder=3,
            )
        elif static_ub is not None:
            # Previously established: show upper bound centred, in muted italic
            ax.text(
                j, i, str(static_ub),
                ha="center", va="center",
                fontsize=5.5, style="italic",
                color="#555555", zorder=3,
            )

# ── Axes cosmetics ────────────────────────────────────────────────────────────
ax.set_xlim(-0.55, ny - 0.45)
ax.set_ylim(-0.55, nx - 0.45)
ax.set_xticks(range(ny))
ax.set_yticks(range(nx))
ax.set_xticklabels([str(v) for v in all_y])
ax.set_yticklabels([str(v) for v in all_x])
ax.set_xlabel("$n$  (columns)", labelpad=6)
ax.set_ylabel("$m$  (rows)", labelpad=6)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# light grid lines between cells
for j in range(ny + 1):
    ax.axvline(j - 0.5, color="#cccccc", lw=0.4, zorder=1)
for i in range(nx + 1):
    ax.axhline(i - 0.5, color="#cccccc", lw=0.4, zorder=1)

ax.invert_yaxis()   # small x at top

# ── Colorbar for gap ──────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=gap_cmap, norm=gap_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.025, aspect=30)
cbar.set_label("Gap  (upper − lower)", labelpad=6, fontsize=8)
cbar.outline.set_linewidth(0.4)
cbar.ax.tick_params(labelsize=7)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=OPTIMAL_COLOR, edgecolor="#999", linewidth=0.4,
                   label="Optimal  (gap = 0)"),
    mpatches.Patch(facecolor=MISSING_COLOR, edgecolor="#999", linewidth=0.4,
                   label="Previously established"),
]
ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.0, -0.10),
    ncol=2,
    frameon=False,
    fontsize=7.5,
)

# Cell annotation key (small note)
"""
fig.text(
    0.13, 0.005,
    "Each cell: bold = lower bound  /  regular = KST upper bound",
    ha="left", va="bottom", fontsize=6.5, color="#555555",
)
"""

fig.tight_layout()

out_pdf = os.path.join(BASE, "zarankiewicz_results.pdf")
out_png = os.path.join(BASE, "zarankiewicz_results.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
