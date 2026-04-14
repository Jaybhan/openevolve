M = 9  # number of rows
N = 23  # number of columns
S = 3   # no K_{S,T} subgraph allowed
T = 3

# EVOLVE-BLOCK-START
import numpy as np


def construct_graphs():
    """
    Construct two M×N 0-1 adjacency matrices:

    G1 — the primary K_{3,3}-free candidate (maximizing valid 1s).
         No 3 rows may share 3 or more common 1-columns.
         This is the graph that counts toward z(11,17;3,3).

    G2 — a dense "prospect" graph (may contain K_{3,3} violations).
         Used to provide gradient signal: even invalid dense graphs
         that are close to K_{3,3}-free earn a partial score bonus.
         G2 should push toward or beyond the upper bound; the evaluator
         rewards G2 for being dense relative to its violation count.

    For z(9,23;3,3): upper bound 104 (target).

    Returns:
        (G1, G2): tuple of np.ndarray, each shape (M, N), dtype int, values in {0, 1}
    """
    # G1: circulant-style baseline — 11 ones per row (row offsets mod N=23)
    # Offsets chosen so no 3 rows share 3+ columns; 11*9=99 ones (target 104).
    offsets_g1 = [0, 1, 2, 4, 7, 10, 13, 15, 17, 19, 21]
    G1 = np.zeros((M, N), dtype=int)
    for i in range(M):
        for d in offsets_g1:
            G1[i, (i + d) % N] = 1

    # G2: denser prospect (12 ones per row = 108 total, above the 104 target)
    # The LLM should evolve G2 to push density while minimising K_{3,3} violations.
    offsets_g2 = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    G2 = np.zeros((M, N), dtype=int)
    for i in range(M):
        for d in offsets_g2:
            G2[i, (i + d) % N] = 1

    return G1, G2


def run_graph():
    """Fixed interface called by the evaluator. Returns (G1, G2)."""
    return construct_graphs()
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    G1, G2 = run_graph()
    print(f"G1 shape: {G1.shape}, ones: {G1.sum()}, ones/row: {G1.sum(axis=1).tolist()}")
    print(f"G2 shape: {G2.shape}, ones: {G2.sum()}, ones/row: {G2.sum(axis=1).tolist()}")
    print("G1:\n", G1)
