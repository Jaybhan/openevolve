N = 17  # matrix dimension (square bipartite: N left vertices, N right vertices)
S = 3   # no K_{S,T} subgraph allowed
T = 3

# EVOLVE-BLOCK-START
import numpy as np


def construct_graphs():
    """
    Construct two N×N 0-1 adjacency matrices:

    G1 — the primary K_{3,3}-free candidate (maximizing valid 1s).
         No 3 rows may share 3 or more common 1-columns.
         This is the graph that counts toward z(17;3).

    G2 — a dense "prospect" graph (may contain K_{3,3} violations).
         Used to provide gradient signal: even invalid dense graphs
         that are close to K_{3,3}-free earn a partial score bonus.
         G2 should push toward or beyond 141 edges; the evaluator
         rewards G2 for being dense relative to its violation count.

    For z(17,17;3,3): upper bound 141 (Collins et al. 2016, open problem).

    Returns:
        (G1, G2): tuple of np.ndarray, each shape (N, N), dtype int, values in {0, 1}
    """
    # G1: circulant baseline — 4 ones per row, known K_{3,3}-free (68 edges)
    # Offsets chosen so no 3 rows share 3+ columns.
    offsets_g1 = [0, 1, 4, 9]
    G1 = np.zeros((N, N), dtype=int)
    for i in range(N):
        for d in offsets_g1:
            G1[i, (i + d) % N] = 1

    # G2: start with the same valid structure as G1, giving g2_bonus=0.5 immediately.
    # The LLM should evolve G2 to be denser (push toward 141 edges) while keeping
    # violations below the expected-random count to maintain a positive bonus.
    offsets_g2 = [0, 1, 4, 9]
    G2 = np.zeros((N, N), dtype=int)
    for i in range(N):
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
