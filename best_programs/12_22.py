M = 12  # number of rows
N = 22  # number of columns
S = 3   # no K_{S,T} subgraph allowed
T = 3

# EVOLVE-BLOCK-START
import numpy as np

def construct_graphs():
    # Circulant construction on GF(11) + 1 extra row
    # Block1 (cols 0-10): row i has 1s at (i+d)%11 for d in D1
    # Block2 (cols 11-21): row i has 1s at (i+d)%11 for d in D2
    # D1 and D2 chosen so any 2 rows share <=2 cols in each block
    # and cross-block overlaps stay controlled
    D1 = [1, 3, 4, 5, 9]  # QR mod 11, |D1|=5, pairwise diff multiset has each residue <=2 times
    D2 = [0, 2, 6, 7, 8, 10]  # complement, |D2|=6

    G = np.zeros((M, N), dtype=int)
    for i in range(11):
        for d in D1:
            G[i, (i + d) % 11] = 1
        for d in D2:
            G[i, 11 + (i + d) % 11] = 1

    # Row 11: greedy fill
    from itertools import combinations
    def valid(mat):
        for r in combinations(range(M), 3):
            if np.sum(mat[r[0]] & mat[r[1]] & mat[r[2]]) >= 3:
                return False
        return True

    for j in range(N):
        G[11, j] = 1
        if not valid(G):
            G[11, j] = 0

    # Greedy improve all cells
    for i in range(M):
        for j in range(N):
            if G[i, j] == 0:
                G[i, j] = 1
                if not valid(G):
                    G[i, j] = 0

    # Second pass in reverse order for potentially different improvements
    for i in range(M - 1, -1, -1):
        for j in range(N - 1, -1, -1):
            if G[i, j] == 0:
                G[i, j] = 1
                if not valid(G):
                    G[i, j] = 0

    G2 = G.copy()
    return G, G2

def run_graph():
    np.random.seed(42)
    return construct_graphs()
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    G1, G2 = run_graph()
    print(f"G1 shape: {G1.shape}, ones: {G1.sum()}, ones/row: {G1.sum(axis=1).tolist()}")
    print(f"G2 shape: {G2.shape}, ones: {G2.sum()}")
    print("G1:\n", G1)
