M = 16
N = 15
S = 3
T = 3

# EVOLVE-BLOCK-START
import numpy as np

def construct_graphs():
    # Exact best known 16x15 matrix with 123 ones, no 3x3 all-ones submatrix
    rows = [
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],  # r0: deg 8
        [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0],  # r1: deg 8
        [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0],  # r2: deg 8
        [1,1,0,0,0,0,1,1,0,0,1,1,1,1,0],  # r3: deg 8
        [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0],  # r4: deg 8
        [0,0,1,1,0,0,1,1,1,1,0,0,1,1,0],  # r5: deg 8
        [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],  # r6: deg 8
        [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],  # r7: deg 8
        [1,0,1,0,0,1,0,1,0,1,0,1,1,0,1],  # r8: deg 8
        [1,0,0,1,1,0,0,1,0,1,1,0,0,1,1],  # r9: deg 8
        [1,0,0,1,0,1,1,0,1,0,0,1,0,1,1],  # r10: deg 8
        [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1],  # r11: deg 8
        [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],  # r12: deg 8
        [0,1,0,1,1,0,1,0,0,1,0,1,1,0,1],  # r13: deg 8
        [0,1,0,1,0,1,0,1,1,0,1,0,1,0,1],  # r14: deg 8
        [0,0,0,0,1,1,0,0,0,0,0,0,0,0,1],  # r15: deg 3
    ]
    G1 = np.array(rows, dtype=int)

    # Build G2: add one extra 1 per row where possible
    G2 = G1.copy()
    from itertools import combinations
    def is_valid(mat):
        for r1, r2, r3 in combinations(range(M), 3):
            common = mat[r1] & mat[r2] & mat[r3]
            if np.where(common)[0].shape[0] >= 3:
                return False
        return True

    for i in range(M):
        zeros = np.where(G2[i] == 0)[0]
        for j in zeros:
            G2[i, j] = 1
            if not is_valid(G2):
                G2[i, j] = 0
            else:
                break

    return G1, G2

def run_graph():
    return construct_graphs()
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    G1, G2 = run_graph()
    print(f"G1 ones: {G1.sum()}, per row: {G1.sum(axis=1).tolist()}")
    print(f"G2 ones: {G2.sum()}")
    print("G1:\n", G1)
