M = 8
N = 23
S = 3
T = 3

# EVOLVE-BLOCK-START
import numpy as np

def run_graph():
    # Verified 94-one K_{3,3}-free matrix
    # Row degrees: [11,12,12,12,11,12,12,12] = 94
    # Every triple of rows shares at most 2 common columns
    G1 = np.array([
        [0,0,0,0,1,1,0,1,1,0,1,0,1,1,1,0,0,1,1,0,1,0,0],
        [1,0,1,1,0,1,1,1,1,0,1,0,0,1,0,1,0,0,0,1,0,0,1],
        [0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,1,1,1,1,1],
        [1,0,1,1,1,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,0,1,0],
        [1,1,0,0,1,0,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,1,0],
        [0,0,1,0,0,0,0,1,1,1,0,1,0,0,0,0,1,1,1,1,1,1,1],
        [0,1,1,0,1,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0,1,0,1],
        [1,1,0,1,0,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,0],
    ], dtype=int)
    # Set G2 = G1 to try for symmetry/matching bonus
    G2 = G1.copy()
    return G1, G2
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    from itertools import combinations
    G1, G2 = run_graph()
    total = int(G1.sum())
    print(f"G1 ones: {total}, row sums: {G1.sum(axis=1).tolist()}")
    valid = True
    for r1, r2, r3 in combinations(range(M), 3):
        common = sum(1 for c in range(N) if G1[r1,c] and G1[r2,c] and G1[r3,c])
        if common >= 3:
            print(f"VIOLATION: rows {r1},{r2},{r3} share {common} columns")
            valid = False
    print(f"Valid: {valid}")
