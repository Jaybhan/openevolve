M = 11
N = 22
S = 3
T = 3

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct_graphs():
    # Circulant construction: two blocks, each 11x11 circulant
    # For K_{3,3}-free: any 3 rows share at most 2 common 1-columns
    # Use sets where triple intersection of shifted copies is ≤ 2

    # Block 1: shift set S1, Block 2: shift set S2
    # Row i, col j in block b: entry = 1 iff (j-i)%11 in S_b

    # Best found: S1 with 7 elements, S2 with 7 elements = 154 ones
    # Need: for any 3 distinct rows (a,b,c), |S1∩(S1+(b-a))∩(S1+(c-a))| + |S2∩(S2+(b-a))∩(S2+(c-a))| ≤ 2

    def check_valid(S1, S2):
        for d1 in range(1, 11):
            for d2 in range(d1+1, 11):
                s1_set = set(S1)
                c1 = len(s1_set & {(x+d1)%11 for x in s1_set} & {(x+d2)%11 for x in s1_set})
                s2_set = set(S2)
                c2 = len(s2_set & {(x+d1)%11 for x in s2_set} & {(x+d2)%11 for x in s2_set})
                if c1 + c2 >= 3:
                    return False
        return True

    best = ([], [], 0)
    for s1_size in range(8, 4, -1):
        for S1 in combinations(range(11), s1_size):
            for s2_size in range(min(22-s1_size*11//11, 8), 4, -1):
                for S2 in combinations(range(11), s2_size):
                    if (len(S1)+len(S2))*11 <= best[2]:
                        continue
                    if check_valid(S1, S2):
                        total = (len(S1)+len(S2))*11
                        if total > best[2]:
                            best = (S1, S2, total)
                            if total >= 154:
                                break
                if best[2] >= 154:
                    break
            if best[2] >= 154:
                break
        if best[2] >= 154:
            break

    S1, S2 = best[0], best[1]
    G1 = np.zeros((M, N), dtype=int)
    for i in range(11):
        for j in range(11):
            if (j - i) % 11 in S1:
                G1[i, j] = 1
            if (j - i) % 11 in S2:
                G1[i, j + 11] = 1

    G2 = G1.copy()
    return G1, G2

def run_graph():
    return construct_graphs()
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    G1, G2 = run_graph()
    print(f"G1 ones: {G1.sum()}, row sums: {G1.sum(axis=1).tolist()}")
