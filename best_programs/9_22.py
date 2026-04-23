M = 9  # number of rows
N = 22  # number of columns
S = 3   # no K_{S,T} subgraph allowed
T = 3

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct_graphs():
    def is_valid(mat):
        for r1, r2, r3 in combinations(range(mat.shape[0]), 3):
            if np.sum(mat[r1] & mat[r2] & mat[r3]) >= 3:
                return False
        return True

    def triple_violations(mat):
        count = 0
        for r1, r2, r3 in combinations(range(mat.shape[0]), 3):
            c = np.sum(mat[r1] & mat[r2] & mat[r3])
            if c >= 3:
                count += c - 2
        return count

    def can_add(mat, i, j):
        others = [r for r in range(M) if r != i and mat[r, j] == 1]
        for r1, r2 in combinations(others, 2):
            common = 0
            for k in range(N):
                if mat[i, k] and mat[r1, k] and mat[r2, k]:
                    common += 1
                    if common >= 3:
                        return False
        return True

    def greedy_fill(mat):
        mat = mat.copy()
        changed = True
        while changed:
            changed = False
            cells = [(mat[i].sum(), i, j) for i in range(M) for j in range(N) if mat[i,j]==0]
            cells.sort()
            for _, i, j in cells:
                if mat[i,j] == 0:
                    mat[i,j] = 1
                    if can_add(mat, i, j):
                        changed = True
                    else:
                        mat[i,j] = 0
        return mat

    # Exhaustive SA-based search from multiple seeds
    best_mat = None
    best_count = 0

    rows97 = [
        [1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,1,1,1,1,1],
        [1,1,1,1,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,1,1],
        [0,1,0,1,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,1],
        [0,0,0,1,0,0,1,1,0,0,0,1,1,0,0,1,1,1,1,0,0,1],
        [1,0,0,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,0,1],
        [0,0,1,0,0,1,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,0],
        [0,0,1,0,1,0,0,1,1,0,1,1,1,0,1,0,1,0,0,0,1,1],
        [1,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,1,0,0],
        [0,1,0,0,1,0,0,1,1,1,0,0,1,0,0,1,0,0,1,0,1,0],
    ]
    base = np.array(rows97, dtype=int)

    np.random.seed(31415)
    for seed_trial in range(20):
        current = base.copy()
        perm = np.random.permutation(N)
        current = current[:, perm]
        current = greedy_fill(current)
        cur_count = int(current.sum())

        for trial in range(2000):
            mat = current.copy()
            ones = list(zip(*np.where(mat==1)))
            np.random.shuffle(ones)
            n_rem = np.random.randint(2, 10)
            for k in range(min(n_rem, len(ones))):
                mat[ones[k]] = 0
            mat = greedy_fill(mat)
            c = int(mat.sum())
            if c >= cur_count:
                current = mat
                cur_count = c
            if c > best_count:
                best_count = c
                best_mat = mat.copy()
                if c >= 100:
                    break
        if best_count >= 100:
            break

    G1 = best_mat if best_mat is not None else greedy_fill(base)
    G2 = np.ones((M, N), dtype=int)
    return G1, G2

def run_graph():
    return construct_graphs()
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    G1, G2 = run_graph()
    print(f"G1 shape: {G1.shape}, ones: {G1.sum()}, ones/row: {G1.sum(axis=1).tolist()}")
