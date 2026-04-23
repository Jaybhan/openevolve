import numpy as np

def construct_graphs():
    M, N = 11, 21

    def is_valid(G):
        for i in range(M):
            for j in range(i+1, M):
                for k in range(j+1, M):
                    if np.sum(G[i] & G[j] & G[k]) >= 3:
                        return False
        return True

    def can_set(G, r, c):
        G[r,c] = 1
        for i in range(M):
            if i == r or G[i,c] == 0: continue
            for j in range(i+1, M):
                if j == r or G[j,c] == 0: continue
                if np.sum(G[r] & G[i] & G[j]) >= 3:
                    G[r,c] = 0
                    return False
        return True

    # Seed from the best known valid matrix (111 ones)
    base = np.array([
        [1,1,1,1,1,1,0,0,0,1,0,0,1,0,0,0,0,0,1,1,0],
        [1,0,0,0,1,1,0,0,1,0,1,1,0,1,0,1,0,0,0,1,1],
        [1,0,0,0,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,0,0],
        [1,0,0,1,0,1,0,1,0,1,0,0,0,1,1,1,1,0,1,0,0],
        [1,0,1,0,0,0,1,1,0,0,1,0,1,1,0,0,0,1,1,1,1],
        [0,1,0,0,0,1,0,1,1,1,0,1,1,0,0,0,1,1,0,1,0],
        [0,1,1,0,0,0,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0],
        [0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,1,1,0,1,1,1],
        [0,1,0,1,1,0,1,0,1,0,0,0,1,1,1,0,1,0,0,0,1],
        [0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,1,0,1,0,0,1],
        [0,0,1,1,1,0,1,0,1,1,0,0,0,0,0,1,0,1,0,1,0],
    ], dtype=int)

    best_G = base.copy()
    best_count = int(np.sum(base))

    for seed in range(300):
        rng = np.random.RandomState(seed * 53 + 17)
        G = base.copy()

        # Try removing k ones and refilling
        ones = list(zip(*np.where(G == 1)))
        rng.shuffle(ones)
        k = rng.randint(3, 12)
        for idx in range(min(k, len(ones))):
            G[ones[idx][0], ones[idx][1]] = 0

        order = [(i,j) for i in range(M) for j in range(N)]
        rng.shuffle(order)
        for i, j in order:
            if G[i,j] == 0:
                can_set(G, i, j)

        for iteration in range(200):
            improved = False
            ones_list = list(zip(*np.where(G == 1)))
            rng.shuffle(ones_list)
            for oi, oj in ones_list:
                G[oi, oj] = 0
                added = []
                zeros = list(zip(*np.where(G == 0)))
                rng.shuffle(zeros)
                for ni, nj in zeros:
                    if can_set(G, ni, nj):
                        added.append((ni, nj))
                        if len(added) >= 2:
                            break
                if len(added) >= 2:
                    improved = True
                else:
                    for ai, aj in added:
                        G[ai, aj] = 0
                    G[oi, oj] = 1
            for i in range(M):
                for j in range(N):
                    if G[i,j] == 0:
                        can_set(G, i, j)
            if not improved:
                break

        count = int(np.sum(G))
        if count > best_count and is_valid(G):
            best_count = count
            best_G = G.copy()
            if best_count >= 116:
                break

    G1 = best_G
    rng2 = np.random.RandomState(999)
    G2 = G1[:, rng2.permutation(N)][rng2.permutation(M), :].copy()
    return G1, G2

def run_graph():
    return construct_graphs()
