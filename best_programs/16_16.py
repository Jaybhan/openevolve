M = 16
N = 16
S = 3
T = 3

# EVOLVE-BLOCK-START
import numpy as np

def construct_graphs():
    def check_add(G, i, j):
        ri = [r for r in range(16) if r != i and G[r, j]]
        ci = [c for c in range(16) if c != j and G[i, c]]
        if len(ri) < 2 or len(ci) < 2:
            return True
        for a in range(len(ri)):
            for b in range(a+1, len(ri)):
                if sum(1 for c in ci if G[ri[a],c] and G[ri[b],c]) >= 2:
                    return False
        for a in range(len(ci)):
            for b in range(a+1, len(ci)):
                if sum(1 for r in ri if G[r,ci[a]] and G[r,ci[b]]) >= 2:
                    return False
        return True

    def is_valid(G):
        for r1 in range(16):
            for r2 in range(r1+1,16):
                common = np.where(G[r1] & G[r2])[0]
                if len(common) < 3: continue
                for r3 in range(r2+1,16):
                    if np.sum(G[r3,common]) >= 3: return False
        return True

    # Best known 119-ones matrix from previous run - use as seed
    R = [[1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,1],[1,1,1,0,0,1,0,0,0,0,0,1,1,0,1,1],[0,1,1,1,0,0,1,0,0,0,0,0,1,1,0,1],[1,0,1,1,1,0,0,1,0,0,0,0,0,1,1,0],[0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,1],[1,0,1,1,1,1,1,0,0,1,0,0,0,0,0,1],[1,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0],[0,1,1,0,1,1,1,1,1,0,0,1,0,0,0,0],[0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,0],[0,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0],[0,0,0,0,1,1,0,1,0,1,1,1,0,0,1,0],[0,0,0,0,0,1,1,0,1,0,1,1,1,0,0,1],[1,0,1,0,0,0,1,1,0,1,1,1,1,1,0,0],[0,1,0,1,0,0,0,1,1,0,1,0,1,1,1,0],[0,0,1,0,1,0,0,0,1,1,0,1,0,1,1,1],[1,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1]]
    best = np.array(R, dtype=int)
    best_sum = best.sum() if is_valid(best) else 0

    for trial in range(100):
        rng = np.random.RandomState(trial + 200)
        G = best.copy()
        ones = list(zip(*np.where(G == 1)))
        rng.shuffle(ones)
        for _ in range(rng.randint(2, 8)):
            if ones:
                idx = rng.randint(len(ones))
                G[ones[idx]] = 0
        zeros = [(i,j) for i in range(16) for j in range(16) if G[i,j]==0]
        rng.shuffle(zeros)
        for i,j in zeros:
            if G[i,j]==0 and check_add(G,i,j):
                G[i,j] = 1
        if is_valid(G) and G.sum() > best_sum:
            best_sum = G.sum()
            best = G.copy()

    G2 = best.copy()
    return best, G2

def run_graph():
    return construct_graphs()
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    G1, G2 = run_graph()
    print(f"G1 ones:{G1.sum()}, density:{G1.sum()/256:.4f}")
    print(G1)
