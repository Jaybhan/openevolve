"""
Evaluator for the graph isomorphism example.

Scoring pipeline:
1. LLM poly-time gate: if evolved code is not polynomial time, score = -1
2. Generate fresh random test cases each evaluation (prevents overfitting)
3. Run evolved solver on each case with timeout
4. Score = fraction of cases with valid answer
"""

import importlib.util
import os
import random
import time
import traceback
import concurrent.futures
from typing import Optional

from openevolve.evaluation_result import EvaluationResult


# ── helpers ──────────────────────────────────────────────────────────────────

# Shared executor so timed-out threads don't accumulate a new pool per call.
# Threads that time out keep running until they naturally finish; we just stop
# waiting on them.  A single bounded pool limits the number of leaked threads
# across the whole evaluation pass.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _run_with_timeout(func, args=(), timeout_seconds=5):
    future = _EXECUTOR.submit(func, *args)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise TimeoutError(f"Function timed out after {timeout_seconds}s")


def _apply_perm(G, perm):
    """Return a new adjacency matrix with vertices permuted by perm.
    Result[i][j] = G[perm[i]][perm[j]]
    """
    n = len(G)
    return [[G[perm[i]][perm[j]] for j in range(n)] for i in range(n)]


def _is_valid_isomorphism(G1, G2, perm):
    """Check if perm is a valid isomorphism: G1[perm[i]][perm[j]] == G2[i][j]."""
    n = len(G1)
    if len(perm) != n:
        return False
    if sorted(perm) != list(range(n)):
        return False
    for i in range(n):
        for j in range(n):
            if G1[perm[i]][perm[j]] != G2[i][j]:
                return False
    return True


# ── graph generators ──────────────────────────────────────────────────────────

def _random_erdos_renyi(n, p, rng):
    """Generate an undirected Erdős–Rényi G(n, p) graph."""
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                G[i][j] = G[j][i] = 1
    return G


def _apply_random_perm(G, rng):
    """Apply a uniformly random permutation to a graph. Returns (G2, perm)
    where perm[i] is the G vertex that maps to G2 vertex i."""
    n = len(G)
    # perm_fwd[k] = new label of old vertex k
    perm_fwd = list(range(n))
    rng.shuffle(perm_fwd)
    # perm_inv[i] = old vertex that maps to new vertex i
    perm_inv = [0] * n
    for k, new_k in enumerate(perm_fwd):
        perm_inv[new_k] = k
    G2 = _apply_perm(G, perm_inv)
    return G2, perm_inv


def _random_regular_graph(n, k, rng, max_attempts=200):
    """Generate a random k-regular graph on n vertices using configuration model."""
    # n*k must be even for a k-regular graph to exist.  If both are odd, bump n
    # by 1: (n+1) is then even, so (n+1)*k is even regardless of k.
    n_local = n + (1 if n * k % 2 != 0 else 0)
    for _ in range(max_attempts):
        stubs = []
        for v in range(n_local):
            stubs.extend([v] * k)
        rng.shuffle(stubs)
        adj = [[0] * n_local for _ in range(n_local)]
        valid = True
        for idx in range(0, len(stubs), 2):
            u, v = stubs[idx], stubs[idx + 1]
            if u == v or adj[u][v]:
                valid = False
                break
            adj[u][v] = adj[v][u] = 1
        if valid:
            return adj
    # fallback: return cycle graph
    adj = [[0] * n_local for _ in range(n_local)]
    for i in range(n_local):
        adj[i][(i + 1) % n_local] = adj[(i + 1) % n_local][i] = 1
    return adj


def _wl_hash(G):
    """1-WL stable coloring hash. Graphs with different hashes are provably non-isomorphic."""
    n = len(G)
    colors = [sum(G[i]) for i in range(n)]
    for _ in range(n):
        new_colors = [hash((colors[v], tuple(sorted(colors[u] for u in range(n) if G[v][u])))) for v in range(n)]
        if new_colors == colors:
            break
        colors = new_colors
    return tuple(sorted(colors))


def _bipartite_regular_graph(n, k, rng, max_attempts=300):
    """Generate a random k-regular bipartite graph on 2n vertices.

    Left partition: 0..n-1, right partition: n..2n-1.
    Bipartite regular graphs are hard for GI because both sides share identical
    degree sequences and the inter-partition structure is the only distinguishing
    information.
    """
    for _ in range(max_attempts):
        left_stubs = []
        for v in range(n):
            left_stubs.extend([v] * k)
        right_stubs = []
        for v in range(n, 2 * n):
            right_stubs.extend([v] * k)
        rng.shuffle(left_stubs)
        rng.shuffle(right_stubs)
        adj = [[0] * (2 * n) for _ in range(2 * n)]
        valid = True
        for l, r in zip(left_stubs, right_stubs):
            if adj[l][r]:
                valid = False
                break
            adj[l][r] = adj[r][l] = 1
        if valid:
            return adj
    # Fallback: cyclic k-regular bipartite (left i connects to right (i+j)%n for j in 0..k-1)
    adj = [[0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(k):
            r = n + (i + j) % n
            adj[i][r] = adj[r][i] = 1
    return adj


def _circulant_graph(n, generators):
    """Generate circulant graph C(n, S).

    Vertex i is connected to (i ± g) mod n for each g in generators.
    Circulant graphs are vertex-transitive and highly symmetric, which makes
    them hard for many GI heuristics.  Two circulants with the same n but
    different generator sets are often non-isomorphic despite identical spectra.
    """
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for g in generators:
            j1 = (i + g) % n
            j2 = (i - g) % n
            adj[i][j1] = adj[j1][i] = 1
            adj[i][j2] = adj[j2][i] = 1
    return adj


def _paley_graph(p):
    """Generate the Paley graph on p vertices (p prime, p ≡ 1 mod 4).

    Vertices: 0..p-1.  Edge (i, j) iff (i-j) is a non-zero quadratic residue mod p.
    Paley graphs are strongly regular (hence WL-stable after one round) and
    self-complementary, making them among the hardest known GI instances.
    """
    qr = set()
    for x in range(1, p):
        qr.add((x * x) % p)
    adj = [[0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1, p):
            if (i - j) % p in qr:
                adj[i][j] = adj[j][i] = 1
    return adj


def _line_graph(G):
    """Construct the line graph L(G).

    Vertices of L(G) are edges of G; two are adjacent iff the corresponding
    edges of G share an endpoint.  Line graphs inherit structural complexity
    from G but have a different vertex count, exposing different solver weaknesses.
    """
    n = len(G)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if G[i][j]]
    m = len(edges)
    L = [[0] * m for _ in range(m)]
    for a in range(m):
        for b in range(a + 1, m):
            if edges[a][0] in edges[b] or edges[a][1] in edges[b]:
                L[a][b] = L[b][a] = 1
    return L


def _random_geometric_graph(n, radius, rng):
    """Random geometric graph: n uniform points in [0,1]^2, edges within radius.

    Geometric graphs have spatial locality that creates subtle structural
    patterns a solver must exploit; two isomorphic geometric graphs look
    identical under relabeling but very different as point clouds.
    """
    points = [(rng.random(), rng.random()) for _ in range(n)]
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if dx * dx + dy * dy < radius * radius:
                adj[i][j] = adj[j][i] = 1
    return adj


def _circulant_non_iso_pair(n, rng):
    """Return a non-isomorphic pair of circulant graphs on n vertices.

    Picks two generator sets with the same cardinality (same degree) so the
    degree sequence gives no information, then verifies non-isomorphism via WL.
    """
    max_gen = n // 2
    for _ in range(150):
        k = rng.randint(1, min(3, max_gen))
        pool = list(range(1, max_gen + 1))
        if len(pool) < k:
            break
        S1 = sorted(rng.sample(pool, k))
        S2 = sorted(rng.sample(pool, k))
        if S1 == S2:
            continue
        G1 = _circulant_graph(n, S1)
        G2 = _circulant_graph(n, S2)
        deg1 = sorted(sum(row) for row in G1)
        deg2 = sorted(sum(row) for row in G2)
        if deg1 != deg2:
            continue
        if _wl_hash(G1) != _wl_hash(G2):
            return G1, G2
    return _non_isomorphic_pair(n, rng)


def _non_isomorphic_pair(n, rng):
    """Return a pair (G1, G2) that are provably NOT isomorphic and have the same n.

    Prefers pairs with identical degree sequences (harder for simple solvers).
    Uses WL hash divergence as a sufficient certificate of non-isomorphism.
    """
    # Strategy: draw two independent random graphs; accept when WL hashes differ
    # (guarantees non-isomorphic) AND degree sequences match (harder rejection case).
    for _ in range(100):
        p = rng.uniform(0.25, 0.55)
        G1 = _random_erdos_renyi(n, p, rng)
        G2 = _random_erdos_renyi(n, p, rng)
        deg1 = sorted(sum(row) for row in G1)
        deg2 = sorted(sum(row) for row in G2)
        if deg1 != deg2:
            continue  # skip easy cases — we want same-degree-sequence pairs
        if _wl_hash(G1) != _wl_hash(G2):
            return G1, G2  # different WL hash → provably non-isomorphic
    # Fallback: path vs cycle (guaranteed non-isomorphic, different degree sequences)
    path = [[0] * n for _ in range(n)]
    cycle = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        path[i][i + 1] = path[i + 1][i] = 1
    for i in range(n):
        cycle[i][(i + 1) % n] = cycle[(i + 1) % n][i] = 1
    return path, cycle


# ── test suite ────────────────────────────────────────────────────────────────

def _generate_test_suite(rng=None):
    """Generate a fresh random test suite every evaluation."""
    if rng is None:
        rng = random.Random()  # unseeded — OS entropy each call

    cases = []  # each: (G1, G2, is_isomorphic)

    # 12 Erdős–Rényi isomorphic pairs, n = 8..20
    for _ in range(12):
        n = rng.randint(8, 20)
        p = rng.uniform(0.2, 0.6)
        G1 = _random_erdos_renyi(n, p, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 10 random regular isomorphic pairs (mixed k=3..5), n = 8..18
    for _ in range(10):
        n = rng.randrange(8, 19, 2)
        k = rng.choice([3, 4, 5])
        G1 = _random_regular_graph(n, k, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 8 bipartite regular isomorphic pairs — hard: both sides identical degree
    for _ in range(8):
        n = rng.randint(5, 10)
        k = rng.randint(2, min(4, n - 1))
        G1 = _bipartite_regular_graph(n, k, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 8 circulant isomorphic pairs — vertex-transitive, fools spectral methods
    for _ in range(8):
        n = rng.randint(10, 20)
        max_gen = n // 2
        k = rng.randint(1, min(3, max_gen))
        S = sorted(rng.sample(range(1, max_gen + 1), k))
        G1 = _circulant_graph(n, S)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 6 Paley graph isomorphic pairs — strongly regular, WL-stable after 1 round
    for p in rng.sample([13, 17, 29, 37, 41, 53], 6):
        G1 = _paley_graph(p)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 6 line graph isomorphic pairs — inherited complexity, different vertex count
    for _ in range(6):
        n = rng.randint(6, 10)
        p = rng.uniform(0.3, 0.5)
        base = _random_erdos_renyi(n, p, rng)
        G1 = _line_graph(base)
        if len(G1) < 4:
            G1 = _line_graph(_random_erdos_renyi(8, 0.4, rng))
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 6 random geometric graph isomorphic pairs — spatial locality is subtle
    for _ in range(6):
        n = rng.randint(10, 18)
        radius = rng.uniform(0.3, 0.5)
        G1 = _random_geometric_graph(n, radius, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 12 non-isomorphic pairs via Erdős–Rényi (same degree seq, different WL)
    for _ in range(12):
        n = rng.randint(8, 16)
        G1, G2 = _non_isomorphic_pair(n, rng)
        cases.append((G1, G2, False))

    # 8 non-isomorphic circulant pairs — same degree, different topology
    for _ in range(8):
        n = rng.randint(12, 20)
        G1, G2 = _circulant_non_iso_pair(n, rng)
        cases.append((G1, G2, False))

    rng.shuffle(cases)
    return cases


def _generate_extended_cases(rng):
    """60 harder cases for programs that pass the 0.8 threshold.

    Larger graphs (n up to 35) and harder families to stress colour-refinement
    based solvers that still struggle with high-symmetry instances.
    """
    cases = []

    # 12 large Erdős–Rényi iso pairs, n = 20..35
    for _ in range(12):
        n = rng.randint(20, 35)
        p = rng.uniform(0.2, 0.5)
        G1 = _random_erdos_renyi(n, p, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 10 high-degree regular iso pairs (k=4,5,6), n = 16..28
    for _ in range(10):
        n = rng.randrange(16, 29, 2)
        k = rng.choice([4, 5, 6])
        G1 = _random_regular_graph(n, k, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 8 large Paley iso pairs — strongly regular, WL-stable
    for p in rng.sample([53, 61, 73, 89, 97], 5):
        G1 = _paley_graph(p)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))
    for p in rng.sample([13, 17, 29, 37, 41], 3):
        G1 = _paley_graph(p)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 10 large circulant iso pairs, n = 20..40
    for _ in range(10):
        n = rng.randint(20, 40)
        max_gen = n // 2
        k = rng.randint(2, min(4, max_gen))
        S = sorted(rng.sample(range(1, max_gen + 1), k))
        G1 = _circulant_graph(n, S)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 10 large non-iso pairs (Erdős–Rényi), n = 16..28
    for _ in range(10):
        n = rng.randint(16, 28)
        G1, G2 = _non_isomorphic_pair(n, rng)
        cases.append((G1, G2, False))

    # 10 large circulant non-iso pairs, n = 18..35
    for _ in range(10):
        n = rng.randint(18, 35)
        G1, G2 = _circulant_non_iso_pair(n, rng)
        cases.append((G1, G2, False))

    rng.shuffle(cases)
    return cases


def _generate_stress_cases(rng):
    """100 stress cases for programs that score 1.0 on the base suite.

    Very large graphs (n up to 60) and the hardest known families (large Paley,
    dense regular, large circulant) to distinguish near-perfect solvers.
    """
    cases = []

    # 20 very large Erdős–Rényi iso pairs, n = 30..60
    for _ in range(20):
        n = rng.randint(30, 60)
        p = rng.uniform(0.2, 0.45)
        G1 = _random_erdos_renyi(n, p, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 15 large regular iso pairs (k=4..7), n = 20..40
    for _ in range(15):
        n = rng.randrange(20, 41, 2)
        k = rng.choice([4, 5, 6, 7])
        G1 = _random_regular_graph(n, k, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 15 very large Paley iso pairs
    for p in rng.sample([101, 109, 113, 137, 149, 157, 173, 181, 193, 197,
                          53, 61, 73, 89, 97], 15):
        G1 = _paley_graph(p)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 15 very large circulant iso pairs, n = 40..60
    for _ in range(15):
        n = rng.randint(40, 60)
        max_gen = n // 2
        k = rng.randint(2, min(5, max_gen))
        S = sorted(rng.sample(range(1, max_gen + 1), k))
        G1 = _circulant_graph(n, S)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 15 large non-iso pairs, n = 20..40
    for _ in range(15):
        n = rng.randint(20, 40)
        G1, G2 = _non_isomorphic_pair(n, rng)
        cases.append((G1, G2, False))

    # 15 large circulant non-iso pairs, n = 30..50
    for _ in range(15):
        n = rng.randint(30, 50)
        G1, G2 = _circulant_non_iso_pair(n, rng)
        cases.append((G1, G2, False))

    # 5 large bipartite regular iso pairs
    for _ in range(5):
        n = rng.randint(12, 20)
        k = rng.randint(3, 5)
        G1 = _bipartite_regular_graph(n, k, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    rng.shuffle(cases)
    return cases


# ── poly-time LLM gate ────────────────────────────────────────────────────────

def _extract_evolve_block(program_path):
    """Extract the EVOLVE-BLOCK code from a program file."""
    with open(program_path) as f:
        content = f.read()
    start = content.find("# EVOLVE-BLOCK-START")
    end = content.find("# EVOLVE-BLOCK-END")
    if start == -1 or end == -1:
        return content
    return content[start:end + len("# EVOLVE-BLOCK-END")]


def _check_polynomial_time(program_path, model=None):
    """
    Ask an LLM whether the evolved algorithm is polynomial time.
    Returns True (pass), False (fail), or None (check skipped due to API error).
    Pass model to override the default (e.g. use a stronger model for high-scoring programs).
    """
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get(
        "API_BASE", "https://openrouter.ai/api/v1"
    )
    if model is None:
        model = os.environ.get("OPENEVOLVE_MODEL", "anthropic/claude-sonnet-4-6")

    if not api_key:
        print("Warning: No API key found, skipping poly-time check")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)
        code = _extract_evolve_block(program_path)
        prompt = (
            "Look at the following Python code\n"
            "Does this code always execute in polynomial time O(n^k)?"
            "Answer with only the single word YES or NO.\n\n"
            f"```python\n{code}\n```"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a code analysis tool. Answer YES or NO only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            timeout=30,
        )
        answer = response.choices[0].message.content.strip().upper()
        is_poly = answer.startswith("YES")
        return is_poly
    except Exception as e:
        print(f"Poly-time check failed (API error): {e}")
        return None  # don't penalize on API errors


def _check_gi_in_p(program_path):
    """Ask Sonnet whether this code constitutes a polynomial-time GI algorithm for ALL inputs.

    This is the jackpot check: called only when a program has scored 1.0 across all
    three test stages.  Returns True only if Sonnet is convinced the algorithm is both
    correct and polynomial for every possible graph, not just the sampled test suite.
    Returns False if it finds a gap, or None if the API call fails.
    """
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get(
        "API_BASE", "https://openrouter.ai/api/v1"
    )

    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)
        code = _extract_evolve_block(program_path)
        prompt = (
            "A program has just scored perfectly on 236 randomized graph isomorphism test "
            "cases spanning Erdős–Rényi, regular, bipartite regular, circulant, Paley "
            "(strongly regular), line graphs, and geometric graphs — at sizes up to n=60 "
            "vertices — while independently passing a polynomial-time complexity check.\n\n"
            "Your task: determine whether this code is a genuine polynomial-time algorithm "
            "that correctly solves graph isomorphism for ALL possible inputs.\n\n"
            "Be a rigorous skeptic. Specifically check:\n"
            "1. Are there graph families or edge cases where this algorithm could return a "
            "wrong answer (false positive isomorphism or missed isomorphism)?\n"
            "2. Are there inputs where the runtime could be super-polynomial "
            "(e.g. hidden exponential branching, recursion depth proportional to n!, etc.)?\n"
            "3. Is the correctness argument complete, or does it rely on heuristics that "
            "could fail on adversarial inputs?\n\n"
            "If and ONLY IF the algorithm is provably correct and provably polynomial for "
            "every possible graph input, answer YES. Otherwise answer NO.\n"
            "Answer with only the single word YES or NO.\n\n"
            f"```python\n{code}\n```"
        )
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4-6",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a theoretical computer scientist reviewing a claimed "
                        "polynomial-time graph isomorphism algorithm. Look hard for "
                        "counterexamples and hidden exponential cases. Answer YES or NO only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            timeout=45,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"GI-in-P check failed (API error): {e}")
        return None


# ── case runner ──────────────────────────────────────────────────────────────

def _run_cases(solve_fn, cases, case_offset=0):
    """Run solve_fn on every case; return counters and failure strings."""
    iso_correct = iso_total = rej_correct = rej_total = timeouts = 0
    failures = []
    for idx, (G1, G2, is_iso) in enumerate(cases):
        try:
            result = _run_with_timeout(solve_fn, args=(G1, G2), timeout_seconds=5)
        except TimeoutError:
            timeouts += 1
            if is_iso:
                iso_total += 1
            else:
                rej_total += 1
            failures.append(f"case {case_offset + idx}: timeout (n={len(G1)}, is_iso={is_iso})")
            continue
        except Exception as e:
            if is_iso:
                iso_total += 1
            else:
                rej_total += 1
            failures.append(f"case {case_offset + idx}: exception {type(e).__name__}: {e}")
            continue

        if is_iso:
            iso_total += 1
            if result is not None and _is_valid_isomorphism(G1, G2, result):
                iso_correct += 1
            else:
                failures.append(
                    f"case {case_offset + idx}: wrong iso answer (n={len(G1)}, "
                    f"returned={result if result is None else 'perm'})"
                )
        else:
            rej_total += 1
            if result is None:
                rej_correct += 1
            elif _is_valid_isomorphism(G1, G2, result):
                # solver found an isomorphism we missed — accept it
                rej_correct += 1
            else:
                failures.append(
                    f"case {case_offset + idx}: non-iso pair but solver returned invalid perm (n={len(G1)})"
                )

    return iso_correct, iso_total, rej_correct, rej_total, timeouts, failures


# ── main evaluate ─────────────────────────────────────────────────────────────

def evaluate(program_path):
    """
    Evaluate the evolved graph isomorphism solver.

    Returns EvaluationResult with:
      combined_score  : fraction of test cases answered correctly [0, 1] or -1 (poly gate fail)
      iso_accuracy    : accuracy on isomorphic pairs
      rejection_accuracy: accuracy on non-isomorphic pairs
      timeout_rate    : fraction of cases that timed out
    """
    # --- load program ---
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "iso_accuracy": 0.0,
                     "rejection_accuracy": 0.0, "timeout_rate": 0.0},
            artifacts={"error_type": "LoadError", "error_message": str(e),
                       "traceback": traceback.format_exc()},
        )

    if not hasattr(program, "solve"):
        return EvaluationResult(
            metrics={"combined_score": 0.0, "iso_accuracy": 0.0,
                     "rejection_accuracy": 0.0, "timeout_rate": 0.0},
            artifacts={"error_type": "MissingFunction",
                       "error_message": "Program is missing required 'solve(G1, G2)' function"},
        )

    rng = random.Random()

    # ── Always: fast poly check with default model ────────────────────────────
    poly_result = _check_polynomial_time(program_path)
    if poly_result is False:
        return EvaluationResult(
            metrics={"combined_score": -1.0, "iso_accuracy": 0.0,
                     "rejection_accuracy": 0.0, "timeout_rate": 0.0},
            artifacts={"stage": "1", "poly_gate": "FAIL",
                       "error_message": "Algorithm is not polynomial time"},
        )

    # ── Stage 1: base suite (76 cases) ───────────────────────────────────────
    cases_s1 = _generate_test_suite(rng)
    ic, it, rc, rt, to, fail = _run_cases(program.solve, cases_s1, case_offset=0)

    total = it + rt
    s1_score = (ic + rc) / total if total > 0 else 0.0

    if s1_score <= 0.8:
        artifacts = {
            "stage": "1",
            "summary": f"correct={ic + rc}/{total}, iso={ic}/{it}, rejection={rc}/{rt}, timeouts={to}",
        }
        if fail:
            artifacts["sample_failures"] = "\n".join(fail[:10])
        return EvaluationResult(
            metrics={"combined_score": s1_score,
                     "iso_accuracy": ic / it if it else 0.0,
                     "rejection_accuracy": rc / rt if rt else 0.0,
                     "timeout_rate": to / total if total else 0.0},
            artifacts=artifacts,
        )

    # ── Stage 2 (score > 0.8): stronger poly check (Sonnet) + 60 harder cases
    poly_sonnet = _check_polynomial_time(program_path, model="anthropic/claude-sonnet-4-6")
    if poly_sonnet is False:
        return EvaluationResult(
            metrics={"combined_score": -1.0, "iso_accuracy": 0.0,
                     "rejection_accuracy": 0.0, "timeout_rate": 0.0},
            artifacts={"stage": "2", "poly_gate": "FAIL",
                       "error_message": "Algorithm is not polynomial time (Sonnet check)"},
        )

    cases_s2 = _generate_extended_cases(rng)
    ic2, it2, rc2, rt2, to2, fail2 = _run_cases(program.solve, cases_s2, case_offset=len(cases_s1))
    ic += ic2; it += it2; rc += rc2; rt += rt2; to += to2; fail += fail2
    stage = "2"

    # ── Stage 3: only if stage-1 score was perfect (captured before stage-2 cases were added)
    if s1_score == 1.0:
        cases_s3 = _generate_stress_cases(rng)
        ic3, it3, rc3, rt3, to3, fail3 = _run_cases(
            program.solve, cases_s3, case_offset=len(cases_s1) + len(cases_s2)
        )
        ic += ic3; it += it3; rc += rc3; rt += rt3; to += to3; fail += fail3
        stage = "3"

    total = it + rt
    combined_score = (ic + rc) / total if total > 0 else 0.0

    # ── Jackpot: perfect score across all stages → ask Sonnet if GI is in P ──
    if combined_score == 1.0 and stage == "3":
        gi_in_p = _check_gi_in_p(program_path)
        if gi_in_p is True:
            return EvaluationResult(
                metrics={"combined_score": 100.0, "iso_accuracy": 1.0,
                         "rejection_accuracy": 1.0, "timeout_rate": 0.0},
                artifacts={
                    "stage": "JACKPOT",
                    "summary": (
                        f"PERFECT on all {total} cases across all three stages. "
                        "Sonnet confirms this is a valid polynomial-time GI algorithm. "
                        "GI IS IN P."
                    ),
                },
            )

    artifacts = {
        "stage": stage,
        "summary": (
            f"correct={ic + rc}/{total}, iso={ic}/{it}, "
            f"rejection={rc}/{rt}, timeouts={to}, poly_gate=pass"
        ),
    }
    if fail:
        artifacts["sample_failures"] = "\n".join(fail[:10])

    return EvaluationResult(
        metrics={
            "combined_score": combined_score,
            "iso_accuracy": ic / it if it else 0.0,
            "rejection_accuracy": rc / rt if rt else 0.0,
            "timeout_rate": to / total if total else 0.0,
        },
        artifacts=artifacts,
    )
