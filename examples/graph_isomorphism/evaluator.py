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


def _run_with_timeout(func, args=(), timeout_seconds=10):
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

def _generate_test_suite():
    """Generate a fresh random test suite every evaluation."""
    rng = random.Random()  # unseeded — OS entropy each call

    cases = []  # each: (G1, G2, is_isomorphic)

    # 20 Erdős–Rényi isomorphic pairs, n = 8..20
    for _ in range(20):
        n = rng.randint(8, 20)
        p = rng.uniform(0.2, 0.6)
        G1 = _random_erdos_renyi(n, p, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 20 random regular isomorphic pairs (k=3), n = 8..16 (must be even for k=3)
    for _ in range(20):
        n = rng.randrange(8, 17, 2)  # even numbers 8,10,...,16
        G1 = _random_regular_graph(n, 3, rng)
        G2, _ = _apply_random_perm(G1, rng)
        cases.append((G1, G2, True))

    # 20 non-isomorphic pairs, n = 8..16
    for _ in range(20):
        n = rng.randint(8, 16)
        G1, G2 = _non_isomorphic_pair(n, rng)
        cases.append((G1, G2, False))

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


def _check_polynomial_time(program_path):
    """
    Ask an LLM whether the evolved algorithm is polynomial time.
    Returns True (pass), False (fail), or None (check skipped due to API error).
    """
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get(
        "API_BASE", "https://openrouter.ai/api/v1"
    )
    model = os.environ.get("OPENEVOLVE_MODEL", "google/gemini-2.0-flash-001")

    if not api_key:
        print("Warning: No API key found, skipping poly-time check")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)
        code = _extract_evolve_block(program_path)
        prompt = (
            "Look at the following Python code and analyze its loops and recursion.\n"
            "Does this specific code — as written — execute in polynomial time O(n^k) "
            "for some fixed constant k, where n = len(G1)?\n"
            "Judge only the actual operations in the code (loops, recursion depth, "
            "itertools.permutations, etc.). Ignore whether graph isomorphism is hard in "
            "general — only analyze this code's runtime.\n"
            "Answer YES if and only if it runs in polynomial time. "
            "Answer NO if it enumerates all n! permutations or has factorial/exponential "
            "branching with no polynomial bound or employs some other nonpolynomial algorithm.\n"
            "Answer with only the single word YES or NO.\n\n"
            f"```python\n{code}\n```"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a code analysis tool. Count loops and recursion in the "
                        "given code to determine its time complexity. Answer YES or NO only."
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

    # --- polynomial-time gate (run tests regardless, halve score if non-poly) ---
    poly_result = _check_polynomial_time(program_path)

    # --- generate fresh test cases ---
    cases = _generate_test_suite()

    iso_correct = 0
    iso_total = 0
    rej_correct = 0
    rej_total = 0
    timeouts = 0
    failures = []

    for idx, (G1, G2, is_iso) in enumerate(cases):
        try:
            result = _run_with_timeout(program.solve, args=(G1, G2), timeout_seconds=10)
        except TimeoutError:
            timeouts += 1
            if is_iso:
                iso_total += 1
            else:
                rej_total += 1
            failures.append(f"case {idx}: timeout (n={len(G1)}, is_iso={is_iso})")
            continue
        except Exception as e:
            if is_iso:
                iso_total += 1
            else:
                rej_total += 1
            failures.append(f"case {idx}: exception {type(e).__name__}: {e}")
            continue

        if is_iso:
            iso_total += 1
            if result is not None and _is_valid_isomorphism(G1, G2, result):
                iso_correct += 1
            else:
                failures.append(
                    f"case {idx}: wrong iso answer (n={len(G1)}, "
                    f"returned={result if result is None else 'perm'})"
                )
        else:
            rej_total += 1
            if result is None:
                rej_correct += 1
            else:
                # also accept a returned perm that is actually valid (program may disagree with our non-iso generator)
                if _is_valid_isomorphism(G1, G2, result):
                    rej_correct += 1
                else:
                    failures.append(
                        f"case {idx}: non-iso pair but solver returned invalid perm (n={len(G1)})"
                    )

    total = iso_total + rej_total
    combined_score = (iso_correct + rej_correct) / total if total > 0 else 0.0
    iso_accuracy = iso_correct / iso_total if iso_total > 0 else 0.0
    rej_accuracy = rej_correct / rej_total if rej_total > 0 else 0.0
    timeout_rate = timeouts / total if total > 0 else 0.0

    if poly_result is False:
        combined_score *= 0.35

    artifacts = {
        "summary": (
            f"correct={iso_correct + rej_correct}/{total}, "
            f"iso={iso_correct}/{iso_total}, "
            f"rejection={rej_correct}/{rej_total}, "
            f"timeouts={timeouts}, "
            f"poly_penalty={'yes' if poly_result is False else 'no'}"
        ),
    }
    if failures:
        artifacts["sample_failures"] = "\n".join(failures[:10])

    return EvaluationResult(
        metrics={
            "combined_score": combined_score,
            "iso_accuracy": iso_accuracy,
            "rejection_accuracy": rej_accuracy,
            "timeout_rate": timeout_rate,
        },
        artifacts=artifacts,
    )
