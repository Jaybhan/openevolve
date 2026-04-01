"""
Evaluator for the Zarankiewicz problem z(17,17;3,3).

Goal: maximize the number of 1s in a 17×17 binary matrix with no 3×3
all-ones submatrix (K_{3,3}-free bipartite graph).

KST upper bound: z(17,17;3,3) ≤ 141 (Collins et al. 2016).
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from itertools import combinations
from math import comb

import numpy as np

# Lazy import — only needed when returning EvaluationResult
def _make_result(metrics, artifacts=None):
    try:
        from openevolve.evaluation_result import EvaluationResult
        return EvaluationResult(metrics=metrics, artifacts=artifacts or {})
    except ImportError:
        return metrics

# Problem parameters (must match initial_program.py)
N = 17
S = 3
T = 3
KST_UPPER_BOUND = 141.0  # upper bound on z(17;3) from Collins et al. 2016 (Table 4, via Lemmas 2/3/4)
# Exact value unknown. Finding a valid 141-edge graph proves z(17;3) = 141 exactly.

# Shared state: track current best valid edge count across evaluations (n_SOTA).
# Stored in a file so worker processes can read it.
_SOTA_FILE = os.path.join(os.path.dirname(__file__), ".n_sota")
_BEST_MATRIX_FILE = os.path.join(os.path.dirname(__file__), ".best_matrix.npy")

def _read_n_sota():
    try:
        with open(_SOTA_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return 0

def _update_n_sota(n, matrix=None):
    current = _read_n_sota()
    if n > current:
        with open(_SOTA_FILE, "w") as f:
            f.write(str(n))
        if matrix is not None:
            np.save(_BEST_MATRIX_FILE, matrix)

def _read_best_matrix():
    try:
        return np.load(_BEST_MATRIX_FILE)
    except Exception:
        return None

def _format_matrix_for_llm(A, label="Best known valid matrix"):
    """Render a binary matrix as a compact string the LLM can read and reason about."""
    n = A.shape[0]
    row_degrees = A.sum(axis=1).astype(int).tolist()
    lines = [
        f"{label} — row degrees: {row_degrees}",
        "     " + " ".join(f"{j:2d}" for j in range(n)),
        "     " + "--" * n,
    ]
    for i, row in enumerate(A.astype(int)):
        bits = " ".join(str(v) for v in row)
        lines.append(f"r{i:2d} | {bits}")
    col_degrees = A.sum(axis=0).astype(int).tolist()
    lines.append("col  " + " ".join(f"{d:2d}" for d in col_degrees))
    return "\n".join(lines)


class TimeoutError(Exception):
    pass


def count_kst_violations(A, s, t):
    """
    Count the number of K_{s,t} subgraphs in the binary matrix A.

    A K_{s,t} exists when s rows share t or more common 1-columns.
    For each s-subset of rows sharing k >= t common 1-columns,
    it contributes C(k, t) violations.

    For N=17, s=t=3: C(17,3) = 680 row-triples to check — fast.
    """
    m, n = A.shape
    count = 0
    for row_subset in combinations(range(m), s):
        # Columns that are 1 in ALL s selected rows
        common = A[row_subset[0]].astype(bool)
        for r in row_subset[1:]:
            common &= A[r].astype(bool)
        shared = int(common.sum())
        if shared >= t:
            count += comb(shared, t)
    return count


def has_kst(A, s, t):
    """Fast short-circuit check: does A contain any K_{s,t}?"""
    m, n = A.shape
    for row_subset in combinations(range(m), s):
        common = A[row_subset[0]].astype(bool)
        for r in row_subset[1:]:
            common &= A[r].astype(bool)
        if common.sum() >= t:
            return True
    return False


def run_with_timeout(program_path, timeout_seconds=20):
    """
    Run the program in a subprocess with timeout.
    Returns (G1, G2) where G1 is the primary K_{3,3}-free candidate and
    G2 is the dense prospect graph.  For backward-compatible programs that
    return a single matrix, G2 is set to None.
    Uses pickle to pass results back from the worker process.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
        temp_file_path = temp_file.name
        results_path = temp_file_path + ".results"
        script = f"""
import sys, os, pickle, traceback
sys.path.insert(0, os.path.dirname('{program_path}'))
try:
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("program", '{program_path}')
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    # Try common entry-point names the LLM might use
    entry_point = None
    for fn_name in ('run_graph', 'construct_graphs', 'construct_graph'):
        if hasattr(program, fn_name) and callable(getattr(program, fn_name)):
            entry_point = getattr(program, fn_name)
            break
    if entry_point is None:
        raise AttributeError("No recognized entry point (run_graph / construct_graphs / construct_graph)")
    result = entry_point()
    # Support both (G1, G2) tuple and legacy single-matrix return
    if isinstance(result, tuple) and len(result) == 2:
        G1, G2 = result
        G1 = np.asarray(G1)
        G2 = np.asarray(G2)
    else:
        G1 = np.asarray(result)
        G2 = None
    with open('{results_path}', 'wb') as f:
        pickle.dump({{'G1': G1, 'G2': G2}}, f)
except Exception as e:
    traceback.print_exc()
    with open('{results_path}', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
"""
        temp_file.write(script)

    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            if stdout:
                print(stdout.decode())
            if stderr:
                print(stderr.decode())
            if process.returncode != 0:
                raise RuntimeError(f"Process exited with code {process.returncode}")
            if not os.path.exists(results_path):
                raise RuntimeError("Results file not found")
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            if "error" in results:
                raise RuntimeError(f"Program error: {results['error']}")
            return results["G1"], results["G2"]
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Timed out after {timeout_seconds}s")
    finally:
        for path in [temp_file_path, results_path]:
            if os.path.exists(path):
                os.unlink(path)


def score_graphs(G1, G2=None):
    """
    Score G1 and G2 following AlphaEvolve Algorithm 1 (two-graph version).

    G1 — primary K_{3,3}-free candidate:
        Valid:   S = 4*e1 if e1>n_SOTA, 2*e1 if e1==n_SOTA, e1 otherwise
        Invalid: S = -1

    G2 — dense prospect graph (may have violations); adds a bonus regardless
        of G1's validity:
        if e2 > 0:
            E_viol = C(N,S)*C(N,T)*(e2/(N*N))^(S*T)   [expected violations]
            S += 0.5 * max(0, 1 - count_viol(G2) / E_viol)

    combined_score = max(0, S) / (4 * KST_UPPER_BOUND)   [normalized to ~[0,1]]
    """
    # --- G1 metrics ---
    num_edges = int(G1.sum())
    edge_density = num_edges / (N * N)
    row_degrees = G1.sum(axis=1)
    row_degree_variance = float(np.var(row_degrees))
    violations = count_kst_violations(G1, S, T)
    valid = violations == 0

    # --- G1 score (Algorithm 1 lines 5-8) ---
    if valid:
        n_sota = _read_n_sota()
        _update_n_sota(num_edges, matrix=G1)
        if num_edges > n_sota:
            raw_score = 4.0 * num_edges
        elif num_edges == n_sota:
            raw_score = 2.0 * num_edges
        else:
            raw_score = float(num_edges)
    else:
        raw_score = -1.0

    # --- G2 prospect bonus (Algorithm 1 lines 9-11) ---
    g2_bonus = 0.0
    if G2 is not None:
        e2 = int(G2.sum())
        if e2 > 0:
            d2 = e2 / (N * N)
            e_expected = comb(N, S) * comb(N, T) * (d2 ** (S * T))
            e_expected = max(e_expected, 1e-9)
            viol2 = count_kst_violations(G2, S, T)
            g2_bonus = 0.5 * max(0.0, 1.0 - viol2 / e_expected)

    # --- Normalize ---
    # Do NOT clamp to 0 — invalid programs with better G2 score slightly less
    # negative than those with worse G2, letting MAP-Elites differentiate them.
    combined_score = (raw_score + g2_bonus) / (4.0 * KST_UPPER_BOUND)

    return {
        "num_edges": float(num_edges),
        "edge_density": float(edge_density),
        "validity": 1.0 if valid else 0.0,
        "target_ratio": float(num_edges / KST_UPPER_BOUND),
        "violation_count": float(violations),
        "row_degree_variance": row_degree_variance,
        "g2_bonus": float(g2_bonus),
        "combined_score": float(combined_score),
    }


# Keep backward-compatible alias
def score_matrix(A):
    return score_graphs(A, G2=None)


def evaluate(program_path):
    """
    Main evaluation function called by OpenEvolve.
    Returns a metrics dict with 'combined_score' as the primary selector.
    """
    start = time.time()
    try:
        G1, G2 = run_with_timeout(program_path, timeout_seconds=25)

        # Validate G1 shape and values
        if G1.shape != (N, N):
            print(f"Wrong shape: {G1.shape}, expected ({N}, {N})")
            return _zero_metrics()
        if not np.all(np.isin(G1, [0, 1])):
            print("G1 contains values other than 0 and 1")
            return _zero_metrics()

        # Validate G2 if provided
        if G2 is not None:
            if G2.shape != (N, N) or not np.all(np.isin(G2, [0, 1])):
                print("G2 invalid shape or values — ignoring G2")
                G2 = None

        metrics = score_graphs(G1, G2)
        elapsed = time.time() - start
        g2_str = f", g2_bonus={metrics['g2_bonus']:.3f}" if G2 is not None else ""
        print(
            f"edges={int(metrics['num_edges'])}, valid={bool(metrics['validity'])}, "
            f"violations={int(metrics['violation_count'])}, "
            f"score={metrics['combined_score']:.4f}{g2_str}, time={elapsed:.2f}s"
        )

        # Build artifact: show the best known valid matrix so the LLM can reason
        # about specific rows/columns to modify in its next attempt.
        artifacts = {}
        best = _read_best_matrix()
        if best is not None:
            artifacts["best_known_matrix"] = _format_matrix_for_llm(
                best, label=f"Best known valid G1 ({int(best.sum())} ones)"
            )
        if metrics["validity"] and best is None:
            # This IS the best — show it directly
            artifacts["best_known_matrix"] = _format_matrix_for_llm(
                G1, label=f"Current G1 ({int(metrics['num_edges'])} ones, valid)"
            )

        return _make_result(metrics, artifacts)

    except TimeoutError:
        print("Evaluation timed out")
        return _zero_metrics()
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        return _zero_metrics()


def evaluate_stage1(program_path):
    """
    Fast cascade stage 1: run program and do a quick violation check on G1.
    Programs with many violations get filtered early.
    """
    try:
        G1, _ = run_with_timeout(program_path, timeout_seconds=15)
        if G1.shape != (N, N):
            return {"combined_score": 0.0, "validity": 0.0, "row_degree_variance": 0.0}
        if not np.all(np.isin(G1, [0, 1])):
            return {"combined_score": 0.0, "validity": 0.0, "row_degree_variance": 0.0}

        # Quick check: just look for any K_{S,T} in G1
        valid = not has_kst(G1, S, T)
        num_edges = int(G1.sum())
        combined_score = (num_edges / KST_UPPER_BOUND) if valid else 0.0
        return {
            "combined_score": float(combined_score),
            "validity": 1.0 if valid else 0.0,
            "num_edges": float(num_edges),
            "row_degree_variance": float(np.var(G1.sum(axis=1))),
        }
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        return {"combined_score": 0.0, "validity": 0.0, "row_degree_variance": 0.0}


def evaluate_stage2(program_path):
    """Full evaluation (same as evaluate)."""
    return evaluate(program_path)


def _zero_metrics():
    return {
        "num_edges": 0.0,
        "edge_density": 0.0,
        "validity": 0.0,
        "target_ratio": 0.0,
        "violation_count": 0.0,
        "row_degree_variance": 0.0,
        "g2_bonus": 0.0,
        "combined_score": 0.0,
    }


if __name__ == "__main__":
    # Quick self-test with the initial program
    import pathlib
    initial = str(pathlib.Path(__file__).parent / "initial_program.py")
    print(f"Testing evaluator with: {initial}")
    result = evaluate(initial)
    print("Metrics:", result)
