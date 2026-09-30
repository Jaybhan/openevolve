"""E28: run the live evaluator on one candidate with a PRIVATE, initially empty gate cache (cold Lean gate), so the
result does not depend on cache/gate/ entries written by the integrator.  Usage:
    python run_eval.py <candidate.py> <gate_cache_dir> <out.json>
"""
import json, os, sys, time

UB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, UB)
os.environ.setdefault("ZAR_UB_NO_LLM", "1")
import zar_ub.lean_gate as lg  # noqa: E402

cand, gdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(gdir, exist_ok=True)
lg.GATE_CACHE_DIR = gdir
import evaluator  # noqa: E402

t0 = time.time()
res = evaluator.evaluate(cand)
el = time.time() - t0
metrics = getattr(res, "metrics", res if isinstance(res, dict) else {})
arts = getattr(res, "artifacts", {}) or {}
sb = arts.get("score_breakdown")
json.dump({"candidate": cand, "seconds": el, "metrics": metrics,
           "score_breakdown": sb if isinstance(sb, (str, dict, list)) else str(sb),
           "gate_cache_files": len([f for f in os.listdir(gdir) if f.endswith(".json")])},
          open(out, "w"), indent=1, default=str)
print(os.path.basename(cand), "combined_score=%.4f" % float(metrics.get("combined_score", float("nan"))), "%.0fs" % el)
