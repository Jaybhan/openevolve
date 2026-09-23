#!/usr/bin/env python3
"""E7a: ONE LLM call on OpenRouter to see what a model's response to the real
OpenEvolve prompt looks like, then apply the diff and score the child.

  python experiments/E7_llm_smoke/probe_one.py --model deepseek/deepseek-v4-flash [--program initial_program.py] [--rewrite]

Writes experiments/E7_llm_smoke/probes/<timestamp>_<model>/{prompt.md,response.md,child.py,metrics.json}
and appends the cost to experiments/cost_ledger.md.
"""
import argparse, json, os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(UB, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, UB)
from openevolve.config import load_config
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import apply_diff, parse_full_rewrite
import evaluator  # noqa: E402


def call_openrouter(model, system, user, max_tokens=16000, temperature=0.6, reasoning="low"):
    key = open(os.path.join(UB, ".openrouter_key")).read().strip()
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "max_tokens": max_tokens, "temperature": temperature, "usage": {"include": True}}
    if reasoning == "none":
        payload["reasoning"] = {"exclude": True, "effort": "low"}
    elif reasoning in ("low", "medium", "high"):
        payload["reasoning"] = {"effort": reasoning}
    body = json.dumps(payload).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "HTTP-Referer": "https://github.com/algorithmicsuperintelligence/openevolve", "X-Title": "zar_ub probe"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--program", default=os.path.join(UB, "initial_program.py"))
    ap.add_argument("--rewrite", action="store_true", help="ask for a full rewrite instead of SEARCH/REPLACE diffs")
    ap.add_argument("--max-tokens", type=int, default=16000); ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--reasoning", default="low", choices=["none", "low", "medium", "high", "default"])
    a = ap.parse_args()
    cfg = load_config(os.path.join(UB, "config.yaml"))
    code = open(a.program).read()
    r = evaluator.evaluate(a.program)
    ps = PromptSampler(cfg.prompt)
    p = ps.build_prompt(current_program=code, parent_program=code, program_metrics=r.metrics, previous_programs=[], top_programs=[],
                        inspirations=[], language="python", evolution_round=1, diff_based_evolution=not a.rewrite,
                        program_artifacts=r.artifacts, feature_dimensions=cfg.database.feature_dimensions)
    out = os.path.join(HERE, "probes", time.strftime("%Y%m%d_%H%M%S") + "_" + a.model.replace("/", "_") + "_r" + a.reasoning)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "prompt.md"), "w").write("# SYSTEM\n\n" + p["system"] + "\n\n# USER\n\n" + p["user"])
    d, secs = call_openrouter(a.model, p["system"], p["user"], a.max_tokens, a.temperature, a.reasoning)
    json.dump(d, open(os.path.join(out, "raw_response.json"), "w"), indent=1)
    if "error" in d or not d.get("choices"):
        print("API error / empty response:", json.dumps(d)[:1500]); return
    msg = d["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text and msg.get("reasoning"):
        text = "[reasoning only]\n" + msg["reasoning"]
    print("finish_reason:", d["choices"][0].get("finish_reason"), "| content chars:", len(msg.get("content") or ""), "| reasoning chars:", len(msg.get("reasoning") or ""))
    usage = d.get("usage", {})
    open(os.path.join(out, "response.md"), "w").write(text)
    if a.rewrite:
        child = parse_full_rewrite(text, "python") or ""
    else:
        child = apply_diff(code, text, cfg.diff_pattern)
    open(os.path.join(out, "child.py"), "w").write(child)
    applied = child.strip() != code.strip() and bool(child.strip())
    res = evaluator.evaluate(os.path.join(out, "child.py")) if applied else None
    summary = {"model": a.model, "seconds": round(secs, 1), "usage": usage, "cost_usd": usage.get("cost"),
               "response_chars": len(text), "diff_applied": applied,
               "parent_metrics": r.metrics, "child_metrics": (res.metrics if res else None),
               "child_artifacts": ({k: v[:2000] for k, v in res.artifacts.items()} if res else None)}
    json.dump(summary, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("child_artifacts", "parent_metrics")}, indent=1))
    if res:
        for k, v in res.artifacts.items():
            print(f"--- artifact {k} ---\n{v[:1500]}")
    os.system(f'"{sys.executable}" "{os.path.join(HERE, "..", "cost.py")}" "E7a probe {a.model}"')


if __name__ == "__main__":
    main()
