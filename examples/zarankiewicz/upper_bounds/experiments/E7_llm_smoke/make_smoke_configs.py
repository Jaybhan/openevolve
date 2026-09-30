#!/usr/bin/env python3
"""Re-derive the configs that are config.yaml plus a few overrides (E27; re-run after config.yaml changes):

  config_smoke_luna.yaml    (E18/E21) gpt-5.6-luna only, 15 iterations, one island, population 20, 2 parallel evals
  config_smoke_sonnet.yaml  (E22)     claude-sonnet-5 only, 20 iterations, one island, population 25, 2 parallel evals
  config_novelty.yaml       (§5.6)    kill_novelty as a third MAP-Elites axis (10 x 6 x 5 bins)

Line-based substitution on the config.yaml TEXT (keeps every comment and the prompt verbatim), then a YAML
parse of the result checks that only the intended keys differ.  experiments/no_llm/make_config_stub.py
derives config_stub.yaml the same way (through yaml, as before).

    python experiments/E7_llm_smoke/make_smoke_configs.py
"""
from __future__ import annotations

import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(UB, "config.yaml")

MODELS_RE = re.compile(r"(  models:\n)((?:    - name: .*\n      weight: .*\n)+)")


def _sub(text: str, pattern: str, repl: str) -> str:
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"pattern not found in config.yaml: {pattern!r}")
    return new


def smoke(text: str, header: str, model: str, iterations: int, population: int) -> str:
    t = header + text
    t = _sub(t, r"^max_iterations: \d+$", f"max_iterations: {iterations}")
    m = MODELS_RE.search(t)
    if not m:
        raise RuntimeError("llm.models block not found")
    t = t[: m.start(2)] + f'    - name: "{model}"\n      weight: 1.0\n' + t[m.end(2):]
    t = _sub(t, r"^  population_size: \d+$", f"  population_size: {population}")
    t = _sub(t, r"^  num_islands: \d+$", "  num_islands: 1")
    t = _sub(t, r"^  parallel_evaluations: \d+$", "  parallel_evaluations: 2")
    return t


def novelty(text: str) -> str:
    lines = text.split("\n")
    lines[0] = ("# OpenEvolve configuration with kill_novelty as a THIRD MAP-Elites axis (design §5.6): use it once\n"
                "# cache/ledger/accepted_prunes.jsonl has >= 5 entries (zar_ub/promote.py prints the switch).\n"
                "# Identical to config.yaml except database.feature_dimensions / feature_bins.")
    t = "\n".join(lines)
    t = _sub(t, r"^  # 10 x 6 = 60 cells for population 60 \(design §5\.6\); kill_novelty becomes a third axis via\n"
                r"  # config_novelty\.yaml once cache/ledger/accepted_prunes\.jsonl has >= 5 entries\.\n"
                r'  feature_dimensions: \["proven_gain", "lean_ladder"\]',
             "  # 10 x 6 x 5 = 300 cells; raise population_size when the ledger is large enough to fill them.\n"
             '  feature_dimensions: ["proven_gain", "lean_ladder", "kill_novelty"]')
    t = _sub(t, r"^    lean_ladder: 6$", "    lean_ladder: 6\n    kill_novelty: 5")
    return t


def main() -> None:
    text = open(SRC, encoding="utf-8").read()
    base = yaml.safe_load(text)
    outs = {
        "config_smoke_luna.yaml": smoke(
            text, "# SMOKE config (E18/E21): luna only, 15 iterations, one island -- derived from config.yaml.\n",
            "openai/gpt-5.6-luna", 15, 20),
        "config_smoke_sonnet.yaml": smoke(
            text, "# SMOKE config (E22): claude-sonnet-5, low reasoning, 20 iterations, one island -- derived from "
                  "config.yaml.\n", "anthropic/claude-sonnet-5", 20, 25),
        "config_novelty.yaml": novelty(text),
    }
    for name, t in outs.items():
        d = yaml.safe_load(t)
        diff = sorted(k for k in set(base) | set(d) if base.get(k) != d.get(k))
        assert d["prompt"] == base["prompt"] and d["evaluator"]["timeout"] == base["evaluator"]["timeout"], name
        with open(os.path.join(UB, name), "w", encoding="utf-8") as f:
            f.write(t)
        print(f"wrote {name} (top-level keys differing from config.yaml: {diff})")


if __name__ == "__main__":
    main()
