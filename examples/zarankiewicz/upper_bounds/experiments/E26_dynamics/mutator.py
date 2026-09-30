"""E26 GenomeMutator: an LLM stand-in (openevolve.llm.base.LLMInterface) that edits the parent's
atom set (genome.py) and answers with ONE SEARCH/REPLACE block over the whole EVOLVE block.

It is blind to scores: every change of population composition comes from OpenEvolve's selection
(the reward + MAP-Elites), which is what E26 measures.  Deterministic: the operator is drawn from
sha1(seed | parent code | #times this parent code was seen), like tools/stub_llm.Chooser.

Operator table (weights; identical for every configuration):
  noop 2.0        same genome, NOTES revision bump (the LLM "tweaks" without changing any rule)
  add:R/D/C/T 1.0 each   add a sound rule atom (no-op revision if already present)
  drop 2.0        remove one present sound atom (noop if none)
  add:B 1.5       append a prune with a broken proof            (-> L1)
  add:U 1.0       add an unsound rule to the Python mirror       (-> battery 0)
  repair 1.0      remove B and U (noop if neither present)
  rewrite 1.5     replace the whole program by a bank entry: one of
                  {initial, R, D, C, T, R+D, B, U}  (the tools/stub_llm.py behaviour)
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(UB, "..", "..", ".."))
for p in (ROOT, UB, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from openevolve.llm.base import LLMInterface  # noqa: E402
import genome as G  # noqa: E402

OPS = [("noop", 2.0), ("add:R", 1.0), ("add:D", 1.0), ("add:C", 1.0), ("add:T", 1.0), ("drop", 2.0),
       ("add:B", 1.5), ("add:U", 1.0), ("repair", 1.0), ("rewrite", 1.5)]
BANK = ["-", "R", "D", "C", "T", "R+D", "B", "U"]
_CURRENT_RE = re.compile(r"# Current Program\s*\n```[a-zA-Z]*\n(.*?)\n```", re.S)


def _u(seed: str, prompt: str, k: int, salt: str) -> float:
    h = hashlib.sha1(f"{seed}|{salt}|{k}|{prompt}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def mutate(genome, rev: int, u1: float, u2: float):
    """(op name, child genome) from two uniforms."""
    tot = sum(w for _, w in OPS)
    x, op = u1 * tot, OPS[-1][0]
    for name, w in OPS:
        if x < w:
            op = name
            break
        x -= w
    g = set(genome)
    if op.startswith("add:"):
        g.add(op[4:])
    elif op == "drop":
        present = [a for a in G.SOUND_ATOMS if a in g]
        if present:
            g.discard(present[int(u2 * len(present)) % len(present)])
    elif op == "repair":
        g.discard("B")
        g.discard("U")
    elif op == "rewrite":
        g = set(G.parse_gstr(BANK[int(u2 * len(BANK)) % len(BANK)]))
    return op, frozenset(g)


class GenomeMutator(LLMInterface):
    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, "name", None) or "e26-genome"
        self.weight = getattr(model_cfg, "weight", 1.0)
        self.seed = os.environ.get("E26_MUTATOR_SEED", "26")
        self.seen: Dict[str, int] = {}
        self.log = os.environ.get("E26_MUTATOR_LOG")

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context("", [{"role": "user", "content": prompt}], **kwargs)

    async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
        user = ""
        for msg in messages:
            if msg.get("role") == "user":
                user = str(msg.get("content", ""))
        m = _CURRENT_RE.search(user)
        cur = m.group(1) if m else user
        parent, prev = G.parse(cur)
        # hash ONLY the parent's code: the rest of the prompt carries wall-clock metrics
        # (eval_seconds, gate_seconds), which made the op sequence non-reproducible (E26 pilot)
        key = cur
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        k = self.seen.get(h, 0)
        self.seen[h] = k + 1
        op, child = mutate(parent, prev, _u(self.seed, key, k, "op"), _u(self.seed, key, k, "arg"))
        rev = int(_u(self.seed, key, k, "rev") * 1e9)
        old = G.block_of(cur)
        if old is None:
            old = G.block_of(G.render(parent, prev))
        new = G.evolve_block(child, rev)
        if self.log:
            try:
                with open(self.log, "a") as f:
                    f.write(f"{G.gstr(parent)}\t{op}\t{G.gstr(child)}\n")
            except OSError:
                pass
        print(f"-> E26 mutator | {G.gstr(parent)} --{op}--> {G.gstr(child)}", flush=True)
        return (f"E26 scripted mutation `{op}`: {G.gstr(parent)} -> {G.gstr(child)}.\n\n"
                f"<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n")


def make_mutator(model_cfg):
    return GenomeMutator(model_cfg)
