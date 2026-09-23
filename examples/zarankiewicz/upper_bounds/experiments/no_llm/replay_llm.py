"""LLM stand-ins for OpenEvolve that never touch the network (design §10.1, T-5).

Both classes implement `openevolve.llm.base.LLMInterface` and are plugged in through
`LLMModelConfig.init_client` (openevolve/config.py: `init_client: Optional[Callable]`,
called by `LLMEnsemble` as `init_client(model_cfg)`; workers receive the config through
pickling, so the factories below are module-level functions).

    SyntheticMutator  -- answers each prompt with a scripted SEARCH/REPLACE diff from the
                         snippet bank (same logic as tools/stub_llm.py, in-process).
    ReplayLLM         -- replays recorded responses from a JSONL file ({"response": str} per
                         line, or OpenEvolve checkpoint `programs/*.json` with "llm_response"),
                         cycling when exhausted.

Configuration reaches the factories through the model name or environment variables:
    name "synthetic[:<bank path>]"      ZAR_UB_STUB_BANK   (default tests/snippets)
    name "replay[:<jsonl path>]"        ZAR_UB_REPLAY_FILE
    ZAR_UB_STUB_MODE = hash | roundrobin
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(UB, "..", "..", ".."))
for p in (ROOT, UB, os.path.join(UB, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from openevolve.llm.base import LLMInterface  # noqa: E402
import stub_llm  # noqa: E402  (tools/stub_llm.py)


def _arg_from_name(name: Optional[str], prefix: str) -> Optional[str]:
    if name and name.startswith(prefix + ":"):
        return name[len(prefix) + 1:]
    return None


class SyntheticMutator(LLMInterface):
    """Scripted mutations from the snippet bank; deterministic given the prompt sequence."""

    def __init__(self, model_cfg=None, bank_path: Optional[str] = None, mode: Optional[str] = None):
        self.model = getattr(model_cfg, "name", None) or "synthetic"
        self.weight = getattr(model_cfg, "weight", 1.0)
        bank_path = bank_path or _arg_from_name(self.model, "synthetic") or os.environ.get("ZAR_UB_STUB_BANK")
        self.bank = stub_llm.load_bank(bank_path)
        self.chooser = stub_llm.Chooser(len(self.bank), mode or os.environ.get("ZAR_UB_STUB_MODE", "hash"))
        self.history: List[Dict[str, object]] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context("", [{"role": "user", "content": prompt}], **kwargs)

    async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
        r = stub_llm.respond(messages, self.bank, self.chooser)
        self.history.append({"entry": r["entry"], "index": r["index"]})
        print(f"-> SYNTHETIC {self.model} | bank entry {r['entry']} ({len(r['content'])} chars)", flush=True)
        return str(r["content"])


class ReplayLLM(LLMInterface):
    """Replay recorded responses in order (cycling)."""

    def __init__(self, model_cfg=None, path: Optional[str] = None):
        self.model = getattr(model_cfg, "name", None) or "replay"
        self.weight = getattr(model_cfg, "weight", 1.0)
        path = path or _arg_from_name(self.model, "replay") or os.environ.get("ZAR_UB_REPLAY_FILE")
        if not path:
            raise ValueError("ReplayLLM needs a path: model name 'replay:<file>' or ZAR_UB_REPLAY_FILE")
        self.responses = self._load(path)
        if not self.responses:
            raise ValueError(f"ReplayLLM: no responses found in {path}")
        self.k = 0

    @staticmethod
    def _load(path: str) -> List[str]:
        out: List[str] = []
        if os.path.isdir(path):  # an OpenEvolve checkpoint: programs/*.json carry llm_response
            for f in sorted(glob.glob(os.path.join(path, "programs", "*.json"))):
                with open(f, encoding="utf-8") as fh:
                    d = json.load(fh)
                if d.get("llm_response"):
                    out.append(str(d["llm_response"]))
            return out
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        try:
            data = json.loads(text)
            items = data if isinstance(data, list) else data.get("responses", [])
            return [str(x["response"] if isinstance(x, dict) else x) for x in items]
        except ValueError:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    d = json.loads(line)
                    out.append(str(d["response"] if isinstance(d, dict) else d))
            return out

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context("", [{"role": "user", "content": prompt}], **kwargs)

    async def generate_with_context(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
        r = self.responses[self.k % len(self.responses)]
        self.k += 1
        print(f"-> REPLAY {self.model} | response {self.k}/{len(self.responses)} ({len(r)} chars)", flush=True)
        return r


# init_client factories (module-level so they pickle into worker processes)
def make_synthetic(model_cfg):
    return SyntheticMutator(model_cfg)


def make_replay(model_cfg):
    return ReplayLLM(model_cfg)


def make_client(model_cfg):
    """Dispatch on the model name: 'replay…' -> ReplayLLM, anything else -> SyntheticMutator."""
    name = getattr(model_cfg, "name", "") or ""
    return ReplayLLM(model_cfg) if name.startswith("replay") else SyntheticMutator(model_cfg)
