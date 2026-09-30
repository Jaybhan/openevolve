"""E25 / R3-critic: shared loading + scoring helpers for the adversarial attacks on R1's VR reward.

Offline only: no SAT, no Lean, no LP.  Reads cache/ tables (never writes them), R1's cached Lean masks
(experiments/E25_reward/masks/*.json), R1's Farkas pool (results/cert_pool.json) and A1's FINAL ground
truth (experiments/E24_difficulty/ground_truth.jsonl).

Label views
  "table"  the labels the live reward uses today (censored cases: min(max(cap, fhat), 20 cap))
  "gt"     A1's final ground truth, in memory: exact d where A1 resolved the case; for still-unknown
           cases d := max(table label, A1 lower bound)  (identical rule to R1's analyze.relabel, but
           pinned to the FINAL ground_truth.jsonl -- R1's *_gtlabels results were computed at 22:20,
           before that file existed at 22:36, i.e. on the initial/interim file)
  "gtlb"   conservative view: exact d where resolved, the LOWER BOUND (conflicts reached) where not.
           This is the only view that never uses an fhat extrapolation.

Realizability.  The gate accepts instance-gated Lean (`if P.m = 10 ∧ P.n = 23 then ... else counting P`,
R1's inst_dgh_10_23 passed L5), and Farkas certificates in SCHEMA_DATA are per (m,n,s,t).  So for any
verified kill set K_I on a table I (a pool certificate, the DGH mask, the recipe mask), a program whose
kill is K_I on the tables it chooses and nothing elsewhere is realizable.  Attack masks marked "real"
are unions of such per-table pieces taken from Lean-gated masks (dgh_lib, schema_recipe_frozen) or from
exactly re-verified Farkas certificates (cert_pool.json; Prune.ofFarkas is sound by farkas_sound).
Masks marked "oracle" are shapes, not programs.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
E25 = os.path.dirname(_HERE)
ROOT = os.path.dirname(os.path.dirname(E25))
sys.path.insert(0, ROOT)
sys.path.insert(0, E25)

import universe as U  # noqa: E402
from zar_ub import reward  # noqa: E402
from zar_ub import reward_variants as RV  # noqa: E402

GT_FINAL = os.path.join(os.path.dirname(E25), "E24_difficulty", "ground_truth.jsonl")
MASKS = os.path.join(E25, "masks")
POOL = os.path.join(E25, "results", "cert_pool.json")

V0 = RV.VARIANTS["V0/S0 current"]
VR = RV.RECOMMENDED


def ordered_universe(include_gt: bool = True):
    uni = U.universe(include_gt=include_gt)
    order = {"train": 0, "band": 1, "wide": 2, "wide_gt": 3, "target": 4, "target_x": 5, "gen": 6}
    keys = sorted(uni, key=lambda k: (min(order.get(r, 9) for r in uni[k]["roles"]), k))
    return keys, uni


def load_gt(path: str = GT_FINAL) -> Dict[tuple, dict]:
    gt = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            gt[(r["cell"], tuple(r["rows"]), tuple(r["cols"]))] = r
    return gt


def gt_of(gt, key, rec):
    cell = key[:-3] if key.endswith("_gt") else key
    return gt.get((cell, tuple(rec.rows), tuple(rec.cols)))


@dataclass
class Env:
    keys: List[str]
    uni: Dict[str, dict]
    tabs: dict
    S: Dict[str, List[int]]                 # library survivors (table baseline mask)
    d: Dict[str, Dict[str, List[float]]]    # view -> key -> per-record label
    exact: Dict[str, List[Optional[bool]]]  # key -> per-record: True exact (table or GT), False censored-everywhere
    src: Dict[str, List[Optional[str]]]     # key -> per-record GT source ("table" | "deepen" | "new_table" | None)


def load_env(include_gt: bool = True) -> Env:
    keys, uni = ordered_universe(include_gt)
    tabs = {k: U.load(k, uni[k]["path"]) for k in keys}
    gt = load_gt()
    S, dv, ex, src = {}, {"table": {}, "gt": {}, "gtlb": {}}, {}, {}
    for k in keys:
        tab = tabs[k]
        S[k] = reward.survivor_indices(tab, None)
        cap = int(getattr(tab, "conf_cap", 0) or 0)
        dt = [reward.rec_difficulty(r, cap) for r in tab.records]
        dg, dl, e = list(dt), list(dt), [None] * len(dt)
        sr = [None] * len(dt)
        for i, r in enumerate(tab.records):
            if r.probe is None:
                continue
            g = gt_of(gt, k, r)
            sr[i] = g["source"] if g else None
            cens = reward.is_censored(r)
            if g is None:
                e[i] = not cens
                if cens:  # no GT row: lower bound = the probe's conflicts reached
                    dl[i] = float(max(1, int(r.probe.get("conflicts") or r.probe.get("budget_cap") or 1)))
                continue
            if g["status"] in ("unsat", "sat"):
                dg[i] = dl[i] = float(max(1, g["d"]))
                e[i] = True
            else:
                dg[i] = max(dt[i], float(g["d"]))
                dl[i] = float(max(1, g["d"]))
                e[i] = False
        dv["table"][k], dv["gt"][k], dv["gtlb"][k], ex[k], src[k] = dt, dg, dl, e, sr
    return Env(keys, uni, tabs, S, dv, ex, src)


# --------------------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------------------
def empty_masks(env: Env) -> Dict[str, List[bool]]:
    return {k: [False] * len(env.tabs[k].records) for k in env.keys}


def r1_mask(name: str, env: Env) -> Dict[str, List[bool]]:
    d = json.load(open(os.path.join(MASKS, f"{name}.json")))
    out = empty_masks(env)
    for k in env.keys:
        t = d["tables"].get(k)
        if t and t.get("lean"):
            out[k] = [bool(x) for x in t["lean"]]
    return out


def pool(env: Env) -> Dict[str, List[dict]]:
    p = json.load(open(POOL))
    return {k: p[k]["certs"] for k in env.keys if k in p}


def union(*ms: Dict[str, List[bool]]) -> Dict[str, List[bool]]:
    out = {}
    for k in ms[0]:
        out[k] = [any(m[k][i] for m in ms) for i in range(len(ms[0][k]))]
    return out


def restrict(m: Dict[str, List[bool]], keep: Callable[[str], bool]) -> Dict[str, List[bool]]:
    return {k: (v if keep(k) else [False] * len(v)) for k, v in m.items()}


def pool_mask(env: Env, keep_table: Callable[[str], bool] = lambda k: True,
              keep_cert: Callable[[str, dict], bool] = lambda k, c: True) -> Dict[str, List[bool]]:
    out = empty_masks(env)
    for k, certs in pool(env).items():
        if not keep_table(k):
            continue
        for c in certs:
            if keep_cert(k, c):
                for j in c["kills"]:
                    out[k][j] = True
    return out


# --------------------------------------------------------------------------------------
# cells (per view, with optional label transform / tail rule) and scoring
# --------------------------------------------------------------------------------------
def top_decile_ties(S, d):
    """Tie-INCLUSIVE top decile: every survivor whose d is >= the k-th largest d (order-free)."""
    if not S:
        return []
    k = max(1, math.ceil(len(S) / 10.0))
    thr = sorted((d[i] for i in S), reverse=True)[k - 1]
    return [i for i in S if d[i] >= thr]


def cells(env: Env, masks: Dict[str, List[bool]], view: str = "table", *, excess: float = 0.0,
          tail_rule: str = "index") -> List[RV.Cell]:
    """RV.Cell list exactly as RV.cell_from_table, but with the chosen label view, an optional
    excess-work transform d' = max(0, d - excess) (cases with d' = 0 leave S_I), and the tail rule
    ("index" = reward.top_decile, ties broken by record index; "ties" = tie-inclusive)."""
    out = []
    for k in env.keys:
        tab = env.tabs[k]
        d = [max(0.0, x - excess) for x in env.d[view][k]]
        S = [i for i in env.S[k] if d[i] > 0]
        mk = masks.get(k)
        W = sum(d[i] for i in S)
        g = reward.gain_I(S, d, mk)
        H = reward.top_decile(S, d) if tail_rule == "index" else top_decile_ties(S, d)
        WH = sum(d[i] for i in H)
        tl = (sum(d[i] for i in H if mk and mk[i]) / WH) if WH > 0 and mk else 0.0
        closed = bool(env.S[k]) and bool(mk) and all(mk[i] for i in env.S[k])  # closure = ALL library survivors
        inst = env.uni[k]
        out.append(RV.Cell(key=k, roles=tuple(inst["roles"]), m=inst["m"], n=inst["n"], s=inst["s"], t=inst["t"],
                           w=inst["w"], family=RV.family_of(inst["m"], inst["n"], inst["s"], inst["t"]), W=W,
                           n_surv=len(S), gain=g, tail=tl, closed=closed))
    return out


def work_removed(env: Env, masks, view="gtlb", keys: Optional[Sequence[str]] = None) -> float:
    """Absolute conflicts removed (sum of labels of killed survivors) over TRAIN∪TARGET(S1) tables."""
    tot = 0.0
    for k in (keys or env.keys):
        roles = set(env.uni[k]["roles"])
        if not roles & {"train", "wide", "wide_gt", "target", "target_x"}:
            continue
        mk = masks[k]
        tot += sum(env.d[view][k][i] for i in env.S[k] if mk[i])
    return tot


def target_work_removed(env, masks, view="gtlb"):
    return work_removed(env, masks, view, [k for k in env.keys if set(env.uni[k]["roles"]) & {"target", "target_x"}])


def score(env, masks, cfg, view="table", **kw) -> float:
    return RV.verified_score(cells(env, masks, view, **kw), cfg)


def comps(env, masks, cfg, view="table", **kw) -> dict:
    return RV.components(cells(env, masks, view, **kw), cfg)
