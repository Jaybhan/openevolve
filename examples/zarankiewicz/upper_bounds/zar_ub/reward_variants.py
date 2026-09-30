"""Reward variants for the verified branch of the score (experiment E25, owner R1).

E27 NOTE: the LIVE reward is zar_ub/reward.py v3 = "VR**": VR below plus R3's two fixes (only work above
20,000 conflicts per case counts; Depth/Close importance from lower-bound work, zero below 1e6 conflicts),
which this module's Config cannot express (see experiments/E25_reward/attacks/fixes.py for the offline
version; experiments/E27_integration/rescore_benchmark.py --check reproduces its numbers with reward.score).
This module is kept as the research record of E25.

Why: reward v2 (zar_ub/reward.py, design §5.3) averages gain_I over mostly near-square cells, so a
rule that helps only specific cells is under-rewarded (DGH, which closes z(11,21) <= 116 with zero
SAT, scores exactly 0.20 = doing nothing).  This module scores a program from PER-CELL RESULTS only
(gain_I, tail_I, W_I, closed_I, role, shape), so variants can be compared offline on cached Lean
masks (experiments/E25_reward) without re-running Lean.  Nothing here touches the unverified
branch or the hard-zero rules: `score()` keeps reward.py's soundness ordering exactly
(hard-zero 0 < unverified <= 0.19 < verified no-op = 0.20 <= verified with gain).

Pure functions only.  Per-cell input (`Cell`):
    key      table id (file stem), roles  subset of {train, band, wide, wide_gt, target, target_x, gen}
    m n s t w, family  square | wide | vwide | gen  (aspect ratio n/m: <1.2, <1.8, >=1.8; (s,t) != (3,3) -> gen)
    W        Σ d over the library survivors S_I (conflicts; censored labels as stored)
    n_surv   |S_I|;  gain = gain_I(K^L);  tail = tail_I(K^L);  closed = n_surv > 0 and every survivor killed

Variants (Config fields; `VARIANTS` holds the named configurations used in E25):
    V0  current formula: 0.20 + 0.80 (0.40 G_train + 0.30 G_target + 0.10 G_gen + 0.20 Tail), plain means
    V1  mixture: each role aggregate = alpha * mean + (1 - alpha) * max over its cells
    V2  power mean M_p over cells (p = 2, 3, 4)
    V3  cell-family mean: average within {square, wide, vwide, gen}, then average the family means;
        used with the WIDE exact cells added to TRAIN (suite "S1")
    V4  work-weighted means: cell weight log(1 + W_I / W0)  (W0 = 1: the literal log(1 + W);
        W0 = 2000: the 2k-conflict probe budget, so a table whose whole work is a few hundred
        conflicts carries almost no weight)
    V5  closure bonus: + w_close * max_{closed I} a_I, a_I = min(1, log(1 + W_I/W0) / log(1 + W_ref/W0)),
        W0 = 2000, W_ref = 1e9 (about the work of the largest target table)
    V6  target focus: one table dominates: 0.20 + 0.80 (f_gain g_T + f_tail tail_T + (1 - f_gain - f_tail) * base)
    VD  depth term: + w_depth * max_I a_I * gain_I  (best importance-weighted single-cell progress)
    VR  the E25 recommendation (see RECOMMENDED): S1 suite, family-balanced log(1 + W/2000)-weighted
        breadth + depth + closure + tail:
        0.20 + 0.80 (0.28 G_train + 0.21 G_target + 0.07 G_gen + 0.14 Tail + 0.15 Depth + 0.15 Close),
        a_I = min(1, ln(1 + W_I/2000) / ln(1 + 1e9/2000)); Depth/Close over TRAIN ∪ TARGET cells only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FLOOR_VERIFIED = 0.20
CAP_UNVERIFIED = 0.19
SPAN = 1.0 - FLOOR_VERIFIED


@dataclass(frozen=True)
class Cell:
    key: str
    roles: Tuple[str, ...]
    m: int
    n: int
    s: int
    t: int
    w: int
    family: str
    W: float
    n_surv: int
    gain: float
    tail: float
    closed: bool = False
    censored_share: float = 0.0  # share of W carried by censored labels (reported, never scored)


def family_of(m: int, n: int, s: int, t: int) -> str:
    if (s, t) != (3, 3):
        return "gen"
    r = max(m, n) / min(m, n)
    if r < 1.2:
        return "square"
    if r < 1.8:
        return "wide"
    return "vwide"


# --------------------------------------------------------------------------------------
# suites: which roles feed which term
# --------------------------------------------------------------------------------------
SUITES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    # the suite of reward v2 today (suite.py: TRAIN_CELLS, DEFAULT_TARGETS, GEN_CELLS)
    "S0": {"train": ("train",), "target": ("target",), "gen": ("gen_default",)},
    # E25 extended suite: WIDE exact cells enter TRAIN, the later targets enter TARGET, the
    # E12 alternative GEN cell (8,9;2,2) w=27 enters GEN
    "S1": {"train": ("train", "wide", "wide_gt"), "target": ("target", "target_x"), "gen": ("gen",)},
}


def _in(cell: Cell, roles: Sequence[str]) -> bool:
    rs = set(cell.roles)
    if "gen_default" in roles and "gen_default" in rs:
        return True
    return bool(rs & set(r for r in roles if r != "gen_default"))


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    name: str = "V0"
    suite: str = "S0"
    # term weights inside the 0.80 span (normalised to sum 1 by `inner`)
    w_train: float = 0.40
    w_target: float = 0.30
    w_gen: float = 0.10
    w_tail: float = 0.20
    w_depth: float = 0.0
    w_close: float = 0.0
    # role aggregate: "mean" | "mix" (alpha*mean + (1-alpha)*max) | "power" (M_p)
    agg: str = "mean"
    alpha: float = 1.0
    p: float = 1.0
    family: bool = False            # V3: average within families, then over families
    work_w0: Optional[float] = None  # V4: cell weight log(1 + W/w0); None = unweighted
    imp_w0: float = 2000.0           # importance a_I for depth / closure / max terms
    imp_wref: float = 1.0e9          # ~ the work of the largest target table: closing it is worth a_I = 1
    imp_max: bool = False            # V1: weight the max term by a_I
    depth_gen: bool = False          # include GEN cells in the depth / closure maxima (default: TRAIN ∪ TARGET only)
    focus: Optional[str] = None      # V6: table key that dominates
    f_gain: float = 0.6
    f_tail: float = 0.2


def importance(W: float, w0: float = 2000.0, wref: float = 1.0e9) -> float:
    """a_I in [0, 1]: log-scaled share of a reference amount of SAT work (1 at W >= wref)."""
    if W <= 0:
        return 0.0
    return min(1.0, math.log1p(W / w0) / math.log1p(wref / w0))


def _weight(c: Cell, cfg: Config) -> float:
    if cfg.work_w0 is None:
        return 1.0
    return math.log1p(max(c.W, 0.0) / cfg.work_w0)


def _agg(vals: List[Tuple[float, float, float]], cfg: Config) -> float:
    """vals = [(value, weight, importance)]; returns the role aggregate (0 when empty)."""
    vals = [v for v in vals if v[1] > 0]
    if not vals:
        return 0.0
    tw = sum(w for _, w, _ in vals)
    if cfg.agg == "power":
        return (sum(w * max(0.0, x) ** cfg.p for x, w, _ in vals) / tw) ** (1.0 / cfg.p)
    mean = sum(w * x for x, w, _ in vals) / tw
    if cfg.agg == "mix":
        mx = max((a if cfg.imp_max else 1.0) * x for x, _, a in vals)
        return cfg.alpha * mean + (1.0 - cfg.alpha) * mx
    return mean


def role_value(cells: Iterable[Cell], cfg: Config, attr: str = "gain") -> Optional[float]:
    """Aggregate of `attr` over cells (None when there is no cell with survivors)."""
    cs = [c for c in cells if c.n_surv > 0]
    if not cs:
        return None
    def trip(c):
        return (getattr(c, attr), _weight(c, cfg), importance(c.W, cfg.imp_w0, cfg.imp_wref))
    if not cfg.family:
        return _agg([trip(c) for c in cs], cfg)
    fams: Dict[str, List[Cell]] = {}
    for c in cs:
        fams.setdefault(c.family, []).append(c)
    return sum(_agg([trip(c) for c in v], cfg) for v in fams.values()) / len(fams)


def components(cells: Sequence[Cell], cfg: Config) -> Dict[str, float]:
    suite = SUITES[cfg.suite]
    tr = [c for c in cells if _in(c, suite["train"])]
    ta = [c for c in cells if _in(c, suite["target"])]
    ge = [c for c in cells if _in(c, suite["gen"])]
    G_train = role_value(tr, cfg) or 0.0
    G_target = role_value(ta, cfg)
    G_target = G_train if G_target is None else G_target
    G_gen = role_value(ge, cfg)
    G_gen = G_train if G_gen is None else G_gen
    # Tail: over TRAIN ∪ TARGET (as reward.py); the same aggregation as the gains
    Tail = role_value(tr + ta, cfg, "tail") or 0.0
    scored = [c for c in tr + ta + (ge if cfg.depth_gen else []) if c.n_surv > 0]
    depth = max((importance(c.W, cfg.imp_w0, cfg.imp_wref) * c.gain for c in scored), default=0.0)
    close = max((importance(c.W, cfg.imp_w0, cfg.imp_wref) for c in scored if c.closed), default=0.0)
    return dict(G_train=G_train, G_target=G_target, G_gen=G_gen, Tail=Tail, Depth=depth, Close=close)


def inner(cells: Sequence[Cell], cfg: Config) -> float:
    """The [0, 1] quantity multiplied by the 0.80 span."""
    comp = components(cells, cfg)
    ws = dict(G_train=cfg.w_train, G_target=cfg.w_target, G_gen=cfg.w_gen, Tail=cfg.w_tail,
              Depth=cfg.w_depth, Close=cfg.w_close)
    tot = sum(ws.values())
    base = sum(ws[k] * comp[k] for k in ws) / tot
    if cfg.focus:
        fc = [c for c in cells if c.key == cfg.focus and c.n_surv > 0]
        if fc:
            f = fc[0]
            base = cfg.f_gain * f.gain + cfg.f_tail * f.tail + (1.0 - cfg.f_gain - cfg.f_tail) * base
    return max(0.0, min(1.0, base))


def verified_score(cells: Sequence[Cell], cfg: Config) -> float:
    return FLOOR_VERIFIED + SPAN * inner(cells, cfg)


def score(cells: Sequence[Cell], cfg: Config, *, ladder: int = 5, hard_zero: bool = False,
          unverified: float = 0.0) -> float:
    """The whole combined_score with the variant's verified branch; the other branches are
    reward.py's unchanged (hard-zero -> 0; ladder < 5 -> the caller's unverified value, capped)."""
    if hard_zero or ladder in (0, 4):
        return 0.0
    if ladder != 5:
        return min(CAP_UNVERIFIED, max(0.0, unverified))
    return verified_score(cells, cfg)


# --------------------------------------------------------------------------------------
# the named variants of E25
# --------------------------------------------------------------------------------------
def _v(name, **kw) -> Config:
    return Config(name=name, **kw)


_CLOSE = dict(w_train=0.32, w_target=0.24, w_gen=0.08, w_tail=0.16, w_close=0.20)
# reward v2's term ratios 0.40 : 0.30 : 0.10 : 0.20 scaled by 0.70, plus 0.15 depth + 0.15 closure: the
# smallest symmetric (depth, closure) weight that passes C1-C5 and the oracle stress test on both label
# views (experiments/E25_reward/results/sensitivity*.md)
_VR = dict(suite="S1", family=True, work_w0=2000.0, imp_w0=2000.0, imp_wref=1.0e9,
           w_train=0.28, w_target=0.21, w_gen=0.07, w_tail=0.14, w_depth=0.15, w_close=0.15)

VARIANTS: Dict[str, Config] = {
    "V0/S0 current": _v("V0/S0 current"),
    "V0/S1 suite only": _v("V0/S1 suite only", suite="S1"),
    "V1 mix a=.5": _v("V1 mix a=.5", suite="S1", agg="mix", alpha=0.5),
    "V1 mix a=.7": _v("V1 mix a=.7", suite="S1", agg="mix", alpha=0.7),
    "V2 p=2": _v("V2 p=2", suite="S1", agg="power", p=2.0),
    "V2 p=3": _v("V2 p=3", suite="S1", agg="power", p=3.0),
    "V2 p=4": _v("V2 p=4", suite="S1", agg="power", p=4.0),
    "V3 family": _v("V3 family", suite="S1", family=True),
    "V4 log(1+W)": _v("V4 log(1+W)", suite="S1", work_w0=1.0),
    "V4 log(1+W/2k)": _v("V4 log(1+W/2k)", suite="S1", work_w0=2000.0),
    "V4 log(1+W/20k)": _v("V4 log(1+W/20k)", suite="S1", work_w0=20000.0),
    "V5 closure": _v("V5 closure", suite="S1", **_CLOSE),
    "V5 closure wref=1e6": _v("V5 closure wref=1e6", suite="S1", imp_wref=1.0e6, **_CLOSE),
    "V3+V4": _v("V3+V4", suite="S1", family=True, work_w0=2000.0),
    "V3+V4+V5": _v("V3+V4+V5", suite="S1", family=True, work_w0=2000.0, **_CLOSE),
    "V1+V3+V4+V5": _v("V1+V3+V4+V5", suite="S1", family=True, work_w0=2000.0, agg="mix", alpha=0.7, imp_max=True,
                      **_CLOSE),
    # the E25 recommendation (experiments/E25_reward/REPORT.md): breadth (family-balanced, log(1+W/2000)-
    # weighted role means) + depth (best a_I * gain_I) + closure (best a_I over closed cells) + tail
    "VR recommended": _v("VR recommended", **_VR),
    "VR w0=20k": _v("VR w0=20k", **dict(_VR, work_w0=20000.0)),
    "VR no family": _v("VR no family", **dict(_VR, family=False)),
    "V6 focus (12,18,109)": _v("V6 focus (12,18,109)", focus="m12_n18_s3_t3_w109", **_VR),
    "V6 focus (11,21,117)p": _v("V6 focus (11,21,117)p", focus="m11_n21_s3_t3_w117_pure", **_VR),
}

RECOMMENDED = VARIANTS["VR recommended"]


# --------------------------------------------------------------------------------------
# per-cell results from a table and a Lean mask (the same quantities as reward.py)
# --------------------------------------------------------------------------------------
def cell_from_table(key: str, roles: Sequence[str], tab, mask: Optional[Sequence[bool]]) -> Cell:
    """gain_I / tail_I / W_I / closed_I of `mask` (the verified kill K^L; None = kills nothing) on a
    CaseTable, with S_I = reward.survivor_indices(tab, None) (the table's baseline_lean_mask)."""
    from . import reward
    inst = tab.inst
    S = reward.survivor_indices(tab, None)
    cap = int(getattr(tab, "conf_cap", 0) or 0)
    d = [reward.rec_difficulty(r, cap) for r in tab.records]
    W = sum(d[i] for i in S)
    g = reward.gain_I(S, d, mask)
    tl = reward.tail_I(S, d, mask)
    closed = bool(S) and bool(mask) and all(i < len(mask) and mask[i] for i in S)
    cens = sum(d[i] for i in S if reward.is_censored(tab.records[i])) / W if W > 0 else 0.0
    return Cell(key=key, roles=tuple(roles), m=inst["m"], n=inst["n"], s=inst["s"], t=inst["t"], w=inst["w"],
                family=family_of(inst["m"], inst["n"], inst["s"], inst["t"]), W=W, n_surv=len(S),
                gain=g, tail=tl, closed=closed, censored_share=cens)


def with_gains(cells: Sequence[Cell], gains: Dict[str, float], tails: Optional[Dict[str, float]] = None) -> List[Cell]:
    """Synthetic programs for property tests: replace gain/tail per key (closed iff gain == 1)."""
    tails = tails if tails is not None else gains
    out = []
    for c in cells:
        g = float(gains.get(c.key, 0.0))
        out.append(replace(c, gain=g, tail=float(tails.get(c.key, 0.0)), closed=(c.n_surv > 0 and g >= 1.0)))
    return out


def cells_from_suite(tables: Dict[str, list], lean_masks: Dict[str, Optional[Sequence[bool]]],
                     library_masks: Optional[Dict[str, Optional[Sequence[bool]]]] = None,
                     include_target: bool = True) -> List[Cell]:
    """Integration helper: the Cells of a `suite.load_suite()` dict and the evaluator's per-tag Lean
    masks, with S_I exactly as reward.score computes it (table baseline mask, else library mask).
    Roles are the suite kinds, so the recommended config (suite "S1") works once suite.py puts the
    WIDE cells in "train" and the later targets in "target" (see experiments/E25_reward/REPORT.md)."""
    from . import reward
    library_masks = library_masks or {}
    out: List[Cell] = []
    for kind in reward.SCORED_KINDS:
        if kind == "target" and not include_target:
            continue
        for inst, tab in tables.get(kind, []):
            tag = inst.tag
            S = reward.survivor_indices(tab, library_masks.get(tag))
            cap = int(getattr(tab, "conf_cap", 0) or 0)
            d = [reward.rec_difficulty(r, cap) for r in tab.records]
            lm = lean_masks.get(tag)
            W = sum(d[i] for i in S)
            closed = bool(S) and bool(lm) and all(i < len(lm) and lm[i] for i in S)
            cens = sum(d[i] for i in S if reward.is_censored(tab.records[i])) / W if W > 0 else 0.0
            roles = (kind, "gen_default") if kind == "gen" else (kind,)
            out.append(Cell(key=f"{kind}:{tag}:{'pure' if not getattr(tab, 'use_table', True) else 'cited'}",
                            roles=roles, m=inst.m, n=inst.n, s=inst.s, t=inst.t, w=inst.w,
                            family=family_of(inst.m, inst.n, inst.s, inst.t), W=W, n_surv=len(S),
                            gain=reward.gain_I(S, d, lm), tail=reward.tail_I(S, d, lm), closed=closed,
                            censored_share=cens))
    return out
