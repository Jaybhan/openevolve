"""E25 / R3-critic: candidate FIXES to R1's VR, scored on the same Cell lists (offline).

Each fix is (cfg, cell_kwargs, post) where cell_kwargs go to common.cells (label view, excess-work
transform, tail rule) and `post` adjusts Depth/Close (minimum-work threshold, importance from
lower-bound work).  The verified-branch formula is otherwise RV.verified_score's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import common as C
from common import RV


@dataclass
class Fix:
    name: str
    cfg: RV.Config = C.VR
    view: str = "table"
    excess: float = 0.0          # d' = max(0, d - excess): only work above the probe budget counts
    tail_rule: str = "index"     # "ties": tie-inclusive top decile (order-free)
    w_min: float = 0.0           # Depth/Close only over cells with W_imp >= w_min
    imp_view: Optional[str] = None  # label view used for W in a_I (None: same as `view`)
    close_target_only: bool = False  # Close only for TARGET (open) cells
    gen_min: float = 0.0          # GEN cells with W < gen_min leave the GEN role (fallback G_gen := G_train)


def inner(env, masks, fx: Fix) -> float:
    cs = C.cells(env, masks, fx.view, excess=fx.excess, tail_rule=fx.tail_rule)
    if fx.gen_min > 0:
        cs = [c if not (set(c.roles) & {"gen", "gen_default"}) or c.W >= fx.gen_min
              else replace(c, n_surv=0) for c in cs]
    comp = RV.components(cs, fx.cfg)
    cfg = fx.cfg
    if fx.w_min > 0 or fx.imp_view or fx.close_target_only:
        imp_cells = {c.key: c for c in C.cells(env, masks, fx.imp_view or fx.view)}  # importance: NO excess transform
        suite = RV.SUITES[cfg.suite]
        scored = [c for c in cs if c.n_surv > 0 and (RV._in(c, suite["train"]) or RV._in(c, suite["target"]))]

        def a(c):
            W = imp_cells[c.key].W
            return 0.0 if W < fx.w_min else RV.importance(W, cfg.imp_w0, cfg.imp_wref)
        comp["Depth"] = max((a(c) * c.gain for c in scored), default=0.0)
        comp["Close"] = max((a(c) for c in scored if c.closed and
                             (not fx.close_target_only or RV._in(c, suite["target"]))), default=0.0)
    ws = dict(G_train=cfg.w_train, G_target=cfg.w_target, G_gen=cfg.w_gen, Tail=cfg.w_tail,
              Depth=cfg.w_depth, Close=cfg.w_close)
    tot = sum(ws.values())
    return max(0.0, min(1.0, sum(ws[k] * comp[k] for k in ws) / tot))


def score(env, masks, fx: Fix) -> float:
    return RV.FLOOR_VERIFIED + RV.SPAN * inner(env, masks, fx)


FIXES: Dict[str, Fix] = {
    "VR (R1)": Fix("VR (R1)"),
    "VR, GT labels": Fix("VR, GT labels", view="gt"),
    "VR, lower-bound labels": Fix("VR, lower-bound labels", view="gtlb"),
    "F1 excess>2k": Fix("F1 excess>2k", excess=2000.0),
    "F2 tie tail": Fix("F2 tie tail", tail_rule="ties"),
    "F3 a_I from lower bound, Wmin 1e6": Fix("F3 a_I from lower bound, Wmin 1e6", imp_view="gtlb", w_min=1.0e6),
    "F4 GEN Wmin 1e5": Fix("F4 GEN Wmin 1e5", gen_min=1.0e5),
    "VR* = F1+F2+F3+F4": Fix("VR* = F1+F2+F3+F4", excess=2000.0, tail_rule="ties", imp_view="gtlb", w_min=1.0e6,
                             gen_min=1.0e5),
    "F1b excess>20k": Fix("F1b excess>20k", excess=20000.0),
    "VR** = F1b+F3": Fix("VR** = F1b+F3", excess=20000.0, imp_view="gtlb", w_min=1.0e6),
    "VR** on GT labels": Fix("VR** on GT labels", view="gt", excess=20000.0, imp_view="gtlb", w_min=1.0e6),
    "VR* on GT labels": Fix("VR* on GT labels", view="gt", excess=2000.0, tail_rule="ties", imp_view="gtlb",
                            w_min=1.0e6, gen_min=1.0e5),
}
