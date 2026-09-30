"""E25 table universe: every cached table the reward variants are evaluated on, with its
roles and shape family.  Read-only over cache/ (never writes a case table).

A table is keyed by its FILE stem (e.g. "m10_n20_s3_t3_w103_pure"), not by Instance.tag,
because the pure and tan2022 tables of one instance share a tag.

roles (a table may carry several):
  train    the current TRAIN cells (suite.TRAIN_CELLS, pure, w = z+1)
  band     suite.BAND_CELLS (pure, w = z+1)
  wide     exactly-known WIDE cells in pure mode (V3 adds them to TRAIN); A1's *_pure_gt tables
           are picked up automatically when they exist
  target   suite.DEFAULT_TARGETS that have cases (tan2022, censored labels)
  target_x the "later" targets of design §5.1 (tan2022, censored)
  gen      suite.GEN_CELLS with content, plus (8,9;2,2) w=27 (the E12 alternative GEN cell; tiny)
  gen_default  the GEN cells of today's suite that have survivors ((9,9;4,4) w=62)
family (V3): by aspect ratio r = n/m of a (3,3) cell: square r < 1.2, wide 1.2 <= r < 1.8,
  vwide r >= 1.8; any (s,t) != (3,3) cell is family "gen".
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
CACHE = os.path.join(ROOT, "cache")

TRAIN = ["m9_n9_s3_t3_w50_pure", "m9_n10_s3_t3_w55_pure", "m10_n10_s3_t3_w61_pure", "m10_n11_s3_t3_w65_pure",
         "m11_n11_s3_t3_w70_pure", "m11_n12_s3_t3_w75_pure", "m12_n12_s3_t3_w81_pure"]
BAND = ["m12_n13_s3_t3_w87_pure", "m13_n13_s3_t3_w93_pure", "m10_n14_s3_t3_w78_pure"]
WIDE = ["m9_n12_s3_t3_w64_pure", "m10_n14_s3_t3_w78_pure", "m10_n20_s3_t3_w103_pure", "m11_n21_s3_t3_w117_pure"]
TARGET = ["m9_n23_s3_t3_w104", "m12_n18_s3_t3_w109"]  # DEFAULT_TARGETS with cases ((13,17),(13,18),(15,17) are empty)
TARGET_X = ["m10_n23_s3_t3_w113", "m11_n23_s3_t3_w124", "m13_n19_s3_t3_w123", "m16_n17_s3_t3_w134",
            "m10_n22_s3_t3_w111"]
GEN = ["m9_n9_s4_t4_w62_pure", "m8_n9_s2_t2_w27_pure"]
GEN_DEFAULT = ["m9_n9_s4_t4_w62_pure"]  # the only default GEN cell with survivors


def parse_key(key: str) -> Dict[str, int]:
    parts = key.split("_")
    d = {p[0]: int(p[1:]) for p in parts[:5]}
    return {"m": d["m"], "n": d["n"], "s": d["s"], "t": d["t"], "w": d["w"]}


def family(m: int, n: int, s: int, t: int) -> str:
    if (s, t) != (3, 3):
        return "gen"
    r = max(m, n) / min(m, n)
    if r < 1.2:
        return "square"
    if r < 1.8:
        return "wide"
    return "vwide"


def gt_tables() -> List[str]:
    """A1's exactly-labelled wide pure tables (cache/case_table_*_pure_gt.json), when they exist."""
    out = []
    for p in sorted(glob.glob(os.path.join(CACHE, "case_table_*_pure_gt.json"))):
        out.append(os.path.basename(p)[len("case_table_"):-len(".json")])
    return out


def universe(include_gt: bool = True) -> Dict[str, dict]:
    """key -> {"path", "roles": [...], "family", "m","n","s","t","w", "trust"}."""
    roles: Dict[str, List[str]] = {}
    for name, keys in (("train", TRAIN), ("band", BAND), ("wide", WIDE), ("target", TARGET),
                       ("target_x", TARGET_X), ("gen", GEN), ("gen_default", GEN_DEFAULT)):
        for k in keys:
            roles.setdefault(k, []).append(name)
    if include_gt:
        for k in gt_tables():
            roles.setdefault(k, []).append("wide_gt")
    out = {}
    for k, rs in roles.items():
        p = os.path.join(CACHE, f"case_table_{k}.json")
        if not os.path.exists(p):
            continue
        inst = parse_key(k)
        out[k] = dict(path=p, roles=rs, family=family(inst["m"], inst["n"], inst["s"], inst["t"]),
                      trust="pure" if "_pure" in k else "tan2022", **inst)
    return out


def load(key: str, path: Optional[str] = None):
    from zar_ub.casetable import table_from_json
    p = path or os.path.join(CACHE, f"case_table_{key}.json")
    with open(p) as f:
        return table_from_json(json.load(f), path=p)
