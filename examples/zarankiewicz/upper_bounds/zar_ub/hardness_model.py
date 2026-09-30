"""The E24 hardness model: a fitted log-linear difficulty estimator for censored cases.

Selected by experiments/E24_difficulty/EVALUATION.md (owner D5-evaluate) from the features of
zar_ub.hardness_progress (CDCL progress + static/LP slack), zar_ub.hardness_lookahead (lookahead /
failed literals) and zar_ub.hardness_sampling (Chivilikhin d-hardness sampling).

    log d_hat = intercept + sum_k coef_k * (T_k(x_k) - mu_k) / sd_k        (natural log)

where x_k are named features ("pr:..." from hardness_progress, "la:..." from hardness_lookahead,
"sa:..." from hardness_sampling, "base:..." from the profile) and T_k is a fixed monotone
transform ("id", "log1p", "slog1p").  The model is meant for cases NOT decided within the
table pipeline's 20,000-conflict probe (the HARD regime, d > 20k): it was fitted on exact hard
labels only, and predict() floors the result at that cap.

API
  fit(feature_dicts, log_d, features, transforms, meta=None) -> HardnessModel
  HardnessModel.predict_features(feats, mean=False) -> float  (d_hat; None if a feature is missing)
  HardnessModel.save(path) / load(path)          (JSON; default experiments/E24_difficulty/hardness_model.json)
  compute_features(inst, rows, cols, model)      -> (feats, cost dict, exact d or None)
  predict(inst, rows, cols, model=None, mean=True, floor=20000, ceiling=None) -> dict
      computes exactly the features the model needs (running the 20k pysat probe itself), returns
      {"d_hat", "log_d", "exact", "features", "cost_conflicts", "cost_propagations", "cost_seconds"}.
      When the 20k probe decides the case, d_hat is the exact conflict count (exact=True).
  predict_from_probe(inst, rows, cols, conflicts, decisions, restarts, propagations=0, ...)
      zero-extra-solver path for a case that an existing 20k pysat cadical195 probe left open
      (needs the probe's accum_stats decisions and restarts; see the TODO in EVALUATION.md).

mean=True multiplies by the Duan smearing factor (use for SUMS of work, e.g. the reward's gain_I);
mean=False is the conditional median (use for ranks and log errors).
Deterministic: every solver run is a fresh deterministic solver; no RNG in the model.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .known import Instance

_HERE = os.path.dirname(os.path.abspath(__file__))
_UB = os.path.dirname(_HERE)
MODEL_PATH = os.path.join(_UB, "experiments", "E24_difficulty", "hardness_model.json")
CAP = 20_000


def _transform(x: float, tr: str) -> float:
    if tr == "log1p":
        return math.log1p(max(x, 0.0))
    if tr == "slog1p":
        return math.copysign(math.log1p(abs(x)), x)
    return x


@dataclass
class HardnessModel:
    features: List[str]
    transforms: Dict[str, str]
    mu: Dict[str, float]
    sd: Dict[str, float]
    coef: Dict[str, float]
    intercept: float
    smear: float = 1.0
    floor: float = float(CAP)
    cap: int = CAP
    meta: Dict[str, object] = field(default_factory=dict)

    # -- prediction from a feature dict ---------------------------------------------------
    def log_d(self, feats: Dict[str, float]) -> Optional[float]:
        z = float(self.intercept)
        for name in self.features:
            v = feats.get(name)
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                return None
            x = _transform(float(v), self.transforms.get(name, "id"))
            z += float(self.coef[name]) * (x - float(self.mu[name])) / float(self.sd[name] or 1.0)
        return z

    def predict_features(self, feats: Dict[str, float], mean: bool = False,
                         floor: Optional[float] = None, ceiling: Optional[float] = None) -> Optional[float]:
        z = self.log_d(feats)
        if z is None:
            return None
        v = math.exp(z) * (self.smear if mean else 1.0)
        v = max(v, self.floor if floor is None else floor)
        if ceiling is not None:
            v = min(v, ceiling)
        return float(v)

    # -- which estimator modules the features need ----------------------------------------
    def needs(self) -> Dict[str, object]:
        la = [f[3:] for f in self.features if f.startswith("la:")]
        return {
            "progress": any(f.startswith("pr:") for f in self.features),
            "binary": any(f.startswith("pr:bin") or f.startswith("pr:prog") for f in self.features),
            "lookahead": bool(la),
            # Knuth probes (kn_*, knfl_*) and pair lookahead (la2_*) need the full lookahead run
            "lookahead_full": any(x.startswith("kn") or x.startswith("la2_") for x in la),
            "lookahead_clause": any(x.startswith("la1_newbin") or x.startswith("la1_wred") or x == "bin_root" for x in la),
            "sampling": any(f.startswith("sa:") for f in self.features),
        }

    # -- persistence ------------------------------------------------------------------------
    def to_json(self) -> dict:
        return asdict(self)

    def save(self, path: str = MODEL_PATH) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.to_json(), fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return path


def load(path: str = MODEL_PATH) -> Optional[HardnessModel]:
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return HardnessModel(**d)


def save(model: HardnessModel, path: str = MODEL_PATH) -> str:
    return model.save(path)


# ---------------------------------------------------------------------------------------------
# fitting (ordinary least squares on standardised transformed features)
# ---------------------------------------------------------------------------------------------
def fit(feature_dicts: Sequence[Dict[str, float]], log_d: Sequence[float], features: Sequence[str],
        transforms: Optional[Dict[str, str]] = None, meta: Optional[dict] = None,
        floor: float = float(CAP)) -> HardnessModel:
    """OLS of log d on the standardised transformed features.  Rows with a missing feature are
    dropped (the count is recorded in meta)."""
    import numpy as np

    features = list(features)
    transforms = dict(transforms or {})
    X, y = [], []
    dropped = 0
    for f, ld in zip(feature_dicts, log_d):
        row = []
        for name in features:
            v = f.get(name)
            if v is None or not math.isfinite(float(v)):
                break
            row.append(_transform(float(v), transforms.get(name, "id")))
        else:
            X.append(row)
            y.append(float(ld))
            continue
        dropped += 1
    X = np.array(X, float).reshape(len(y), len(features))
    y = np.array(y, float)
    mu = X.mean(0) if len(features) else np.zeros(0)
    sd = X.std(0) if len(features) else np.zeros(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    A = np.hstack([np.ones((len(y), 1)), Z])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ beta
    m = dict(meta or {})
    m.update({"n_fit": int(len(y)), "n_dropped": int(dropped), "rmse_in_sample": float(np.sqrt(np.mean(res ** 2)))})
    return HardnessModel(
        features=features,
        transforms={k: transforms.get(k, "id") for k in features},
        mu={k: float(v) for k, v in zip(features, mu)},
        sd={k: float(v) for k, v in zip(features, sd)},
        coef={k: float(v) for k, v in zip(features, beta[1:])},
        intercept=float(beta[0]),
        smear=float(np.mean(np.exp(res))),
        floor=float(floor),
        meta=m,
    )


# ---------------------------------------------------------------------------------------------
# feature computation for one case
# ---------------------------------------------------------------------------------------------
def base_features(inst: Instance, rows: Sequence[int], cols: Sequence[int]) -> Dict[str, float]:
    return {
        "base:log2_volume": float(sum(math.log2(math.comb(inst.n, r)) for r in rows)),
        "base:aspect": inst.n / inst.m,
        "base:mn": float(inst.m * inst.n),
    }


def compute_features(inst: Instance, rows: Sequence[int], cols: Sequence[int],
                     model: HardnessModel) -> Tuple[Dict[str, float], Dict[str, float], Optional[float]]:
    """Compute the features `model` needs.  Returns (feats, cost, exact_d): exact_d is the exact
    conflict count when the 20k pysat probe decided the case (then nothing else is run)."""
    need = model.needs()
    feats: Dict[str, float] = base_features(inst, rows, cols)
    cost = {"conflicts": 0, "propagations": 0, "seconds": 0.0}
    t0 = time.time()
    exact = None
    if need["progress"]:
        from . import hardness_progress as hp
        o = hp.estimate(inst, rows, cols, tier="20k", binary=bool(need["binary"]))
        feats.update({"pr:" + k: float(v) for k, v in o["features"].items()})
        cost["conflicts"] += int(o["cost_conflicts"])
        cost["propagations"] += int(o["cost_propagations"])
        if o["features"].get("ps20k_solved", 0.0) > 0:
            exact = float(max(1.0, o["features"]["ps20k_conflicts"]))
            cost["seconds"] = time.time() - t0
            return feats, cost, exact
    if need["lookahead"]:
        from . import hardness_lookahead as hl
        kw = dict(calibrated=False)
        if not need["lookahead_full"]:
            kw.update(n_pairs=0, n_probes=0, clause_measures=bool(need["lookahead_clause"]))
        o = hl.estimate(inst, rows, cols, **kw)
        feats.update({"la:" + k: float(v) for k, v in o["features"].items() if isinstance(v, (int, float))})
        cost["conflicts"] += int(o["cost_conflicts"])
        cost["propagations"] += int(o["cost_propagations"])
    if need["sampling"]:
        from . import hardness_sampling as hs
        o = hs.estimate(inst, rows, cols)
        feats.update({"sa:" + k: float(v) for k, v in o["features"].items()})
        raw = o.get("d_hat_raw")
        feats["sa:log_mu"] = math.log(max(float(raw), 1.0)) if raw else float("nan")
        cost["conflicts"] += int(o["cost_conflicts"])
        cost["propagations"] += int(o["cost_propagations"])
    cost["seconds"] = time.time() - t0
    return feats, cost, exact


def predict(inst: Instance, rows: Sequence[int], cols: Sequence[int], model: Optional[HardnessModel] = None,
            mean: bool = True, floor: Optional[float] = None, ceiling: Optional[float] = None) -> dict:
    """d_hat for one case, computing the needed features itself (see module doc)."""
    model = model or load()
    if model is None:
        raise FileNotFoundError(MODEL_PATH)
    feats, cost, exact = compute_features(inst, rows, cols, model)
    if exact is not None:
        d_hat, ld = exact, math.log(exact)
    else:
        d_hat = model.predict_features(feats, mean=mean, floor=floor, ceiling=ceiling)
        ld = model.log_d(feats)
    return {"d_hat": d_hat, "log_d": ld, "exact": exact is not None, "features": feats,
            "cost_conflicts": cost["conflicts"], "cost_propagations": cost["propagations"],
            "cost_seconds": cost["seconds"]}


def estimate(inst: Instance, rows: Sequence[int], cols: Sequence[int], **kw) -> dict:
    """E24 estimator contract wrapper around predict()."""
    o = predict(inst, rows, cols, **kw)
    return {"features": o["features"], "d_hat": o["d_hat"], "cost_conflicts": o["cost_conflicts"],
            "cost_propagations": o["cost_propagations"], "cost_seconds": o["cost_seconds"]}


def predict_from_probe(inst: Instance, rows: Sequence[int], cols: Sequence[int], conflicts: int, decisions: int,
                       restarts: int, propagations: int = 0, model: Optional[HardnessModel] = None,
                       mean: bool = True, floor: Optional[float] = None, ceiling: Optional[float] = None,
                       extra: Optional[Dict[str, float]] = None) -> Optional[float]:
    """Zero-extra-conflict d_hat for a case left open by an existing 20k pysat cadical195 probe.
    Static/LP features are recomputed (no solver); lookahead features (BCP only) are computed if
    the model needs them; returns None if the model needs sampling or binary features that are
    not supplied in `extra`."""
    model = model or load()
    if model is None:
        return None
    from . import hardness_progress as hp
    feats: Dict[str, float] = base_features(inst, rows, cols)
    feats.update({"pr:" + k: float(v) for k, v in hp.static_features(inst, rows, cols, lp=True).items()})
    r = {"status": "unknown", "conflicts": int(conflicts), "decisions": int(decisions),
         "propagations": int(propagations), "restarts": int(restarts), "wall": 0.0}
    feats.update({"pr:" + k: float(v) for k, v in hp._pysat_features(hp._capname(CAP), r).items()})
    need = model.needs()
    if need["lookahead"]:
        from . import hardness_lookahead as hl
        kw = dict(calibrated=False)
        if not need["lookahead_full"]:
            kw.update(n_pairs=0, n_probes=0, clause_measures=bool(need["lookahead_clause"]))
        o = hl.estimate(inst, rows, cols, **kw)
        feats.update({"la:" + k: float(v) for k, v in o["features"].items() if isinstance(v, (int, float))})
    if extra:
        feats.update(extra)
    return model.predict_features(feats, mean=mean, floor=floor, ceiling=ceiling)
