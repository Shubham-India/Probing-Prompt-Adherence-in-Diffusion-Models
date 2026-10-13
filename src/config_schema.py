"""Strict schema check for experiment configs. Unknown or missing keys are errors; the pilot config stays valid."""
from __future__ import annotations

import json
import re
from pathlib import Path

UNPINNED = "UNPINNED-DRAFT"   # allowed only in a draft config; check_frozen() rejects it

PILOT_KEYS = {"model_id": str, "revision": (str, type(None)), "image_size": int, "cfg_scales": list, "steps": list, "seeds": list}

V2_EXTRA = {
    "schema_version": int,
    "scheduler": dict,        # name, eta
    "restart": dict,          # grid, D, equal_ratio_target, D_sensitivity, full_resample_control
    "seeds_core": list, "seeds_branch_bases": list,
    "num_shards": int,
    "split_manifest": (str, type(None)),
    "freeze": dict,           # addendum_hash (None until the freeze), test_sealed
}
RESTART_KEYS = {"grid": list, "D": int, "equal_ratio_target": float, "D_sensitivity": list, "full_resample_control": bool}


def _check(d, spec, where):
    unknown = set(d) - set(spec)
    missing = set(spec) - set(d)
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    for k, typ in spec.items():
        if not isinstance(d[k], typ) or (typ is int and isinstance(d[k], bool)):
            raise ValueError(f"{where}.{k}: expected {typ}, got {type(d[k]).__name__}")


def validate(cfg: dict) -> str:
    """Returns 'pilot' or 'v2'. Raises ValueError otherwise."""
    if "schema_version" not in cfg:
        _check(cfg, PILOT_KEYS, "pilot config")
        return "pilot"
    _check(cfg, {**PILOT_KEYS, **V2_EXTRA}, "v2 config")
    _check(cfg["restart"], RESTART_KEYS, "restart")
    n = max(cfg["steps"])
    if cfg["restart"]["D"] < 0 or any(not 0 < g <= n for g in cfg["restart"]["grid"]):
        raise ValueError("restart grid/D outside [0, steps]")
    if not set(cfg["seeds_branch_bases"]) <= set(cfg["seeds_core"]):
        raise ValueError("branch base seeds must be a subset of core seeds (C-6)")
    if cfg["revision"] != UNPINNED and not re.fullmatch(r"[0-9a-f]{40}", str(cfg["revision"])):
        raise ValueError("v2 revision must be a 40-hex commit or the explicit draft marker (D-11)")
    return "v2"


def check_frozen(cfg: dict) -> None:
    """Gate used before any TEST-path read (D-06): pinned revision, split manifest and Addendum hash present."""
    if validate(cfg) != "v2":
        raise ValueError("only a v2 config can be frozen")
    if cfg["revision"] == UNPINNED:
        raise ValueError("model revision is not pinned")
    f = cfg["freeze"]
    if not cfg["split_manifest"] or not f.get("addendum_hash") or not re.fullmatch(r"[0-9a-f]{64}", f["addendum_hash"]):
        raise ValueError("freeze incomplete: split manifest and SHA-256 Addendum hash required")


def load(path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(cfg)
    return cfg
