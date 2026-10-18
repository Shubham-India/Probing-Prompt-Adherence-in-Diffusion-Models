"""Pre-registered duplicate handling for pooled estimates (variant review H3, OWNER_DECISIONS Q2 = A, DL-33). Benchmark and manifests unchanged.
Identical prompt text + CFG + steps + seed + the same noise draw (r3_run.init_latents) gives a byte-identical image, so such rows are ONE generation.
Rules, applied before every pooled DEV or TEST estimate:
  R1 generation de-duplication: core rows of one split with the same generation key (normalised prompt text, CFG, steps, seed) count once. The row kept is the
     highest priority: canonical (CAN) before any other variant, main benchmark before the PS family, then the smaller prompt id. Branch, switch and
     CAG rows built on a dropped base (base_job) are dropped with it.
  R2 duplicate canonical scenes: when two scenes of one split have the same normalised CANONICAL prompt text (main sg_0091 = PS psD05_P), the
     lower-priority scene (PS family, then larger scene id) is dropped from pooled estimates entirely (all its rows), so the scene contributes once
     to DEV-derived constants and to every scene bootstrap.
  R3 DEV/TEST overlap (variant-level S2 only): TEST rows whose normalised prompt text also occurs as a DEV prompt are excluded from TEST
     variant-level estimates; they stay in the data and in the catalogue.
Every drop is listed in the returned log."""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILES = ("data/benchmark_v2/prompts_v2.json", "data/benchmark_v2/prompts_ps.json")
_INDEX = None


def norm(text):
    return " ".join(re.findall(r"[a-z]+", str(text).lower()))


def prompt_index(root=ROOT):
    global _INDEX
    if _INDEX is None or root != ROOT:
        idx = {p["id"]: p for f in PROMPT_FILES for p in json.loads((Path(root) / f).read_text())}
        if root != ROOT:
            return idx
        _INDEX = idx
    return _INDEX


def is_ps(pid):
    return str(pid).startswith("ps")


def priority(row, prompts):
    p = prompts.get(row.get("prompt_id"), {})
    return (0 if (row.get("variant") or p.get("variant")) == "CAN" else 1, 1 if is_ps(row.get("prompt_id")) else 0, str(row.get("prompt_id")))


def generation_key(row, prompts):
    p = prompts.get(row.get("prompt_id"))
    if p is not None:
        text = norm(p["prompt"])
    elif row.get("prompt"):
        text = norm(row["prompt"])
    else:                                    # prompt unknown: never merged with another row
        text = ("unknown", row.get("prompt_id"), row.get("scene_id"), row.get("variant"), row.get("job_id"))
    return (row.get("split"), text, float(row["cfg"]), int(row["steps"]), int(row["seed"]))   # never merged across splits (R3 handles DEV/TEST)


def duplicate_scenes(prompts):
    """{dropped scene_id: kept scene_id} for scenes of one split sharing a normalised canonical prompt (R2)."""
    by = defaultdict(list)
    for p in prompts.values():
        if p["variant"] == "CAN":
            by[(p["split"], norm(p["prompt"]))].append(p)
    out = {}
    for group in by.values():
        if len(group) > 1:
            group.sort(key=lambda p: (1 if is_ps(p["id"]) else 0, p["scene_id"]))
            for p in group[1:]:
                out[p["scene_id"]] = group[0]["scene_id"]
    return out


def dedup(rows, prompts=None):
    """Apply R2 then R1 to a list of analysis rows (any kinds). Returns (kept rows, log)."""
    prompts = prompt_index() if prompts is None else prompts
    log = dict(dropped_scenes={}, dropped_generations=[], dropped_dependents=[])
    dup_sc = duplicate_scenes(prompts)
    present = {r.get("scene_id") for r in rows}
    drop_sc = {s: k for s, k in dup_sc.items() if s in present}
    rows = [r for r in rows if r.get("scene_id") not in drop_sc]
    log["dropped_scenes"] = drop_sc
    core = [r for r in rows if r.get("kind", "core") == "core"]
    best = {}
    for r in core:
        k = generation_key(r, prompts)
        if k not in best or priority(r, prompts) < priority(best[k], prompts):
            best[k] = r
    keep_ids = {id(r) for r in best.values()}
    dropped_jobs = {r["job_id"] for r in core if id(r) not in keep_ids}
    log["dropped_generations"] = sorted(dropped_jobs)
    out = []
    for r in rows:
        if r.get("kind", "core") == "core":
            if id(r) in keep_ids:
                out.append(r)
        elif r.get("base_job") in dropped_jobs:
            log["dropped_dependents"].append(r["job_id"])
        else:
            out.append(r)
    return out, log


def dev_texts(prompts=None):
    prompts = prompt_index() if prompts is None else prompts
    return {norm(p["prompt"]) for p in prompts.values() if p["split"] == "DEV"}


def exclude_dev_overlap(test_rows, prompts=None):
    """R3: drop TEST rows whose prompt text also occurs in DEV. Returns (kept, excluded job ids)."""
    prompts = prompt_index() if prompts is None else prompts
    dev = dev_texts(prompts)
    kept, excl = [], []
    for r in test_rows:
        p = prompts.get(r.get("prompt_id"))
        if r.get("split") == "TEST" and p is not None and norm(p["prompt"]) in dev:
            excl.append(r["job_id"])
        else:
            kept.append(r)
    return kept, excl
