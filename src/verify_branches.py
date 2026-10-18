"""Integrity check of a restart-branch run (run_branches.py output) before any analysis; torch-free. Exit code 0 only if all checks pass.
Derives the expected job list from the core run + config exactly as run_branches.py does, then checks: coverage and uniqueness, image
readability/size/SHA-256, orphan and temp files, that each record's restart plan (requested/effective jump, ratio, clipped flag, target timestep)
equals the plan recomputed from the scheduler, that noise seeds follow the documented rule, that the source (core) record exists, and
optionally that the core latent files still match the hashes recorded in the core run.
Usage: python src/verify_branches.py --branches <dir> --core-run <dir> --config configs/experiment_v2.json --prompts data/benchmark_v2/prompts_v2.json
       --variants D8 ER30 FULL --M 2 [--split DEV] [--check-core-latents]"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

import branch as B
import jobs as J
import run_branches as R
from verify_outputs import sha


def verify(branches, core, cfg, prompts, variants, M, split=None, check_core_latents=False):
    branches, core = Path(branches), Path(core)
    errs = []
    core_recs = [json.loads(l) for l in (core / "samples.jsonl").read_text().splitlines() if l.strip()]
    if split:
        core_recs = [r for r in core_recs if r.get("split") == split]
    core_by = {r["name"]: r for r in core_recs}
    sch = B.Schedule(cfg["steps"][0])
    jobs = R.enumerate_branch_jobs(core_recs, cfg["restart"]["grid"], variants, M, cfg["seeds_branch_bases"])
    exp = {f"{r['name']}__br_{v}_t{t}_m{m}": (r, t, v, m) for r, t, v, m in jobs}
    files = sorted(branches.glob("samples_shard*of*.jsonl")) or [branches / "samples.jsonl"]
    recs = [json.loads(l) for f in files if f.exists() for l in f.read_text().splitlines() if l.strip()]
    names = Counter(r["name"] for r in recs)
    errs += [f"duplicate record: {n} x{c}" for n, c in names.items() if c > 1]
    missing, extra = sorted(set(exp) - set(names)), sorted(set(names) - set(exp))
    errs += [f"missing branch: {n}" for n in missing[:20]] + ([f"... {len(missing) - 20} more missing"] if len(missing) > 20 else [])
    errs += [f"unexpected branch: {n}" for n in extra[:20]]
    for r in recs:
        if r["name"] not in exp:
            continue
        core_rec, t, v, m = exp[r["name"]]
        plan = R.plan_for(v, sch, t, cfg["restart"])
        for key in ("requested_jump", "effective_jump", "clipped", "ts_target"):
            if r.get(key) != plan[key]:
                errs.append(f"plan mismatch {key}: {r['name']} recorded {r.get(key)} expected {plan[key]}")
        if abs(r.get("ratio", -1) - plan["ratio"]) > 1e-9:
            errs.append(f"plan mismatch ratio: {r['name']}")
        if r.get("noise_seed") != R.noise_seed(core_rec["name"], t, v, m):
            errs.append(f"noise seed rule violated: {r['name']}")
        if r.get("core_name") != core_rec["name"] or r.get("t") != t or r.get("variant") != v or r.get("m") != m:
            errs.append(f"record fields disagree with the job: {r['name']}")
        if r.get("revision") != cfg.get("revision"):
            errs.append(f"revision differs from config: {r['name']}")
        p = branches / "images" / f"{r['name']}.png"
        if not p.exists():
            errs.append(f"image missing: {r['name']}")
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if im.size != (r["size"], r["size"]):
                    errs.append(f"wrong size {im.size}: {r['name']}")
        except Exception as e:
            errs.append(f"corrupt image {r['name']}: {type(e).__name__}")
            continue
        if sha(p) != r.get("image_sha256"):
            errs.append(f"hash mismatch: {r['name']}")
    imgdir = branches / "images"
    on_disk = {p.stem for p in imgdir.glob("*.png")} if imgdir.exists() else set()
    errs += [f"orphan image: {n}" for n in sorted(on_disk - set(names))[:20]]
    errs += [f"leftover temp file: {p.name}" for p in (imgdir.glob("*.tmp") if imgdir.exists() else [])]
    if check_core_latents:
        for name, r in core_by.items():
            lp = core / "latents" / f"{name}.pt"
            if not lp.exists():
                errs.append(f"core latents missing: {name}")
            elif r.get("latents_sha256") and sha(lp) != r["latents_sha256"]:
                errs.append(f"core latents hash mismatch: {name}")
    return errs, dict(records=len(recs), unique=len(names), expected=len(exp))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", required=True)
    ap.add_argument("--core-run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--split", default=None, choices=["DEV", "TEST"])
    ap.add_argument("--check-core-latents", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    errs, stats = verify(a.branches, a.core_run, cfg, J.load_prompts(a.prompts), a.variants, a.M, a.split, a.check_core_latents)
    print(json.dumps(stats), f"errors={len(errs)}")
    for e in errs[:50]:
        print(" ", e)
    sys.exit(1 if errs else 0)
