"""Integrity check of a generation run before any analysis. Exit code 0 only if every check passes.
Checks: expected job list (from config + prompts + shards) fully covered, no duplicate names, every recorded image exists,
decodes as PNG, has the recorded SHA-256 and size, no orphan images, latents present/valid when requested.
Usage: python src/verify_outputs.py --run outputs/x --config configs/experiment.json [--prompts FILE] [--split DEV] [--variants CAN] [--latents]"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

import jobs as J


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify(run, cfg_all, prompts, shards=None, need_latents=False, seeds=None):
    run = Path(run)
    errs = []
    files = sorted(run.glob("samples_shard*of*.jsonl")) or [run / "samples.jsonl"]
    recs = [json.loads(l) for f in files if f.exists() for l in f.read_text().splitlines() if l.strip()]
    names = Counter(r["name"] for r in recs)
    errs += [f"duplicate record: {n} x{c}" for n, c in names.items() if c > 1]
    expected = [J.sample_name(p["id"], c, s, sd) for p, c, s, sd in J.enumerate_jobs(prompts, cfg_all, seeds)]
    want = set(expected)   # shards partition the full list; the union of all shard files must cover it
    missing, extra = sorted(want - set(names)), sorted(set(names) - want)
    errs += [f"missing sample: {n}" for n in missing[:20]] + ([f"... {len(missing) - 20} more missing"] if len(missing) > 20 else [])
    errs += [f"unexpected sample: {n}" for n in extra[:20]]
    for r in recs:
        p = run / "images" / f"{r['name']}.png"
        if not p.exists():
            errs.append(f"image missing: {r['name']}")
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if im.size != (r["size"], r["size"]):
                    errs.append(f"wrong size {im.size}: {r['name']}")
        except Exception as e:  # corrupt / truncated PNG
            errs.append(f"corrupt image {r['name']}: {type(e).__name__}")
            continue
        if "image_sha256" in r and sha(p) != r["image_sha256"]:
            errs.append(f"hash mismatch: {r['name']}")
        if need_latents:
            lp = run / "latents" / f"{r['name']}.pt"
            if not lp.exists():
                errs.append(f"latents missing: {r['name']}")
            elif "latents_sha256" in r and sha(lp) != r["latents_sha256"]:
                errs.append(f"latents hash mismatch: {r['name']}")
    on_disk = {p.stem for p in (run / "images").glob("*.png")} if (run / "images").exists() else set()
    errs += [f"orphan image: {n}" for n in sorted(on_disk - set(names))[:20]]
    tmp = list((run / "images").glob("*.tmp")) if (run / "images").exists() else []
    errs += [f"leftover temp file: {p.name}" for p in tmp]
    return errs, dict(records=len(recs), unique=len(names), expected=len(want))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent.parent / "data" / "prompts.json"))
    ap.add_argument("--limit-prompts", default=None)
    ap.add_argument("--split", default=None, choices=["DEV", "TEST"])
    ap.add_argument("--variants", default=None, help="same filters as generate.py")
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--latents", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    prompts = J.load_prompts(a.prompts)
    if a.limit_prompts:
        prompts = [p for p in prompts if p["id"] in set(a.limit_prompts.split(","))]
    if a.split:
        prompts = [p for p in prompts if p.get("split") == a.split]
    if a.variants:
        prompts = [p for p in prompts if p.get("variant") in set(a.variants.split(","))]
    errs, stats = verify(a.run, cfg, prompts, a.shards, a.latents, cfg["seeds"][: a.seeds] if a.seeds else None)
    print(json.dumps(stats), f"errors={len(errs)}")
    for e in errs[:50]:
        print(" ", e)
    sys.exit(1 if errs else 0)
