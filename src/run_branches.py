"""Restart-branch generation for the restart-persistence experiments (E23 / E23b). Needs a core run made with --save-latents-at.
For every base sample (CAN prompt, branch-base seeds), grid step t, variant and branch index m it re-noises the saved latent
(branch.restart_plan / equal_ratio_plan) and re-denoises with the pipeline-equivalent loop (branch.denoise_from).
Variants: D<k> fixed step jump (e.g. D8 = design value, D4/D12 sensitivity), ER<pct> equal noise ratio (e.g. ER30), FULL = fresh N(0, I)
start at step 0 (full-resample control, D-02). Every record logs requested/effective jump, ratio, clipped flag and timesteps.
Usage: python src/run_branches.py --core-run outputs/dev_core --config configs/experiment_v2.json --prompts data/benchmark_v2/prompts_v2.json \\
         --out outputs/dev_branches --variants D8 ER30 FULL --M 2 [--shard 0/2] [--split DEV] [--limit N]"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import branch as B
import jobs as J


def noise_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def plan_for(variant, sch, t, cfg_restart):
    if variant == "FULL":
        return dict(t=t, requested_jump=None, target_idx=0, effective_jump=t, clipped=False, ts_t=int(sch.ts[t]) if t < sch.n else None,
                    ts_target=int(sch.ts[0]), abar_t=sch.abar_level(t), abar_target=sch.abar_level(0), ratio=0.0)
    m = re.fullmatch(r"D(\d+)", variant)
    if m:
        return B.restart_plan(sch, t, int(m.group(1)))
    m = re.fullmatch(r"ER(\d+)", variant)
    if m:
        return B.equal_ratio_plan(sch, t, int(m.group(1)) / 100)
    raise ValueError(f"unknown variant {variant}")


def enumerate_branch_jobs(core_records, grid, variants, M, base_seeds, can_only=True):
    out = []
    for r in core_records:
        if r["seed"] not in base_seeds or (can_only and r.get("variant", "CAN") != "CAN"):
            continue
        for t in grid:
            for v in variants:
                for m in range(M):
                    out.append((r, t, v, m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["D8"])
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--split", default=None, choices=["DEV", "TEST"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    import torch
    import generate as G
    cfg = json.loads(Path(a.config).read_text())
    prompts = {p["id"]: p for p in J.load_prompts(a.prompts)}
    core = Path(a.core_run)
    recs = [json.loads(l) for l in (core / "samples.jsonl").read_text().splitlines() if l.strip()]
    if a.split:
        recs = [r for r in recs if r.get("split") == a.split]
    J.guard_test_access([prompts[r["prompt_id"]] for r in recs], cfg)
    steps, cfgscale = cfg["steps"][0], cfg["cfg_scales"][0]
    sch = B.Schedule(steps)
    grid = cfg["restart"]["grid"]
    jobs = enumerate_branch_jobs(recs, grid, a.variants, a.M, cfg["seeds_branch_bases"])
    shard = J.parse_shard(a.shard) if a.shard else None
    jobs = J.shard_jobs(jobs, shard)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta_path = out / (f"samples_shard{shard[0]}of{shard[1]}.jsonl" if shard else "samples.jsonl")
    done = G.done_names(meta_path)
    name_of = lambda r, t, v, m: f"{r['name']}__br_{v}_t{t}_m{m}"
    jobs = [j for j in jobs if name_of(*j) not in done or not (out / "images" / (name_of(*j) + ".png")).exists()]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} branches to generate ({len(done)} already done)", flush=True)
    if not jobs:
        return
    revision = None if cfg.get("revision") == "UNPINNED-DRAFT" else cfg.get("revision")
    pipe = G.load_pipeline(cfg["model_id"], revision)
    cache, embeds = {}, {}
    with open(meta_path, "a") as meta:
        for i, (r, t, v, m) in enumerate(jobs, 1):
            name = name_of(r, t, v, m)
            t0 = time.time()
            if r["name"] not in cache:
                cache.clear()
                cache[r["name"]] = torch.load(core / "latents" / f"{r['name']}.pt")
            lat = cache[r["name"]]["latents_fp16"][t].to("cuda")
            if r["prompt"] not in embeds:
                embeds[r["prompt"]] = B.encode_prompt_pair(pipe, r["prompt"], "cuda")
            plan = plan_for(v, sch, t, cfg["restart"])
            ns = noise_seed(r["name"], t, v, m)
            if v == "FULL":
                x = B.branch_noise(lat.shape, ns, lat.dtype, lat.device)
                image = B.denoise_from(pipe, embeds[r["prompt"]], x, 0, steps, cfgscale)
            else:
                image = B.branch_generate(pipe, embeds[r["prompt"]], lat, t, plan, ns, steps, cfgscale)
            path = out / "images" / f"{name}.png"
            G.atomic_save_image(image, path)
            rec = {"name": name, "core_name": r["name"], "prompt_id": r["prompt_id"], "category": r["category"], "prompt": r["prompt"],
                   "scene_id": r.get("scene_id"), "split": r.get("split"), "seed": r["seed"], "cfg": cfgscale, "steps": steps, "size": cfg["image_size"],
                   "t": t, "m": m, "variant": v, "noise_seed": ns, "requested_jump": plan["requested_jump"], "effective_jump": plan["effective_jump"],
                   "ratio": plan["ratio"], "clipped": plan["clipped"], "ts_target": plan["ts_target"], "scheduler": "DDIM(eta=0)",
                   "model_id": cfg["model_id"], "revision": cfg.get("revision"), "image_sha256": G.file_sha256(path), "run_id": a.run_id,
                   "seconds": round(time.time() - t0, 2), "torch": torch.__version__}
            meta.write(json.dumps(rec) + "\n")
            meta.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {name}", flush=True)


if __name__ == "__main__":
    main()
