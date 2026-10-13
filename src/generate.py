"""Generates the Prompt x CFG x steps x seed grid with Stable Diffusion 1.5 and saves images + metadata.

Usage: python src/generate.py --config configs/experiment.json --out outputs/main [--limit-prompts ID,ID] [--seeds 2]
Re-running skips every sample already listed in <out>/samples.jsonl whose image file exists.
Optional (M4; defaults reproduce the pilot behaviour): --prompts FILE, --shard i/n (writes samples_shard<i>of<n>.jsonl),
--save-latents-at 6,12,.. (fp16 latents after t completed steps, one .pt per image), --limit N (first N jobs).
Images are written atomically (tmp file + rename) and their SHA-256 is recorded; TEST prompts need a frozen v2 config (D-06).
"""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

ROOT = Path(__file__).resolve().parent.parent
import jobs as J  # noqa: E402  (torch-free helpers)


def load_pipeline(model_id, revision=None):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False, cache_dir=ROOT / "models")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def initial_latents(seed, size, device="cuda"):
    """Noise is drawn on the CPU so a seed means the same noise on any machine."""
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((1, 4, size // 8, size // 8), generator=g).to(device, torch.float16)


def sample_name(prompt_id, cfg, steps, seed):
    return J.sample_name(prompt_id, cfg, steps, seed)


def generate_image(pipe, prompt, cfg, steps, seed, size, **extra):
    return pipe(prompt, guidance_scale=cfg, num_inference_steps=steps, height=size, width=size,
                latents=initial_latents(seed, size), negative_prompt="", **extra).images[0]


def done_names(meta_path):
    if not meta_path.exists():
        return set()
    return {json.loads(line)["name"] for line in meta_path.read_text().splitlines() if line.strip()}


def generate_with_latents(pipe, prompt, cfg, steps, seed, size, save_at):
    """Same call as generate_image plus a callback that keeps the fp16 latents after t completed steps (t in save_at).
    callback_on_step_end fires after step i with the latents of step i, i.e. after i + 1 completed steps."""
    kept = {}

    def cb(_pipe, i, _t, kw):
        if (i + 1) in save_at:
            kept[i + 1] = kw["latents"].detach().to("cpu", torch.float16).clone()
        return kw
    image = pipe(prompt, guidance_scale=cfg, num_inference_steps=steps, height=size, width=size,
                 latents=initial_latents(seed, size), negative_prompt="", callback_on_step_end=cb,
                 callback_on_step_end_tensor_inputs=["latents"]).images[0]
    return image, kept


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save_image(image, path):
    tmp = path.with_suffix(".png.tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=str(ROOT / "data" / "prompts.json"))
    ap.add_argument("--limit-prompts", default=None, help="comma-separated prompt ids")
    ap.add_argument("--split", default=None, choices=["DEV", "TEST"], help="keep only prompts of this split (V2 prompt files)")
    ap.add_argument("--seeds", type=int, default=None, help="use only the first N seeds")
    ap.add_argument("--shard", default=None, help="i/n: this process handles jobs k with k %% n == i")
    ap.add_argument("--save-latents-at", default=None, help="comma-separated completed-step counts, e.g. 6,12,18,24,30,36")
    ap.add_argument("--limit", type=int, default=None, help="only the first N jobs (after sharding)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg_all = json.loads(Path(args.config).read_text())
    prompts = J.load_prompts(args.prompts)
    if args.limit_prompts:
        keep = set(args.limit_prompts.split(","))
        prompts = [p for p in prompts if p["id"] in keep]
    if args.split:
        prompts = [p for p in prompts if p.get("split") == args.split]
    J.guard_test_access(prompts, cfg_all)
    seeds = cfg_all["seeds"][: args.seeds] if args.seeds else cfg_all["seeds"]
    shard = J.parse_shard(args.shard) if args.shard else None
    save_at = {int(x) for x in args.save_latents_at.split(",")} if args.save_latents_at else set()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if save_at:
        (out / "latents").mkdir(exist_ok=True)
    meta_path = out / (f"samples_shard{shard[0]}of{shard[1]}.jsonl" if shard else "samples.jsonl")
    finished = done_names(meta_path)

    def pending(p, c, s, sd):
        n = sample_name(p["id"], c, s, sd)
        return n not in finished or not (out / "images" / (n + ".png")).exists() or (save_at and not (out / "latents" / (n + ".pt")).exists())
    all_jobs = J.shard_jobs(J.enumerate_jobs(prompts, cfg_all, seeds), shard)
    jobs = [j for j in all_jobs if pending(*[j[0], j[1], j[2], j[3]])]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} images to generate ({len(finished)} already done)")
    if not jobs:
        return

    revision = cfg_all.get("revision")
    if revision == "UNPINNED-DRAFT":    # draft config: use the default revision, recorded as unpinned in every sample record
        print("WARNING: model revision is not pinned (draft config); results are not eligible for the freeze", flush=True)
        revision = None
    pipe = load_pipeline(cfg_all["model_id"], revision)
    size = cfg_all["image_size"]
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version()}
    with open(meta_path, "a") as meta:
        for i, (p, c, s, sd) in enumerate(jobs, 1):
            name = sample_name(p["id"], c, s, sd)
            t0 = time.time()
            kept = {}
            if save_at:
                image, kept = generate_with_latents(pipe, p["prompt"], c, s, sd, size, save_at)
            else:
                image = generate_image(pipe, p["prompt"], c, s, sd, size)
            img_path = out / "images" / f"{name}.png"
            atomic_save_image(image, img_path)
            record = {"name": name, "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"],
                      "cfg": c, "steps": s, "seed": sd, "size": size, "scheduler": "DDIM(eta=0)",
                      "model_id": cfg_all["model_id"], "negative_prompt": "", "seconds": round(time.time() - t0, 2),
                      "torch": torch.__version__}
            if args.shard or save_at or args.run_id:   # additive provenance fields; absent in default (pilot-style) runs
                record.update({"image_sha256": file_sha256(img_path), "run_id": args.run_id, "revision": cfg_all.get("revision"), "env": env})
            record.update({k: p[k] for k in ("scene_id", "variant", "split") if k in p})   # V2 prompt provenance (absent for pilot prompts)
            if shard:
                record["shard"] = args.shard
            if save_at:
                lat_path = out / "latents" / f"{name}.pt"
                torch.save({"steps_completed": sorted(kept), "latents_fp16": {t: kept[t] for t in sorted(kept)}}, str(lat_path) + ".tmp")
                os.replace(str(lat_path) + ".tmp", lat_path)
                record["latents_sha256"] = file_sha256(lat_path)
            meta.write(json.dumps(record) + "\n")
            meta.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {name} {record['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
