"""Generates the Prompt x CFG x steps x seed grid with Stable Diffusion 1.5 and saves images + metadata.

Usage: python src/generate.py --config configs/experiment.json --out outputs/main [--limit-prompts ID,ID] [--seeds 2]
Re-running skips every sample already listed in <out>/samples.jsonl whose image file exists.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

ROOT = Path(__file__).resolve().parent.parent


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
    return f"{prompt_id}__cfg{cfg:g}__st{steps}__seed{seed}"


def generate_image(pipe, prompt, cfg, steps, seed, size, **extra):
    return pipe(prompt, guidance_scale=cfg, num_inference_steps=steps, height=size, width=size,
                latents=initial_latents(seed, size), negative_prompt="", **extra).images[0]


def done_names(meta_path):
    if not meta_path.exists():
        return set()
    return {json.loads(line)["name"] for line in meta_path.read_text().splitlines() if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-prompts", default=None, help="comma-separated prompt ids")
    ap.add_argument("--seeds", type=int, default=None, help="use only the first N seeds")
    args = ap.parse_args()

    cfg_all = json.loads(Path(args.config).read_text())
    prompts = json.loads((ROOT / "data" / "prompts.json").read_text())
    if args.limit_prompts:
        keep = set(args.limit_prompts.split(","))
        prompts = [p for p in prompts if p["id"] in keep]
    seeds = cfg_all["seeds"][: args.seeds] if args.seeds else cfg_all["seeds"]

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta_path = out / "samples.jsonl"
    finished = done_names(meta_path)

    jobs = [(p, c, s, sd) for p in prompts for c in cfg_all["cfg_scales"]
            for s in cfg_all["steps"] for sd in seeds
            if sample_name(p["id"], c, s, sd) not in finished or not (out / "images" / (sample_name(p["id"], c, s, sd) + ".png")).exists()]
    print(f"{len(jobs)} images to generate ({len(finished)} already done)")
    if not jobs:
        return

    pipe = load_pipeline(cfg_all["model_id"], cfg_all.get("revision"))
    size = cfg_all["image_size"]
    with open(meta_path, "a") as meta:
        for i, (p, c, s, sd) in enumerate(jobs, 1):
            name = sample_name(p["id"], c, s, sd)
            t0 = time.time()
            image = generate_image(pipe, p["prompt"], c, s, sd, size)
            image.save(out / "images" / f"{name}.png")
            record = {"name": name, "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"],
                      "cfg": c, "steps": s, "seed": sd, "size": size, "scheduler": "DDIM(eta=0)",
                      "model_id": cfg_all["model_id"], "negative_prompt": "", "seconds": round(time.time() - t0, 2),
                      "torch": torch.__version__}
            meta.write(json.dumps(record) + "\n")
            meta.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {name} {record['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
