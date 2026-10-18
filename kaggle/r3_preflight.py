"""Revision 3 hardware / numerical preflight (notebook 00, STAGE = "preflight"). Measures, on the actual GPU, everything the runtime plan needs and
checks the identities the Revision 3 code relies on. Writes preflight.json. Never generates experiment data (prompts are pilot/DEV prompts only).
Measured: model load time; 40-step image time (plain, with latents, with attention recording); hook-on vs hook-off image identity; branch time;
CAG loop vs pipeline identity at constant w = 7.5 and at w = 1 (+ UNet evaluation counts); peak VRAM; J1 evaluator time per image; DINOv2 load and
embedding time; PNG / latent / attention file sizes; optionally 1 vs 2 concurrent worker throughput (--two-workers).
Usage: python kaggle/r3_preflight.py --out preflight.json [--n 4] [--two-workers] [--fake-pipeline]"""
import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--two-workers", action="store_true")
    ap.add_argument("--fake-pipeline", action="store_true")
    a = ap.parse_args()
    import numpy as np
    import torch
    import attn_record as AR
    import branch as B
    import cag as CAG
    import nbtools
    import pipeline_factory as PF
    import r3_run as RR
    import run_branches as RB
    cfg = json.loads((ROOT / "configs" / "experiment_r3.json").read_text())
    rep = dict(env=nbtools.env_report(), fake_pipeline=a.fake_pipeline, checks={}, timing={})
    t0 = time.time()
    pipe, info = PF.load(cfg, a.fake_pipeline)
    rep["timing"]["model_load_s"] = round(time.time() - t0, 1)
    size, side, dev, dt = info["image_size"], info["image_size"] // pipe.vae_scale_factor, info["device"], info["dtype"]
    prompts = [p for p in json.loads((ROOT / "data" / "benchmark_v2" / "prompts_v2.json").read_text()) if p["split"] == "DEV" and p["variant"] == "CAN"][: a.n]
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() and not a.fake_pipeline else (lambda: None)
    counter = RR.UnetCounter(pipe)
    def timed(fn):
        sync(); t = time.time(); r = fn(); sync()
        return r, time.time() - t
    plain, attn, lat = [], [], []
    ident = []
    grid = set(cfg["commitment"]["grid"])
    res = tuple(cfg["attention"]["resolutions"]) if not a.fake_pipeline else (8,)
    sizes = {}
    for k, p in enumerate(prompts):
        x = RR.init_latents(k, side, dev, dt)
        img0, s0 = timed(lambda: pipe(p["prompt"], guidance_scale=7.5, num_inference_steps=40, height=size, width=size, latents=x.clone(), negative_prompt="").images[0])
        plain.append(s0)
        counter.take()
        (img1, maps, words, meta, kept), s1 = timed(lambda: AR.generate_with_attention_groups(pipe, p["prompt"], p["constraints"], 7.5, 40, x.clone(), size, res, grid))
        attn.append(s1)
        ident.append(bool(np.array_equal(np.asarray(img0), np.asarray(img1))))
        buf = io.BytesIO(); img0.save(buf, format="PNG"); sizes.setdefault("png_bytes", []).append(len(buf.getvalue()))
        buf = io.BytesIO(); torch.save({"latents_fp16": kept}, buf); sizes.setdefault("latent_bytes", []).append(len(buf.getvalue()))
        tmp = Path(a.out).with_suffix(f".attn{k}.npz"); AR.save_npz(tmp, maps, words, meta); sizes.setdefault("attention_bytes", []).append(tmp.stat().st_size); tmp.unlink()
        if k == 0:
            emb = B.encode_prompt_pair(pipe, p["prompt"], dev)
            sch = B.Schedule(40)
            plan = RB.plan_for("D8", sch, 12, cfg["commitment"])
            lt = kept[12].to(dev, getattr(torch, dt))
            _, sb = timed(lambda: B.branch_generate(pipe, emb, lt, 12, plan, 123, 40, 7.5))
            rep["timing"]["branch_t12_D8_s"] = round(sb, 2)
            imgc, _ = CAG.sample_with_schedule(pipe, emb, x.clone(), [7.5] * 40)
            rep["checks"]["cag_constant_equals_pipeline"] = bool(np.array_equal(np.asarray(imgc), np.asarray(img0)))
            counter.take()
            img_w1, calls = CAG.sample_with_schedule(pipe, emb, x.clone(), [1.0] * 40)
            ref_w1 = pipe(p["prompt"], guidance_scale=1.0, num_inference_steps=40, height=size, width=size, latents=x.clone(), negative_prompt="").images[0]
            rep["checks"]["cag_w1_equals_pipeline_w1"] = bool(np.array_equal(np.asarray(img_w1), np.asarray(ref_w1)))
            rep["checks"]["cag_w1_unet_evaluations"] = calls
            counter.take()
    rep["checks"]["attention_hook_image_identical"] = all(ident)
    rep["timing"].update(plain_s_median=round(float(np.median(plain[1:] or plain)), 2), with_attention_and_latents_s_median=round(float(np.median(attn[1:] or attn)), 2))
    rep["timing"]["attention_overhead_fraction"] = round(rep["timing"]["with_attention_and_latents_s_median"] / rep["timing"]["plain_s_median"] - 1, 3)
    rep["storage_bytes_mean"] = {k: int(np.mean(v)) for k, v in sizes.items()}
    if torch.cuda.is_available() and not a.fake_pipeline:
        rep["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        try:
            import evaluate as E
            from PIL import Image
            t = time.time()
            rv = cfg["evaluator_revisions"]
            models = E.load_models("cuda", rv["google/owlvit-base-patch32"], rv["openai/clip-vit-base-patch32"])
            rep["timing"]["judge_load_s"] = round(time.time() - t, 1)
            t = time.time()
            for p in prompts:
                E.detect(img0, E.required_objects(p["constraints"]), models, "cuda")
            rep["timing"]["judge_s_per_image"] = round((time.time() - t) / len(prompts), 3)
            import diversity as DV
            t = time.time(); d = DV.DinoEmbedder("cuda", cache_dir=ROOT / "models", revision=rv["facebook/dinov2-small"]); rep["timing"]["dino_load_s"] = round(time.time() - t, 1)
            t = time.time(); d([img0] * 4); rep["timing"]["dino_s_per_image"] = round((time.time() - t) / 4, 3)
        except Exception as e:
            rep["checks"]["evaluator_error"] = f"{type(e).__name__}: {e}"
    if a.two_workers and nbtools.gpu_count() >= 2:
        probe = Path(a.out).parent / "_two_worker_probe"
        jobs = []
        import manifests as M
        main_p, _ = M.load_prompts(cfg)
        dev = [q for q in main_p if q["split"] == "DEV" and q["variant"] == "PAR"][:8]
        jobs = [M.core_job(q, 7.5, 40, 0, "preflight", "PREFLIGHT") for q in dev]
        mf = probe / "m.jsonl"; probe.mkdir(parents=True, exist_ok=True)
        mf.write_text("".join(json.dumps(j) + "\n" for j in jobs))
        t = time.time()
        codes = nbtools.run_workers(ROOT, mf, probe / "out", "preflight", "p")
        rep["timing"]["two_workers_8_images_wall_s"] = round(time.time() - t, 1)
        rep["checks"]["two_worker_exit_codes"] = codes
    rep["checks"]["all_identities_hold"] = bool(rep["checks"].get("attention_hook_image_identical") and rep["checks"].get("cag_constant_equals_pipeline")
                                                and rep["checks"].get("cag_w1_equals_pipeline_w1") and rep["checks"].get("cag_w1_unet_evaluations") == 40)
    failures = [] if rep["checks"]["all_identities_hold"] else ["identities"]
    if not a.fake_pipeline:
        if not torch.cuda.is_available():
            failures.append("cuda_unavailable")
        if "evaluator_error" in rep["checks"] or "judge_s_per_image" not in rep["timing"] or "dino_s_per_image" not in rep["timing"]:
            failures.append("evaluator")
    if "two_worker_exit_codes" in rep["checks"] and any(c != 0 for c in rep["checks"]["two_worker_exit_codes"].values()):
        failures.append("two_worker_probe")
    if a.two_workers and nbtools.gpu_count() >= 2 and "two_worker_exit_codes" not in rep["checks"]:
        failures.append("two_worker_probe_not_run")
    rep["checks"]["preflight_failures"] = failures
    rep["checks"]["preflight_ok"] = not failures
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["checks"]["preflight_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
