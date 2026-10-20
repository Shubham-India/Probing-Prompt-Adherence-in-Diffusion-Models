"""Revision 3 worker: executes the jobs of one manifest shard (src/manifests.py) and writes verifiable outputs. Resume-safe and idempotent.
  core   : full generation (optional saved latents at the commitment grid; optional attention maps for HF)
  branch : restart from a saved latent (re-noising, branch-specific noise, deterministic DDIM to the end)
  switch : continue a counterfactual trajectory from its saved latent with the canonical prompt
  cag    : DDIM with a per-step guidance schedule (w = 1 steps skip the unconditional pass)
Every output is written to a temporary name and renamed; the record is appended (and flushed) only after all files of the job exist; at start-up
a job counts as done only if its record exists AND every file it names exists with the recorded SHA-256. TEST jobs are refused unless
protocol/protocol_freeze.json verifies (src/protocol_r3.py). UNet evaluations are counted by wrapping the UNet, not inferred.
Usage: python src/r3_run.py --manifest m.jsonl --out DIR --run-id RID --worker-id W [--upstream DIR ...] [--protocol FILE]
                            [--gpu-subshard i/n] [--fake-pipeline] [--limit N] [--config configs/experiment_r3.json]"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
KIND_ORDER = {"core": 0, "branch": 1, "switch": 2, "cag": 3}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def init_latents(seed, side, device, dtype):
    """Same draw as generate.initial_latents (CPU generator, shape [1, 4, side, side]) cast to the pipeline's device/dtype."""
    import torch
    g = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn((1, 4, side, side), generator=g).to(device, getattr(torch, dtype))


class UnetCounter:
    def __init__(self, pipe):
        self.n, self.pipe, self.orig = 0, pipe, pipe.unet.forward

        def fwd(sample, *a, **k):
            self.n += int(sample.shape[0])
            return self.orig(sample, *a, **k)
        pipe.unet.forward = fwd

    def take(self):
        n, self.n = self.n, 0
        return n


def load_records(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []


def verified_done(out, records):
    done = {}
    for r in records:
        files = r.get("files", {})
        if files and all((out / p).exists() and sha(out / p) == h for p, h in files.items()):
            done[r["job_id"]] = r
    return done


def find_latent(job_id, dirs):
    for d in dirs:
        p = Path(d) / "latents" / f"{job_id}.pt"
        if p.exists():
            return p
    return None


def subshard(jobs, spec):
    if not spec:
        return jobs
    i, n = (int(x) for x in spec.split("/"))
    groups = sorted({j["shard_group"] for j in jobs})
    mine = {g for k, g in enumerate(groups) if k % n == i}
    return [j for j in jobs if j["shard_group"] in mine]


def guard_test(jobs, protocol):
    if any(j["split"] == "TEST" for j in jobs):
        import protocol_r3
        problems = protocol_r3.verify(protocol) if protocol else ["no --protocol file given"]
        if problems:
            sys.exit("TEST jobs refused (protocol not frozen/verified): " + "; ".join(problems[:5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--upstream", nargs="*", default=[], help="verified output folders that hold base latents")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--gpu-subshard", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment_r3.json"))
    ap.add_argument("--fake-pipeline", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-per-kind", type=int, default=None, help="dry-run only: at most N jobs of each kind, only jobs whose base latent exists")
    ap.add_argument("--max-errors", type=int, default=20)
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    jobs = [json.loads(l) for l in Path(a.manifest).read_text().splitlines() if l.strip()]
    manifest_sha = hashlib.sha256(Path(a.manifest).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    jobs = sorted(subshard(jobs, a.gpu_subshard), key=lambda j: (KIND_ORDER[j["kind"]], j["job_id"]))
    guard_test(jobs, a.protocol)
    out = Path(a.out)
    for d in ("images", "latents", "attention"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rec_path, err_path = out / f"records_{a.worker_id}.jsonl", out / f"errors_{a.worker_id}.jsonl"
    done = verified_done(out, [r for f in sorted(out.glob("records_*.jsonl")) for r in load_records(f)])   # any worker's verified output counts
    todo = [j for j in jobs if j["job_id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    if a.limit_per_kind:
        cores = [j for j in todo if j["kind"] == "core"][: a.limit_per_kind]
        made = {j["job_id"] for j in cores if j.get("save_latents")}
        pick = list(cores)
        for kind in ("branch", "switch", "cag"):
            cand = [j for j in todo if j["kind"] == kind]
            if kind != "cag":       # only jobs whose base latent this limited run can resolve (never pad the subset with jobs that must fail)
                cand = [j for j in cand if j["base_job"] in made or find_latent(j["base_job"], [out] + list(a.upstream))]
            pick += cand[: a.limit_per_kind]
        todo = pick
    print(json.dumps(dict(worker=a.worker_id, jobs=len(jobs), already_verified=len(done), to_run=len(todo),
                          image_equivalents_to_run=round(sum(j["unet_calls"] for j in todo) / 80, 1))), flush=True)
    if not todo:
        return 0

    import torch
    import attn_record as AR
    import branch as B
    import cag as CAG
    import generate as G
    import pipeline_factory as PF
    import run_branches as RB
    import switch as SW
    pipe, info = PF.load(cfg, a.fake_pipeline)
    side = info["image_size"] // pipe.vae_scale_factor
    size = info["image_size"]
    counter = UnetCounter(pipe)
    sch = B.Schedule(cfg["steps"])
    prompts = {p["id"]: p for f in (cfg["benchmark"]["prompts"], cfg["benchmark"]["ps_prompts"]) for p in json.loads((ROOT / f).read_text())}
    grid = set(cfg["commitment"]["grid"])
    res = tuple(cfg["attention"]["resolutions"]) if not info["fake"] else (8,)
    env = {"torch": torch.__version__, "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and not info["fake"] else "cpu"}
    import diffusers
    env["diffusers"] = diffusers.__version__
    embeds = {}

    def emb(prompt):
        if prompt not in embeds:
            embeds.clear()
            embeds[prompt] = B.encode_prompt_pair(pipe, prompt, info["device"])
        return embeds[prompt]

    def base_latent(job_id, t):
        p = find_latent(job_id, [out] + list(a.upstream))
        if p is None:
            raise FileNotFoundError(f"base latent {job_id}.pt not found in {[str(out)] + a.upstream}")
        return torch.load(p, map_location="cpu", weights_only=True)["latents_fp16"][t].to(info["device"], getattr(torch, info["dtype"]))

    n_err, t_start = 0, time.time()
    with open(rec_path, "a") as rf, open(err_path, "a") as ef:
        for k, j in enumerate(todo, 1):
            t0 = time.time()
            counter.take()
            files = {}
            try:
                kept, maps = {}, None
                if j["kind"] == "core":
                    x = init_latents(j["seed"], side, info["device"], info["dtype"])
                    if j["attention"]:
                        image, maps, words, meta, kept = AR.generate_with_attention_groups(
                            pipe, j["prompt"], prompts[j["prompt_id"]]["constraints"], j["cfg"], j["steps"], x, size, res, grid if j["save_latents"] else ())
                    else:
                        def cb(_p, i, _t, kw):
                            if j["save_latents"] and (i + 1) in grid:
                                kept[i + 1] = kw["latents"].detach().to("cpu", torch.float16).clone()
                            return kw
                        image = pipe(j["prompt"], guidance_scale=j["cfg"], num_inference_steps=j["steps"], height=size, width=size, latents=x,
                                     negative_prompt="", callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
                elif j["kind"] == "branch":
                    lat = base_latent(j["base_job"], j["t"])
                    plan = RB.plan_for(j["restart"], sch, j["t"], cfg["commitment"])
                    assert plan["target_idx"] == j["target_idx"], "restart plan differs from the manifest"
                    if j["restart"] == "FULL":
                        image = B.denoise_from(pipe, emb(j["prompt"]), B.branch_noise(lat.shape, j["noise_seed"], lat.dtype, lat.device), 0, j["steps"], j["cfg"])
                    else:
                        image = B.branch_generate(pipe, emb(j["prompt"]), lat, j["t"], plan, j["noise_seed"], j["steps"], j["cfg"])
                elif j["kind"] == "switch":
                    image = SW.switch_generate(pipe, emb(j["prompt"]), base_latent(j["base_job"], j["t"]), j["t"], j["steps"], j["cfg"])
                elif j["kind"] == "cag":
                    if j.get("schedule") is None:
                        raise ValueError(j.get("not_estimable", "no schedule"))
                    image, _ = CAG.sample_with_schedule(pipe, emb(j["prompt"]), init_latents(j["seed"], side, info["device"], info["dtype"]), j["schedule"])
                else:
                    raise ValueError(j["kind"])
                img = out / "images" / f"{j['job_id']}.png"
                G.atomic_save_image(image, img)
                files[f"images/{j['job_id']}.png"] = sha(img)
                if kept:
                    lp = out / "latents" / f"{j['job_id']}.pt"
                    torch.save({"steps_completed": sorted(kept), "latents_fp16": {t: kept[t] for t in sorted(kept)}}, str(lp) + ".tmp")
                    os.replace(str(lp) + ".tmp", lp)
                    files[f"latents/{j['job_id']}.pt"] = sha(lp)
                if j["kind"] == "core" and j["save_latents"] and sorted(kept) != sorted(grid):
                    raise RuntimeError(f"latents saved at {sorted(kept)}, expected {sorted(grid)}")
                if maps:
                    ap_ = out / "attention" / f"{j['job_id']}.npz"
                    AR.save_npz(ap_, maps, words, dict(meta, job_id=j["job_id"], scene_id=j["scene_id"], prompt_id=j["prompt_id"], seed=j["seed"], cfg=j["cfg"], steps=j["steps"]))
                    files[f"attention/{j['job_id']}.npz"] = sha(ap_)
                rec = {k2: v for k2, v in j.items() if k2 != "schedule"}
                if j["kind"] == "cag":
                    rec["schedule_sha256"] = hashlib.sha256(json.dumps(j["schedule"]).encode()).hexdigest()
                rec.update(files=files, image_sha256=files[f"images/{j['job_id']}.png"], unet_calls_measured=counter.take(), seconds=round(time.time() - t0, 2),
                           run_id=a.run_id, worker_id=a.worker_id, manifest_sha256=manifest_sha, model_id=cfg["model_id"], revision=cfg["revision"],
                           env=env, fake_pipeline=info["fake"], image_size=size)
                rf.write(json.dumps(rec) + "\n")
                rf.flush()
                os.fsync(rf.fileno())
            except Exception as e:
                n_err += 1
                ef.write(json.dumps(dict(job_id=j["job_id"], error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:], worker_id=a.worker_id)) + "\n")
                ef.flush()
                print(f"ERROR {j['job_id']}: {type(e).__name__}: {e}", flush=True)
                if n_err >= a.max_errors:
                    print("too many errors; stopping", flush=True)
                    break
            if k % 20 == 0 or k == len(todo):
                el = time.time() - t_start
                print(f"[{k}/{len(todo)}] {j['job_id']} {round(time.time() - t0, 2)}s  elapsed {el / 60:.1f} min", flush=True)
    summary = dict(worker=a.worker_id, run_id=a.run_id, ran=len(todo), errors=n_err, seconds=round(time.time() - t_start, 1), env=env,
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() and not info["fake"] else None, fake_pipeline=info["fake"])
    (out / f"worker_{a.worker_id}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
