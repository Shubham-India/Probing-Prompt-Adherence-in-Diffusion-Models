"""Independent audit of a core generation run that saved latents (e.g. R2a). Read-only; CPU only; no model download.
It re-derives expectations from the config + prompts files instead of trusting the run's own verifier.
Usage: python src/audit_run.py --run <dir with images/ latents/ samples*.jsonl> --config configs/experiment_v2.json
       --prompts data/benchmark_v2/prompts_v2.json --split DEV --variants CAN [--checksums <checksums.sha256>] [--out audit.json]
Exit code 0 only if every hard check passes. Soft observations are reported but do not fail."""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobs as J

GRID = [6, 12, 18, 24, 30, 36]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def init_noise(seed, size=512):
    """Same call as generate.initial_latents, on CPU, float32 -> fp16 (generate.py casts the CPU tensor to fp16)."""
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((1, 4, size // 8, size // 8), generator=g).to(torch.float16).float()


def corr(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--checksums", default=None, help="sha256sum-style file; paths are relative to its parent directory")
    ap.add_argument("--grid", default=",".join(map(str, GRID)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    grid = [int(x) for x in a.grid.split(",")]
    cfg = json.loads(Path(a.config).read_text())
    prompts = J.load_prompts(a.prompts)
    if a.split:
        prompts = [p for p in prompts if p.get("split") == a.split]
    if a.variants:
        prompts = [p for p in prompts if p.get("variant") in set(a.variants.split(","))]
    pmap = {p["id"]: p for p in prompts}
    R, hard, soft = {}, [], []

    def check(name, ok, detail=""):
        R[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            hard.append(f"{name}: {detail}")

    # ---- manifests
    shard_files = sorted(run.glob("samples_shard*of*.jsonl"))
    merged = [json.loads(l) for l in (run / "samples.jsonl").read_text().splitlines() if l.strip()]
    shard_recs = [json.loads(l) for f in shard_files for l in f.read_text().splitlines() if l.strip()]
    names = [r["name"] for r in merged]
    check("merged_names_unique", len(names) == len(set(names)), f"{len(names)} records, {len(set(names))} unique")
    check("shard_union_equals_merged", {r["name"]: r for r in shard_recs} == {r["name"]: r for r in merged},
          f"shards={len(shard_recs)} merged={len(merged)}")
    check("shards_disjoint", len(shard_recs) == len({r['name'] for r in shard_recs}), "")
    expected_jobs = J.enumerate_jobs(prompts, cfg)
    exp_names = [J.sample_name(p["id"], c, s, sd) for p, c, s, sd in expected_jobs]
    check("expected_count_derived_from_config", len(exp_names) == len(prompts) * len(cfg["cfg_scales"]) * len(cfg["steps"]) * len(cfg["seeds"]),
          f"{len(prompts)} prompts x {len(cfg['cfg_scales'])} cfg x {len(cfg['steps'])} steps x {len(cfg['seeds'])} seeds = {len(exp_names)}")
    check("coverage_exact", set(exp_names) == set(names), f"missing={len(set(exp_names) - set(names))} extra={len(set(names) - set(exp_names))}")
    nsh = cfg.get("num_shards", 2)
    bad_shard = [n for k, n in enumerate(exp_names) if next((r['shard'] for r in merged if r['name'] == n), None) != f"{k % nsh}/{nsh}"]
    check("round_robin_shard_assignment", not bad_shard, f"{len(bad_shard)} records on the wrong shard")
    meta_bad = []
    run_ids, envs, revs = set(), Counter(), set()
    for r in merged:
        p = pmap.get(r["prompt_id"])
        if p is None or r["prompt"] != p["prompt"] or r["category"] != p["category"] or r.get("scene_id") != p["scene_id"] \
                or r.get("variant") != p["variant"] or r.get("split") != p["split"]:
            meta_bad.append(r["name"])
        if r["size"] != cfg["image_size"] or r["steps"] not in cfg["steps"] or r["cfg"] not in cfg["cfg_scales"] or r["seed"] not in cfg["seeds"] \
                or r["model_id"] != cfg["model_id"] or r["negative_prompt"] != "" or r["scheduler"] != "DDIM(eta=0)":
            meta_bad.append(r["name"] + " (settings)")
        run_ids.add(r.get("run_id")); revs.add(r.get("revision")); envs[json.dumps(r.get("env"), sort_keys=True)] += 1
    check("metadata_matches_prompts_and_config", not meta_bad, f"{len(meta_bad)} mismatches: {meta_bad[:3]}")
    check("single_run_id", len(run_ids) == 1, str(run_ids))
    check("revision_pinned_and_equal_config", revs == {cfg["revision"]} and cfg["revision"] not in (None, "UNPINNED-DRAFT"), f"{revs} vs {cfg['revision']}")
    R["environments"] = dict(envs)
    secs = defaultdict(list)
    for r in merged:
        secs[r["shard"]].append(r["seconds"])
    R["seconds_per_image_by_shard"] = {k: {"median": float(np.median(v)), "min": min(v), "max": max(v)} for k, v in secs.items()}

    # ---- images
    img_dir, lat_dir = run / "images", run / "latents"
    on_disk_img, on_disk_lat = {p.stem for p in img_dir.glob("*.png")}, {p.stem for p in lat_dir.glob("*.pt")}
    check("no_orphan_or_missing_images", on_disk_img == set(names), f"orphans={len(on_disk_img - set(names))} missing={len(set(names) - on_disk_img)}")
    check("no_orphan_or_missing_latents", on_disk_lat == set(names), f"orphans={len(on_disk_lat - set(names))} missing={len(set(names) - on_disk_lat)}")
    check("no_temp_files", not list(run.rglob("*.tmp")), str(list(run.rglob("*.tmp"))[:3]))
    img_hash, stats, sizebad, decbad, hashbad = {}, {}, [], [], []
    for r in merged:
        p = img_dir / f"{r['name']}.png"
        try:
            with Image.open(p) as im:
                im.load()
                if im.size != (512, 512) or im.mode != "RGB":
                    sizebad.append((r["name"], im.size, im.mode))
                arr = np.asarray(im, dtype=np.float32)
            stats[r["name"]] = (float(arr.mean()), float(arr.std()))
        except Exception as e:
            decbad.append((r["name"], type(e).__name__))
            continue
        img_hash[r["name"]] = sha(p)
        if img_hash[r["name"]] != r.get("image_sha256"):
            hashbad.append(r["name"])
    check("images_decode", not decbad, str(decbad[:3]))
    check("images_512_rgb", not sizebad, str(sizebad[:3]))
    check("image_sha_matches_manifest", not hashbad, f"{len(hashbad)} mismatches")
    dups = [h for h, c in Counter(img_hash.values()).items() if c > 1]
    check("no_duplicate_image_bytes", not dups, f"{len(dups)} duplicated hashes")
    flat = [n for n, (m, s) in stats.items() if s < 8]
    check("no_blank_images", not flat, f"{len(flat)} images with pixel std < 8: {flat[:3]}")
    ext = [n for n, (m, s) in stats.items() if m < 15 or m > 240]
    if ext:
        soft.append(f"{len(ext)} images are very dark/bright (mean <15 or >240): {ext[:3]}")
    R["image_pixel_std"] = {"min": min(s for m, s in stats.values()), "max": max(s for m, s in stats.values())}
    # same-scene different seeds must differ (checked by hash above) -- also across all scenes

    # ---- latents (format, content, relation to the seed)
    latbad, fmt, hashlat, per_step_std = [], Counter(), [], defaultdict(list)
    corr_match, corr_mismatch, mono = defaultdict(list), defaultdict(list), 0
    noise_cache = {}
    seeds = sorted({r["seed"] for r in merged})
    for sd in seeds:
        noise_cache[sd] = init_noise(sd, cfg["image_size"])
    for r in merged:
        lp = lat_dir / f"{r['name']}.pt"
        if sha(lp) != r.get("latents_sha256"):
            hashlat.append(r["name"])
        try:
            obj = torch.load(lp, map_location="cpu", weights_only=True)
        except Exception as e:
            latbad.append((r["name"], f"load: {type(e).__name__}"))
            continue
        fmt[tuple(sorted(obj.keys())) if isinstance(obj, dict) else type(obj).__name__] += 1
        d = obj.get("latents_fp16", {})
        if sorted(obj.get("steps_completed", [])) != grid or sorted(d.keys()) != grid or not all(isinstance(k, int) for k in d):
            latbad.append((r["name"], f"steps {obj.get('steps_completed')} keys {list(d)[:8]}"))
            continue
        for t in grid:
            x = d[t]
            if x.dtype != torch.float16 or tuple(x.shape) != (1, 4, cfg["image_size"] // 8, cfg["image_size"] // 8) or not torch.isfinite(x.float()).all():
                latbad.append((r["name"], f"t={t} dtype/shape/finite {x.dtype} {tuple(x.shape)}"))
                break
            per_step_std[t].append(float(x.float().std()))
        else:
            for t in (6, 12):
                corr_match[t].append(corr(d[t].float(), noise_cache[r["seed"]]))
                other = noise_cache[seeds[(seeds.index(r["seed"]) + 1) % len(seeds)]]
                corr_mismatch[t].append(corr(d[t].float(), other))
            if len({tuple(d[t].flatten()[:8].tolist()) for t in grid}) < len(grid):
                latbad.append((r["name"], "identical checkpoints"))
    check("latents_sha_matches_manifest", not hashlat, f"{len(hashlat)} mismatches")
    check("latents_loadable_weights_only_and_valid", not latbad, f"{len(latbad)} bad: {latbad[:3]}")
    R["latent_file_format"] = {str(k): v for k, v in fmt.items()}
    R["latent_std_by_step_mean"] = {t: float(np.mean(v)) for t, v in sorted(per_step_std.items())}
    for t in (6, 12):
        if corr_match[t]:
            m, o = float(np.mean(corr_match[t])), float(np.mean(corr_mismatch[t]))
            R[f"corr_latent_t{t}_with_own_seed_noise"] = {"matched_mean": m, "matched_min": float(np.min(corr_match[t])), "other_seed_mean": o,
                                                         "other_seed_absmax": float(np.max(np.abs(corr_mismatch[t])))}
    if corr_match[6]:
        check("latents_linked_to_their_seed", np.min(corr_match[6]) > 0.5 and np.max(np.abs(corr_mismatch[6])) < 0.2,
              f"t=6: matched min {np.min(corr_match[6]):.3f}, other-seed |max| {np.max(np.abs(corr_mismatch[6])):.3f}")

    # noise-level check: Var(x_t) = abar*s2 + (1-abar) for x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps. Fit s2 (variance of the clean latent) and
    # compare how well each assumed step offset explains the saved checkpoints (offset 0 = "latents after t completed steps").
    if len(per_step_std) == len(grid):
        betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2        # SD 1.5 scaled_linear schedule
        abar = np.cumprod(1.0 - betas)
        ts = (np.arange(40) * 25)[::-1] + 1                                  # DDIM, 40 steps, leading spacing, offset 1
        var = np.array([np.mean(per_step_std[t]) ** 2 for t in grid])
        rss = {}
        for off in (-2, -1, 0, 1, 2):
            ab = np.array([abar[ts[min(max(t + off, 0), 39)]] for t in grid])
            s2 = float(np.sum((var - (1 - ab)) * ab) / np.sum(ab * ab))
            rss[off] = float(np.sum((var - (ab * s2 + 1 - ab)) ** 2))
        R["noise_level_fit_rss_by_assumed_offset"] = rss
        best = min(rss, key=rss.get)
        # OBSERVATION ONLY, not a pass/fail check: the fit assumes the sampled trajectory has the forward-process marginal variance, which the
        # model's blurrier x0 predictions violate; on R2a it prefers offset -1 although a direct comparison with a local regeneration
        # (local_audit diag, corr 0.99999 at the same step vs 0.9998 at neighbours) shows the key t is "after t completed steps".
        # Use a direct regeneration (tests/gpu_identity.py reload_resume_maxdiff == 0) to decide offsets, not this fit.
        soft.append(f"variance-vs-schedule fit prefers assumed offset {best} (rss {rss}); not a reliable offset detector, see comment in audit_run.py")

    # ---- external checksum file
    if a.checksums:
        base = Path(a.checksums).parent
        lines = [l.split(None, 1) for l in Path(a.checksums).read_text().splitlines() if l.strip()]
        bad, missing, covered, normalised = [], [], set(), []
        for h, rel in lines:
            f = base / rel.strip().lstrip("*")
            if not f.exists():
                missing.append(rel)
            elif sha(f) != h:
                # text files restored through git with autocrlf carry CRLF; accept only if the LF-normalised content matches
                if f.suffix in (".jsonl", ".json", ".log", ".md") and hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == h:
                    normalised.append(rel)
                else:
                    bad.append(rel)
            covered.add(f.name)
        check("checksums_file_all_match", not bad and not missing, f"{len(lines)} entries, bad={len(bad)}, missing={len(missing)}")
        uncovered = [n for n in ([f"{x}.png" for x in names] + [f"{x}.pt" for x in names]) if n not in covered]
        check("checksums_cover_every_image_and_latent", not uncovered, f"{len(uncovered)} uncovered")
        R["checksum_text_files_matching_only_after_CRLF_to_LF"] = normalised
        if normalised:
            soft.append(f"{len(normalised)} text files match their checksum only after CRLF->LF (line endings changed by the local restore, not by the run)")
        R["checksum_entries_not_images_or_latents"] =[rel for h, rel in lines if not rel.strip().endswith((".png", ".pt"))]

    out = {"run": str(run), "hard_failures": hard, "soft_observations": soft, "checks": R}
    print(json.dumps(out, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
