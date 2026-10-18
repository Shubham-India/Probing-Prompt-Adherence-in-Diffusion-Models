"""Judge (J1: OWL-ViT + CLIP, src/evaluate.py) applied to merged Revision 3 records. One row per image with the legacy constraint results, the
presence-conditional and joint outcomes (incl. count_ok, size_given_pres, joint_size), the CLIP score, and for prompt-switch images also the
outcomes of the COUNTERFACTUAL prompt's constraints (prefix cf_). Optional DINOv2 embeddings for seed diversity (--diversity).
--fake-judge replaces detection with deterministic pseudo-detections (hash of the job id) so the CPU dry-run can exercise the scoring code; rows are
flagged fake_judge and analysis refuses them unless explicitly allowed. Resume-safe (skips job ids already in the output).
Usage: python src/r3_evaluate.py --records merged_records.jsonl --out eval_r3.jsonl [--diversity embeddings.npz] [--fake-judge]"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import evaluate as E   # noqa: E402


def prompts_index(cfg):
    return {p["id"]: p for f in (cfg["benchmark"]["prompts"], cfg["benchmark"]["ps_prompts"]) for p in json.loads((ROOT / f).read_text())}


def fake_detections(job_id, objects, colours):
    rng = random.Random(int(hashlib.sha256(job_id.encode()).hexdigest()[:12], 16))
    dets = {}
    for o in objects:
        boxes = []
        for _ in range(rng.choice([0, 1, 1, 1, 2])):
            x0, y0 = rng.uniform(0, 300), rng.uniform(0, 300)
            w = rng.uniform(40, 200)
            col = {c: rng.random() for c in E.COLORS}
            if colours.get(o) and rng.random() < 0.6:
                col[colours[o]] = 2.0
            s = sum(col.values())
            boxes.append({"box": [x0, y0, x0 + w, y0 + w], "score": rng.uniform(0.11, 0.9), "colors": {k: v / s for k, v in col.items()}})
        dets[o] = boxes
    return dets


def outcome_row(cons, dets, prefix=""):
    res = E.score(cons, dets)
    row = {f"{prefix}constraint_results": res, f"{prefix}all_satisfied": int(all(res))}
    row.update({f"{prefix}{k}": v for k, v in E.conditional_outcomes(cons, dets).items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment_r3.json"))
    ap.add_argument("--fake-judge", action="store_true")
    ap.add_argument("--diversity", default=None, help="write DINOv2 CLS embeddings of every image to this .npz")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    P = prompts_index(cfg)
    recs = [json.loads(l) for l in Path(a.records).read_text().splitlines() if l.strip()]
    done = {json.loads(l)["job_id"] for l in Path(a.out).read_text().splitlines() if l.strip()} if Path(a.out).exists() else set()
    todo = [r for r in recs if r["job_id"] not in done]
    print(f"{len(todo)} images to judge ({len(done)} done)", flush=True)
    models = device = None
    if todo and not a.fake_judge:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rv = cfg.get("evaluator_revisions", {})
        models = E.load_models(device, rv.get("google/owlvit-base-patch32"), rv.get("openai/clip-vit-base-patch32"))
    from PIL import Image
    emb_ids, embs, dino = [], [], None
    if a.diversity:
        import diversity as DV
        dino = DV.DinoEmbedder("cuda", cache_dir=E.model_cache_dir(), revision=cfg.get("evaluator_revisions", {}).get(DV.DINO_ID)) if not a.fake_judge else None
    with open(a.out, "a") as f:
        for i, r in enumerate(todo, 1):
            img_path = next(v for k, v in r["abs_files"].items() if k.endswith(".png"))
            image = Image.open(img_path).convert("RGB")
            cons = P[r["prompt_id"]]["constraints"]
            cf = P[r["cf_prompt_id"]]["constraints"] if r.get("cf_prompt_id") else []
            objects = E.required_objects(cons + cf)
            if a.fake_judge:
                colours = {c["object"]: c["color"] for c in cons if c["type"] == "binding"}
                dets, clip = fake_detections(r["job_id"], objects, colours), None
            else:
                dets = E.detect(image, objects, models, device)
                clip, _ = E.clip_score(image, r["prompt"], models[2], models[3], device)
            row = {k: r.get(k) for k in ("job_id", "kind", "experiment", "stage", "split", "scene_id", "shard_group", "prompt_id", "variant", "category", "cfg",
                                         "steps", "seed", "t", "restart", "m", "arm", "w_lo", "s_star", "cf_prompt_id", "cf_type", "base_job", "unet_calls_measured",
                                         "seconds", "fake_pipeline", "effective_jump", "ratio", "clipped", "mean_guidance")}
            row.update(outcome_row(cons, dets))
            if cf:
                row.update(outcome_row(cf, dets, "cf_"))
            row["clip_score"] = None if clip is None else round(float(clip), 3)
            row["fake_judge"] = bool(a.fake_judge)
            f.write(json.dumps(row) + "\n")
            if a.diversity:
                emb_ids.append(r["job_id"])
                if dino is not None:
                    embs.append(dino([image])[0])
                else:
                    rng = random.Random(int(hashlib.sha256(r["job_id"].encode()).hexdigest()[:12], 16))
                    embs.append([rng.gauss(0, 1) for _ in range(16)])
            if i % 100 == 0:
                print(f"[{i}/{len(todo)}]", flush=True)
    if a.diversity and emb_ids:
        import numpy as np
        old_ids, old = [], None
        if Path(a.diversity).exists():
            z = np.load(a.diversity)
            old_ids, old = list(z["ids"]), z["emb"]
        ids = old_ids + emb_ids
        arr = np.asarray(embs, dtype=np.float32) if old is None else np.concatenate([old, np.asarray(embs, dtype=np.float32)])
        np.savez(a.diversity, ids=np.array(ids), emb=arr, model="facebook/dinov2-small" if not a.fake_judge else "FAKE")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
