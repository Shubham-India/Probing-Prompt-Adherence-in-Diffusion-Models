"""Scores generated images against the constraints stored in data/prompts.json.

Two steps, kept separate so thresholds can be changed without re-running the models:
  detect()  : OWL-ViT boxes per queried object + CLIP colour probabilities for each box + CLIP image-text score
  score()   : turns the raw detections into per-constraint 0/1 results
Usage: python src/evaluate.py --run outputs/main      (writes <run>/detections.jsonl and <run>/eval.jsonl)
       python src/evaluate.py --run outputs/main --rescore   (only step 2)
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import box_iou, nms
from transformers import CLIPModel, CLIPProcessor, OwlViTForObjectDetection, OwlViTProcessor

ROOT = Path(__file__).resolve().parent.parent
COLORS = ["red", "blue", "yellow", "green", "orange", "purple", "white", "black", "brown", "pink"]
# Thresholds were fixed a priori; a sweep on the manually labelled validation subset (report/validation.md) showed accuracy was insensitive to them.
THRESHOLDS = {"presence": 0.10, "count": 0.15, "distinct_iou": 0.5, "nms_iou": 0.3}
SIZE_RATIO = 1.5   # Revision 3 J-size: the intended larger object's box area must be >= 1.5x the other's (RB 24.6); both objects detected
MAX_BOXES = 10


from model_cache import model_cache_dir  # noqa: E402


def load_models(device, owl_revision=None, clip_revision=None):
    """Default (revision None) reproduces the pilot behaviour; Revision 3 passes the pinned revisions from configs/experiment_r3.json."""
    cache = model_cache_dir()
    owl_p = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir=cache, revision=owl_revision)
    owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir=cache, revision=owl_revision).to(device).eval()
    clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache, revision=clip_revision)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache, revision=clip_revision).to(device).eval()
    return owl_p, owl, clip_p, clip


@torch.no_grad()
def detect(image, objects, models, device):
    """Return {object: [{score, box(xyxy), colors{name: prob}} ...]} and the CLIP image-text inputs."""
    owl_p, owl, clip_p, clip = models
    queries = [[f"a photo of a {o}" for o in objects]]
    inputs = owl_p(text=queries, images=image, return_tensors="pt").to(device)
    out = owl(**inputs)
    target = torch.tensor([image.size[::-1]], device=device)
    result = owl_p.post_process_grounded_object_detection(out, threshold=0.05, target_sizes=target)[0]
    detections = {}
    for idx, obj in enumerate(objects):
        keep = result["labels"] == idx
        boxes, scores = result["boxes"][keep], result["scores"][keep]
        order = nms(boxes, scores, THRESHOLDS["nms_iou"])[:MAX_BOXES] if len(boxes) else []
        detections[obj] = [{"score": float(scores[i]), "box": [float(v) for v in boxes[i]],
                            "colors": colour_probs(image, boxes[i], clip_p, clip, device)} for i in order]
    return detections


@torch.no_grad()
def colour_probs(image, box, clip_p, clip, device):
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    crop = image.crop((max(x0, 0), max(y0, 0), max(x1, x0 + 2), max(y1, y0 + 2)))
    texts = [f"a photo of a {c} object" for c in COLORS]
    inputs = clip_p(text=texts, images=crop, return_tensors="pt", padding=True).to(device)
    probs = clip(**inputs).logits_per_image.softmax(-1)[0]
    return {c: round(float(p), 4) for c, p in zip(COLORS, probs)}


@torch.no_grad()
def clip_score(image, prompt, clip_p, clip, device):
    inputs = clip_p(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    out = clip(**inputs)
    return float(out.logits_per_image[0, 0]), [round(float(v), 4) for v in out.image_embeds[0]]


def boxes_above(dets, obj, thr):
    return [d for d in dets.get(obj, []) if d["score"] >= thr]


def distinct_best_boxes(dets, objects):
    """Greedy one-box-per-object assignment so one detection cannot satisfy two different objects."""
    chosen, used = {}, []
    ranked = sorted(((d["score"], o, d) for o in objects for d in boxes_above(dets, o, THRESHOLDS["presence"])),
                    key=lambda t: -t[0])
    for _, obj, d in ranked:
        if obj in chosen:
            continue
        box = torch.tensor([d["box"]])
        if any(float(box_iou(box, torch.tensor([u]))) > THRESHOLDS["distinct_iou"] for u in used):
            continue
        chosen[obj] = d
        used.append(d["box"])
    return chosen


def score(constraints, dets):
    """One 0/1 result per constraint, in the same order as the constraints list."""
    objects = sorted({c["object"] for c in constraints if "object" in c} |
                     {c[k] for c in constraints for k in ("left", "right", "larger", "smaller") if k in c})
    best = distinct_best_boxes(dets, objects)
    counted = {c["object"] for c in constraints if c["type"] == "count"}
    results = []
    for c in constraints:
        t = c["type"]
        if t == "presence":
            ok = c["object"] in best
        elif t == "count":
            ok = len(boxes_above(dets, c["object"], THRESHOLDS["count"])) == c["n"]
        elif t == "binding":
            if c["object"] in counted:  # several instances requested: majority of boxes must have the colour
                votes = [max(b["colors"], key=b["colors"].get) == c["color"]
                         for b in boxes_above(dets, c["object"], THRESHOLDS["count"])]
                ok = len(votes) > 0 and sum(votes) > len(votes) / 2
            else:
                b = best.get(c["object"])
                ok = b is not None and max(b["colors"], key=b["colors"].get) == c["color"]
        elif t == "spatial":
            l, r = best.get(c["left"]), best.get(c["right"])
            centre = lambda d: (d["box"][0] + d["box"][2]) / 2
            ok = l is not None and r is not None and centre(l) < centre(r)
        elif t == "size":   # Revision 3: relative size; an undetected object makes the constraint fail (conditional version below)
            big, small = best.get(c["larger"]), best.get(c["smaller"])
            area = lambda d: max(d["box"][2] - d["box"][0], 0) * max(d["box"][3] - d["box"][1], 0)
            ok = big is not None and small is not None and area(small) > 0 and area(big) >= SIZE_RATIO * area(small)
        else:
            raise ValueError(f"unknown constraint type {t}")
        results.append(int(bool(ok)))
    return results


def run_detection(run, prompts, device):
    models = load_models(device)
    det_path = run / "detections.jsonl"
    done = {json.loads(l)["name"] for l in det_path.read_text().splitlines()} if det_path.exists() else set()
    samples = [json.loads(l) for l in (run / "samples.jsonl").read_text().splitlines()]
    todo = [s for s in samples if s["name"] not in done]
    print(f"{len(todo)} images to detect")
    with open(det_path, "a") as f:
        for i, s in enumerate(todo, 1):
            image = Image.open(run / "images" / f"{s['name']}.png").convert("RGB")
            constraints = prompts[s["prompt_id"]]["constraints"]
            objects = required_objects(constraints)
            dets = detect(image, objects, models, device)
            sim, embedding = clip_score(image, s["prompt"], models[2], models[3], device)
            f.write(json.dumps({"name": s["name"], "detections": dets, "clip_score": round(sim, 3),
                                "clip_image_embedding": embedding}) + "\n")
            if i % 50 == 0:
                print(f"[{i}/{len(todo)}]", flush=True)


def required_objects(constraints):
    return sorted({c["object"] for c in constraints if "object" in c} | {c[k] for c in constraints for k in ("left", "right", "larger", "smaller") if k in c})


UNCERTAIN_BAND = 0.03   # a required object's best detection within +/- band of the presence threshold is flagged (does not change any outcome)


def conditional_outcomes(constraints, dets):
    """D-04 outcomes next to (never instead of) the legacy per-constraint results. Every decision reuses score().
    pres_all: every required object has a distinct detection; <key>_given_pres is null when not eligible;
    joint_<key>: pres_all AND the constraint family satisfied; uncertain: some required object's best score is near the threshold."""
    objs = required_objects(constraints)
    pres = score([{"type": "presence", "object": o} for o in objs], dets)
    pres_all = int(all(pres))
    res = score(constraints, dets)
    out = {"n_required_objects": len(objs), "objects_present": pres_all}
    for t, key in (("binding", "bind"), ("spatial", "spatial")):
        fam = [r for c, r in zip(constraints, res) if c["type"] == t]
        if fam:
            ok = int(all(fam))
            out[f"{key}_given_pres"] = ok if pres_all else None
            out[f"joint_{key}"] = int(pres_all and ok)
        else:
            out[f"{key}_given_pres"] = None
            out[f"joint_{key}"] = None
    # Revision 3 additions (appended after the D-04 fields; absent constraint families give null)
    cnt = [r for c, r in zip(constraints, res) if c["type"] == "count"]
    out["count_ok"] = int(all(cnt)) if cnt else None
    siz = [r for c, r in zip(constraints, res) if c["type"] == "size"]
    out["size_given_pres"] = (int(all(siz)) if pres_all else None) if siz else None
    out["joint_size"] = int(pres_all and all(siz)) if siz else None
    best = [max((d["score"] for d in dets.get(o, [])), default=0.0) for o in objs]
    out["uncertain"] = int(any(abs(b - THRESHOLDS["presence"]) <= UNCERTAIN_BAND for b in best))
    return out


def run_scoring(run, prompts, conditional=False, out_name="eval.jsonl"):
    """Default call is identical to the pilot behaviour. conditional=True appends the D-04 fields AFTER all legacy columns."""
    samples = {json.loads(l)["name"]: json.loads(l) for l in (run / "samples.jsonl").read_text().splitlines()}
    with open(run / out_name, "w") as f:
        for line in (run / "detections.jsonl").read_text().splitlines():
            d = json.loads(line)
            s = samples[d["name"]]
            cons = prompts[s["prompt_id"]]["constraints"]
            res = score(cons, d["detections"])
            row = {**s, "constraint_results": res, "all_satisfied": int(all(res)), "clip_score": d["clip_score"]}
            if conditional:
                row.update(conditional_outcomes(cons, d["detections"]))
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "prompts.json"))
    ap.add_argument("--conditional", action="store_true", help="append D-04 conditional/joint fields (legacy columns unchanged)")
    ap.add_argument("--out-name", default="eval.jsonl")
    args = ap.parse_args()
    run = Path(args.run)
    prompts = {p["id"]: p for p in json.loads(Path(args.prompts).read_text())}
    if not args.rescore:
        run_detection(run, prompts, "cuda" if torch.cuda.is_available() else "cpu")
    run_scoring(run, prompts, args.conditional, args.out_name)
    print("wrote", run / args.out_name)
