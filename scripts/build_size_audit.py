"""Blinded human audit package for the relative-SIZE judge (Revision 3 PS family; Addendum C: size is confirmatory only with >= 140 audited
judgements and accuracy >= 0.85). Census of every DEV size judgement in a VERIFIED DEV_CORE output folder (no TEST data):
1. J1 (OWL-ViT + CLIP at the pinned revisions, src/evaluate.py) judges the DEV PS-family images that carry a size constraint;
2. src/audit_dev.build writes the package with the existing blinded HTML (annotator sees only an opaque image and a neutral question: which
   object looks larger), a practice round, and a re-annotation subset (B).
Output (git-ignored): local_audit/audit_size_dev/{annotator_A, annotator_B, key, j1_size_eval.jsonl}. Score after labelling:
  python src/audit_dev.py score --key local_audit/audit_size_dev/key/audit_key.csv --a labels_size_A.csv [--b labels_size_B.csv] --out size_audit_report.json
Usage: python scripts/build_size_audit.py --dev-core <verified DEV_CORE folder> [--out local_audit/audit_size_dev] [--device cuda|cpu]"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-core", required=True)
    ap.add_argument("--out", default=str(ROOT / "local_audit" / "audit_size_dev"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    import torch
    from PIL import Image
    import audit_dev as AD
    import evaluate as E
    src, out = Path(a.dev_core), Path(a.out)
    if (out / "key" / "audit_key.csv").exists():
        sys.exit(f"{out} already holds a package (it may be in use); choose another --out")
    guard = json.loads((src / "guard.json").read_text())
    if guard.get("stage") != "DEV_CORE" or guard.get("ok") is not True:
        sys.exit("not a verified DEV_CORE output folder (guard.json)")
    cfg = json.loads((ROOT / "configs" / "experiment_r3.json").read_text())
    prompts = {p["id"]: p for p in json.loads((ROOT / "data" / "benchmark_v2" / "prompts_ps.json").read_text())}
    recs = [json.loads(l) for f in sorted(src.glob("records_*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r["prompt_id"] in prompts and r["split"] == "DEV" and any(c["type"] == "size" for c in prompts[r["prompt_id"]]["constraints"])]
    if any(r["split"] != "DEV" for r in recs):
        sys.exit("TEST rows present")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rv = cfg["evaluator_revisions"]
    models = E.load_models(dev, rv["google/owlvit-base-patch32"], rv["openai/clip-vit-base-patch32"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, r in enumerate(recs):
        img_path = src / "images" / f"{r['job_id']}.png"
        image = Image.open(img_path).convert("RGB")
        cons = prompts[r["prompt_id"]]["constraints"]
        dets = E.detect(image, E.required_objects(cons), models, dev)
        res = E.score(cons, dets)
        cond = E.conditional_outcomes(cons, dets)
        rows.append(dict(name=r["job_id"], prompt_id=r["prompt_id"], split=r["split"], image_sha256=r["image_sha256"], constraint_results=res,
                         objects_present=cond.get("objects_present"), uncertain=cond.get("uncertain"), size_given_pres=cond.get("size_given_pres")))
        if k % 40 == 0:
            print(f"[{k}/{len(recs)}] judged", flush=True)
    (out / "j1_size_eval.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    man = AD.build(rows, prompts, src / "images", out, a.seed, quotas={"size": 0}, n_practice=10, n_double=40, types={"size"},
                   args_for_manifest=dict(source_dev_core=str(src), dev_core_run=guard.get("plan", {}).get("manifest_sha256"), n_images=len(rows),
                                          judge_device=dev, evaluator_revisions=rv, purpose="size judge audit (census of DEV size judgements)"))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
