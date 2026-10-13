"""Blinded, stratified human-audit sample (RB 12.4). Unit = one (image, constraint) judgement.
Strata: constraint type x judge prediction (0/1) x uncertain flag. Draw is seeded and reproducible; uncertain items are oversampled
and flagged in the key so headline accuracy can be reported with and without them.
Exports (never containing arm/variant/CFG/judge verdict in the annotator-facing file):
  <out>/audit_items.csv   item_id, image_file, question, label (empty, to fill: 1 / 0 / amb)
  <out>/audit_key.csv     item_id, name, constraint_index, type, judge_pred, uncertain, stratum, cfg, steps, variant   (keep away from annotators)
  <out>/images/<item_id>.png  copies, if the run has images
Usage: python src/audit_sample.py --run outputs/x --eval eval.jsonl --prompts data/prompts.json --out outputs/x/audit --n 300 --seed 1"""
import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def question(c):
    t = c["type"]
    if t == "presence":
        return f"Is there at least one {c['object']} in the image?"
    if t == "count":
        return f"Are there exactly {c['n']} {c['object']}(s) in the image?"
    if t == "binding":
        return f"Is the {c['object']} {c['color']}? (if several {c['object']}s are shown: are most of them {c['color']}?)"
    if t == "spatial":
        return f"Is the {c['left']} to the LEFT of the {c['right']} (comparing their horizontal centres)?"
    raise ValueError(t)


def draw(rows, prompts, n, seed, uncertain_share=0.25):
    items = []
    for r in rows:
        cons = prompts[r["prompt_id"]]["constraints"]
        for i, (c, pred) in enumerate(zip(cons, r["constraint_results"])):
            items.append(dict(name=r["name"], i=i, type=c["type"], pred=int(pred), unc=int(r.get("uncertain", 0)), c=c,
                              cfg=r.get("cfg"), steps=r.get("steps"), variant=r.get("variant", r.get("category"))))
    rng = random.Random(seed)
    strata = defaultdict(list)
    for it in items:
        strata[(it["type"], it["pred"], it["unc"])].append(it)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    n_unc = min(int(n * uncertain_share), sum(len(v) for k, v in strata.items() if k[2] == 1))   # no uncertain items => whole quota goes to the others
    chosen = []
    for want_unc, quota in ((1, n_unc), (0, n - n_unc)):
        ks = [k for k in keys if k[2] == want_unc]
        taken = {k: 0 for k in ks}
        while sum(taken.values()) < quota and ks:           # round-robin over non-empty strata
            progressed = False
            for k in ks:
                if sum(taken.values()) >= quota:
                    break
                if taken[k] < len(strata[k]):
                    taken[k] += 1
                    progressed = True
            if not progressed:
                break
        for k in ks:
            chosen += strata[k][: taken[k]]
    rng.shuffle(chosen)
    return chosen


def export(run, rows, prompts, out, n, seed):
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    chosen = draw(rows, prompts, n, seed)
    rng = random.Random(seed + 1)
    ids = rng.sample(range(10**6), len(chosen))
    with open(out / "audit_items.csv", "w", newline="") as fi, open(out / "audit_key.csv", "w", newline="") as fk:
        wi, wk = csv.writer(fi), csv.writer(fk)
        wi.writerow(["item_id", "image_file", "question", "label"])
        wk.writerow(["item_id", "name", "constraint_index", "type", "judge_pred", "uncertain", "stratum", "cfg", "steps", "variant"])
        for it, num in zip(chosen, ids):
            iid = f"a{num:06d}"
            src = Path(run) / "images" / f"{it['name']}.png"
            if src.exists():
                shutil.copy(src, out / "images" / f"{iid}.png")
            wi.writerow([iid, f"images/{iid}.png", question(it["c"]), ""])
            wk.writerow([iid, it["name"], it["i"], it["type"], it["pred"], it["unc"], f"{it['type']}|pred{it['pred']}|unc{it['unc']}", it["cfg"], it["steps"], it["variant"]])
    return len(chosen)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", default="eval.jsonl")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run = Path(a.run)
    rows = [json.loads(l) for l in (run / a.eval).read_text().splitlines()]
    prompts = {p["id"]: p for p in json.loads(Path(a.prompts).read_text())}
    print("exported", export(run, rows, prompts, a.out, a.n, a.seed), "items")
