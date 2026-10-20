"""Judge-vs-human agreement from a filled audit file (RB 12.4). Reports per constraint type: n, accuracy with Wilson 95% CI,
precision/recall, ambiguous share, inter-annotator Cohen's kappa (when two annotation files are given), and the EXPLORATORY rule
(accuracy < 0.85 => that type stays exploratory). Uncertain-flag items are reported as a separate stratum.
Labels: 1 / 0 / amb (or empty = not annotated). With two annotators only items on which both gave 1/0 and agree are used as consensus.
Usage: python src/validate_human.py --key audit_key.csv --a annot_A.csv [--b annot_B.csv]"""
import argparse
import csv
import json
import math
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def kappa(a, b):
    """Cohen's kappa on paired binary labels."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def read_labels(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("label") or "").strip().lower()
            if v in ("1", "0", "amb"):
                out[r["item_id"]] = v
    return out


def analyse(key_rows, la, lb=None):
    stats = defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0, amb=0, n_items=0))
    pairs = []
    for r in key_rows:
        i = r["item_id"]
        if i not in la:
            continue
        labels = [la[i]] + ([lb[i]] if lb and i in lb else [])
        if lb is not None and i in lb:
            if la[i] in "10" and lb[i] in "10" and la[i] != "amb" and lb[i] != "amb":
                pairs.append((int(la[i]), int(lb[i])))
        for t in (r["type"], "ALL", f"unc{r['uncertain']}"):
            s = stats[t]
            s["n_items"] += 1
            if "amb" in labels or (len(labels) == 2 and labels[0] != labels[1]):
                s["amb"] += 1
                continue
            m, p = int(labels[0]), int(r["judge_pred"])
            s["tp" if m and p else "fp" if p else "fn" if m else "tn"] += 1
    rep = {}
    for t, s in sorted(stats.items()):
        n = s["tp"] + s["fp"] + s["fn"] + s["tn"]
        lo, hi = wilson(s["tp"] + s["tn"], n)
        acc = (s["tp"] + s["tn"]) / n if n else float("nan")
        rep[t] = dict(n=n, accuracy=acc, ci95=[lo, hi], precision=s["tp"] / max(s["tp"] + s["fp"], 1), recall=s["tp"] / max(s["tp"] + s["fn"], 1),
                      ambiguous_or_disagree_share=s["amb"] / max(s["n_items"], 1), exploratory=bool(n and acc < 0.85) if t in ("presence", "count", "binding", "spatial") else None, **{k: s[k] for k in ("tp", "fp", "fn", "tn")})
    if lb is not None:
        rep["inter_annotator_kappa"] = dict(n_pairs=len(pairs), kappa=kappa([a for a, _ in pairs], [b for _, b in pairs]))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    x = ap.parse_args()
    key = list(csv.DictReader(open(x.key, newline="")))
    print(json.dumps(analyse(key, read_labels(x.a), read_labels(x.b) if x.b else None), indent=1))
