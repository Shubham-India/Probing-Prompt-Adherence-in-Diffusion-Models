"""Compares automatic constraint results with the manual labels in data/validation_labels.json.
Usage: python src/validate_metrics.py --run outputs/main"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_labels():
    return json.loads((ROOT / "data" / "validation_labels.json").read_text())["labels"]


def compare(run, labels):
    names = json.loads((ROOT / "data" / "validation_subset.json").read_text())
    prompts = {p["id"]: p for p in json.loads((ROOT / "data" / "prompts.json").read_text())}
    auto = {json.loads(l)["name"]: json.loads(l) for l in (run / "eval.jsonl").read_text().splitlines()}
    table = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for i, name in enumerate(names):
        rec = auto[name]
        for c, manual, pred in zip(prompts[rec["prompt_id"]]["constraints"], labels[str(i)], rec["constraint_results"]):
            if manual is None:
                continue
            key = ("tp" if pred else "fn") if manual else ("fp" if pred else "tn")
            table[c["type"]][key] += 1
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    for t, v in sorted(compare(run, load_labels()).items()):
        n = sum(v.values())
        print(f"{t:9s} n={n:3d} acc={(v['tp'] + v['tn']) / n:.2f} "
              f"precision={v['tp'] / max(v['tp'] + v['fp'], 1):.2f} recall={v['tp'] / max(v['tp'] + v['fn'], 1):.2f} {v}")
