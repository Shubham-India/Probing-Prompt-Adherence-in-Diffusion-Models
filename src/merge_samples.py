"""Merge samples_shard*.jsonl (and an existing samples.jsonl) of one run into samples.jsonl.
Identical duplicate records are collapsed; conflicting duplicates (same name, different image hash) abort the merge.
Usage: python src/merge_samples.py --run outputs/x [--out samples.jsonl]"""
import argparse
import json
import sys
from pathlib import Path


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def merge(run, out_name="samples.jsonl"):
    run = Path(run)
    files = sorted(run.glob("samples_shard*of*.jsonl")) + ([run / "samples.jsonl"] if (run / "samples.jsonl").exists() and out_name != "samples.jsonl" else [])
    if out_name == "samples.jsonl" and (run / "samples.jsonl").exists():
        files.append(run / "samples.jsonl")
    by_name, conflicts = {}, []
    for f in files:
        for r in read(f):
            old = by_name.get(r["name"])
            if old is None:
                by_name[r["name"]] = r
            elif old.get("image_sha256") != r.get("image_sha256"):
                conflicts.append(r["name"])
    if conflicts:
        raise SystemExit(f"conflicting duplicates (different image hash): {conflicts[:10]}")
    tmp = run / (out_name + ".tmp")
    tmp.write_text("".join(json.dumps(by_name[n]) + "\n" for n in sorted(by_name)))
    tmp.replace(run / out_name)
    return len(by_name), len(files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="samples.jsonl")
    a = ap.parse_args()
    n, f = merge(a.run, a.out)
    print(f"merged {f} files into {a.run}/{a.out}: {n} unique samples")
