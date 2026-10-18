"""Merge Revision 3 worker outputs against the stage manifest and reject anything that is not exactly the planned, verified set.
Rejects: manifest jobs with no record (missing), records not in the manifest (unexpected), the same job with different image hashes (conflict),
records lacking a file the manifest job requires (image; latents if save_latents; attention if attention) (mismatch),
files whose SHA-256 differs or PNGs that do not decode (corrupt), records whose settings differ from the manifest (config mismatch), model
revision different from the config, and fake-pipeline outputs (unless --allow-fake, used only by the CPU dry-run).
Identical duplicate records of one job (a resumed worker) are collapsed. Output: <out>/merged_records.jsonl (absolute paths of every file;
--copy also copies the files) and <out>/merge_report.json. Exit code 0 only if ok.
Usage: python src/r3_merge.py --manifest stage_manifest.jsonl --inputs DIR [DIR ...] --out MERGED [--copy] [--allow-fake]"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CHECK_KEYS = ("kind", "prompt_id", "seed", "cfg", "steps", "t", "restart", "m", "arm", "noise_seed", "base_job", "split", "scene_id")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def merge(manifest, inputs, out, copy=False, allow_fake=False, revision=None, allow_missing=False):
    plan = {j["job_id"]: j for j in (json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip())}
    seen = defaultdict(list)
    for d in inputs:
        for f in sorted(Path(d).rglob("records_*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    r["_dir"] = str(f.parent)
                    seen[r["job_id"]].append(r)
    rep = dict(manifest_jobs=len(plan), records=sum(len(v) for v in seen.values()), missing=[], unexpected=[], conflicts=[], corrupt=[], mismatch=[],
               fake=[], wrong_revision=[], collapsed_duplicates=0)
    merged = []
    for jid, recs in seen.items():
        if jid not in plan:
            rep["unexpected"].append(jid)
            continue
        if any(not isinstance(r.get("files"), dict) or not r.get("image_sha256") for r in recs):
            rep["corrupt"].append(jid)
            continue
        hashes = {r["image_sha256"] for r in recs}
        if len(hashes) > 1:
            rep["conflicts"].append(jid)
            continue
        rep["collapsed_duplicates"] += len(recs) - 1
        r = recs[0]
        p = plan[jid]
        bad = [k for k in CHECK_KEYS if k in p and p.get(k) != r.get(k)]
        if p.get("kind") == "cag" and r.get("schedule_sha256") != hashlib.sha256(json.dumps(p["schedule"]).encode()).hexdigest():
            bad.append("schedule")
        need = [f"images/{jid}.png"] + ([f"latents/{jid}.pt"] if p.get("save_latents") else []) + ([f"attention/{jid}.npz"] if p.get("attention") else [])
        if any(f not in r["files"] for f in need) or r["files"].get(f"images/{jid}.png") != r["image_sha256"]:
            bad.append("files")
        if bad:
            rep["mismatch"].append({jid: bad})
            continue
        if r.get("fake_pipeline") and not allow_fake:
            rep["fake"].append(jid)
            continue
        if revision and r.get("revision") != revision:
            rep["wrong_revision"].append(jid)
            continue
        abs_files, ok = {}, True
        for rel, h in r["files"].items():
            fp = Path(r["_dir"]) / rel
            if not fp.exists() or sha(fp) != h:
                ok = False
                break
            if rel.endswith(".png"):
                try:
                    with Image.open(fp) as im:
                        im.verify()
                except Exception:
                    ok = False
                    break
            abs_files[rel] = str(fp.resolve())
        if not ok:
            rep["corrupt"].append(jid)
            continue
        if copy:
            for rel, src in list(abs_files.items()):
                dst = Path(out) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                abs_files[rel] = str(dst.resolve())
        merged.append({k: v for k, v in r.items() if k != "_dir"} | {"abs_files": abs_files})
    rep["missing"] = sorted(set(plan) - {r["job_id"] for r in merged} - set(rep["conflicts"]) - set(rep["corrupt"]) - {list(m)[0] for m in rep["mismatch"]} - set(rep["fake"]) - set(rep["wrong_revision"]))
    rep["merged"] = len(merged)
    checks = ("unexpected", "conflicts", "corrupt", "mismatch", "fake", "wrong_revision") + (() if allow_missing else ("missing",))
    rep["ok"] = all(not rep[k] for k in checks) and (allow_missing or len(merged) == len(plan))
    rep["allow_missing_dry_run"] = bool(allow_missing)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "merged_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(merged, key=lambda r: r["job_id"])))
    short = {k: (v[:50] if isinstance(v, list) else v) for k, v in rep.items()}
    (Path(out) / "merge_report.json").write_text(json.dumps(short, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--allow-fake", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="CPU dry-run only (limited subsets); never for research runs")
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment_r3.json"))
    a = ap.parse_args()
    rev = json.loads(Path(a.config).read_text())["revision"]
    if a.allow_missing and not a.allow_fake:
        sys.exit("--allow-missing is only accepted together with --allow-fake (dry-run)")
    rep = merge(a.manifest, a.inputs, a.out, a.copy, a.allow_fake, None if a.allow_fake else rev, a.allow_missing)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()}, indent=1))
    sys.exit(0 if rep["ok"] else 1)
