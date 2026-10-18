"""Revision 3 protocol freeze (Addendum C). Writes protocol/protocol_freeze.json, which TEST workers and the confirmatory analysis verify before
they run. It binds: the code (src/*.py, kaggle/*.py), the configs (experiment_r3.json, analysis_plan_r3.json, benchmark_v2.json), every benchmark and
split file, the judge definitions, the DEV-derived constants (t50 per outcome kind at w = 7.5, s* rule inputs, chosen w_lo), the evaluator-audit
verdict per constraint type, the TEST manifests' hashes and the human-readable docs/ADDENDUM_C.md.
  python src/protocol_r3.py --draft  --constants dev_constants.json --audit audit_report.json     -> protocol/protocol_freeze.json (status DRAFT)
  python src/protocol_r3.py --sign "Owner Name"                                                   -> status FROZEN (refused while any check fails)
  python src/protocol_r3.py --verify                                                              -> exit 0 only if FROZEN and nothing changed
All hashes are SHA-256 of LF-normalised text (Windows and Linux copies agree)."""
import argparse
import os
import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "protocol" / "protocol_freeze.json"
BOUND_GLOBS = ["src/*.py", "kaggle/*.py", "scripts/*.py", "configs/experiment_r3.json", "configs/analysis_plan_r3.json", "configs/benchmark_v2.json",
               "data/benchmark_v2/scene_graphs.json", "data/benchmark_v2/prompts_v2.json", "data/benchmark_v2/splits.json", "data/benchmark_v2/lexicon.json",
               "data/benchmark_v2/ps_family.json", "data/benchmark_v2/prompts_ps.json", "data/benchmark_v2/splits_ps.json", "docs/ADDENDUM_C.md",
               "requirements-kaggle.txt"]
KINDS = ["presence", "count", "bind_given_pres", "spatial_given_pres", "size_given_pres"]


def sha_text(path):
    b = Path(path).read_bytes()
    if Path(path).suffix in (".py", ".json", ".md", ".txt", ".jsonl", ".csv"):
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def bound_files(root=ROOT):
    files = []
    for g in BOUND_GLOBS:
        files += [Path(p).relative_to(root).as_posix() for p in glob.glob(str(root / g))]
    return sorted(set(files))


def judge_definition():
    import evaluate as E
    return dict(thresholds=E.THRESHOLDS, size_ratio=E.SIZE_RATIO, uncertain_band=E.UNCERTAIN_BAND, colours=E.COLORS)


def check_constants(c):
    problems = []
    t50 = (c or {}).get("t50_dev_w7.5")
    if not t50:
        return ["constants: t50_dev_w7.5 missing"]
    for k in KINDS[:4]:      # size is needed only for prompts with a size constraint; CAG runs on main-benchmark prompts, which have none
        if k not in t50:
            problems.append(f"constants: t50 for {k} missing (use '>36' when unresolved, never omit)")
    if c.get("w_lo") not in (1.0, 3.0):
        problems.append("constants: w_lo must be 1.0 (primary) or 3.0 (pre-registered secondary)")
    return problems


def check_audit(a):
    if not a:
        return ["evaluator audit verdicts missing"]
    probs = []
    for t in ("presence", "count", "binding", "spatial", "size"):
        if t not in a or "criterion_met" not in a[t]:
            probs.append(f"audit: no verdict for {t}")
    return probs


def manifest_hashes(constants):
    import manifests as M
    jobs, notes, _ = M.build(constants=constants)
    return {st: M.manifest_hash(M.select(jobs, st)) for st in ("TEST_CORE", "TEST_COMMIT", "TEST_CAG_SIZE")}


def body_hash(d):
    core = {k: v for k, v in d.items() if k not in ("status", "approved_by", "approved_utc", "freeze_sha256")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def draft(constants, audit, out=DEFAULT, root=ROOT, dry_run=False):
    sys.path.insert(0, str(root / "src"))
    d = dict(schema="protocol-r3-1", created_utc=datetime.now(timezone.utc).isoformat(), status="DRAFT", approved_by=None, approved_utc=None,
             files={f: sha_text(root / f) for f in bound_files(root)}, judge=judge_definition(), constants=constants, audit=audit,
             test_manifests=manifest_hashes(constants) if not check_constants(constants) else None,
             hypotheses=["HA", "HD", "HE", "HF"], multiplicity="Holm over HA, HD, HE, HF (configs/analysis_plan_r3.json)", dry_run=bool(dry_run))
    d["problems_at_draft"] = check_constants(constants) + check_audit(audit)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(d, indent=1, sort_keys=True))
    return d


def verify(path=DEFAULT, root=ROOT):
    p = Path(path)
    if not p.exists():
        return [f"{p} missing"]
    try:
        d = json.loads(p.read_text())
    except ValueError as e:
        return [f"{p} is not valid JSON: {e}"]
    if not isinstance(d, dict):
        return [f"{p} is not a JSON object"]
    problems = []
    if d.get("dry_run") and os.environ.get("GBC_ALLOW_DRYRUN_PROTOCOL") != "1":
        problems.append("dry-run protocol (fake constants) is never valid for real runs")
    if d.get("status") != "FROZEN" or not d.get("approved_by"):
        problems.append("protocol not signed (status must be FROZEN with approved_by)")
    if d.get("freeze_sha256") != body_hash(d):
        problems.append("freeze_sha256 does not match the protocol body (file edited after signing)")
    now = {f: sha_text(root / f) for f in bound_files(root)}
    for f, h in d.get("files", {}).items():
        if f not in now:
            problems.append(f"bound file missing: {f}")
        elif now[f] != h:
            problems.append(f"changed after freeze: {f}")
    problems += [f"new file not bound: {f}" for f in now if f not in d.get("files", {})]
    problems += check_constants(d.get("constants")) + check_audit(d.get("audit"))
    if d.get("test_manifests") is None:
        problems.append("TEST manifest hashes missing")
    return problems


def sign(name, path=DEFAULT):
    d = json.loads(Path(path).read_text())
    d["status"], d["approved_by"], d["approved_utc"] = "FROZEN", name, datetime.now(timezone.utc).isoformat()
    d["freeze_sha256"] = body_hash(d)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
    problems = verify(tmp)
    if problems:
        tmp.unlink()
        sys.exit("refusing to sign:\n  " + "\n  ".join(problems[:20]))
    tmp.replace(path)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--constants", default=None)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--sign", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--path", default=str(DEFAULT))
    ap.add_argument("--dry-run", action="store_true", help="CPU dry-run only: marks the protocol dry_run (rejected unless GBC_ALLOW_DRYRUN_PROTOCOL=1)")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    if a.draft:
        c = json.loads(Path(a.constants).read_text()) if a.constants else None
        au = json.loads(Path(a.audit).read_text()) if a.audit else None
        d = draft(c, au, a.path, dry_run=a.dry_run)
        print(json.dumps(dict(status=d["status"], files=len(d["files"]), problems=d["problems_at_draft"]), indent=1))
    elif a.sign:
        sign(a.sign, a.path)
        print("FROZEN")
    else:
        p = verify(a.path)
        print("VERIFIED" if not p else "NOT VERIFIED:\n  " + "\n  ".join(p[:30]))
        sys.exit(1 if p else 0)
