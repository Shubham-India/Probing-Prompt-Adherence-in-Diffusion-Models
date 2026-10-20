"""Closes a Kaggle (or local) generation stage: runs the matching verifier, writes checksums and a compact machine-readable run report, and only if
verification passed writes the COMPLETE marker and the ZIP to download. On failure it still writes run_report.json (ok=false, reasons) but NO marker,
so a half-finished folder can never be mistaken for a finished run.
Usage: python repo/kaggle/finish_run.py --stage {r2a,r2b,r3} --run-dir <dir with images/ + samples*.jsonl> --config repo/configs/experiment_v2.json
         --prompts repo/data/benchmark_v2/prompts_v2.json --run-id <RUN> [--split DEV] [--variants CAN] [--latents]
         [--core-run <dir>] [--branch-variants D8 ER30 FULL] [--M 2] [--setup-report <setup_report.json>] [--zip-out /kaggle/working/<RUN>.zip]"""
import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run(a):
    run_dir = Path(a.run_dir)
    cfg = json.loads(Path(a.config).read_text())
    import jobs as J
    prompts = J.load_prompts(a.prompts)
    report = dict(stage=a.stage, run_id=a.run_id, utc=datetime.now(timezone.utc).isoformat(), run_dir=str(run_dir), ok=False, errors=[])
    if a.setup_report and Path(a.setup_report).exists():
        s = json.loads(Path(a.setup_report).read_text())
        report.update(commit=s.get("commit_in_dataset") or s.get("commit_checked_out"), requirements_sha256=s.get("requirements_sha256"))
    report["config_sha256"] = hashlib.sha256(Path(a.config).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    report["model_revision"] = cfg.get("revision")
    if a.stage in ("r2a", "r2b"):
        import verify_outputs
        if a.split:
            prompts = [p for p in prompts if p.get("split") == a.split]
        if a.variants:
            prompts = [p for p in prompts if p.get("variant") in set(a.variants.split(","))]
        errs, stats = verify_outputs.verify(run_dir, cfg, prompts, need_latents=a.latents)
    else:
        import verify_branches
        errs, stats = verify_branches.verify(run_dir, a.core_run, cfg, prompts, a.branch_variants, a.M, a.split, check_core_latents=True)
    report["verifier_stats"], report["errors"] = stats, errs[:50]
    files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name not in ("checksums.sha256", "run_report.json", "COMPLETE"))
    report["files"] = len(files)
    report["bytes"] = sum(p.stat().st_size for p in files)
    if not errs:
        (run_dir / "checksums.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(run_dir).as_posix()}\n" for p in files))
        report["ok"] = True
    (run_dir / "run_report.json").write_text(json.dumps(report, indent=1))
    if report["ok"]:
        (run_dir / "COMPLETE").write_text(a.run_id + "\n")
        if a.zip_out:
            with zipfile.ZipFile(a.zip_out, "w", zipfile.ZIP_STORED) as z:
                for p in sorted(p for p in run_dir.rglob("*") if p.is_file()):
                    z.write(p, p.relative_to(run_dir).as_posix())
            report["zip"] = str(a.zip_out)
    print(json.dumps({k: report[k] for k in ("stage", "run_id", "ok", "verifier_stats", "files", "bytes")} | {"errors": report["errors"][:5]}, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["r2a", "r2b", "r3"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--variants", default=None, help="comma-separated prompt variants for r2a/r2b")
    ap.add_argument("--latents", action="store_true")
    ap.add_argument("--core-run", default=None)
    ap.add_argument("--branch-variants", nargs="+", default=["D8", "ER30", "FULL"])
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--setup-report", default=None)
    ap.add_argument("--zip-out", default=None)
    sys.exit(run(ap.parse_args()))
