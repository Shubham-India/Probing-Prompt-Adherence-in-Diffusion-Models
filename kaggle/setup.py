"""Kaggle setup cell logic (R0/R1/...). Obtains the code at an EXACT commit, installs pins, creates the run folder.
Two code sources (the repository is private; never make it public to simplify this):
  A) --source dataset : code uploaded as a private Kaggle dataset (see docs/EXPERIMENT_RUNBOOK.md 3b); no token needed
  B) --source github  : clone with a read token taken from the Kaggle Secret named GITHUB_READ_TOKEN (never printed or logged)
Usage (in a notebook cell):  !python setup.py --source dataset --dataset-dir /kaggle/input/<name> --commit <SHA> --run-id <ID>
This script is UNTESTED on Kaggle; the output `setup_report.json` states what actually happened."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dataset", "github"], required=True)
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--repo", default="Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workdir", default="/kaggle/working")
    a = ap.parse_args()
    work = Path(a.workdir)
    code = work / "repo"
    report = {"source": a.source, "commit_requested": a.commit, "run_id": a.run_id}
    if a.source == "dataset":
        src = Path(a.dataset_dir)
        assert (src / "COMMIT").exists(), "dataset must contain a COMMIT file written by make_bundle"
        report["commit_in_dataset"] = (src / "COMMIT").read_text().strip()
        assert report["commit_in_dataset"] == a.commit, "dataset commit differs from the requested commit"
        run(["cp", "-r", str(src / "repo"), str(code)])
    else:
        token = os.environ.get("GITHUB_READ_TOKEN")
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                token = UserSecretsClient().get_secret("GITHUB_READ_TOKEN")
            except Exception:
                sys.exit("GITHUB_READ_TOKEN secret not available; use --source dataset")
        run(["git", "clone", "-q", f"https://x-access-token:{token}@github.com/{a.repo}.git", str(code)])
        run(["git", "-C", str(code), "remote", "set-url", "origin", f"https://github.com/{a.repo}.git"])   # drop the token from .git/config
        run(["git", "-C", str(code), "checkout", "-q", a.commit])
        report["commit_checked_out"] = run(["git", "-C", str(code), "rev-parse", "HEAD"]).stdout.strip()
        assert report["commit_checked_out"].startswith(a.commit[:7])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(code / "requirements-kaggle.txt")])
    outdir = work / "outputs" / a.run_id
    outdir.mkdir(parents=True, exist_ok=True)
    report["outdir"] = str(outdir)
    report["requirements_sha256"] = hashlib.sha256((code / "requirements-kaggle.txt").read_bytes()).hexdigest()
    (outdir / "setup_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
