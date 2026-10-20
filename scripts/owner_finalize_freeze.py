"""OWNER command: finalize the Revision 3 freeze in one step, under the owner's own name. Run it yourself; nothing here may be run for you.
Fail-closed order (stops at the first failure):
 1. variant review: replace the placeholder reviewer name in the review sheets by YOUR name (`adopt_candidate_review.py --rename-reviewer`),
    then `check_variant_review.py` must exit 0;
 2. set `variant_review_complete` = true in configs/stage_gates.json (only after step 1 passed; this file is not bound by the freeze);
 3. sign the drafted protocol (`protocol_r3.py --sign NAME`) and verify it (must print VERIFIED);
 4. commit the listed paths (no `git add .`) and build the release; the build must report dirty=False and VALIDATION: PASSED.
Usage: python scripts/owner_finalize_freeze.py --name "<your full name>" [--no-commit]"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PLACEHOLDER = re.compile(r"your\s*(full\s*)?name|<[^>]*>|^\s*(name|reviewer|owner|tbd|todo|xxx+|n/?a|-+)\s*$", re.I)
COMMIT_PATHS = ["configs/stage_gates.json", "protocol/protocol_freeze.json", "docs/evidence/variant_review", "docs/evidence/audit_20261019",
                "docs/evidence/dev_analysis_20261020", "docs/evidence/hps_guard_20261020", "docs/ADDENDUM_C.md", "docs/R3_DECISION_LOG.md", "docs/DEV_NEXT_STEPS.md",
                "configs/analysis_plan_r3.json", "src/r3_analyze.py", "scripts/final_report.py", "scripts/owner_finalize_freeze.py", "scripts/paper_dev_tables.py",
                "tests/test_r3_analyze.py", "tests/test_final_report.py", "paper"]


def run(*cmd, check=True):
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    print(f"$ {' '.join(map(str, cmd))}\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
    if check and r.returncode != 0:
        sys.exit(f"STOPPED: exit {r.returncode}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-commit", action="store_true")
    a = ap.parse_args()
    name = a.name.strip()
    if not name or PLACEHOLDER.search(name):
        sys.exit("--name must be your real name")
    run(PY, "scripts/adopt_candidate_review.py", "--rename-reviewer", "YOUR FULL NAME", name)
    run(PY, "scripts/check_variant_review.py")
    gp = ROOT / "configs" / "stage_gates.json"
    g = json.loads(gp.read_text())
    g["variant_review_complete"] = True
    gp.write_text(json.dumps(g, indent=2) + "\n")
    print("gate variant_review_complete = true (variant-review checker passed)")
    run(PY, "src/protocol_r3.py", "--sign", name)
    v = run(PY, "src/protocol_r3.py", "--verify")
    if v.stdout.strip().splitlines()[:1] != ["VERIFIED"]:
        sys.exit("STOPPED: protocol does not verify")
    if a.no_commit:
        return
    run("git", "add", "--", *COMMIT_PATHS)
    run("git", "commit", "-m", f"Revision 3 protocol freeze signed by {name}; variant review complete")
    b = run(PY, "scripts/build_release.py")
    if "dirty=False" not in b.stdout or "VALIDATION: PASSED" not in b.stdout:
        sys.exit("STOPPED: release is not clean or did not validate")
    print("FROZEN, committed and released: upload release/guide_before_commitment/source_package.zip as a new gbc-source version")


if __name__ == "__main__":
    main()
