import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import build_draft as BD


def test_escape_and_formatting():
    assert BD.esc("a_b & 5% #x") == r"a\_b \& 5\% \#x"
    assert BD.ci(0.1, 0.25) == "[0.100, 0.250]" and BD.f3(None) == "n/a"


def test_build_is_traceable_and_never_invents_results():
    R, rows, macros = BD.build(ROOT)
    counts = json.loads((ROOT / "outputs" / "main" / "results" / "counts.json").read_text())
    assert macros["pilotImages"] == counts["images_evaluated"] == 1008
    ids = {r["id"]: r for r in rows}
    for h in ("HA", "HD", "HE", "HF"):
        assert ids[h]["estimate"] == "NOT ESTIMABLE" and ids[h]["status"] == "PLANNED CONFIRMATORY TEST"
    assert all(s["sha256"] != "MISSING" for r in rows for s in r["sources"] if r["id"].startswith(("P", "B")))
    if not (ROOT / "local_audit" / "audit_dev" / "report.json").exists():                # no labels returned: must stay pending
        assert R["dev_audit"] == "PENDING" and "PENDING" in (ROOT / "paper" / "tables" / "dev_audit.tex").read_text()
    assert r"\newcommand{\pilotImages}{1008}" in (ROOT / "paper" / "tables" / "numbers.tex").read_text()


def test_paper_static_checks_pass():
    r = subprocess.run([sys.executable, str(ROOT / "paper" / "check_paper.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
