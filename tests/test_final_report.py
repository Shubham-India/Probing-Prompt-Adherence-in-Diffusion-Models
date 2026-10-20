"""scripts/final_report.py renders tables, figures and the generated TEST section from a confirmatory.json, and refuses non-confirmatory input."""
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / "kaggle")]
DRY = sorted(glob.glob(str(ROOT / "local_audit" / "dryrun*" / "work_04_test" / "final_*")))


def test_refuses_dry_run_output():
    if not DRY:
        pytest.skip("no local dry-run TEST output")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "final_report.py"), "--final", DRY[-1]], capture_output=True, text=True)
    assert r.returncode != 0 and "not CONFIRMATORY" in (r.stdout + r.stderr)


def test_renders_from_a_confirmatory_result(tmp_path, monkeypatch):
    if not DRY:
        pytest.skip("no local dry-run TEST output")
    import final_report as FR
    import protocol_r3
    src = Path(DRY[-1])
    d = tmp_path / "final"
    shutil.copytree(src, d, ignore=shutil.ignore_patterns("merged", "merge_in", "attention_all", "*.npz"))
    c = json.loads((d / "confirmatory.json").read_text())
    c["status"] = "CONFIRMATORY"                              # rendering test only: the real input must come from a frozen TEST run
    (d / "confirmatory.json").write_text(json.dumps(c))
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "paper", root / "paper")
    monkeypatch.setattr(FR, "ROOT", root)
    for k, sub in (("TABLES", "paper/tables"), ("FIGS", "paper/figures"), ("SECS", "paper/sections"), ("REPORTS", "reports")):
        monkeypatch.setattr(FR, k, root / sub)
    monkeypatch.setattr(protocol_r3, "verify", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", ["final_report.py", "--final", str(d)])
    with pytest.raises(SystemExit) as e:
        FR.main()
    for t in ("test_hypotheses", "test_ha", "test_hd", "test_he", "test_hf", "test_s2"):
        assert (root / "paper" / "tables" / f"{t}.tex").exists(), t
    sec = (root / "paper" / "sections" / "results_test.tex").read_text(encoding="utf-8")
    assert r"\input{tables/test_hypotheses}" in sec
    main_tex = (root / "paper" / "main.tex").read_text(encoding="utf-8")
    assert "sections/results_test" in main_tex and "sections/results_HA" not in main_tex
    assert json.loads((root / "reports" / "final_results.json").read_text())["results"]["HA"]
    assert (ROOT / "paper" / "main.tex").read_text(encoding="utf-8").count("results_test") == 0    # the real manuscript is untouched
