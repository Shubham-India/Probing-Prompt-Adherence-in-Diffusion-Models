"""Candidate review (proposals) and owner adoption: proposals never touch the human sheets; adoption needs the owner's name and date, never
overwrites a decision, skips candidates whose owner question is unanswered, refuses stale candidates; FAIL candidates keep the checker from passing."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_variant_review as V  # noqa: E402

PY = sys.executable


def run(script, *args):
    return subprocess.run([PY, str(ROOT / "scripts" / script), *map(str, args)], capture_output=True, text=True)


@pytest.fixture(scope="module")
def proposed(tmp_path_factory):
    d = tmp_path_factory.mktemp("vr")
    assert run("build_variant_review.py", "--out", d).returncode == 0
    r = run("propose_variant_review.py", "--dir", d)
    assert r.returncode == 0, r.stderr[-2000:]
    return d


def fresh(proposed, tmp_path):
    d = tmp_path / "vr"
    d.mkdir()
    for f in ("variant_review.csv", "lexicon_review.csv", "review_inputs.json", "candidate_review.csv", "candidate_summary.json"):
        (d / f).write_bytes((proposed / f).read_bytes())
    return d


def test_proposals_cover_every_row_and_leave_the_sheets_empty(proposed):
    cand = V.read_csv(proposed / "candidate_review.csv")
    rows, lrows, _ = V.build_rows()
    assert sorted(c["item_id"] for c in cand) == sorted([r["item_id"] for r in rows] + [r["item_id"] for r in lrows])
    assert all(c["proposed_decision"] in ("PASS", "FAIL", "AMBIGUOUS") and c["proposed_reason"] for c in cand)
    assert all(not r["decision"] and not r["reviewer"] for r in V.read_csv(proposed / "variant_review.csv"))
    prompts = [c["prompt"] for c in cand if c["sheet"] == "variant_review"]
    assert not any("golden" in t.split() or "side of" in t for t in prompts)          # amendment 2026-10-18 (Q1=A)
    lemon = [c for c in cand if c["sheet"] == "variant_review" and "lemon" in c["prompt"].split()]
    rel = [c for c in cand if c["sheet"] == "variant_review" and ("on the left of" in c["prompt"] or "on the right of" in c["prompt"])]
    assert len(lemon) == 47 and len(rel) == 82
    assert all(c["proposed_decision"] == "PASS" and "Q1" in c["depends_on"] for c in lemon + rel)
    assert all("R-REL-SYN" in c["rules"] for c in rel)


def test_adoption_requires_owner_identity(proposed, tmp_path):
    d = fresh(proposed, tmp_path)
    assert run("adopt_candidate_review.py", "--dir", d, "--reviewer", "", "--date", "2026-10-19").returncode != 0
    assert run("adopt_candidate_review.py", "--dir", d, "--reviewer", "owner", "--date", "19/10/2026").returncode != 0


def test_adoption_skips_unanswered_questions_and_never_overwrites(proposed, tmp_path):
    d = fresh(proposed, tmp_path)
    rows = V.read_csv(d / "variant_review.csv")
    keep = rows[0]["item_id"]
    rows[0].update(decision="FAIL", reason="my own call", reviewer="owner", review_date="2026-10-18")
    import csv
    with open(d / "variant_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=V.FIELDS)
        w.writeheader()
        w.writerows(rows)
    r = run("adopt_candidate_review.py", "--dir", d, "--reviewer", "owner", "--date", "2026-10-19")
    assert r.returncode == 0, r.stderr
    after = {x["item_id"]: x for x in V.read_csv(d / "variant_review.csv")}
    assert after[keep]["reason"] == "my own call"
    cand = {c["item_id"]: c for c in V.read_csv(d / "candidate_review.csv")}
    for i, x in after.items():
        if i == keep:
            continue
        if cand[i]["depends_on"]:
            assert x["decision"] == ""                                  # Q1 / Q2 not answered
        else:
            assert x["decision"] == cand[i]["proposed_decision"] and x["reason"].startswith("[adopted candidate") and x["reviewer"] == "owner"


def test_full_adoption_in_a_copy_is_accepted_by_the_unchanged_checker(proposed, tmp_path):
    d = fresh(proposed, tmp_path)
    assert run("adopt_candidate_review.py", "--dir", d, "--reviewer", "owner", "--date", "2026-10-19", "--answers", "Q1=A,Q2=A").returncode == 0
    r = run("check_variant_review.py", "--dir", d)
    ev = json.loads((d / "variant_review_evidence.json").read_text())
    assert r.returncode == 0 and ev["complete"] is True and ev["criteria_met"] is True and ev["rejection_rate"] == 0
    assert all(x["reason"].startswith("[adopted candidate") for x in V.read_csv(d / "variant_review.csv"))


def test_partial_adoption_without_answers_is_not_accepted(proposed, tmp_path):
    d = fresh(proposed, tmp_path)
    assert run("adopt_candidate_review.py", "--dir", d, "--reviewer", "owner", "--date", "2026-10-19").returncode == 0
    r = run("check_variant_review.py", "--dir", d)
    assert r.returncode == 1 and json.loads((d / "variant_review_evidence.json").read_text())["complete"] is False


def test_stale_candidates_are_refused(proposed, tmp_path):
    d = fresh(proposed, tmp_path)
    s = json.loads((d / "candidate_summary.json").read_text())
    s["benchmark_inputs"]["prompts_v2.json"] = "0" * 64
    (d / "candidate_summary.json").write_text(json.dumps(s))
    r = run("adopt_candidate_review.py", "--dir", d, "--reviewer", "owner", "--date", "2026-10-19")
    assert r.returncode != 0 and "different benchmark files" in (r.stdout + r.stderr)


def test_placeholder_reviewer_is_refused_and_can_be_renamed(proposed, tmp_path):
    d = fresh(proposed, tmp_path)
    assert run("adopt_candidate_review.py", "--dir", d, "--reviewer", "YOUR FULL NAME", "--date", "2026-10-19", "--answers", "Q1=A,Q2=A").returncode != 0
    import csv
    for sheet, fields in (("variant_review.csv", V.FIELDS), ("lexicon_review.csv", V.LEX_FIELDS)):    # as if adopted with the placeholder
        rows = V.read_csv(d / sheet)
        cand = {c["item_id"]: c for c in V.read_csv(d / "candidate_review.csv")}
        for r in rows:
            r.update(decision=cand[r["item_id"]]["proposed_decision"], reason="[adopted candidate] x", reviewer="YOUR FULL NAME", review_date="2026-10-19")
        with open(d / sheet, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    r = run("check_variant_review.py", "--dir", d)
    ev = json.loads((d / "variant_review_evidence.json").read_text())
    assert r.returncode == 1 and any("placeholder" in p for p in ev["problems"])
    assert run("adopt_candidate_review.py", "--dir", d, "--rename-reviewer", "YOUR FULL NAME", "<your name>").returncode != 0
    assert run("adopt_candidate_review.py", "--dir", d, "--rename-reviewer", "YOUR FULL NAME", "A. Person").returncode == 0
    assert run("check_variant_review.py", "--dir", d).returncode == 0
    assert {x["reviewer"] for x in V.read_csv(d / "variant_review.csv")} == {"A. Person"}
