"""Variant / lexicon review sheets (H3): the builder covers every prompt and lexicon entry without touching the benchmark, its automatic checks
fire on broken prompts, and the checker refuses incomplete, edited or stale reviews and applies the < 20% rejection criterion."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_variant_review as V  # noqa: E402

PY = sys.executable


def test_rows_cover_every_prompt_and_lexicon_entry_and_leave_the_benchmark_unchanged():
    before = {f: V.sha(V.B / f) for f in V.INPUTS}
    rows, lrows, meta = V.build_rows()
    ids = [p["id"] for f in ("prompts_v2.json", "prompts_ps.json") for p in json.loads((V.B / f).read_text())]
    assert sorted(r["item_id"] for r in rows) == sorted(ids) and len(set(ids)) == len(ids)
    L = json.loads((V.B / "lexicon.json").read_text())
    assert len(lrows) == len(L["nouns"]) + len(L["colours"]) + len(L["colour_synonyms"]) + len(L["noun_synonyms"]) + len(L["relations"]) \
        + len(L["relation_synonyms"]) + len(L["conflicts"])
    assert all(r["decision"] == "" for r in rows + lrows)
    assert {f: V.sha(V.B / f) for f in V.INPUTS} == before == meta["inputs"]
    assert V.build_rows()[0] == rows                                  # deterministic (the checker rebuilds and compares)


def test_checks_fire_on_broken_prompts():
    lex = V.Lex(json.loads((V.B / "lexicon.json").read_text()))
    can = dict(id="x_CAN", variant="CAN", prompt="an orange book and a red cup", n_constraints=4, n_objects=2,
               constraints=[{"type": "presence", "object": "book"}, {"type": "presence", "object": "cup"},
                            {"type": "binding", "object": "book", "color": "orange"}, {"type": "binding", "object": "cup", "color": "red"}])
    assert V.text_checks(can, lex) == []
    assert any(f.startswith("TEXT-BIND") for f in V.text_checks(dict(can, prompt="a red book and an orange cup"), lex))
    assert any(f.startswith("GRAMMAR") for f in V.text_checks(dict(can, prompt="a orange book and a red cup"), lex))
    assert any(f.startswith("TEXT-EXTRA") for f in V.text_checks(dict(can, prompt="an orange book and a red blue cup"), lex))
    rel = dict(can, prompt="an orange book to the left of a red cup", constraints=can["constraints"] + [{"type": "spatial", "left": "cup", "right": "book"}])
    assert any(f.startswith("TEXT-REL") for f in V.text_checks(rel, lex))
    cfc = dict(can, id="x_CFC", variant="CF-C", constraints=[dict(c, color="blue") if c["type"] == "binding" else c for c in can["constraints"]])
    assert any("new colour" in f for f in V.change_checks(cfc, can, lex)[1])
    par = dict(can, id="x_PAR", variant="PAR", prompt="an orange tome and a red cup")
    assert any("not in the lexicon" in f for f in V.change_checks(par, can, lex)[1])
    ord_ = dict(can, id="x_ORD", variant="ORD", prompt="a red cup and an orange book")
    assert V.change_checks(ord_, can, lex)[1] == []


@pytest.fixture()
def review(tmp_path):
    out = tmp_path / "vr"
    r = subprocess.run([PY, str(ROOT / "scripts" / "build_variant_review.py"), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def fill(path, decide):
    rows = V.read_csv(path)
    for i, r in enumerate(rows):
        r.update(decide(i, r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def ok(i, r):
    return dict(decision="PASS", reason="flag checked, not invalid" if r["auto_flags"] else "", reviewer="owner", review_date="2026-10-19")


def check(out):
    r = subprocess.run([PY, str(ROOT / "scripts" / "check_variant_review.py"), "--dir", str(out)], capture_output=True, text=True)
    return r.returncode, json.loads((out / "variant_review_evidence.json").read_text())


def test_empty_review_is_refused(review):
    rc, ev = check(review)
    assert rc == 1 and ev["complete"] is False and ev["criteria_met"] is False


def test_complete_passing_review_meets_the_criterion(review):
    fill(review / "variant_review.csv", ok)
    fill(review / "lexicon_review.csv", ok)
    rc, ev = check(review)
    assert rc == 0 and ev["complete"] and ev["criteria_met"] and ev["rejection_rate"] == 0
    gates = json.loads((ROOT / "configs" / "stage_gates.json").read_text())
    assert gates["variant_review_complete"] is False                   # the checker never sets a gate


@pytest.mark.parametrize("decide, needle", [
    (lambda i, r: dict(ok(i, r), reason="") if r["auto_flags"] else ok(i, r), "PASS on an auto-flagged row"),
    (lambda i, r: dict(ok(i, r), decision="FAIL", reason="") if i == 0 else ok(i, r), "FAIL without a reason"),
    (lambda i, r: dict(ok(i, r), decision="maybe") if i == 0 else ok(i, r), "must be PASS, FAIL or AMBIGUOUS"),
    (lambda i, r: dict(ok(i, r), prompt=r["prompt"] + " x") if i == 0 else ok(i, r), "non-decision columns edited"),
    (lambda i, r: dict(ok(i, r), reviewer="") if i == 0 else ok(i, r), "reviewer or review_date missing"),
])
def test_invalid_reviews_are_refused(review, decide, needle):
    fill(review / "variant_review.csv", decide)
    fill(review / "lexicon_review.csv", ok)
    rc, ev = check(review)
    assert rc == 1 and not ev["criteria_met"] and any(needle in p for p in ev["problems"]), ev["problems"][:3]


def test_rejection_threshold_and_ambiguous(review):
    n = len(V.read_csv(review / "variant_review.csv"))
    fill(review / "lexicon_review.csv", ok)
    fill(review / "variant_review.csv", lambda i, r: dict(ok(i, r), decision="FAIL", reason="invalid") if i < int(0.2 * n) + 1 else ok(i, r))
    rc, ev = check(review)
    assert rc == 1 and ev["complete"] and not ev["criteria_met"] and ev["rejection_rate"] >= 0.2
    fill(review / "variant_review.csv", lambda i, r: dict(ok(i, r), decision="AMBIGUOUS", reason="unclear") if i == 0 else ok(i, r))
    rc, ev = check(review)
    assert rc == 1 and ev["complete"] and not ev["criteria_met"] and ev["requires_owner_action"]["ambiguous_items"]


def test_stale_review_is_refused(review):
    fill(review / "variant_review.csv", ok)
    fill(review / "lexicon_review.csv", ok)
    meta = json.loads((review / "review_inputs.json").read_text())
    meta["inputs"]["prompts_v2.json"] = "0" * 64
    (review / "review_inputs.json").write_text(json.dumps(meta))
    rc, ev = check(review)
    assert rc == 1 and any("benchmark file changed" in p for p in ev["problems"])


def test_builder_never_overwrites_decisions_and_carries_over(review, tmp_path):
    fill(review / "variant_review.csv", ok)
    fill(review / "lexicon_review.csv", ok)
    r = subprocess.run([PY, str(ROOT / "scripts" / "build_variant_review.py"), "--out", str(review)], capture_output=True, text=True)
    assert r.returncode != 0 and "exists" in (r.stdout + r.stderr)
    new = tmp_path / "vr_new"
    r = subprocess.run([PY, str(ROOT / "scripts" / "build_variant_review.py"), "--out", str(new), "--carry-over", str(review)], capture_output=True, text=True)
    assert r.returncode == 0
    assert all(x["decision"] == "PASS" for x in V.read_csv(new / "variant_review.csv"))
