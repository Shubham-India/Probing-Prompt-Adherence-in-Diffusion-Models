"""OWNER tool: adopt candidate decisions (candidate_review.csv) into the human review sheets. Run it yourself, with your own name and date; it is
how you accept the proposed decisions you agree with. It never overwrites a decision already in the sheets, never adopts a candidate that depends on
an owner question you have not answered (--answers), never adopts AMBIGUOUS candidates, and refuses when the candidates were built from different
benchmark files than the sheets. Each adopted reason is prefixed "[adopted candidate <rules>]" so the evidence shows what was accepted in bulk.
Usage: python scripts/adopt_candidate_review.py --reviewer "<your name>" --date YYYY-MM-DD [--answers Q1=A,Q2=A] [--only-rules R-OK,R-LEX-NOUN]
       [--exclude ITEM_ID ...] [--dry-run]
Then: python scripts/check_variant_review.py"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_variant_review as V  # noqa: E402

D = ROOT / "docs" / "evidence" / "variant_review"


def write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--answers", default="", help="answered owner questions, e.g. Q1=A,Q2=A (see OWNER_DECISIONS.md)")
    ap.add_argument("--only-rules", default="", help="comma-separated rule ids; adopt only candidates whose rules are all in this list")
    ap.add_argument("--exclude", nargs="*", default=[], help="item ids you want to decide yourself")
    ap.add_argument("--dir", default=str(D))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.date) or not a.reviewer.strip():
        sys.exit("--reviewer and --date YYYY-MM-DD are required")
    d = Path(a.dir)
    answers = dict(x.split("=", 1) for x in a.answers.split(",") if "=" in x)
    summ = json.loads((d / "candidate_summary.json").read_text())
    inputs = json.loads((d / "review_inputs.json").read_text())["inputs"]
    if summ["benchmark_inputs"] != inputs:
        sys.exit("candidate_review.csv was built from different benchmark files than the review sheets; re-run scripts/propose_variant_review.py")
    accepted = {q: answers[q] for q in ("Q1", "Q2") if answers.get(q)}
    only = {r for r in a.only_rules.split(",") if r}
    cand = {c["item_id"]: c for c in V.read_csv(d / "candidate_review.csv")}
    stats = Counter()
    for sheet, fields in (("variant_review.csv", V.FIELDS), ("lexicon_review.csv", V.LEX_FIELDS)):
        rows = V.read_csv(d / sheet)
        for r in rows:
            c = cand.get(r["item_id"])
            if c is None:
                stats["no_candidate"] += 1
                continue
            if (r.get("decision") or "").strip():
                stats["kept_existing_decision"] += 1
                continue
            deps = [q for q in c["depends_on"].split(",") if q]
            rules = [x.strip() for x in c["rules"].split(";") if x.strip()]
            if r["item_id"] in a.exclude or c["proposed_decision"] not in ("PASS", "FAIL") or (only and not set(rules) <= only):
                stats["skipped_by_choice_or_ambiguous"] += 1
                continue
            if any(accepted.get(q) != "A" for q in deps):
                stats["skipped_question_not_answered_A"] += 1      # option A = the recommended option the candidates assume
                continue
            r.update(decision=c["proposed_decision"], reason=f"[adopted candidate {'; '.join(rules)}] {c['proposed_reason']}", reviewer=a.reviewer.strip(), review_date=a.date)
            stats[f"adopted_{c['proposed_decision']}"] += 1
        if not a.dry_run:
            write(d / sheet, rows, fields)
    print(json.dumps(dict(stats, dry_run=a.dry_run), indent=1))


if __name__ == "__main__":
    main()
