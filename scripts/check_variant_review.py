"""Check a completed variant / lexicon review (blueprint H3) and write its evidence file. Never edits the benchmark or configs/stage_gates.json.
Refuses (exit 1) when: the benchmark files changed since the sheets were built; rows were added, removed or their non-decision columns edited;
any decision is empty or not PASS / FAIL / AMBIGUOUS; a FAIL or AMBIGUOUS row, or a PASS on an auto-flagged row, has no reason; reviewer or
date is missing. Reports the rejection rate (FAIL / prompt rows, overall and per split / source / variant) against the H3 criterion (< 20%).
`criteria_met` is true only when the review is complete, no AMBIGUOUS decision remains, no lexicon entry FAILs and the prompt rejection rate is
below 20%; any FAIL item still needs a benchmark amendment before generation (listed under `requires_owner_action`).
Usage: python scripts/check_variant_review.py [--dir docs/evidence/variant_review]"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_variant_review as V  # noqa: E402

DECISIONS = {"PASS", "FAIL", "AMBIGUOUS"}
THRESHOLD = 0.20


def check_sheet(done, expected, fields, label):
    probs = []
    exp = {r["item_id"]: r for r in expected}
    got = {}
    for r in done:
        if r.get("item_id") in got:
            probs.append(f"{label}: duplicate row {r.get('item_id')}")
        got[r.get("item_id")] = r
    probs += [f"{label}: row missing {i}" for i in sorted(set(exp) - set(got))[:20]]
    probs += [f"{label}: unexpected row {i}" for i in sorted(set(got) - set(exp))[:20]]
    for i in sorted(set(exp) & set(got)):
        e, g = exp[i], got[i]
        edited = [c for c in fields if c not in V.DECISION_COLS and str(e[c]) != (g.get(c) or "")]
        if edited:
            probs.append(f"{label}: {i}: non-decision columns edited {edited} (only decision, reason, reviewer, review_date may change)")
        d = (g.get("decision") or "").strip().upper()
        if d not in DECISIONS:
            probs.append(f"{label}: {i}: decision {g.get('decision')!r} (must be PASS, FAIL or AMBIGUOUS)")
        reason = (g.get("reason") or "").strip()
        if d in ("FAIL", "AMBIGUOUS") and not reason:
            probs.append(f"{label}: {i}: {d} without a reason")
        if d == "PASS" and (e.get("auto_flags") or "").strip() and not reason:
            probs.append(f"{label}: {i}: PASS on an auto-flagged row without a reason (say why the flag does not invalidate it)")
        if not (g.get("reviewer") or "").strip() or not (g.get("review_date") or "").strip():
            probs.append(f"{label}: {i}: reviewer or review_date missing")
    return probs, got


def rates(rows, key):
    by = defaultdict(Counter)
    for r in rows:
        by[key(r)][(r.get("decision") or "").strip().upper() or "EMPTY"] += 1
    return {k: dict(v, n=sum(v.values()), fail_rate=round(v["FAIL"] / sum(v.values()), 4)) for k, v in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(V.ROOT / "docs" / "evidence" / "variant_review"))
    a = ap.parse_args()
    d = Path(a.dir)
    meta = json.loads((d / "review_inputs.json").read_text())
    probs = [f"benchmark file changed since the sheets were built: {f}" for f, h in meta["inputs"].items() if V.sha(V.B / f) != h]
    rows, lrows, _ = V.build_rows()
    vdone, ldone = V.read_csv(d / "variant_review.csv"), V.read_csv(d / "lexicon_review.csv")
    p1, vgot = check_sheet(vdone, rows, V.FIELDS, "variant_review.csv")
    p2, lgot = check_sheet(ldone, lrows, V.LEX_FIELDS, "lexicon_review.csv")
    probs += p1 + p2
    dec = Counter((r.get("decision") or "").strip().upper() or "EMPTY" for r in vgot.values())
    n = len(rows)
    rate = dec["FAIL"] / n if n else 1.0
    fails = sorted(i for i, r in vgot.items() if (r.get("decision") or "").strip().upper() == "FAIL")
    amb = sorted(i for i, r in vgot.items() if (r.get("decision") or "").strip().upper() == "AMBIGUOUS")
    lfail = sorted(i for i, r in lgot.items() if (r.get("decision") or "").strip().upper() == "FAIL")
    lamb = sorted(i for i, r in lgot.items() if (r.get("decision") or "").strip().upper() == "AMBIGUOUS")
    complete = not probs
    criteria = complete and not amb and not lamb and not lfail and rate < THRESHOLD
    ev = dict(checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), complete=complete, criteria_met=criteria,
              criterion=f"H3: FAIL / prompt rows < {THRESHOLD:.0%}; no AMBIGUOUS left; no lexicon FAIL",
              prompt_rows=n, decisions=dict(dec), rejection_rate=round(rate, 4), lexicon_decisions=dict(Counter((r.get("decision") or "").strip().upper() or "EMPTY" for r in lgot.values())),
              by_split=rates(vgot.values(), lambda r: r.get("split")), by_source=rates(vgot.values(), lambda r: r.get("source")),
              by_variant=rates(vgot.values(), lambda r: r.get("variant")), reviewers=sorted({(r.get("reviewer") or "").strip() for r in list(vgot.values()) + list(lgot.values())} - {""}),
              requires_owner_action=dict(fail_items=fails, ambiguous_items=amb, lexicon_fail=lfail, lexicon_ambiguous=lamb,
                                         note="FAIL items must be fixed or removed by a dated benchmark amendment (decision log), then the sheets rebuilt with --carry-over and the changed items re-reviewed"),
              benchmark_inputs=meta["inputs"], sheets_sha256={f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ("variant_review.csv", "lexicon_review.csv")},
              problems=probs[:200], n_problems=len(probs),
              gate_note="This script never sets configs/stage_gates.json. The owner sets variant_review_complete = true only if criteria_met is true and no FAIL item remains unaddressed.")
    (d / "variant_review_evidence.json").write_text(json.dumps(ev, indent=1))
    print(json.dumps({k: ev[k] for k in ("complete", "criteria_met", "prompt_rows", "decisions", "rejection_rate", "lexicon_decisions", "n_problems")}, indent=1))
    for p in probs[:25]:
        print("  -", p)
    sys.exit(0 if criteria else 1)


if __name__ == "__main__":
    main()
