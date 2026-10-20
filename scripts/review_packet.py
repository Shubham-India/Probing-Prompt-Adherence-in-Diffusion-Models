"""Owner review packet for the variant review (H3): every candidate rule group with its count, proposed decision, confidence, reason, evidence and
representative examples (DEV and TEST, every variant type present, amended rows first), so the owner can check groups instead of 1,747 rows.
Reads candidate_review.csv / candidate_summary.json / amendment_dispositions.csv; writes docs/evidence/variant_review/OWNER_REVIEW_PACKET.md.
It records no decision. Usage: python scripts/review_packet.py"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "docs" / "evidence" / "variant_review"


def read(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def esc(t):
    return str(t).replace("|", "\\|").replace("\n", " ")


def main():
    cand = read(D / "candidate_review.csv")
    summ = json.loads((D / "candidate_summary.json").read_text())
    amended = {r["item_id"] for r in read(D / "amendment_dispositions.csv")} if (D / "amendment_dispositions.csv").exists() else set()
    groups = defaultdict(list)
    for c in cand:
        groups[(c["sheet"], c["rules"], c["proposed_decision"], c["confidence"], c["depends_on"])].append(c)
    L = ["# Owner review packet: variant and lexicon review (H3)", "",
         "Candidate decisions are proposals derived from rules, evidence and the approved owner answers (Q1 = A, Q2 = A, 2026-10-18). **None of them is your decision until you adopt it.** "
         "Check each group below (the examples are representative; open `candidate_review.csv` filtered on `rules` for all rows), then adopt the groups you agree with "
         "(command at the end). Rows you disagree with: put their item ids in `--exclude` and decide them in the sheets yourself.", "",
         "**What the confidence means.** `high` = every automatic check passes (text vs constraints, variant definition, grammar, duplicates, confusable pairs; "
         "checks verified to fire on broken prompts). The checks cannot judge naturalness, odd wording or a misleading reading: read at least the examples of every "
         "group and a random sample of the large R-OK group before adopting it.", "",
         f"Prompt rows: {summ['prompt_rows']} (proposed " + ", ".join(f"{k} {v}" for k, v in summ['proposed'].items()) + f"); lexicon rows: {summ['lexicon_rows']} (proposed "
         + ", ".join(f"{k} {v}" for k, v in summ['proposed_lexicon'].items()) + "); "
         f"projected rejection rate {summ['projected_rejection_rate']:.1%} (criterion < 20%). Rows changed by the 2026-10-18 amendment: 120 PAR prompts "
         f"(dispositions of the 77 former FAIL candidates: `amendment_dispositions.csv`).", "",
         "| # | Sheet | Rules | Decision | Confidence | Depends on | Rows |", "|---|---|---|---|---|---|---|"]
    order = sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1])))
    for i, (k, rows) in enumerate(order, 1):
        L.append(f"| G{i:02d} | {k[0]} | {esc(k[1])} | {k[2]} | {k[3]} | {k[4] or '-'} | {len(rows)} |")
    L.append("")
    for i, (k, rows) in enumerate(order, 1):
        L += [f"## G{i:02d}: {k[1]} ({k[0]}, {len(rows)} rows, proposed {k[2]}, confidence {k[3]})", "",
              f"**Reason (first row):** {esc(rows[0]['proposed_reason'])}", ""]
        if rows[0]["evidence"]:
            L += [f"**Evidence:** {esc(rows[0]['evidence'])}", ""]
        picks, seen = [], set()
        for c in sorted(rows, key=lambda c: (c["item_id"] not in amended, c["split"] != "DEV", c["variant"], c["item_id"])):
            key = (c["split"], c["variant"], c["item_id"] in amended)
            if key not in seen and len(picks) < 6:
                picks.append(c)
                seen.add(key)
        for c in rows:
            if len(picks) >= 4:
                break
            if c not in picks:
                picks.append(c)
        L += ["| Item | Split | Variant | Text | Flags |", "|---|---|---|---|---|"]
        L += [f"| {c['item_id']}{' (amended)' if c['item_id'] in amended else ''} | {c['split']} | {c['variant']} | {esc(c['prompt'])} | {esc(c['auto_flags'])[:160]} |" for c in picks]
        L.append("")
    L += ["## Adopt", "", "After checking the groups (and excluding any rows you want to decide yourself):", "",
          "`python scripts/adopt_candidate_review.py --reviewer \"<your name>\" --date YYYY-MM-DD --answers Q1=A,Q2=A [--exclude ITEM_ID ...]`", "",
          "`python scripts/check_variant_review.py`  (must exit 0 with `criteria_met: true`)", ""]
    (D / "OWNER_REVIEW_PACKET.md").write_text("\n".join(L), encoding="utf-8")
    print(f"{len(order)} groups -> {D / 'OWNER_REVIEW_PACKET.md'}")


if __name__ == "__main__":
    main()
