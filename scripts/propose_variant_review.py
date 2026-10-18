"""CANDIDATE decisions for the variant / lexicon review (H3), proposed by rule with evidence and confidence. They are NOT owner decisions:
this script never writes the human sheets' decision / reason / reviewer / review_date columns. The owner adopts or overrides them
(scripts/adopt_candidate_review.py, which records the owner's name and date only when the owner runs it).
Inputs: the rows rebuilt by build_variant_review.build_rows(), docs/evidence/variant_review/synonym_evidence.json (scripts/synonym_evidence.py) and
the full generation manifests (src/manifests.py, to see whether duplicate prompt texts receive identical generation settings).
Outputs (docs/evidence/variant_review/): candidate_review.csv (one row per prompt and lexicon entry), candidate_summary.json (counts per rule).
Usage: python scripts/propose_variant_review.py [--dir docs/evidence/variant_review]"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import build_variant_review as V  # noqa: E402

OUT = ROOT / "docs" / "evidence" / "variant_review"
FIELDS = ["item_id", "sheet", "split", "source", "variant", "prompt", "auto_flags", "proposed_decision", "rules", "proposed_reason", "evidence",
          "confidence", "depends_on"]
# Colour-synonym verdicts (Q1). Policy from the specification: PAR is meaning-preserving with constraints unchanged (blueprint 1806) and the judge scores
# the canonical colour as the argmax over its basic colour terms (src/evaluate.py COLORS); a synonym is acceptable when it names a shade INSIDE the
# canonical basic-colour category and adds no object or material that changes a constraint. A narrower shade is acceptable, a different category is not.
SYN_VERDICT = {
    "crimson": ("PASS", "high", "deep red: inside red; blueprint's own PAR example; nearest basic colour red in all occurrences in both text encoders"),
    "azure": ("PASS", "high", "bright/sky blue: inside blue; nearest basic colour blue in 47/47 (SD) and 46/47 (judge CLIP) amended PAR prompts"),
    "violet": ("PASS", "high", "bluish purple: inside purple; nearest purple in all occurrences in both encoders; colour reading stronger than the flower reading"),
    "tangerine": ("PASS", "high", "reddish orange: inside orange; nearest orange in all occurrences in both encoders; colour reading stronger than the fruit reading"),
    "emerald": ("PASS", "medium", "bright green: inside green; nearest green in all occurrences in both encoders; gemstone reading about as strong as the colour reading in the judge's CLIP (material only, colour unchanged)"),
    "ivory": ("PASS", "medium", "off-white: inside white; nearest white in SD for all and in the judge's CLIP for 44/47 occurrences; strong 'made of ivory' reading (material only, still white)"),
    "ebony": ("PASS", "medium", "very dark/black: inside black; nearest black in SD for all, in the judge's CLIP for 42/47 (5 brown); colour reading stronger than the wood reading"),
    "lemon": ("PASS", "medium", "lemon (focal) yellow: inside yellow; nearest yellow in 47/47 (SD) and 45/47 (judge CLIP) PAR prompts; colour reading beats an extra-fruit reading for 40/40 nouns (proposed replacement for 'golden', OWNER_DECISIONS Q1)"),
    "golden": ("FAIL", "medium", "'golden' names the colour of gold (metallic yellow-brown), not basic yellow; the judge's CLIP text space puts it nearest BROWN in 33/47 PAR prompts and yellow in only 10/47, so PAR-yellow rows would test the synonym, not the paraphrase"),
}


INTRINSIC = {"dog", "cat", "horse", "bird", "cow", "sheep", "car", "bus", "bicycle", "boat", "chair", "bench", "laptop", "guitar"}   # objects with a front/sides
_LEX = json.loads((ROOT / "data" / "benchmark_v2" / "lexicon.json").read_text())
PLURAL_TO_NOUN = {pl: n for n, pl in _LEX["nouns"].items()} | {"bike": "bicycle", "bikes": "bicycle"}


def norm(t):
    return " ".join(re.findall(r"[a-z]+", t.lower()))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(OUT))
    out_dir = Path(ap.parse_args().dir)
    rows, lrows, meta = V.build_rows()
    ev_path = OUT / "synonym_evidence.json"
    syn_ev = json.loads(ev_path.read_text())["synonyms"] if ev_path.exists() else {}
    import manifests as M
    jobs, _, _ = M.build()
    settings = defaultdict(set)
    for j in jobs:
        if j["kind"] == "core":
            settings[j["prompt_id"]].add((norm(j["prompt"]), j["cfg"], j["steps"], j["seed"]))
    text_members = defaultdict(list)
    for r in rows:
        text_members[norm(r["prompt"])].append(r)
    out = []
    for r in rows:
        flags = [f for f in r["auto_flags"].split(" || ") if f]
        rules, reasons, evidence, conf, deps, decision = [], [], [], [], [], "PASS"
        kinds = {f.split(":")[0] for f in flags}
        unknown = kinds - {"SYNONYM-CHECK", "DUPLICATE", "DUPLICATE-CROSS-SPLIT", "METADATA"}
        if unknown:
            rules.append("R-UNRESOLVED")
            reasons.append(f"automatic flags {sorted(unknown)} need the owner's judgement")
            decision, conf = "AMBIGUOUS", ["low"]
        if not flags:
            rules.append("R-OK")
            reasons.append("text expresses exactly the listed constraints and nothing else; the variant makes only its defined change; grammar checks pass"
                           + ("; unusual colour-noun pairs are intended for binding (DATASHEET)" if r["variant"] == "CAN" else ""))
            evidence.append("build_variant_review text/change/grammar/duplicate/conflict checks: no flag")
            conf.append("high")
        used = [w for w in re.findall(r"[a-z]+", r["prompt"].lower()) if w in SYN_VERDICT] if r["variant"] == "PAR" else []
        if used:
            for w in used:
                d, c, why = SYN_VERDICT[w]
                rules.append(f"R-SYN-{w}")
                reasons.append(f"'{w}': {why}")
                e = syn_ev.get(w, {})
                if e:
                    evidence.append(f"{w}: SD nearest {e.get('sd_nearest_colour')}, judge nearest {e.get('judge_nearest_colour')}")
                conf.append(c)
                if d == "FAIL":
                    decision = "FAIL"
            deps.append("Q1")
        rel = re.search(r"on the (left|right) side of (an? )?(.*?)( on an? |$)", r["prompt"])
        rel_new = re.search(r"\bon the (left|right) of\b", r["prompt"])
        if rel_new:
            rules.append("R-REL-SYN")
            reasons.append(f"relation synonym 'on the {rel_new.group(1)} of' (amendment 2026-10-18, Q1=A): viewer-frame phrase without 'side of'; keeps the "
                           "canonical direction in 82/82 PAR prompts in SD's text encoder (synonym_candidates.json); direction checked against the constraint")
            conf.append("medium")
            deps.append("Q1")
        if rel:
            ref = next((PLURAL_TO_NOUN.get(w, w) for w in re.findall(r"[a-z]+", rel.group(3)) if PLURAL_TO_NOUN.get(w, w) in INTRINSIC), None)
            if ref:
                rules.append("R-REL-INTRINSIC")
                reasons.append(f"relation synonym 'on the {rel.group(1)} side of' a {ref} admits the {ref}'s own (intrinsic) left/right and contact, "
                               "while the judge scores image left/right; not reliably meaning-preserving (fixed by the Q1 relation amendment)")
                evidence.append(f"reference object '{ref}' has an intrinsic front/side; 30 of 82 relation-synonym prompts")
                conf.append("medium")
                decision = "FAIL"
            else:
                rules.append("R-REL-OK")
                reasons.append("relation synonym with a reference object that has no intrinsic left/right side: viewer frame is the only reading")
                conf.append("medium")
            deps.append("Q1")
        if kinds & {"DUPLICATE", "DUPLICATE-CROSS-SPLIT"}:
            twins = [q for q in text_members[norm(r["prompt"])] if q["item_id"] != r["item_id"]]
            same = [q["item_id"] for q in twins if settings.get(q["item_id"]) and settings.get(q["item_id"]) & settings.get(r["item_id"], set())]
            cross = sorted({q["split"] for q in twins} - {r["split"]})
            rules.append("R-DUP-CROSS" if cross else "R-DUP")
            reasons.append("prompt is valid; its text equals " + ", ".join(f"{q['item_id']} ({q['split']})" for q in twins[:4])
                           + (f"; identical generation settings with {len(same)} of them -> byte-identical images" if same else "")
                           + ("; no confirmatory analysis (HA/HD/HE/HF) uses CF-O/CF-A images" if r["variant"] in ("CF-O", "CF-A") else "")
                           + "; handled by the Q2 de-duplication rule for variant-level summaries")
            evidence.append(f"{len(twins)} same-text item(s); shared settings: {len(same)}; splits {sorted({q['split'] for q in twins})}")
            conf.append("medium")
            deps.append("Q2")
        if "METADATA" in kinds:
            rules.append("R-NOBJ")
            reasons.append("n_objects is the SCENE's object count by the generator's definition (src/benchmark_v2.py flat_prompts: len(scene objects)), "
                           "identical for every variant of a scene; the prompt's own object count (1) follows from its constraints; no R3 code reads the field")
            evidence.append("all 90 CF-O rows: n_objects = 2 = scene object count; constraint objects = 1")
            conf.append("high")
        order = {"low": 0, "medium": 1, "high": 2}
        out.append(dict(item_id=r["item_id"], sheet="variant_review", split=r["split"], source=r["source"], variant=r["variant"], prompt=r["prompt"],
                        auto_flags=r["auto_flags"], proposed_decision=decision, rules="; ".join(rules), proposed_reason=" | ".join(reasons),
                        evidence=" | ".join(evidence), confidence=min(conf, key=order.get) if conf else "low", depends_on=",".join(sorted(set(deps)))))
    for r in lrows:
        kind, entry, value = r["kind"], r["entry"], r["value"]
        dep = ""
        if kind == "colour_synonym":
            d, c, why = SYN_VERDICT.get(value, ("AMBIGUOUS", "low", "no verdict"))
            rule, reason, dep = f"R-SYN-{value}", why, "Q1"
        elif kind == "noun":
            d, c, rule = "PASS", "high", "R-LEX-NOUN"
            reason = f"plural '{value}' is correct English; detector validity of the noun is assessed by the evaluator audit / E22, not by H3"
        elif kind == "colour":
            d, c, rule, reason = "PASS", "high", "R-LEX-COLOUR", "basic colour term; also a judge colour (src/evaluate.py COLORS)"
        elif kind == "noun_synonym":
            d, c, rule, reason = "PASS", "high", "R-LEX-NSYN", f"'{value}' is the common synonym of '{entry}' and the same detector object"
        elif kind == "relation":
            if entry.endswith("_synonym") and "side of" not in value:
                d, c, rule, dep = "PASS", "medium", "R-REL-SYN", "Q1"
                reason = f"'{value}' replaces 'side of' (amendment 2026-10-18, Q1=A): viewer frame; canonical direction kept in 82/82 PAR prompts (SD text encoder)"
            elif entry.endswith("_synonym"):
                d, c, rule, dep = "FAIL", "medium", "R-REL-INTRINSIC", "Q1"
                reason = f"'{value}' admits the reference object's intrinsic frame and contact for oriented objects (30 of 82 prompts); replace (Q1)"
            else:
                d, c, rule = "PASS", "high", "R-LEX-REL"
                reason = f"'{value}' is the canonical relation phrase; the judge scores image left/right; direction checked in every prompt"
        else:
            co = int(r["used_in_prompts"])
            d, c, rule = ("PASS", "high", "R-LEX-CONFLICT") if co == 0 else ("AMBIGUOUS", "low", "R-LEX-CONFLICT")
            reason = f"confusable pair kept out of every scene (co-occurs in {co} prompts)"
        out.append(dict(item_id=r["item_id"], sheet="lexicon_review", split="", source="lexicon", variant=kind, prompt=f"{entry} -> {value}",
                        auto_flags=r["auto_flags"], proposed_decision=d, rules=rule, proposed_reason=reason, evidence="", confidence=c, depends_on=dep))
    with open(out_dir / "candidate_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    groups = Counter((o["sheet"], o["rules"], o["proposed_decision"], o["confidence"], o["depends_on"]) for o in out)
    vrows = [o for o in out if o["sheet"] == "variant_review"]
    summ = dict(note="CANDIDATE decisions proposed by rule; not owner decisions. Adopt with scripts/adopt_candidate_review.py.",
                benchmark_inputs=meta["inputs"], prompt_rows=len(vrows), lexicon_rows=len(out) - len(vrows),
                proposed=dict(Counter(o["proposed_decision"] for o in vrows)), proposed_lexicon=dict(Counter(o["proposed_decision"] for o in out if o["sheet"] == "lexicon_review")),
                projected_rejection_rate=round(sum(o["proposed_decision"] == "FAIL" for o in vrows) / len(vrows), 4),
                groups=[dict(sheet=k[0], rules=k[1], decision=k[2], confidence=k[3], depends_on=k[4], n=n) for k, n in sorted(groups.items(), key=lambda kv: -kv[1])])
    (out_dir / "candidate_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("groups", "benchmark_inputs")}, indent=1))
    for g in summ["groups"]:
        print(f"{g['n']:5d}  {g['sheet']:15s} {g['decision']:9s} {g['confidence']:6s} deps={g['depends_on'] or '-':8s} {g['rules']}")


if __name__ == "__main__":
    main()
