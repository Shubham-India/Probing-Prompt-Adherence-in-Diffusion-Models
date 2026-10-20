"""Human review sheets for the benchmark variants and lexicon (blueprint H3; gate `variant_review_complete`).
Writes, into --out (default docs/evidence/variant_review):
  variant_review.csv   one row per prompt (main benchmark + PS family, DEV then TEST, each scene's canonical prompt followed by its variants)
  lexicon_review.csv   one row per lexicon entry (nouns, plurals, colours, synonyms, relations, conflict pairs)
  review_inputs.json   SHA-256 of every benchmark file the sheets were built from (the checker refuses a review of a changed benchmark)
Every row carries the intended change derived from the constraint difference to the scene's canonical prompt and automatic FLAGS. Flags are
pre-checks for the reviewer, never corrections: nothing in the benchmark is modified. Decision columns are left empty for the human reviewer.
Usage: python scripts/build_variant_review.py [--out DIR] [--carry-over OLD_DIR] [--force]
Completed sheets are checked with scripts/check_variant_review.py (instructions: docs/evidence/variant_review/REVIEW_INSTRUCTIONS.md)."""
import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "data" / "benchmark_v2"
INPUTS = ["prompts_v2.json", "prompts_ps.json", "lexicon.json", "scene_graphs.json", "ps_family.json", "splits.json", "splits_ps.json"]
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
SIZE = {"large", "small", "big", "tiny"}
EXPECTED = {"CAN": "reference prompt of the scene", "PAR": "meaning-preserving synonym paraphrase: constraints identical to CAN",
            "ORD": "mention-order swap (converse relation where needed): constraints identical to CAN", "CF-C": "counterfactual: colour assignment changed, nothing else",
            "CF-A": "counterfactual: one colour attribute inserted or removed, nothing else", "CF-R": "counterfactual: spatial relation reversed, nothing else",
            "CF-N": "counterfactual: requested count changed, nothing else", "CF-O": "counterfactual: one object (and its constraints) deleted or inserted",
            "CF-S": "counterfactual: relative size swapped, nothing else"}
FIELDS = ["item_id", "split", "source", "scene_id", "category", "variant", "canonical_prompt", "prompt", "intended_change", "constraints", "auto_flags",
          "decision", "reason", "reviewer", "review_date"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def cstr(c):
    t = c["type"]
    if t == "presence":
        return f"presence({c['object']})"
    if t == "binding":
        return f"colour({c['object']}={c['color']})"
    if t == "count":
        return f"count({c['object']}={c['n']})"
    if t == "spatial":
        return f"spatial({c['left']} left of {c['right']})"
    if t == "size":
        return f"size({c['larger']} larger than {c['smaller']})"
    return json.dumps(c, sort_keys=True)


def key(c):
    return json.dumps(c, sort_keys=True)


def words(text):
    return re.findall(r"[a-z]+", text.lower())


class Lex:
    def __init__(self, L):
        self.L = L
        self.plural = L["nouns"]
        self.colour_syn = L["colour_synonyms"]
        self.noun_syn = L["noun_synonyms"]
        self.colour_words = set(L["colours"]) | set(self.colour_syn.values())
        self.rel = {k: v for k, v in L["relations"].items()} | {k + "#syn": v for k, v in L["relation_synonyms"].items()}

    def noun_forms(self, n):
        f = {n, self.plural.get(n, n + "s")}
        if n in self.noun_syn:
            s = self.noun_syn[n]
            f |= {s, s + "s"}
        return f

    def colour_forms(self, c):
        return {c, self.colour_syn.get(c, c)}


def text_checks(p, lex):
    """Flags where the prompt text does not express exactly its constraints (binding adjacency, counts, relation direction, size words)."""
    fl = []
    t = p["prompt"].lower()
    w = words(t)
    cons = p["constraints"]
    objs = {c.get("object") for c in cons if c.get("object")} | {c[k] for c in cons for k in ("left", "right", "larger", "smaller") if k in c}
    for o in objs:
        if not lex.noun_forms(o) & set(w):
            fl.append(f"TEXT-NOUN: '{o}' not mentioned")
    for c in cons:
        if c["type"] == "binding":
            forms = lex.colour_forms(c["color"])
            nouns = lex.noun_forms(c["object"])
            ok = any(w[i] in forms and any(w[j] in nouns for j in range(i + 1, min(i + 3, len(w)))) for i in range(len(w)))
            if not ok:
                fl.append(f"TEXT-BIND: colour '{c['color']}' not directly before '{c['object']}'")
        if c["type"] == "count":
            nouns = lex.noun_forms(c["object"])
            ok = any(w[i] in NUM and NUM[w[i]] == c["n"] and any(w[j] in nouns for j in range(i + 1, min(i + 4, len(w)))) for i in range(len(w)))
            if not ok:
                fl.append(f"TEXT-COUNT: '{c['n']} {c['object']}' not expressed")
        if c["type"] == "size":
            for obj, wanted in ((c["larger"], {"large", "big"}), (c["smaller"], {"small", "tiny"})):
                nouns = lex.noun_forms(obj)
                ok = any(w[i] in wanted and any(w[j] in nouns for j in range(i + 1, min(i + 3, len(w)))) for i in range(len(w)))
                if not ok:
                    fl.append(f"TEXT-SIZE: '{obj}' lacks its size word")
        if c["type"] == "spatial":
            hit = None
            for k, phrase in lex.rel.items():
                if phrase in t:
                    hit = (k.split("#")[0], phrase)
            if not hit:
                fl.append("TEXT-REL: no relation phrase for the spatial constraint")
            else:
                before = words(t.split(hit[1])[0])
                subj = next((o for o in objs if lex.noun_forms(o) & set(before)), None)
                want = c["left"] if hit[0] == "left_of" else c["right"]
                if subj != want:
                    fl.append(f"TEXT-REL: text says {subj} {hit[1].strip()} the other, constraint wants {c['left']} left of {c['right']}")
    n_col = sum(1 for x in w if x in lex.colour_words)
    n_bind = sum(1 for c in cons if c["type"] == "binding")
    if n_col != n_bind:
        fl.append(f"TEXT-EXTRA: {n_col} colour words in text vs {n_bind} colour constraints")
    if not any(c["type"] == "size" for c in cons) and SIZE & set(w):
        fl.append("TEXT-EXTRA: size word without a size constraint")
    n_num = [NUM[x] for x in w if x in NUM]
    if sorted(n_num) != sorted(c["n"] for c in cons if c["type"] == "count"):
        fl.append(f"TEXT-EXTRA: number words {n_num} vs count constraints")
    if not any(c["type"] == "spatial" for c in cons) and any(ph in t for ph in lex.rel.values()):
        fl.append("TEXT-EXTRA: relation phrase without a spatial constraint")
    for m in re.finditer(r"\ba ([aeiou]\w*)", t):
        fl.append(f"GRAMMAR: 'a {m.group(1)}'")
    for m in re.finditer(r"\ban ([^aeiou\W]\w*)", t):
        fl.append(f"GRAMMAR: 'an {m.group(1)}'")
    plurals = set(lex.plural.values()) - set(lex.plural)
    for i, x in enumerate(w):
        if x in NUM and NUM[x] >= 2:
            nxt = next((y for y in w[i + 1:i + 4] if y in lex.plural or y in plurals), None)
            if nxt in lex.plural and nxt not in plurals and lex.plural[nxt] != nxt:
                fl.append(f"GRAMMAR: '{x} ... {nxt}' (singular after a number)")
        if x in ("a", "an"):
            nxt = next((y for y in w[i + 1:i + 4] if y in lex.plural or y in plurals), None)
            if nxt in plurals:
                fl.append(f"GRAMMAR: '{x} ... {nxt}' (article before a plural)")
    return fl


def change_checks(p, can, lex):
    """(intended change text, flags) from the constraint difference to the canonical prompt of the same scene."""
    v = p["variant"]
    if v == "CAN":
        return EXPECTED["CAN"], []
    if can is None:
        return EXPECTED.get(v, v), ["NO-CANONICAL: scene has no CAN prompt to compare with"]
    a, b = {key(c): c for c in can["constraints"]}, {key(c): c for c in p["constraints"]}
    rem = [a[k] for k in a if k not in b]
    add = [b[k] for k in b if k not in a]
    diff = "; ".join([f"-{cstr(c)}" for c in rem] + [f"+{cstr(c)}" for c in add]) or "no constraint change"
    text = f"{EXPECTED.get(v, v)} | constraint diff: {diff}"
    fl = []
    types = {c["type"] for c in rem + add}
    if v in ("PAR", "ORD"):
        if rem or add:
            fl.append("CHANGE: meaning-preserving variant changes the constraints")
        wc, wp = Counter(words(can["prompt"])), Counter(words(p["prompt"]))
        for art in ("a", "an"):
            wc.pop(art, None), wp.pop(art, None)
        if v == "ORD" and any(c["type"] == "spatial" for c in p["constraints"]):
            for x in (wc, wp):                       # the converse relation (left <-> right) is part of a mention-order swap
                x.pop("left", None), x.pop("right", None)
        if v == "ORD" and wc != wp:
            fl.append(f"CHANGE: ORD changes words {sorted((wc - wp).elements())} -> {sorted((wp - wc).elements())}")
        if v == "PAR":
            gone, new = sorted((wc - wp).elements()), sorted((wp - wc).elements())
            syn = {**lex.colour_syn, **lex.noun_syn}
            relsyn = " ".join(lex.L["relation_synonyms"].values())
            singular = {pl: n for n, pl in lex.plural.items()}
            def syn_of(g):
                n = singular.get(g, g)
                return syn.get(g) or (syn.get(n) + "s" if syn.get(n) and n != g else None)
            synforms = set(syn.values()) | {x + "s" for x in lex.noun_syn.values()}
            unexplained = [g for g in gone if syn_of(g) not in new and g not in " ".join(lex.L["relations"].values()).split()]
            unexplained_new = [n for n in new if n not in synforms and n not in relsyn.split()]
            if unexplained or unexplained_new:
                fl.append(f"CHANGE: PAR substitution not in the lexicon synonym lists: {unexplained} -> {unexplained_new}")
            if not gone and not new:
                fl.append("CHANGE: PAR identical to CAN")
            text += f" | substitutions: {gone} -> {new}"
    elif v == "CF-C":
        if types != {"binding"} or len(rem) != len(add) or not rem:
            fl.append("CHANGE: CF-C changes more than colour assignments")
        elif sorted(c["color"] for c in rem) != sorted(c["color"] for c in add):
            fl.append("CHANGE: CF-C introduces a new colour (not a swap)")
    elif v == "CF-A":
        if len(rem) + len(add) != 1 or (rem + add)[0]["type"] != "binding":
            fl.append("CHANGE: CF-A should insert or remove exactly one colour constraint")
    elif v == "CF-R":
        if types != {"spatial"} or len(rem) != 1 or len(add) != 1 or (rem[0]["left"], rem[0]["right"]) != (add[0]["right"], add[0]["left"]):
            fl.append("CHANGE: CF-R should only reverse the spatial relation")
    elif v == "CF-N":
        if types != {"count"} or len(rem) != 1 or len(add) != 1 or rem[0]["object"] != add[0]["object"] or rem[0]["n"] == add[0]["n"]:
            fl.append("CHANGE: CF-N should only change one requested count")
    elif v == "CF-S":
        if types != {"size"} or len(rem) != 1 or len(add) != 1 or (rem[0]["larger"], rem[0]["smaller"]) != (add[0]["smaller"], add[0]["larger"]):
            fl.append("CHANGE: CF-S should only swap the relative size")
    elif v == "CF-O":
        before = {c.get("object") for c in can["constraints"] if c.get("object")}
        left = {c.get("object") for c in p["constraints"] if c.get("object")}
        if len(before ^ left) != 1 or (rem and add):
            fl.append("CHANGE: CF-O should delete or insert exactly one object with its constraints")
        if p.get("n_objects") != len(left):
            fl.append(f"METADATA: n_objects={p.get('n_objects')} but {len(left)} object(s) remain (field unused by the R3 analysis)")
    else:
        fl.append(f"UNKNOWN-VARIANT: {v}")
    if p.get("n_constraints") != len(p["constraints"]):
        fl.append(f"METADATA: n_constraints={p.get('n_constraints')} vs {len(p['constraints'])}")
    return text, fl


LEX_FIELDS = ["item_id", "kind", "entry", "value", "used_in_prompts", "auto_flags", "decision", "reason", "reviewer", "review_date"]
DECISION_COLS = ("decision", "reason", "reviewer", "review_date")


def build_rows():
    """(variant rows, lexicon rows, meta) from the current benchmark files; decision columns empty."""
    L = json.loads((B / "lexicon.json").read_text())
    lex = Lex(L)
    main_p = json.loads((B / "prompts_v2.json").read_text())
    ps_p = json.loads((B / "prompts_ps.json").read_text())
    allp = [dict(p, _source="main") for p in main_p] + [dict(p, _source="ps_family") for p in ps_p]
    can = {p["scene_id"]: p for p in allp if p["variant"] == "CAN"}
    norm = defaultdict(list)
    for p in allp:
        norm[" ".join(words(p["prompt"]))].append(p)
    conflicts = [set(c) for c in L["conflicts"]]
    order = {"CAN": 0, "PAR": 1, "ORD": 2}
    rows = []
    for p in sorted(allp, key=lambda p: (p["split"] != "DEV", p["_source"] != "main", p["scene_id"], order.get(p["variant"], 3), p["variant"])):
        c = can.get(p["scene_id"])
        change, fl = change_checks(p, c, lex)
        fl += text_checks(p, lex)
        twins = [q for q in norm[" ".join(words(p["prompt"]))] if q["id"] != p["id"]]
        if twins:
            cross = sorted({q["split"] for q in twins} - {p["split"]})
            fl.append(("DUPLICATE-CROSS-SPLIT" if cross else "DUPLICATE") + ": same text as " + ", ".join(f"{q['id']} ({q['split']})" for q in twins[:4]))
        objs = {x.get("object") for x in p["constraints"] if x.get("object")}
        for pair in conflicts:
            if pair <= objs:
                fl.append(f"CONFLICT-PAIR: {sorted(pair)} is listed as a confusable pair in lexicon.conflicts")
        syn_used = [s for s in words(p["prompt"]) if s in {"golden", "lemon", "ivory", "ebony", "violet", "emerald", "tangerine", "azure"}]
        if syn_used:
            fl.append(f"SYNONYM-CHECK: {syn_used} may read as a material/noun or a narrower hue (see lexicon_review.csv)")
        rows.append(dict(item_id=p["id"], split=p["split"], source=p["_source"], scene_id=p["scene_id"], category=p["category"], variant=p["variant"],
                         canonical_prompt=c["prompt"] if c else "", prompt=p["prompt"], intended_change=change,
                         constraints="; ".join(cstr(x) for x in p["constraints"]), auto_flags=" || ".join(fl), decision="", reason="", reviewer="", review_date=""))
    lrows = []
    hints = {"golden": "can read as gold/metallic material rather than yellow", "ivory": "material connotation; off-white", "ebony": "material (wood) connotation",
             "violet": "also a flower noun", "emerald": "gemstone; narrower hue than green", "azure": "light/sky blue, narrower than blue",
             "tangerine": "also a fruit noun", "lemon": "also a fruit noun (replaces 'golden', amendment 2026-10-18)", "crimson": "darker red; usually acceptable"}
    used = Counter(x for p in allp for x in words(p["prompt"]))
    for n, pl in L["nouns"].items():
        lrows.append(dict(item_id=f"LEX-NOUN-{n}", kind="noun", entry=n, value=pl, used_in_prompts=used[n] + used[pl],
                          auto_flags="" if pl.endswith("s") or pl == n else "irregular plural: check", decision="", reason="", reviewer="", review_date=""))
    for c in L["colours"]:
        lrows.append(dict(item_id=f"LEX-COLOUR-{c}", kind="colour", entry=c, value=c, used_in_prompts=used[c], auto_flags="", decision="", reason="", reviewer="", review_date=""))
    for c, s in L["colour_synonyms"].items():
        lrows.append(dict(item_id=f"LEX-CSYN-{c}", kind="colour_synonym", entry=c, value=s, used_in_prompts=used[s], auto_flags=hints.get(s, ""),
                          decision="", reason="", reviewer="", review_date=""))
    for n, s in L["noun_synonyms"].items():
        lrows.append(dict(item_id=f"LEX-NSYN-{n}", kind="noun_synonym", entry=n, value=s, used_in_prompts=used[s], auto_flags="", decision="", reason="", reviewer="", review_date=""))
    for k, v in list(L["relations"].items()) + [(k + "_synonym", v) for k, v in L["relation_synonyms"].items()]:
        lrows.append(dict(item_id=f"LEX-REL-{k}", kind="relation", entry=k, value=v, used_in_prompts=sum(1 for p in allp if v in p["prompt"]),
                          auto_flags="", decision="", reason="", reviewer="", review_date=""))
    for pair in L["conflicts"]:
        co = sum(1 for p in allp if set(pair) <= {x.get("object") for x in p["constraints"] if x.get("object")})
        lrows.append(dict(item_id=f"LEX-CONFLICT-{pair[0]}-{pair[1]}", kind="conflict_pair", entry=pair[0], value=pair[1], used_in_prompts=co,
                          auto_flags=f"co-occurs in {co} prompts" if co else "", decision="", reason="", reviewer="", review_date=""))
    flagged = Counter(r["variant"] for r in rows if r["auto_flags"])
    kinds = Counter(f.split(":")[0] for r in rows for f in r["auto_flags"].split(" || ") if f)
    meta = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), inputs={f: sha(B / f) for f in INPUTS},
                lexicon_status=L.get("status"), n_variant_rows=len(rows), n_lexicon_rows=len(lrows),
                rows_by_split_variant={f"{s}/{v}": n for (s, v), n in sorted(Counter((r["split"], r["variant"]) for r in rows).items())},
                rows_with_flags_by_variant=dict(sorted(flagged.items())), flag_kinds=dict(sorted(kinds.items())),
                acceptance="blueprint H3: fewer than 20% of reviewed variant items rejected (FAIL); see docs/evidence/variant_review/REVIEW_INSTRUCTIONS.md")
    return rows, lrows, meta


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def carry_over(rows, old_rows, content_cols):
    """Copy decisions from an earlier review for items whose reviewed content is unchanged (after a benchmark amendment). Changed or new items
    stay empty and must be reviewed again. Returns the number carried over."""
    old = {r["item_id"]: r for r in old_rows}
    n = 0
    for r in rows:
        o = old.get(r["item_id"])
        if o and all(o.get(c, "") == str(r[c]) for c in content_cols) and (o.get("decision") or "").strip():
            r.update({c: o.get(c, "") for c in DECISION_COLS})
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "evidence" / "variant_review"))
    ap.add_argument("--carry-over", default=None, help="folder of an earlier (partly) completed review: keep decisions for unchanged items")
    ap.add_argument("--force", action="store_true", help="overwrite existing sheets in --out")
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "variant_review.csv").exists() and not a.force and not a.carry_over:
        raise SystemExit(f"{out / 'variant_review.csv'} exists (it may hold review decisions); use --carry-over or --force, or another --out")
    rows, lrows, meta = build_rows()
    if a.carry_over:
        co = Path(a.carry_over)
        meta["carried_over"] = dict(source=str(co), variant=carry_over(rows, read_csv(co / "variant_review.csv"), [c for c in FIELDS if c not in DECISION_COLS]),
                                    lexicon=carry_over(lrows, read_csv(co / "lexicon_review.csv"), [c for c in LEX_FIELDS if c not in DECISION_COLS]))
    out.mkdir(parents=True, exist_ok=True)
    for name, fields, data in (("variant_review.csv", FIELDS, rows), ("lexicon_review.csv", LEX_FIELDS, lrows)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    (out / "review_inputs.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main()
