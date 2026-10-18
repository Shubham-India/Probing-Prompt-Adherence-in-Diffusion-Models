"""Benchmark V2: scene graphs -> prompt variants with structured constraints (RB 11; milestone M3).

Deterministic given the config seed. Writes data/benchmark_v2/{scene_graphs.json, prompts_v2.json, splits.json, lexicon.json,
design_counts.json, hashes.json}. Constraints use the legacy evaluator format (presence / count / binding / spatial), so
src/evaluate.py can score V2 prompts unchanged.

Binding-bearing (D-05): a scene is binding-bearing iff its CANONICAL constraint list contains >= 1 `binding` item, i.e. an
explicit, evaluable attribute-to-object assignment. A stricter subset, `binding_competitive`, additionally requires >= 2
distinct coloured objects (a colour can be mis-assigned to the other object). Both counts are DERIVED from the generated
graphs and written to design_counts.json; quotas in the config are only category quotas.

The lexicon is a DRAFT: noun/colour validity for the detector (E22, E03) is NOT yet verified; it is frozen later (D-05 C-8).
Usage: python src/benchmark_v2.py --build   |   python src/benchmark_v2.py --validate
"""
import argparse
import copy
import hashlib
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "benchmark_v2"
NUM = {1: "one", 2: "two", 3: "three", 4: "four"}

NOUNS = {  # singular -> plural (DRAFT lexicon, ~40 concrete objects)
    "dog": "dogs", "cat": "cats", "horse": "horses", "bird": "birds", "cow": "cows", "sheep": "sheep", "bicycle": "bicycles",
    "car": "cars", "bus": "buses", "boat": "boats", "chair": "chairs", "bench": "benches", "vase": "vases", "cup": "cups",
    "book": "books", "clock": "clocks", "umbrella": "umbrellas", "bottle": "bottles", "bowl": "bowls", "backpack": "backpacks",
    "laptop": "laptops", "suitcase": "suitcases", "kite": "kites", "skateboard": "skateboards", "teapot": "teapots",
    "lamp": "lamps", "hat": "hats", "shoe": "shoes", "ball": "balls", "apple": "apples", "banana": "bananas", "cube": "cubes",
    "sphere": "spheres", "pillow": "pillows", "basket": "baskets", "candle": "candles", "mirror": "mirrors", "guitar": "guitars",
    "fork": "forks", "box": "boxes"}
CONFLICTS = [{"cube", "box"}, {"ball", "sphere"}, {"car", "bus"}, {"cup", "bowl"}, {"backpack", "suitcase"}, {"bench", "chair"},
             {"cup", "teapot"}, {"bottle", "vase"}, {"cube", "sphere"}, {"ball", "apple"}, {"sheep", "cow"}, {"cat", "dog"}]
COLOURS = ["red", "blue", "yellow", "green", "orange", "purple", "white", "black"]   # no brown/pink (RB 11.4)
COLOUR_SYN = {"red": "crimson", "blue": "azure", "yellow": "golden", "green": "emerald", "orange": "tangerine",
              "purple": "violet", "white": "ivory", "black": "ebony"}
NOUN_SYN = {"bicycle": "bike"}
REL_TEXT = {"left_of": "to the left of", "right_of": "to the right of"}
REL_SYN = {"left_of": "on the left side of", "right_of": "on the right side of"}
CONVERSE = {"left_of": "right_of", "right_of": "left_of"}
PILOT_PAIRS_PATH = ROOT / "data" / "prompts.json"


def art(word):
    return "an" if word[0] in "aeiou" else "a"


def conflict(a, b):
    return {a, b} in CONFLICTS


# ----- scene graph -> constraints / text ---------------------------------------------------------------------------
def constraints_of(sg):
    cons, objs = [], sg["objects"]
    for o in objs:
        cons.append({"type": "count", "object": o["noun"], "n": o["count"]} if o["count"] > 1 else {"type": "presence", "object": o["noun"]})
    for o in objs:
        if o["attributes"].get("colour"):
            cons.append({"type": "binding", "object": o["noun"], "color": o["attributes"]["colour"]})
    by = {o["oid"]: o["noun"] for o in objs}
    for r in sg["relations"]:
        l, rr = (r["subject"], r["object"]) if r["rel"] == "left_of" else (r["object"], r["subject"])
        cons.append({"type": "spatial", "left": by[l], "right": by[rr]})
    return cons


def canon(cons):
    return sorted(json.dumps(c, sort_keys=True) for c in cons)


def phrase(o, syn=False):
    noun = o["noun"]
    if syn and noun in NOUN_SYN:
        noun = NOUN_SYN[noun]
    col = o["attributes"].get("colour")
    if col and syn:
        col = COLOUR_SYN[col]
    if o["count"] > 1:
        plural = NOUNS[o["noun"]] if not (syn and noun != o["noun"]) else noun + "s"
        return f"{NUM[o['count']]} {col + ' ' if col else ''}{plural}"
    w = f"{col + ' ' if col else ''}{noun}"
    return f"{art(w)} {w}"


def render(sg, syn=False, mention_swap=False):
    objs = sg["objects"][::-1] if mention_swap else sg["objects"]
    rel = sg["relations"][0] if sg["relations"] else None
    if rel:
        by = {o["oid"]: o for o in sg["objects"]}
        s_o, o_o = by[rel["subject"]], by[rel["object"]]
        r = rel["rel"]
        if mention_swap:
            s_o, o_o, r = o_o, s_o, CONVERSE[r]
        text = f"{phrase(s_o, syn)} {(REL_SYN if syn else REL_TEXT)[r]} {phrase(o_o, syn)}"
    else:
        text = " and ".join(phrase(o, syn) for o in objs)
    if sg["background"]:
        text += " " + sg["background"]
    return text


# ----- scene generation --------------------------------------------------------------------------------------------
def make_obj(i, noun, count=1, colour=None):
    return {"oid": f"o{i}", "noun": noun, "detector_query": noun, "count": count, "attributes": {"colour": colour} if colour else {}}


def finish(sg):
    cons = constraints_of(sg)
    objs = sg["objects"]
    sg["complexity"] = {"n_objects": len(objs), "n_attributes": sum(1 for o in objs if o["attributes"].get("colour")),
                        "n_relations": len(sg["relations"]), "n_count_constraints": sum(1 for c in cons if c["type"] == "count"),
                        "n_constraints": len(cons)}
    sg["object_pair_group"] = "|".join(sorted(o["noun"] for o in objs))
    return sg


def build_variants(sg, cfg):
    v = []
    base = constraints_of(sg)
    v.append(dict(type="CAN", prompt=render(sg), constraints=base))
    par = render(sg, syn=True)
    if par != v[0]["prompt"]:
        v.append(dict(type="PAR", rewrite_rule="synonym_lexicon", prompt=par, constraints=base))
    if len(sg["objects"]) == 2:
        sw = render(sg, mention_swap=True)
        if sw != v[0]["prompt"]:
            v.append(dict(type="ORD", rewrite_rule="mention_order", prompt=sw, constraints=base))
    cands = []
    objs = sg["objects"]
    coloured = [o for o in objs if o["attributes"].get("colour")]
    if len(coloured) == 2 and coloured[0]["attributes"]["colour"] != coloured[1]["attributes"]["colour"]:
        e = copy.deepcopy(sg)
        a, b = e["objects"][0], e["objects"][1]
        a["attributes"]["colour"], b["attributes"]["colour"] = b["attributes"]["colour"], a["attributes"]["colour"]
        cands.append(("CF-C", e, {"field": "colour", "swap": [objs[0]["oid"], objs[1]["oid"]]}))
    if sg["relations"]:
        e = copy.deepcopy(sg)
        e["relations"][0]["rel"] = CONVERSE[e["relations"][0]["rel"]]
        cands.append(("CF-R", e, {"field": "relation", "reverse": 0}))
    counted = [o for o in objs if o["count"] > 1]
    if counted:
        e = copy.deepcopy(sg)
        o = next(x for x in e["objects"] if x["count"] > 1)
        lo, hi = cfg["counting_range"]
        o["count"] = o["count"] + 1 if o["count"] == lo else o["count"] - 1 if o["count"] == hi else o["count"] + (1 if (len(sg["scene_id"]) + o["count"]) % 2 else -1)
        cands.append(("CF-N", e, {"field": "count", "oid": o["oid"], "from": next(x for x in objs if x["oid"] == o["oid"])["count"], "to": o["count"]}))
    if len(objs) == 2:
        e = copy.deepcopy(sg)
        e["objects"] = e["objects"][:1]
        e["relations"] = []
        cands.append(("CF-O", e, {"field": "objects", "delete": objs[1]["oid"]}))
    if coloured:
        e = copy.deepcopy(sg)
        tgt = next(x for x in reversed(e["objects"]) if x["attributes"].get("colour"))
        tgt["attributes"] = {}
        cands.append(("CF-A", e, {"field": "colour", "remove": tgt["oid"]}))
    elif objs:
        e = copy.deepcopy(sg)
        e["objects"][0]["attributes"] = {"colour": COLOURS[len(sg["scene_id"]) % len(COLOURS)]}
        cands.append(("CF-A", e, {"field": "colour", "insert": objs[0]["oid"]}))
    prio = {"binding": ["CF-C", "CF-A", "CF-O"], "multi": ["CF-C", "CF-R", "CF-N", "CF-A"], "counting": ["CF-N", "CF-A"],
            "spatial": ["CF-R", "CF-O"], "presence": ["CF-O", "CF-A"]}[sg["category"]]
    chosen = [c for p in prio for c in cands if c[0] == p][: cfg["max_cf_per_scene"]]
    for typ, e, edit in chosen:
        e = finish(e)
        v.append(dict(type=typ, target_edit=edit, prompt=render(e), constraints=constraints_of(e),
                      preserve=[c for c in canon(base) if c in canon(constraints_of(e))]))
    for x in v:
        x["variant_id"] = f"{sg['scene_id']}_{x['type'].replace('-', '')}"
        x["review"] = {"reviewer": None, "status": "pending"}
    return v


def generate(cfg):
    rng = random.Random(cfg["seed"])
    nouns = sorted(NOUNS)
    pilot = json.loads(PILOT_PAIRS_PATH.read_text())
    pilot_groups = {"|".join(sorted({c["object"] for c in p["constraints"] if "object" in c} | {c[k] for c in p["constraints"] for k in ("left", "right") if k in c})) for p in pilot}
    pilot_nouns = {n for g in pilot_groups for n in g.split("|")}
    # nouns reserved for counting (singleton groups), split-disjoint
    pool = sorted(nouns)
    rng.shuffle(pool)
    cn = {s: [pool.pop() for _ in range(k)] for s, k in cfg["counting_nouns_per_split"].items()}
    allowed = [(a, b) for a, b in itertools.combinations(nouns, 2) if not conflict(a, b)]
    rng.shuffle(allowed)
    pilot_pairs = [p for p in allowed if "|".join(sorted(p)) in pilot_groups]
    fresh = [p for p in allowed if "|".join(sorted(p)) not in pilot_groups]
    scenes, used_groups, n = [], set(), 0
    pair_cursor = {"DEV": iter(pilot_pairs + fresh[: len(fresh) // 3]), "TEST": iter(fresh[len(fresh) // 3:])}  # pilot pairs only in DEV
    colour_cycle = {s: itertools.cycle(rng.sample(COLOURS, len(COLOURS))) for s in ("DEV", "TEST")}
    for split in ("DEV", "TEST"):
        for cat, q in cfg["quotas"][split].items():
            for i in range(q):
                n += 1
                sid = f"sg_{n:04d}"
                bg = rng.choice(cfg["backgrounds"])

                def two_colours():
                    a = next(colour_cycle[split]); b = next(colour_cycle[split])
                    while b == a:
                        b = next(colour_cycle[split])
                    return a, b
                if cat == "counting":
                    noun = cn[split][i % len(cn[split])]
                    c = next(colour_cycle[split])
                    lo, hi = cfg["counting_range"]
                    objs = [make_obj(1, noun, rng.randint(lo, hi), c)]
                    rels = []
                else:
                    a, b = next(pair_cursor[split])
                    while a in sum(cn.values(), []) or b in sum(cn.values(), []):
                        a, b = next(pair_cursor[split])
                    if rng.random() < 0.5:
                        a, b = b, a
                    tmpl = None
                    if cat == "multi":
                        w = cfg["multi_templates"]
                        tmpl = rng.choices(list(w), weights=list(w.values()))[0]
                    if cat == "presence":
                        objs, rels = [make_obj(1, a), make_obj(2, b)], []
                    elif cat == "binding":
                        ca, cb = two_colours()
                        objs, rels = [make_obj(1, a, 1, ca), make_obj(2, b, 1, cb)], []
                    elif cat == "spatial":
                        objs, rels = [make_obj(1, a), make_obj(2, b)], [{"rel": rng.choice(list(REL_TEXT)), "subject": "o1", "object": "o2"}]
                    elif tmpl == "bind_spatial":
                        ca, cb = two_colours()
                        objs, rels = [make_obj(1, a, 1, ca), make_obj(2, b, 1, cb)], [{"rel": rng.choice(list(REL_TEXT)), "subject": "o1", "object": "o2"}]
                    elif tmpl == "bind_count":
                        ca, cb = two_colours()
                        lo, hi = cfg["counting_range"]
                        objs, rels = [make_obj(1, a, rng.randint(lo, hi), ca), make_obj(2, b, 1, cb)], []
                    else:  # count_spatial_nocolour
                        lo, hi = cfg["counting_range"]
                        objs, rels = [make_obj(1, a, rng.randint(lo, hi)), make_obj(2, b)], [{"rel": rng.choice(list(REL_TEXT)), "subject": "o1", "object": "o2"}]
                sg = finish({"scene_id": sid, "split": split, "category": cat, "objects": objs, "relations": rels, "background": bg})
                sg["variants"] = build_variants(sg, cfg)
                scenes.append(sg)
    return scenes


# ----- design counts (D-05) ------------------------------------------------------------------------------------------
def is_binding_bearing(sg):
    return any(c["type"] == "binding" for c in sg["variants"][0]["constraints"])


def is_binding_competitive(sg):
    return len({o["attributes"]["colour"] for o in sg["objects"] if o["attributes"].get("colour")}) >= 2


def design_counts(scenes):
    out = {}
    for split in ("DEV", "TEST"):
        S = [s for s in scenes if s["split"] == split]
        out[split] = dict(
            scenes=len(S), by_category=dict(Counter(s["category"] for s in S)),
            binding_bearing=sum(map(is_binding_bearing, S)),
            binding_bearing_by_category=dict(Counter(s["category"] for s in S if is_binding_bearing(s))),
            binding_competitive=sum(map(is_binding_competitive, S)),
            spatial_bearing=sum(1 for s in S if s["relations"]),
            variants=sum(len(s["variants"]) for s in S), variants_by_type=dict(Counter(v["type"] for s in S for v in s["variants"])),
            n_constraints_distribution=dict(sorted(Counter(s["complexity"]["n_constraints"] for s in S).items())))
    return out


# ----- validators ---------------------------------------------------------------------------------------------------
def graph_diff(a, b):
    """Set of declared-field names in which two scene graphs differ."""
    d = set()
    if [o["noun"] for o in a["objects"]] != [o["noun"] for o in b["objects"]]:
        d.add("objects")
    else:
        for x, y in zip(a["objects"], b["objects"]):
            if x["count"] != y["count"]:
                d.add("count")
            if x["attributes"] != y["attributes"]:
                d.add("colour")
    if a["relations"] != b["relations"]:
        d.add("relation")
    if a["background"] != b["background"]:
        d.add("background")
    return d


def validate(scenes, cfg):
    errs = []
    ids = [s["scene_id"] for s in scenes]
    if len(ids) != len(set(ids)):
        errs.append("duplicate scene ids")
    groups = defaultdict(set)
    for s in scenes:
        groups[s["object_pair_group"]].add(s["split"])
    errs += [f"group {g} in several splits {sorted(v)}" for g, v in groups.items() if len(v) > 1]
    pilot = json.loads(PILOT_PAIRS_PATH.read_text())
    pg = {"|".join(sorted({c["object"] for c in p["constraints"] if "object" in c} | {c[k] for c in p["constraints"] for k in ("left", "right") if k in c})) for p in pilot}
    errs += [f"pilot group {s['object_pair_group']} in TEST" for s in scenes if s["split"] == "TEST" and s["object_pair_group"] in pg]
    texts = Counter(re.sub(r"\s+", " ", v["prompt"].lower()).strip() for s in scenes for v in s["variants"] if v["type"] == "CAN")
    errs += [f"duplicate canonical prompt: {t}" for t, c in texts.items() if c > 1]
    for s in scenes:
        nouns = [o["noun"] for o in s["objects"]]
        if len(nouns) != len(set(nouns)) or any(conflict(a, b) for a, b in itertools.combinations(nouns, 2)):
            errs.append(f"{s['scene_id']}: repeated or conflicting nouns")
        can = s["variants"][0]
        if can["type"] != "CAN" or canon(can["constraints"]) != canon(constraints_of(s)):
            errs.append(f"{s['scene_id']}: CAN constraints inconsistent with graph")
        for v in s["variants"][1:]:
            if v["type"] in ("PAR", "ORD") and canon(v["constraints"]) != canon(can["constraints"]):
                errs.append(f"{v['variant_id']}: meaning-preserving variant changes constraints")
            if v["type"] == "ORD" and v["prompt"] == can["prompt"]:
                errs.append(f"{v['variant_id']}: ORD identical to CAN")
            if v["type"].startswith("CF"):
                if canon(v["constraints"]) == canon(can["constraints"]):
                    errs.append(f"{v['variant_id']}: counterfactual does not change constraints")
        for v in s["variants"]:   # every constrained noun/colour must be present in the text (modulo declared synonyms)
            low = v["prompt"].lower()
            for c in v["constraints"]:
                for k in ("object", "left", "right"):
                    if k in c and c[k] not in low and NOUN_SYN.get(c[k], "\0") not in low:
                        errs.append(f"{v['variant_id']}: noun {c[k]} missing from text")
                if c["type"] == "binding" and c["color"] not in low and COLOUR_SYN[c["color"]] not in low:
                    errs.append(f"{v['variant_id']}: colour {c['color']} missing from text")
    # counterfactual: re-derive the edit and require exactly one declared field class to differ
    for s in scenes:
        for v in s["variants"]:
            if not v["type"].startswith("CF"):
                continue
            e = copy.deepcopy(s)
            ed = v["target_edit"]
            if v["type"] == "CF-C":
                a, b = e["objects"][0], e["objects"][1]
                a["attributes"], b["attributes"] = b["attributes"], a["attributes"]
            elif v["type"] == "CF-R":
                e["relations"][0]["rel"] = CONVERSE[e["relations"][0]["rel"]]
            elif v["type"] == "CF-N":
                next(o for o in e["objects"] if o["oid"] == ed["oid"])["count"] = ed["to"]
            elif v["type"] == "CF-O":
                e["objects"] = e["objects"][:1]
                e["relations"] = []
            elif v["type"] == "CF-A":
                t = next(o for o in e["objects"] if o["oid"] == (ed.get("remove") or ed.get("insert")))
                t["attributes"] = {} if "remove" in ed else {"colour": json.loads(json.dumps(v["constraints"]))[-1].get("color") or next(c["color"] for c in v["constraints"] if c["type"] == "binding" and c["object"] == t["noun"])}
            diff = graph_diff(s, e)
            if v["type"] == "CF-O":   # a relation that referenced the deleted object disappears with it (dependent, not a second edit)
                diff.discard("relation")
            if len(diff) != 1 or ed["field"] not in diff:
                errs.append(f"{v['variant_id']}: edit differs in {sorted(diff)}, declared {ed['field']}")
            if canon(constraints_of(e)) != canon(v["constraints"]):
                errs.append(f"{v['variant_id']}: constraints do not follow from the edited graph")
            if v["type"] == "CF-N" and not (cfg["counting_range"][0] <= ed["to"] <= cfg["counting_range"][1]):
                errs.append(f"{v['variant_id']}: count outside range")
    for split, q in cfg["quotas"].items():
        got = Counter(s["category"] for s in scenes if s["split"] == split)
        if dict(got) != q:
            errs.append(f"{split} quotas {dict(got)} != {q}")
    return errs


def sha(path):
    """SHA-256 with CRLF normalised to LF for text files, so a Windows (autocrlf) checkout and the Linux Kaggle copy agree."""
    b = Path(path).read_bytes()
    if Path(path).suffix in (".json", ".md", ".py", ".txt", ".jsonl"):
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def flat_prompts(scenes):
    rows = []
    for s in scenes:
        for v in s["variants"]:
            rows.append({"id": v["variant_id"], "scene_id": s["scene_id"], "split": s["split"], "category": s["category"], "variant": v["type"],
                         "prompt": v["prompt"], "constraints": v["constraints"], "pair_group": s["object_pair_group"],
                         "n_objects": len(s["objects"]), "n_constraints": len(v["constraints"])})
    return rows


def build():
    cfg = json.loads((ROOT / "configs" / "benchmark_v2.json").read_text())
    scenes = generate(cfg)
    errs = validate(scenes, cfg)
    if errs:
        sys.exit("validation failed at build:\n" + "\n".join(errs[:30]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scene_graphs.json").write_text(json.dumps(scenes, indent=1))
    (OUT / "prompts_v2.json").write_text(json.dumps(flat_prompts(scenes), indent=1))
    (OUT / "splits.json").write_text(json.dumps({s: [x["scene_id"] for x in scenes if x["split"] == s] for s in ("DEV", "TEST")}, indent=1))
    (OUT / "lexicon.json").write_text(json.dumps({"status": "DRAFT", "nouns": NOUNS, "conflicts": [sorted(c) for c in CONFLICTS], "colours": COLOURS,
                                                   "colour_synonyms": COLOUR_SYN, "noun_synonyms": NOUN_SYN, "relations": REL_TEXT, "relation_synonyms": REL_SYN}, indent=1))
    (OUT / "design_counts.json").write_text(json.dumps(design_counts(scenes), indent=1))
    (OUT / "hashes.json").write_text(json.dumps({f: sha(OUT / f) for f in ("scene_graphs.json", "prompts_v2.json", "splits.json", "lexicon.json", "design_counts.json")}
                                                 | {"generator": sha(__file__), "config": sha(ROOT / "configs" / "benchmark_v2.json")}, indent=1))
    print(json.dumps(design_counts(scenes), indent=1))


def check():
    cfg = json.loads((ROOT / "configs" / "benchmark_v2.json").read_text())
    scenes = json.loads((OUT / "scene_graphs.json").read_text())
    errs = validate(scenes, cfg)
    regen = json.dumps(generate(cfg), indent=1) == json.dumps(scenes, indent=1)
    hashes = json.loads((OUT / "hashes.json").read_text())
    stale = [f for f in ("scene_graphs.json", "prompts_v2.json", "splits.json", "lexicon.json", "design_counts.json") if sha(OUT / f) != hashes[f]]
    print(f"validation errors: {len(errs)}; regenerates identically: {regen}; stale hashes: {stale}")
    for e in errs[:30]:
        print(" ", e)
    sys.exit(0 if not errs and regen and not stale else 1)

# ===== Revision 3: problem-statement (PS) flagship family with relative size (RB 24.6) ===================================
# Additive: written to separate files (ps_family.json, prompts_ps.json, splits_ps.json, hashes_ps.json) so that the main benchmark
# files, their hashes and every run already made from them (R2a) stay valid. Size exists only in this family.
PS_SUBSETS = ["P", "C", "S", "R", "CS", "CR", "SR", "CSR"]          # on top of presence: Colour, Size, spatial Relation
TABLETOP = ["cup", "book", "vase", "clock", "bottle", "bowl", "teapot", "lamp", "hat", "shoe", "ball", "apple", "banana", "cube",
            "sphere", "pillow", "basket", "candle", "box", "laptop", "backpack", "mirror"]


def ps_phrase(o, use_colour, use_size):
    words = [o["attributes"]["size"]] if use_size else []
    words += [o["attributes"]["colour"]] if use_colour else []
    w = " ".join(words + [o["noun"]])
    return f"{art(w)} {w}"


def ps_render(sg):
    sub = sg["ps"]["subset"]
    a, b = sg["objects"]
    uc, us = "C" in sub, "S" in sub
    if "R" in sub:
        r = sg["relations"][0]["rel"]
        return f"{ps_phrase(a, uc, us)} {REL_TEXT[r]} {ps_phrase(b, uc, us)} {sg['background']}"
    return f"{ps_phrase(a, uc, us)} and {ps_phrase(b, uc, us)} {sg['background']}"


def ps_constraints(sg):
    sub = sg["ps"]["subset"]
    a, b = sg["objects"]
    cons = [{"type": "presence", "object": a["noun"]}, {"type": "presence", "object": b["noun"]}]
    if "C" in sub:
        cons += [{"type": "binding", "object": o["noun"], "color": o["attributes"]["colour"]} for o in (a, b)]
    if "S" in sub:
        big, small = (a, b) if a["attributes"]["size"] == "large" else (b, a)
        cons.append({"type": "size", "larger": big["noun"], "smaller": small["noun"]})
    if "R" in sub:
        r = sg["relations"][0]
        left, right = (a, b) if r["rel"] == "left_of" else (b, a)
        cons.append({"type": "spatial", "left": left["noun"], "right": right["noun"]})
    return cons


def ps_variants(sg):
    """CAN for every subset; counterfactuals (exactly one declared field) only for the full subset CSR, which E31 uses:
    CF-C colour swap, CF-S size swap, CF-R relation reversal."""
    v = [dict(type="CAN", prompt=ps_render(sg), constraints=ps_constraints(sg))]
    if sg["ps"]["subset"] == "CSR":
        for typ, field in (("CF-C", "colour"), ("CF-S", "size"), ("CF-R", "relation")):
            e = copy.deepcopy(sg)
            a, b = e["objects"]
            if field == "relation":
                e["relations"][0]["rel"] = CONVERSE[e["relations"][0]["rel"]]
            else:
                a["attributes"][field], b["attributes"][field] = b["attributes"][field], a["attributes"][field]
            v.append(dict(type=typ, target_edit={"field": field}, prompt=ps_render(e), constraints=ps_constraints(e), _graph=e))
    for x in v:
        x["variant_id"] = f"{sg['scene_id']}_{x['type'].replace('-', '')}"
        x["review"] = {"reviewer": None, "status": "pending"}
    return v


def ps_graph_diff(a, b):
    d = set()
    if [o["noun"] for o in a["objects"]] != [o["noun"] for o in b["objects"]]:
        d.add("objects")
    for x, y in zip(a["objects"], b["objects"]):
        for k in ("colour", "size"):
            if x["attributes"].get(k) != y["attributes"].get(k):
                d.add(k)
    if a["relations"] != b["relations"]:
        d.add("relation")
    return d


def generate_ps(cfg, main_scenes):
    p = cfg["ps_family"]
    rng = random.Random(p["seed"])
    main_groups = defaultdict(set)
    for x in main_scenes:
        main_groups[x["split"]].add(x["object_pair_group"])
    pilot = json.loads(PILOT_PAIRS_PATH.read_text())
    pilot_groups = {"|".join(sorted({c["object"] for c in q["constraints"] if "object" in c} | {c[k] for c in q["constraints"] for k in ("left", "right") if k in c})) for q in pilot}
    pairs = [tuple(sorted(x)) for x in itertools.combinations(sorted(TABLETOP), 2) if not conflict(*x)]
    rng.shuffle(pairs)
    chosen = {"DEV": [], "TEST": []}
    for split, other in (("DEV", "TEST"), ("TEST", "DEV")):
        for pr in pairs:
            g = "|".join(pr)
            taken = {q for s in chosen.values() for q in s}
            if len(chosen[split]) >= p["pairs"][split] or pr in taken or g in main_groups[other] or (split == "TEST" and g in pilot_groups):
                continue
            chosen[split].append(pr)
    scenes = []
    for split in ("DEV", "TEST"):
        for k, (n1, n2) in enumerate(chosen[split]):
            pid = f"ps{split[0]}{k:02d}"
            ca, cb = rng.sample(COLOURS, 2)
            big_first = rng.random() < 0.5
            rel = rng.choice(list(REL_TEXT))
            objs = [make_obj(1, n1, 1, ca), make_obj(2, n2, 1, cb)]
            objs[0]["attributes"]["size"], objs[1]["attributes"]["size"] = ("large", "small") if big_first else ("small", "large")
            for sub in PS_SUBSETS:
                sg = {"scene_id": f"{pid}_{sub}", "split": split, "category": "ps_family", "ps": {"pair_id": pid, "subset": sub},
                      "objects": copy.deepcopy(objs), "relations": [{"rel": rel, "subject": "o1", "object": "o2"}] if "R" in sub else [],
                      "background": p["background"], "object_pair_group": "|".join(sorted((n1, n2))), "shard_group": pid}
                sg["size_relation"] = ({"rel": "larger_than", "subject": "o1" if big_first else "o2", "object": "o2" if big_first else "o1"}
                                       if "S" in sub else None)
                sg["variants"] = ps_variants(sg)
                scenes.append(sg)
    return scenes


def validate_ps(ps, main_scenes, cfg):
    errs = []
    ids = [s["scene_id"] for s in ps] + [s["scene_id"] for s in main_scenes]
    if len(ids) != len(set(ids)):
        errs.append("scene ids not unique across main benchmark and PS family")
    pids = [v["variant_id"] for s in ps for v in s["variants"]] + [v["variant_id"] for s in main_scenes for v in s["variants"]]
    if len(pids) != len(set(pids)):
        errs.append("prompt ids not unique")
    by_group = defaultdict(set)
    for s in ps + main_scenes:
        by_group[s["object_pair_group"]].add(s["split"])
    errs += [f"pair group {g} in several splits" for g, v in by_group.items() if len(v) > 1]
    pairs = defaultdict(set)
    for s in ps:
        pairs[s["ps"]["pair_id"]].add(s["split"])
    for pid in pairs:
        subs = sorted(x["ps"]["subset"] for x in ps if x["ps"]["pair_id"] == pid)
        if subs != sorted(PS_SUBSETS):
            errs.append(f"{pid}: subsets {subs} incomplete")
    errs += [f"PS pair {k} in several splits" for k, v in pairs.items() if len(v) > 1]
    for split in ("DEV", "TEST"):
        n = len({s["ps"]["pair_id"] for s in ps if s["split"] == split})
        if n != cfg["ps_family"]["pairs"][split]:
            errs.append(f"{split}: {n} PS pairs, expected {cfg['ps_family']['pairs'][split]}")
    want = {"C": {"binding"}, "S": {"size"}, "R": {"spatial"}}
    for s in ps:
        can = s["variants"][0]
        types = {c["type"] for c in can["constraints"]} - {"presence"}
        if types != set().union(*[want[ch] for ch in s["ps"]["subset"] if ch in want]):
            errs.append(f"{can['variant_id']}: constraint types {sorted(types)} do not match subset {s['ps']['subset']}")
        for v in s["variants"]:
            low = v["prompt"].lower()
            for c in v["constraints"]:
                for k in ("object", "left", "right", "larger", "smaller"):
                    if k in c and c[k] not in low:
                        errs.append(f"{v['variant_id']}: noun {c[k]} missing from text")
                if c["type"] == "binding" and c["color"] not in low:
                    errs.append(f"{v['variant_id']}: colour {c['color']} missing from text")
        words = can["prompt"].lower().split()
        if ("S" in s["ps"]["subset"]) != ("small" in words and "large" in words):
            errs.append(f"{can['variant_id']}: size words inconsistent with subset")
        for v in s["variants"][1:]:
            if "_graph" in v and ps_graph_diff(s, v["_graph"]) != {v["target_edit"]["field"]}:
                errs.append(f"{v['variant_id']}: edit differs in {sorted(ps_graph_diff(s, v['_graph']))}, declared {v['target_edit']['field']}")
            if canon(v["constraints"]) == canon(can["constraints"]):
                errs.append(f"{v['variant_id']}: counterfactual does not change constraints")
    return errs


def ps_flat(ps):
    rows = []
    for s in ps:
        for v in s["variants"]:
            rows.append({"id": v["variant_id"], "scene_id": s["scene_id"], "split": s["split"], "category": "ps_family", "variant": v["type"],
                         "prompt": v["prompt"], "constraints": v["constraints"], "pair_group": s["object_pair_group"], "shard_group": s["shard_group"],
                         "ps_pair_id": s["ps"]["pair_id"], "ps_subset": s["ps"]["subset"], "n_objects": 2, "n_constraints": len(v["constraints"])})
    return rows


PS_FILES = ("ps_family.json", "prompts_ps.json", "splits_ps.json")


def ps_clean(ps):
    return [dict(s, variants=[{k: x for k, x in v.items() if k != "_graph"} for v in s["variants"]]) for s in ps]


def build_ps():
    cfg = json.loads((ROOT / "configs" / "benchmark_v2.json").read_text())
    main = json.loads((OUT / "scene_graphs.json").read_text())
    ps = generate_ps(cfg, main)
    errs = validate_ps(ps, main, cfg)
    if errs:
        sys.exit("PS family validation failed:\n" + "\n".join(errs[:30]))
    (OUT / "ps_family.json").write_text(json.dumps(ps_clean(ps), indent=1))
    (OUT / "prompts_ps.json").write_text(json.dumps(ps_flat(ps), indent=1))
    (OUT / "splits_ps.json").write_text(json.dumps({sp: sorted({s["ps"]["pair_id"] for s in ps if s["split"] == sp}) for sp in ("DEV", "TEST")}, indent=1))
    (OUT / "hashes_ps.json").write_text(json.dumps({f: sha(OUT / f) for f in PS_FILES} | {"generator": sha(__file__), "config": sha(ROOT / "configs" / "benchmark_v2.json")}, indent=1))
    print(f"PS family: {len(ps)} scenes, {sum(len(s['variants']) for s in ps)} prompts")


def check_ps():
    cfg = json.loads((ROOT / "configs" / "benchmark_v2.json").read_text())
    main = json.loads((OUT / "scene_graphs.json").read_text())
    regen = generate_ps(cfg, main)
    errs = validate_ps(regen, main, cfg)
    same = json.loads((OUT / "prompts_ps.json").read_text()) == ps_flat(regen)
    hashes = json.loads((OUT / "hashes_ps.json").read_text())
    stale = [f for f in PS_FILES if sha(OUT / f) != hashes[f]]
    print(f"PS family validation errors: {len(errs)}; regenerates identically: {same}; stale hashes: {stale}")
    for e in errs[:30]:
        print(" ", e)
    return not errs and same and not stale


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--build-ps", action="store_true", help="Revision 3: build the PS flagship family with relative size (separate files)")
    ap.add_argument("--validate-ps", action="store_true")
    a = ap.parse_args()
    if a.build_ps:
        build_ps()
    elif a.validate_ps:
        sys.exit(0 if check_ps() else 1)
    else:
        build() if a.build else check()
