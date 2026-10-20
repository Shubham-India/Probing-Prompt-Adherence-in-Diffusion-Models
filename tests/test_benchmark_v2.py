import copy, json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import benchmark_v2 as B

CFG = json.loads((ROOT / "configs/benchmark_v2.json").read_text())
SCENES = json.loads((ROOT / "data/benchmark_v2/scene_graphs.json").read_text())


def test_saved_benchmark_validates_and_regenerates():
    assert B.validate(SCENES, CFG) == []
    assert json.dumps(B.generate(CFG), indent=1) == json.dumps(SCENES, indent=1)


def test_quotas_and_split_disjointness():
    for split, q in CFG["quotas"].items():
        assert {c: sum(1 for s in SCENES if s["split"] == split and s["category"] == c) for c in q} == q
    groups = {}
    for s in SCENES:
        groups.setdefault(s["object_pair_group"], set()).add(s["split"])
    assert all(len(v) == 1 for v in groups.values())


def test_binding_bearing_counts_derived_independently():
    """Recount from the structured objects (not from the constraint strings the module uses)."""
    for split in ("DEV", "TEST"):
        S = [s for s in SCENES if s["split"] == split]
        indep = sum(1 for s in S if any(o["attributes"].get("colour") for o in s["objects"]))
        assert B.design_counts(SCENES)[split]["binding_bearing"] == indep
    dc = json.loads((ROOT / "data/benchmark_v2/design_counts.json").read_text())
    assert dc == json.loads(json.dumps(B.design_counts(SCENES)))


def test_nonbinding_multi_template_is_not_counted():
    cfg = copy.deepcopy(CFG)
    cfg["multi_templates"] = {"bind_spatial": 0.0, "bind_count": 0.0, "count_spatial_nocolour": 1.0}
    sc = B.generate(cfg)
    assert B.validate(sc, cfg) == []
    multi = [s for s in sc if s["category"] == "multi"]
    assert multi and not any(B.is_binding_bearing(s) for s in multi)


def test_seeded_corruptions_are_detected():
    s = copy.deepcopy(SCENES)
    cf = next(v for sc in s for v in sc["variants"] if v["type"] == "CF-C")
    cf["constraints"] = [c for c in cf["constraints"] if c["type"] != "binding"]          # constraints no longer follow from the edit
    assert any("CF-C" in e or "CFC" in e for e in B.validate(s, CFG))
    s = copy.deepcopy(SCENES)
    next(v for sc in s for v in sc["variants"] if v["type"] == "PAR")["constraints"] = []   # meaning-preserving variant changed
    assert any("meaning-preserving" in e for e in B.validate(s, CFG))
    s = copy.deepcopy(SCENES)
    t = next(sc for sc in s if sc["split"] == "TEST" and sc["category"] == "binding")
    t["object_pair_group"] = next(sc for sc in s if sc["split"] == "DEV" and sc["category"] == "binding")["object_pair_group"]
    assert any("several splits" in e for e in B.validate(s, CFG))
    s = copy.deepcopy(SCENES)
    s[1]["variants"][0]["prompt"] = s[0]["variants"][0]["prompt"]                            # duplicate canonical prompt
    assert any("duplicate canonical" in e for e in B.validate(s, CFG))


def test_pilot_pairs_never_in_test():
    pilot = json.loads((ROOT / "data/prompts.json").read_text())
    pg = {p["pair_group"] for p in pilot}
    assert pg and not any(s["split"] == "TEST" and s["object_pair_group"] in {"|".join(sorted(g.split("_")[1:])) for g in pg} for s in SCENES)


def test_legacy_scorer_accepts_v2_constraints():
    for v in json.loads((ROOT / "data/benchmark_v2/prompts_v2.json").read_text())[:200]:
        assert all(c["type"] in {"presence", "count", "binding", "spatial"} for c in v["constraints"])
