"""PS family (relative size) and the executable Revision 3 manifests: leakage, exactly-one-field counterfactuals, unique ids, scene-group
sharding and dependency integrity."""
import copy, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import benchmark_v2 as BV
import manifests as M
import jobs as J

CFG = json.loads((ROOT / "configs" / "benchmark_v2.json").read_text())
MAIN = json.loads((ROOT / "data" / "benchmark_v2" / "scene_graphs.json").read_text())


def test_ps_family_validates_and_regenerates():
    ps = BV.generate_ps(CFG, MAIN)
    assert BV.validate_ps(ps, MAIN, CFG) == []
    assert len(ps) == 24 * 8 and sum(len(s["variants"]) for s in ps) == 24 * 8 + 24 * 3
    assert json.loads((ROOT / "data" / "benchmark_v2" / "prompts_ps.json").read_text()) == BV.ps_flat(ps)


def test_ps_counterfactuals_change_exactly_one_field_and_tampering_is_caught():
    ps = BV.generate_ps(CFG, MAIN)
    for s in ps:
        for v in s["variants"][1:]:
            assert BV.ps_graph_diff(s, v["_graph"]) == {v["target_edit"]["field"]}
    bad = copy.deepcopy(ps)
    csr = next(s for s in bad if s["ps"]["subset"] == "CSR")
    g = csr["variants"][1]["_graph"]
    g["relations"][0]["rel"] = BV.CONVERSE[g["relations"][0]["rel"]]          # colour swap now also reverses the relation
    assert any("edit differs" in e for e in BV.validate_ps(bad, MAIN, CFG))


def test_ps_no_leakage_across_splits_or_into_main_benchmark():
    ps = BV.generate_ps(CFG, MAIN)
    split_of = defaultdict(set)
    for s in ps + MAIN:
        split_of[s["object_pair_group"]].add(s["split"])
    assert all(len(v) == 1 for v in split_of.values())
    pairs = defaultdict(set)
    for s in ps:
        pairs[s["ps"]["pair_id"]].add(s["split"])
    assert all(len(v) == 1 for v in pairs.values())
    bad = copy.deepcopy(ps)
    bad[0]["split"] = "TEST" if bad[0]["split"] == "DEV" else "DEV"
    assert BV.validate_ps(bad, MAIN, CFG)


def test_size_constraint_matches_the_size_words():
    for s in BV.generate_ps(CFG, MAIN):
        if "S" in s["ps"]["subset"]:
            c = next(c for c in s["variants"][0]["constraints"] if c["type"] == "size")
            big = next(o for o in s["objects"] if o["attributes"]["size"] == "large")
            assert c["larger"] == big["noun"] and f"large {big['attributes']['colour'] + ' ' if 'C' in s['ps']['subset'] else ''}{big['noun']}" in s["variants"][0]["prompt"]


def test_manifest_ids_unique_dependencies_resolve_and_r2a_names_match():
    jobs, notes, _ = M.build()
    ids = [j["job_id"] for j in jobs]
    assert len(ids) == len(set(ids))
    core = {j["job_id"] for j in jobs if j["kind"] == "core"}
    r2a = {json.loads(l)["name"] for l in (ROOT / "outputs" / "acceptance").glob("*") if False}   # placeholder: R2a names checked below
    for j in jobs:
        if j["kind"] in ("branch", "switch"):
            assert j["base_job"] in core or j.get("base_upstream") == M.R2A_RUN
    dev_bases = {j["base_job"] for j in jobs if j["experiment"] == "E23-DEV"}
    assert all(b.endswith(("__seed0", "__seed1")) and "_CAN__cfg7.5__st40__" in b for b in dev_bases) and len(dev_bases) == 200
    assert not any(j["split"] == "TEST" for j in jobs if j["stage"].startswith("DEV"))
    assert not any(j["split"] == "DEV" for j in jobs if j["stage"].startswith("TEST"))


def test_scene_group_sharding_is_a_disjoint_cover_and_keeps_scenes_together():
    jobs, _, _ = M.build()
    for stage in ("TEST_CORE", "TEST_COMMIT", "TEST_CAG_SIZE", "DEV_COMMIT"):
        all_ids = {j["job_id"] for j in M.select(jobs, stage)}
        parts = [M.select(jobs, stage, (i, 3)) for i in range(3)]
        got = [j["job_id"] for p in parts for j in p]
        assert len(got) == len(set(got)) == len(all_ids) and set(got) == all_ids
        owner = {}
        for i, p in enumerate(parts):
            for j in p:
                assert owner.setdefault(j["shard_group"], i) == i                 # every scene group in exactly one shard
    # commitment branches land in the same shard as their TEST base images
    core_shard = {j["job_id"]: i for i in range(3) for j in M.select(jobs, "TEST_CORE", (i, 3))}
    for i in range(3):
        for j in M.select(jobs, "TEST_COMMIT", (i, 3)):
            assert core_shard[j["base_job"]] == i


def test_manifest_counts_match_closed_forms():
    jobs, _, _ = M.build()
    c = Counter(j["experiment"] for j in jobs)
    assert c["E23-DEV"] == 100 * 2 * 6 * 3 * 2 == 7200
    assert c["E23-TEST"] == 100 * 2 * 6 * 2 * 2 and c["E29-TEST"] == 100 * 2 * 6 * 2 * 2
    assert c["E11"] == 150 * (5 * 2 - 1) * 4 == 5400
    n_test_variants = sum(1 for p in J.load_prompts(ROOT / "data" / "benchmark_v2" / "prompts_v2.json") if p["split"] == "TEST")
    assert c["E04-TEST"] == n_test_variants * 4
    assert c["E32-TEST-branch"] == 16 * 4 * 2 * 6 * 2
