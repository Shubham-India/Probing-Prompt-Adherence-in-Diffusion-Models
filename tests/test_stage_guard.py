import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import stage_guard as G


def run(*args):
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stage_guard.py"), *args], capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout)


def test_every_true_gate_is_backed_by_a_recorded_owner_decision():
    """Gates default to false; a gate may be true only when an owner-decision row of docs/R3_DECISION_LOG.md names it (e.g. DL-39)."""
    gates = json.loads((ROOT / "configs" / "stage_gates.json").read_text())
    true = [k for k, v in gates.items() if not k.startswith("_") and v is True]
    assert all(v in (True, False) for k, v in gates.items() if not k.startswith("_"))
    owner_rows = [l for l in (ROOT / "docs" / "R3_DECISION_LOG.md").read_text(encoding="utf-8").splitlines()
                  if l.startswith("| DL-") and "Owner decisions" in l and l.rstrip().endswith("| owner |")]
    assert [k for k in true if not any(f"`{k}`" in row for row in owner_rows)] == []


def test_every_stage_is_refused_while_gates_are_false():
    for stage in ("r2b", "r3", "e22", "r4"):
        code, out = run("--stage", stage)
        assert code == 1 and out["ok"] is False and out["blockers"]


def test_r4_refused_without_freeze_even_if_gates_were_true(tmp_path):
    g = tmp_path / "g.json"
    g.write_text(json.dumps({k: True for k in G.REQUIRED_GATES["r4"]}))
    code, out = run("--stage", "r4", "--gates", str(g))
    assert code == 1 and any(b.startswith("freeze:") for b in out["blockers"])


def test_r2b_plan_excludes_canonical_variants(tmp_path):
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"variant_review_complete": True}))
    cfg = json.loads((ROOT / "configs" / "experiment_v2.json").read_text())
    prompts = json.loads((ROOT / "data" / "benchmark_v2" / "prompts_v2.json").read_text())
    plan = G.plan_r2b(prompts, cfg)
    n_can = sum(1 for p in prompts if p["split"] == "DEV" and p["variant"] == "CAN")
    n_dev = sum(1 for p in prompts if p["split"] == "DEV")
    assert plan["prompts"] == n_dev - n_can == 370 and plan["jobs"] == 370 * 4


def test_r3_plan_matches_run_branches_enumeration():
    cfg = json.loads((ROOT / "configs" / "experiment_v2.json").read_text())
    recs = [{"name": f"s{s}", "seed": s, "variant": "CAN"} for s in range(4)]
    p = G.plan_r3(recs, cfg, ["D8", "ER30", "FULL"], 2)
    assert p["jobs"] == 2 * 6 * 3 * 2 and p["base_seeds"] == [0, 1]
    p4 = G.plan_r3(recs, cfg, ["D8", "ER30", "FULL"], 2, bases=4)
    assert p4["jobs"] == 2 * p["jobs"] and p4["wall_hours_2_workers"][1] > p["wall_hours_2_workers"][1]
    # per-100-scene value reproduces the figure in docs/R0_R1_AUDIT.md (7,200 jobs, 5,730 image-equivalents)
    big = [{"name": f"s{i}_{s}", "seed": s, "variant": "CAN"} for i in range(100) for s in range(4)]
    pb = G.plan_r3(big, cfg, ["D8", "ER30", "FULL"], 2)
    assert pb["jobs"] == 7200 and pb["image_equivalents"] == 5730.0
