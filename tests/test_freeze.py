import copy, json, shutil, subprocess, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import freeze as FZ, commitment as C, evaluate as E, config_schema as CS


def stage(tmp_path):
    for rel in FZ.collect(ROOT):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    return tmp_path


def freeze_it(root):
    cfgp = root / "configs/experiment_v2.json"
    cfg = json.loads(cfgp.read_text()); cfg["revision"] = "a" * 40; cfg["split_manifest"] = "data/benchmark_v2/splits.json"
    cfgp.write_text(json.dumps(cfg, indent=1))
    man = FZ.build_manifest(root)
    (root / "docs/freeze").mkdir(parents=True, exist_ok=True)
    (root / "docs/freeze/freeze_manifest.json").write_text(json.dumps(man))
    h = FZ.manifest_hash(man)
    (root / "docs/freeze/ADDENDUM_HASH").write_text(h + "\n")
    cfg["freeze"]["addendum_hash"] = h
    cfgp.write_text(json.dumps(cfg, indent=1))                    # storing the hash must not invalidate it (canonical config bytes)
    return h


def test_repository_is_not_frozen_today():
    assert FZ.verify(ROOT)                                          # no manifest / unpinned revision => problems reported


def test_freeze_verify_roundtrip_and_tamper_detection(tmp_path):
    root = stage(tmp_path)
    freeze_it(root)
    assert FZ.verify(root) == []
    CS.check_frozen(json.loads((root / "configs/experiment_v2.json").read_text()))
    (root / "src/commitment.py").write_text((root / "src/commitment.py").read_text() + "\n# tweak after freeze\n")
    assert any("changed after freeze: src/commitment.py" in p for p in FZ.verify(root))


def test_threshold_or_split_change_is_detected(tmp_path):
    root = stage(tmp_path); freeze_it(root)
    plan = json.loads((root / "configs/analysis_plan.json").read_text()); plan["HA"]["practical_effect_steps"] = 2.0
    (root / "configs/analysis_plan.json").write_text(json.dumps(plan))
    s = json.loads((root / "data/benchmark_v2/splits.json").read_text()); s["TEST"] = s["TEST"][:-1]
    (root / "data/benchmark_v2/splits.json").write_text(json.dumps(s))
    p = FZ.verify(root)
    assert any("analysis_plan" in x for x in p) and any("splits.json" in x for x in p)


def test_unpinned_revision_blocks_verification(tmp_path):
    root = stage(tmp_path); freeze_it(root)
    cfg = json.loads((root / "configs/experiment_v2.json").read_text()); cfg["revision"] = "UNPINNED-DRAFT"
    (root / "configs/experiment_v2.json").write_text(json.dumps(cfg))
    assert any("changed after freeze" in x or "not pinned" in x for x in FZ.verify(root))


def test_analysis_plan_matches_code_defaults():
    plan = json.loads((ROOT / "configs/analysis_plan.json").read_text())
    for k, v in plan["judge_thresholds"].items():
        if k == "uncertain_band":
            assert v == E.UNCERTAIN_BAND
        else:
            assert E.THRESHOLDS[k] == v
    import inspect
    sig = inspect.signature(C.classify_HA).parameters
    assert sig["practical"].default == plan["HA"]["practical_effect_steps"] and sig["null_bound"].default == plan["HA"]["bounded_null_halfwidth_steps"]
    assert list(C.GRID) == plan["HA"]["grid"]
    cfg = json.loads((ROOT / "configs/experiment_v2.json").read_text())
    assert cfg["restart"]["grid"] == plan["HA"]["grid"] and cfg["restart"]["D"] == plan["HA"]["D"] and cfg["restart"]["equal_ratio_target"] == plan["HA"]["equal_ratio_target"]
    assert cfg["restart"]["D_sensitivity"] == plan["HA"]["D_sensitivity"]
    import edit_probe as EP
    assert abs(EP.mass_ratio_ok.__defaults__[0] - plan["HC"]["bias_mass_tolerance"]) < 1e-12
    import math
    assert math.log(plan["HC"]["beta_max_scale"]) == pytest.approx(math.log(2))
    assert inspect.signature(C.windows_from_t50_dev).parameters["window_len"].default == plan["HC"]["window_length"]
