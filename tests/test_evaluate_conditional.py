import json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import evaluate as E

PROMPTS = {p["id"]: p for p in json.loads((ROOT / "data/prompts.json").read_text())}


def stage(tmp_path):
    for f in ("samples.jsonl", "detections.jsonl"):
        shutil.copy(ROOT / "outputs/main" / f, tmp_path / f)
    return tmp_path


def test_default_rescoring_is_byte_identical_to_pilot(tmp_path):
    run = stage(tmp_path)
    E.run_scoring(run, PROMPTS)
    assert (run / "eval.jsonl").read_bytes() == (ROOT / "outputs/main/eval.jsonl").read_bytes()


def test_conditional_appends_after_legacy_columns(tmp_path):
    run = stage(tmp_path)
    E.run_scoring(run, PROMPTS, conditional=True, out_name="eval_cond.jsonl")
    old = [json.loads(l) for l in (ROOT / "outputs/main/eval.jsonl").read_text().splitlines()]
    new = [json.loads(l) for l in (run / "eval_cond.jsonl").read_text().splitlines()]
    assert len(old) == len(new) == 1008
    for o, n in zip(old, new):
        assert list(n)[: len(o)] == list(o) and all(n[k] == o[k] for k in o)
        assert {"objects_present", "bind_given_pres", "joint_bind", "spatial_given_pres", "joint_spatial", "uncertain"} <= set(n)
        if n["objects_present"] == 0:
            assert n["bind_given_pres"] is None and n["spatial_given_pres"] is None


def test_uncertain_flag_band():
    cons = [{"type": "presence", "object": "cup"}]
    near = {"cup": [{"score": 0.11, "box": [0, 0, 5, 5], "colors": {"red": 1.0}}]}
    far = {"cup": [{"score": 0.5, "box": [0, 0, 5, 5], "colors": {"red": 1.0}}]}
    assert E.conditional_outcomes(cons, near)["uncertain"] == 1 and E.conditional_outcomes(cons, far)["uncertain"] == 0
