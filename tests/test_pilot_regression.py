"""Regression: the Revision 3 evaluator changes (relative size, count/size conditional fields) must leave every pilot result unchanged.
Re-scores the saved pilot detections (no models) and compares with outputs/main/eval.jsonl."""
import json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import evaluate as E

DET, EVAL = ROOT / "outputs" / "main" / "detections.jsonl", ROOT / "outputs" / "main" / "eval.jsonl"


@pytest.mark.skipif(not DET.exists(), reason="pilot detections not present")
def test_pilot_rescoring_is_identical():
    prompts = {p["id"]: p for p in json.loads((ROOT / "data" / "prompts.json").read_text())}
    old = {json.loads(l)["name"]: json.loads(l) for l in EVAL.read_text().splitlines() if l.strip()}
    n = 0
    for line in DET.read_text().splitlines():
        d = json.loads(line)
        r = old[d["name"]]
        res = E.score(prompts[r["prompt_id"]]["constraints"], d["detections"])
        assert res == r["constraint_results"] and int(all(res)) == r["all_satisfied"], d["name"]
        cond = E.conditional_outcomes(prompts[r["prompt_id"]]["constraints"], d["detections"])
        assert cond["size_given_pres"] is None and cond["joint_size"] is None      # no size constraint in the pilot
        n += 1
    assert n == 1008


def test_size_rule():
    box = lambda x0, y0, x1, y1: {"box": [x0, y0, x1, y1], "score": 0.9, "colors": {"red": 1.0}}
    cons = [{"type": "size", "larger": "sphere", "smaller": "cube"}]
    assert E.score(cons, {"sphere": [box(0, 0, 30, 30)], "cube": [box(40, 0, 60, 20)]}) == [1]       # 900 >= 1.5 * 400
    assert E.score(cons, {"sphere": [box(0, 0, 22, 22)], "cube": [box(40, 0, 60, 20)]}) == [0]       # 484 < 600
    assert E.score(cons, {"sphere": [box(0, 0, 30, 30)]}) == [0]                                      # missing object fails the joint rule
    c = E.conditional_outcomes(cons, {"sphere": [box(0, 0, 30, 30)]})
    assert c["objects_present"] == 0 and c["size_given_pres"] is None and c["joint_size"] == 0        # conditional is not estimable, never a 0
