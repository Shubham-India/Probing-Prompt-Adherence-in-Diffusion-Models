import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from analyze_pilot_conditional import conditional_outcomes, required_objects


def det(score_, box, color):
    return {"score": score_, "box": box, "colors": {"red": 0.0, "blue": 0.0, color: 1.0}}


CONS = [{"type": "presence", "object": "cube"}, {"type": "presence", "object": "sphere"},
        {"type": "binding", "object": "cube", "color": "red"}, {"type": "binding", "object": "sphere", "color": "blue"}]


def test_all_present_correct_binding():
    d = {"cube": [det(.5, [0, 0, 100, 100], "red")], "sphere": [det(.5, [200, 0, 300, 100], "blue")]}
    o = conditional_outcomes(CONS, d)
    assert (o["pres_all"], o["bind_given_pres"], o["joint_bind"], o["bind_legacy"]) == (1, 1, 1, 1)
    assert np.isnan(o["spatial_given_pres"])


def test_present_but_swapped_binding():
    d = {"cube": [det(.5, [0, 0, 100, 100], "blue")], "sphere": [det(.5, [200, 0, 300, 100], "red")]}
    o = conditional_outcomes(CONS, d)
    assert (o["pres_all"], o["bind_given_pres"], o["joint_bind"]) == (1, 0, 0)


def test_missing_object_is_ineligible_not_wrong():
    d = {"cube": [det(.5, [0, 0, 100, 100], "red")], "sphere": []}
    o = conditional_outcomes(CONS, d)
    assert o["pres_all"] == 0 and np.isnan(o["bind_given_pres"]) and o["joint_bind"] == 0 and o["bind_legacy"] == 0


def test_required_objects_include_spatial_operands():
    assert required_objects([{"type": "spatial", "left": "cup", "right": "book"}]) == ["book", "cup"]
