"""DEV CAG quality guard computation (src/hps_guard.py): predeclared rule (drop <= 0.5 HPS points) and the w_lo decision."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import hps_guard as H  # noqa: E402


def world(d1, d3, n=40):
    recs, scores = [], {}
    for i in range(n):
        pid = f"sg_{i:04d}_CAN"
        for s in (0, 1):
            scores[H.a1_job(pid, s)] = 0.27
            for w, d in ((1.0, d1), (3.0, d3)):
                jid = f"{pid}__cag_A2_wlo{w:g}__seed{s}"
                recs.append(dict(job_id=jid, prompt_id=pid, scene_id=f"sg_{i:04d}", seed=s, w_lo=w))
                scores[jid] = 0.27 + d / 100
    return recs, scores


@pytest.mark.parametrize("d1, d3, w_lo", [(-0.2, -0.1, 1.0), (-0.5, -0.9, 1.0), (-0.8, -0.3, 3.0), (-1.0, -0.7, None), (+0.4, 0.0, 1.0)])
def test_rule_and_decision(d1, d3, w_lo):
    recs, scores = world(d1, d3)
    g = H.guard(recs, scores, n_boot=200)
    assert g["decision"]["w_lo"] == w_lo
    assert abs(g["per_w_lo"]["1"]["mean_change_points"] - d1) < 1e-9 and g["per_w_lo"]["1"]["n_pairs"] == 80


def test_missing_scores_are_an_error():
    recs, scores = world(-0.1, -0.1)
    scores.pop(H.a1_job("sg_0003_CAN", 1))
    with pytest.raises(ValueError, match="without an HPS score"):
        H.guard(recs, scores)
