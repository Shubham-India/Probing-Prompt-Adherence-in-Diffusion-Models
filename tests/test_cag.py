"""CAG schedule rules and the guidance loop. Loop identities are checked on a tiny RANDOM diffusers pipeline (CPU): they validate the code path,
not Stable Diffusion 1.5 numerics (that is the Kaggle preflight's job)."""
import sys
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import cag as CAG
import branch as B
from test_branch_loop import tiny, arr, pipeline_image, x0   # noqa: F401  (pytest fixture reuse)


def test_s_star_rules():
    t50 = {"presence": 9.2, "bind_given_pres": 22.1, "spatial_given_pres": 15.0, "count": 14.0, "size_given_pres": 11.0}
    assert CAG.s_star(["presence", "binding"], t50, 2) == (25, "ok")          # ceil(22.1) + 2
    assert CAG.s_star(["presence"], t50, 2) == (12, "ok")
    s, note = CAG.s_star(["presence", "binding"], dict(t50, bind_given_pres=">36"), 2)
    assert s == 40 and "unresolved" in note
    assert CAG.s_star(["binding"], {"bind_given_pres": 39.5}, 2)[0] == 40     # capped at the number of steps
    assert CAG.s_star_global({"a": 9.0, "b": 15.0, "c": 22.0}, 2) == 17


def test_schedules_mean_matching_and_reversal():
    sch, mean = CAG.schedules(24, 7.5, 1.0)
    assert sch["A2"][:24] == [7.5] * 24 and sch["A2"][24:] == [1.0] * 16
    assert abs(sum(sch["A4"]) / 40 - mean) < 1e-12 and abs(sum(sch["A6"]) / 40 - mean) < 1e-12
    assert sch["A4"][:24] == [1.0] * 24 and sch["A4"][24] > 7.5                # guidance moved after the switch
    assert CAG.schedules(40, 7.5, 1.0)[0]["A4"] is None                         # undefined when CAG never switches
    assert CAG.unet_calls(sch["A2"]) == 2 * 24 + 16


def test_constant_schedule_equals_pipeline(tiny):
    pipe, emb = tiny
    z = x0()
    ref = arr(pipeline_image(pipe, emb, z.clone(), cfg=7.5))
    img, calls = CAG.sample_with_schedule(pipe, emb, z.clone(), [7.5] * 40)
    assert np.array_equal(ref, arr(img)) and calls == 80                      # s* = 40: identical to constant CFG
    assert np.array_equal(arr(B.denoise_from(pipe, emb, z.clone(), 0, 40, 7.5)), arr(img))


def test_w1_skips_the_unconditional_pass_and_matches_the_pipeline(tiny):
    pipe, emb = tiny
    z = x0()
    calls_seen = []
    orig = pipe.unet.forward
    def counting(*a, **k):
        calls_seen.append(a[0].shape[0])
        return orig(*a, **k)
    pipe.unet.forward = counting
    try:
        img, calls = CAG.sample_with_schedule(pipe, emb, z.clone(), [1.0] * 40)
    finally:
        pipe.unet.forward = orig
    assert calls == 40 and set(calls_seen) == {1}                              # one conditional evaluation per step, batch of 1
    ref = arr(pipeline_image(pipe, emb, z.clone(), cfg=1.0))                  # diffusers also skips the unconditional pass at w = 1
    assert np.array_equal(ref, arr(img))


def test_switch_schedule_runs_and_counts(tiny):
    pipe, emb = tiny
    sch, _ = CAG.schedules(10, 7.5, 1.0)
    img, calls = CAG.sample_with_schedule(pipe, emb, x0(), sch["A2"])
    assert calls == 2 * 10 + 30
