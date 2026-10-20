import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import branch as B

GRID = [6, 12, 18, 24, 30, 36]
# docs/evidence/sched_check_output.txt (executed with diffusers DDIMScheduler during the audit)
EVIDENCE_RATIO = dict(zip(GRID, [0.2099, 0.1843, 0.2795, 0.3994, 0.5381, 0.6834]))


@pytest.fixture(scope="module")
def sch():
    return B.Schedule(40)


def test_timesteps_match_ddim_leading(sch):
    assert sch.ts[:4].tolist() == [976, 951, 926, 901] and sch.ts[-3:].tolist() == [51, 26, 1] and len(sch.ts) == 40
    assert sch.abar_level(0) == pytest.approx(0.006124, abs=1e-6)
    assert sch.final == pytest.approx(0.99915, abs=1e-5)


def test_ratios_equal_audit_evidence(sch):
    for t, r in EVIDENCE_RATIO.items():
        assert B.restart_plan(sch, t, 8)["ratio"] == pytest.approx(r, abs=1e-3)


def test_boundary_clipping_flagged_and_effective_jump(sch):
    p = B.restart_plan(sch, 6, 8)
    assert p["clipped"] and p["target_idx"] == 0 and p["effective_jump"] == 6 and p["requested_jump"] == 8
    for t in GRID[1:]:
        q = B.restart_plan(sch, t, 8)
        assert not q["clipped"] and q["effective_jump"] == 8


def test_t_equals_n_endpoint(sch):
    p = B.restart_plan(sch, 40, 8)
    assert p["target_idx"] == 32 and p["ts_t"] is None and p["ratio"] == pytest.approx(0.7908, abs=1e-3)


def test_D0_identity(sch):
    x = np.random.default_rng(0).standard_normal((4, 64, 64)).astype(np.float32)
    noise = np.random.default_rng(1).standard_normal(x.shape).astype(np.float32)
    for t in GRID:
        p = B.restart_plan(sch, t, 0)
        assert p["ratio"] == 1.0 and p["effective_jump"] == 0
        assert B.restart(x, p["ratio"], noise) is x


def test_ratio_decreasing_in_jump(sch):
    for t in GRID:
        rs = [B.restart_plan(sch, t, D)["ratio"] for D in range(0, t + 1)]
        assert all(a >= b for a, b in zip(rs, rs[1:]))


def test_equal_ratio_plan(sch):
    for t in GRID[1:]:
        p = B.equal_ratio_plan(sch, t, 0.30)
        assert abs(p["ratio"] - 0.30) < 0.06 and p["target_idx"] <= t
    # at t=6 the ratio 0.30 is unreachable (min ratio 0.21 at index 0): must still be a valid plan, not an error
    assert B.equal_ratio_plan(sch, 6, 0.30)["ratio"] >= 0.2099 - 1e-6


def test_moment_test_synthetic_gaussian(sch):
    """x0 ~ N(1, 0.25); noise to level t, restart to level j, compare with the exact marginal at j.
    Gaussian synthetic only: says nothing about real SD 1.5 latents."""
    rng = np.random.default_rng(0)
    n = 200_000
    x0 = rng.standard_normal(n) * 0.5 + 1.0
    for t in (12, 30):
        p = B.restart_plan(sch, t, 8)
        a_t, a_j = p["abar_t"], p["abar_target"]
        xt = a_t**.5 * x0 + (1 - a_t) ** .5 * rng.standard_normal(n)
        xr = B.restart(xt, p["ratio"], rng.standard_normal(n))
        res = xr - a_j**.5 * x0
        assert res.var() == pytest.approx(1 - a_j, rel=0.03)
        assert abs(res.mean()) < 0.01


def test_matches_diffusers_scheduler_if_available(sch):
    diffusers = pytest.importorskip("diffusers")
    s = diffusers.DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    s.set_timesteps(40)
    assert s.timesteps.tolist() == sch.ts.tolist()
    assert np.allclose(s.alphas_cumprod.numpy(), sch.ac, rtol=1e-6)
    assert float(s.final_alpha_cumprod) == pytest.approx(sch.final, rel=1e-6)
