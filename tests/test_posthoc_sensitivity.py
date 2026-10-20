"""Tests of the post-hoc sensitivity tools (partial/robustness.py) and of the registered estimator under a full-resample null
(diagnostics/full_control.py). None of these change a registered analysis."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "partial"))
import commitment as C  # noqa: E402
import r3_analyze as RA  # noqa: E402
import robustness as RB  # noqa: E402


def _agree(y):
    n, o = len(y), sum(y)
    return (o * (o - 1) + (n - o) * (n - o - 1)) / (n * (n - 1))


def test_error_correction_is_identity_at_zero():
    g = np.random.default_rng(0)
    num, den = g.normal(size=(20, 6)), g.uniform(0, 1, size=(20, 6))
    cnt = (g.uniform(size=(20, 6)) > 0.2).astype(float)
    den = den * cnt
    n2, d2 = RB.corrected(num, den, cnt, 0.0)
    assert np.allclose(n2, num) and np.allclose(d2, den)
    n3, d3 = RB.corrected(num, den, cnt, 0.0, delta=0.1)
    s = d3.sum(0)
    assert np.allclose(C.curve_from_arrays(n3, d3), C.curve_from_arrays(num, den) - 0.1 * (s > 1e-12) / np.where(s > 1e-12, 1, 1))


def test_error_correction_recovers_true_persistence():
    """Scenes with base rate p, branch keeps the base outcome with probability q (true C known in expectation); outcomes are then flipped
    independently at rate e. The corrected pooled C must be close to the uncorrupted pooled C, and the uncorrected one attenuated."""
    g = np.random.default_rng(1)
    e, S, K, B = 0.12, 3000, 4, 6
    num_t, den_t, num_o, den_o = [], [], [], []
    for _ in range(S):
        p = g.uniform(0.2, 0.8)
        seeds = (g.uniform(size=K) < p).astype(int)
        base = seeds[0]
        branch = np.where(g.uniform(size=B) < 0.6, base, (g.uniform(size=B) < p).astype(int))

        def flip(y):
            return np.where(g.uniform(size=np.shape(y)) < e, 1 - y, y)
        a0_t, a_t = _agree(list(seeds)), float(np.mean(branch == base))
        seeds_o, branch_o = flip(seeds), flip(branch)
        a0_o, a_o = _agree(list(seeds_o)), float(np.mean(branch_o == seeds_o[0]))
        num_t.append(a_t - a0_t); den_t.append(1 - a0_t); num_o.append(a_o - a0_o); den_o.append(1 - a0_o)
    true_c = np.sum(num_t) / np.sum(den_t)
    raw = np.sum(num_o) / np.sum(den_o)
    n2, d2 = RB.corrected(np.array(num_o)[:, None], np.array(den_o)[:, None], np.ones((S, 1)), e)
    corr = float(C.curve_from_arrays(n2, d2)[0])
    assert raw < true_c - 0.1           # attenuation without correction
    assert abs(corr - true_c) < 0.08    # the correction recovers the uncorrupted value


def test_registered_estimator_unbiased_under_full_resample_null():
    """FULL-style design (4 base seeds per scene, branches from seeds 0 and 1, 12 independent redraws each): with exchangeable outcomes the
    registered prepare() + pooled C must be ~0 at every grid step (averaged over replicates)."""
    g = np.random.default_rng(2)
    curves = []
    for rep in range(30):
        core, br = [], []
        for s in range(100):
            p = g.uniform(0.1, 0.9)
            for k in range(4):
                core.append(dict(job_id=f"s{s}_k{k}", name=f"s{s}_k{k}", scene_id=f"s{s}", prompt_id=f"s{s}_CAN", cfg=7.5, steps=40, seed=k,
                                 objects_present=int(g.uniform() < p), variant="CAN"))
            for k in (0, 1):
                for t in C.GRID:
                    for m in (0, 1):
                        br.append(dict(base_job=f"s{s}_k{k}", scene_id=f"s{s}", seed=k, t=t, m=m, restart="FULL", cfg=7.5, steps=40,
                                       objects_present=int(g.uniform() < p)))
        adf, a0 = RA.prepare(core, br, "FULL")
        scenes = sorted(set(adf.scene) | set(a0.scene))
        num, den, _ = C._arrays(adf, a0, "presence", scenes, RA.GRID)
        curves.append(C.curve_from_arrays(num, den))
    m = np.nanmean(np.array(curves), axis=0)
    assert np.all(np.abs(m) < 0.06), m
