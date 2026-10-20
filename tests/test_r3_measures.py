"""Revision 3 measurement code on synthetic data with known answers: prompt-switch estimator, attention velocity / t_stab, HF rule,
seed diversity. None of these numbers is a research result."""
import math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import switch as SW, attn_stability as AS, diversity as DV

GRID = (6, 12, 18, 24, 30, 36)


def synth_switch(commit_step, n_scenes=40, seeds=2, w0=0.1, w1=0.9, rng=np.random.default_rng(0)):
    sw, can, cf = [], [], []
    for s in range(n_scenes):
        for sd in range(seeds):
            for t in GRID:
                p = w1 if t >= commit_step else w0               # before commitment the canonical prompt wins
                sw.append(dict(scene=s, seed=sd, t=t, y=int(rng.random() < p)))
            cf.append(dict(scene=s, seed=sd, y=int(rng.random() < w1)))
        for sd in range(4):
            can.append(dict(scene=s, seed=sd, y=int(rng.random() < w0)))
    return sw, can, cf


def test_switch_estimator_recovers_a_known_commitment_step():
    tables = SW.scene_tables(*synth_switch(19))
    c = SW.curve(tables)
    assert c["t50_status"] == "ok" and 18 <= c["t50"] <= 24              # true switch between grid 18 and 24 -> interpolated crossing there
    assert c["C"][0] < 0.3 and c["C"][-1] > 0.7
    b = SW.bootstrap_t50(tables, n_boot=200)
    assert b["ci95"][0] <= c["t50"] <= b["ci95"][1]


def test_switch_never_committed_is_right_censored():
    tables = SW.scene_tables(*synth_switch(99))
    assert SW.curve(tables)["t50"] == ">36"


def test_velocity_and_t_stab_on_a_map_that_settles():
    rng = np.random.default_rng(1)
    final = rng.random((16, 16))
    maps = []
    for i in range(40):
        noise = rng.random((16, 16)) * (1.0 if i < 10 else 0.0)        # moves for 10 steps, then frozen
        maps.append(final + noise)
    v = AS.velocity_map(np.array(maps))
    assert len(v) == 39 and v[12:].max() < 1e-12
    ts = AS.t_stab_path(v)
    assert ts is not None and 1 <= ts <= 11
    assert AS.t_stab_threshold(v) is not None and AS.t_stab_threshold(v) <= 11
    assert AS.t_stab_path(np.zeros(39)) is None                         # degenerate: reported, not imputed


def test_t_stab_path_interpolation_exact():
    v = np.array([1.0, 1.0, 1.0, 1.0])                                    # half of 4 at the end of v[1] -> position 1 + 1 + offset 1
    assert AS.t_stab_path(v) == 3.0


def test_signals_and_concordance():
    a = np.zeros((3, 16, 16)); b = np.zeros((3, 16, 16))
    a[:, 2:6, 2:6] = 1; b[:, 2:6, 10:14] = 1
    ang = AS.spatial_signal(a, b)
    assert np.allclose(ang, 0.0)                                           # b is to the right of a
    assert AS.size_signal(a, b).tolist() == [1.0, 1.0, 1.0]
    assert AS.count_signal(a + b).tolist() == [2.0, 2.0, 2.0]
    conc, n = AS.concordance({"p": 5, "s": 7, "b": 9}, {"p": 10, "s": 12, "b": 20})
    assert conc == 1.0 and n == 3
    assert AS.concordance({"p": 9, "s": 7, "b": 5}, {"p": 10, "s": 12, "b": 20})[0] == 0.0
    assert AS.hf_decision(0.9, (0.7, 1.0), {"p": (0.5, 0.2, 0.7), "s": (0.4, 0.1, 0.6)}) == "supported"
    assert AS.hf_decision(0.4, (0.2, 0.6), {"p": (0.5, 0.2, 0.7)}).startswith("rejected")
    assert AS.hf_decision(0.9, (0.7, 1.0), {"p": (0.5, 0.2, 0.7), "s": (0.1, -0.2, 0.3)}) == "inconclusive"


def test_diversity_metric():
    e = np.eye(3)
    assert abs(DV.mean_pairwise_cosine_distance(e) - 1.0) < 1e-12
    assert DV.mean_pairwise_cosine_distance(np.ones((4, 5))) < 1e-12
    assert math.isnan(DV.mean_pairwise_cosine_distance(np.ones((1, 5))))
