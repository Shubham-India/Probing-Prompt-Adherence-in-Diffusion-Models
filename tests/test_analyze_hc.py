import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import analyze_hc as H


def make(d_early=0.0, d_late=0.0, n=150, seeds=4, rng_seed=0, mass_bad=False, p_present=0.9):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s in range(n):
        base_p = rng.uniform(0.2, 0.6)
        for sd in range(seeds):
            for onset, eff in ((0, d_early), (28, d_late)):
                u = rng.random()
                pres_t, pres_r = int(rng.random() < p_present), int(rng.random() < p_present)
                for arm, p, pres in (("always_on", base_p + eff, pres_t), ("random_pair", base_p, pres_r)):
                    y = int(u < min(max(p, 0), 1))             # shared uniform per (scene, seed, onset): paired design
                    rows.append(dict(scene_id=f"s{s}", seed=sd, arm=arm, onset=onset, objects_present=pres,
                                     bind_given_pres=y if pres else None, joint_bind=int(pres and y),
                                     edit={"bias_mass_l1": 10.0 * (1.3 if (mass_bad and arm == "random_pair") else 1.0)}))
    return rows


def test_time_gated_effect_is_supported():
    r = H.run(make(0.20, 0.0), n_boot=400)
    assert r["primary_conditional"]["D_early"] > 0.12 and abs(r["primary_conditional"]["D_late"]) < 0.06
    assert r["final"] == "support"


def test_null_effect_is_rejected_as_no_handle():
    r = H.run(make(0.0, 0.0, n=400), n_boot=400)
    assert r["final"].startswith("reject: no handle") or r["final"] == "inconclusive"
    assert abs(r["primary_conditional"]["D_early"]) < 0.05


def test_ungated_effect_is_not_called_time_gated():
    r = H.run(make(0.20, 0.20, n=300), n_boot=400)
    assert r["primary_conditional"]["D_early"] > 0.1 and r["final"] != "support"


def test_mass_mismatch_excludes_pairs_and_is_reported():
    p, info = H.paired(make(0.2, 0.0, n=20, mass_bad=True), 0, True)
    assert info["pairs_used"] == 0 and info["mass_violations"] == info["pairs_total"] == 80
    p2, info2 = H.paired(make(0.2, 0.0, n=20), 0, True)
    assert info2["mass_violations"] == 0 and 0 < info2["pairs_used"] <= 80
