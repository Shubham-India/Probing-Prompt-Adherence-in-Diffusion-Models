import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dev_gate_observed as D


def toy(n=30, k=4, seed=0):
    rng = np.random.default_rng(seed)
    pres = (rng.random((n, k)) < 0.5).astype(int)
    bind = np.where(pres == 1, (rng.random((n, k)) < 0.5).astype(int), -1)
    return pres, bind


def test_noise_free_world_copies_base_after_commitment():
    pres, bind = toy()
    br = D.simulate_branches(np.random.default_rng(1), pres, bind, tau_p=12, tau_b=12, n_bases=2, m=2)
    late = br[(br.t >= 12) & (br.kind == "presence")]
    assert (late.y == late.base_y).all()                         # full persistence, no evaluator error
    assert len(br) == 2 * 6 * 30 * 2 * 2                         # kinds x grid x scenes x bases x M


def test_evaluator_noise_lowers_agreement_and_respects_rates():
    pres, bind = toy(200)
    rng = np.random.default_rng(2)
    clean = D.simulate_branches(rng, pres, bind, 12, 12, 2, 2)
    noisy = D.simulate_branches(rng, pres, bind, 12, 12, 2, 2, flip=0.15)
    a = lambda df: float((df[(df.t >= 12) & (df.kind == "presence")].eval("y == base_y")).mean())
    assert a(clean) == 1.0 and 0.65 < a(noisy) < 0.95            # roughly 1 - 2*0.15*0.85 = 0.745 for symmetric error, independent per look
    # posterior of the latent truth respects Bayes: a certain evaluator leaves the observation unchanged
    z = D._posterior_truth(rng, np.array([1, 0, 1, 0] * 50), 0.5, 0.0, 0.0)
    assert (z == np.array([1, 0, 1, 0] * 50)).all()
