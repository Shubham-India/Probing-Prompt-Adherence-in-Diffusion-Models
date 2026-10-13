import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import features as F, predict_failure as P


def test_noise_features_match_generator_noise():
    import generate as G
    z32 = G.initial_latents(3, 512, device="cpu").float().numpy()[0]          # fp16-cast pipeline noise
    z64 = F.initial_noise(3, 512)
    assert np.allclose(z32, z64, atol=2e-3) and z64.shape == (4, 64, 64)
    f = F.noise_features(3)
    assert f == F.noise_features(3) and len(f) == 4 * (2 + 16 + 4)
    assert F.noise_features(3) != F.noise_features(4)


def test_prompt_features():
    cons = [{"type": "presence", "object": "cup"}, {"type": "presence", "object": "book"}, {"type": "binding", "object": "cup", "color": "red"}, {"type": "binding", "object": "book", "color": "blue"}]
    f = F.prompt_features(cons, "a red cup and a blue book on a wooden table")
    assert f["p_n_constraints"] == 4 and f["p_n_binding"] == 2 and f["p_has_background"] == 1 and f["p_attr_noun_dist"] == 4.5


def synthetic(n_groups=60, per=8, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    base = rng.normal(size=n_groups)[g]
    extra = rng.normal(size=len(g))
    logit = base + signal * extra
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-logit))).astype(int)
    X1 = pd.DataFrame({"base": base})
    X2 = X1.assign(extra=extra)
    return X1, X2, y, g


def test_real_signal_is_detected_and_label_permutation_gives_zero():
    X1, X2, y, g = synthetic(signal=2.0)
    res, _ = P.nested_compare({"a": X1, "b": X2}, y, g, {"d": ("b", "a")}, n_boot=200)
    assert res["delta_auc"]["d"] > 0.05 and res["delta_auc_ci95"]["d"][0] > 0
    null = P.permutation_delta({"a": X1, "b": X2}, y, g, ("b", "a"), n_perm=10)
    assert abs(null.mean()) < 0.03


def test_no_signal_gives_zero_delta_and_groups_are_disjoint_in_cv():
    X1, X2, y, g = synthetic(signal=0.0, seed=1)
    res, _ = P.nested_compare({"a": X1, "b": X2}, y, g, {"d": ("b", "a")}, n_boot=200)
    assert abs(res["delta_auc"]["d"]) < 0.03 and res["delta_auc_ci95"]["d"][0] < 0 < res["delta_auc_ci95"]["d"][1]
    from sklearn.model_selection import GroupKFold
    for tr, te in GroupKFold(7).split(X1, y, g):
        assert not set(g[tr]) & set(g[te])
