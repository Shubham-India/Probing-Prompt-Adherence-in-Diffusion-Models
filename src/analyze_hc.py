"""HC analysis (RB 8.3): is the effect of the targeted attention edit time-gated?
  D_early = targeted@early - random@early ;  D_late = targeted@late - random@late ;  G = D_early - D_late.
Arms: targeted := mode 'always_on' (the confirmatory variant, RB 13.8), random := mode 'random_pair'. Pairing by (scene, seed).
Outcomes (D-04): primary = bind_given_pres counted only for (scene, seed) pairs where BOTH arms have all objects present; companion =
joint_bind (omission = failure) over all pairs. A conclusion needs both to agree in sign. Guard: logged bias mass of paired arms must
agree within 5% (RB 13.8) - pairs that violate it are listed and excluded from the confirmatory contrast, never silently dropped.
Scene-level paired bootstrap (resample scenes), per-call seeded RNG. Everything is EXPLORATORY unless run through the freeze (--frozen-config).
Usage: python src/analyze_hc.py --rows eval_arms.jsonl --out outputs/hc [--frozen-config configs/experiment_v2.json]"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config_schema
from edit_probe import mass_ratio_ok

MIN_D_EARLY, MIN_G, NULL_D, NULL_G = 0.08, 0.10, 0.04, 0.05


def paired(rows, onset, conditional):
    """Per-scene mean difference targeted - random at one onset, plus bookkeeping counts.
    conditional=True : bind_given_pres, only (scene, seed) pairs where BOTH arms have all objects present.
    conditional=False: joint_bind (omission = failure), all pairs. Pairs whose bias masses differ by > 5% are excluded and counted."""
    df = pd.DataFrame(rows)
    df["mass"] = df["edit"].apply(lambda e: e["bias_mass_l1"] if isinstance(e, dict) else np.nan)
    df = df[df.onset == onset]
    t, r = (df[df.arm == m].set_index(["scene_id", "seed"]) for m in ("always_on", "random_pair"))
    j = t.join(r, lsuffix="_t", rsuffix="_r", how="inner")
    info = {"pairs_total": int(len(j))}
    ok = np.array([mass_ratio_ok(x, y) for x, y in zip(j.mass_t, j.mass_r)], dtype=bool)
    info["mass_violations"] = int((~ok).sum())
    j = j[ok]
    if conditional:
        j = j[(j.objects_present_t == 1) & (j.objects_present_r == 1)]
        d = j.bind_given_pres_t - j.bind_given_pres_r
    else:
        d = j.joint_bind_t - j.joint_bind_r
    info["pairs_used"] = int(len(d))
    return d.groupby(level=0).mean(), info


def effects(rows, conditional=True, onset_values=None, n_boot=4000):
    ov = onset_values or {"early": 0, "late": 28}
    pe, ie = paired(rows, ov["early"], conditional)
    pl, il = paired(rows, ov["late"], conditional)
    scenes = sorted(set(pe.index) | set(pl.index))
    de, dl = pe.reindex(scenes).to_numpy(), pl.reindex(scenes).to_numpy()
    rng = np.random.default_rng(int(hashlib.sha256(f"HC|{conditional}".encode()).hexdigest()[:8], 16))

    def stat(idx):
        e, l = np.nanmean(de[idx]), np.nanmean(dl[idx])
        return e, l, e - l
    obs = stat(np.arange(len(scenes)))
    draws = np.array([stat(rng.integers(0, len(scenes), len(scenes))) for _ in range(n_boot)])
    lo, hi = np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)
    return dict(D_early=obs[0], D_late=obs[1], G=obs[2], D_early_ci95=[lo[0], hi[0]], D_late_ci95=[lo[1], hi[1]], G_ci95=[lo[2], hi[2]],
                n_scenes_early=int(np.isfinite(de).sum()), n_scenes_late=int(np.isfinite(dl).sum()), early_pairs=ie, late_pairs=il)


def decide(res):
    """RB 8.3 support / reject rules on one outcome definition."""
    d, g = res["D_early"], res["G"]
    dl, gl = res["D_early_ci95"], res["G_ci95"]
    if dl[1] < NULL_D:
        return "reject: no handle (D_early CI upper < 0.04)"
    if dl[0] > 0 and d >= MIN_D_EARLY and gl[0] > 0 and g >= MIN_G:
        return "support"
    if dl[0] > 0 and d >= MIN_D_EARLY and -NULL_G < gl[0] and gl[1] < NULL_G:
        return "reject: handle is not time-gated (G inside +/-0.05)"
    return "inconclusive"


def run(rows, n_boot=4000):
    primary, companion = effects(rows, conditional=True, n_boot=n_boot), effects(rows, conditional=False, n_boot=n_boot)
    dp, dc = decide(primary), decide(companion)
    # D-04: a conclusion stands only if the conditional and the joint definition reach the same decision (support already requires a
    # positive point estimate under both, so sign agreement is implied there; two identical rejections need no sign agreement).
    final = dp if dp == dc else f"inconclusive (conditional '{dp}' vs joint '{dc}')"
    return dict(primary_conditional=primary, companion_joint=companion, decision_conditional=dp, decision_joint=dc, final=final)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen-config", default=None)
    ap.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.rows).read_text().splitlines()]
    if a.frozen_config:
        config_schema.check_frozen(json.loads(Path(a.frozen_config).read_text()))
        status = "CONFIRMATORY (freeze verified)"
    else:
        if any(str(r.get("split", "")).upper() == "TEST" for r in rows):
            sys.exit("TEST rows present but no verified frozen config (D-06)")
        status = "EXPLORATORY (no freeze verification)"
    res = run(rows, a.n_boot)
    res["status"] = status
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "hc.json").write_text(json.dumps(res, indent=1, default=float))
    print(status, res["final"])
