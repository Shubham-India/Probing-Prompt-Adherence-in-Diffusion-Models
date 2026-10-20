"""Restart-persistence statistics (HA, RB 8.0/8.1) with the working decisions D-02, D-03, D-04, D-12.

Terminology (D-03): this measures PERSISTENCE of a judge-scored outcome under bounded local re-noising, relative to the agreement of
independent seeds. It does not show that the model internally "decided" a constraint; the symbol C and t50 are kept for continuity
with the blueprint and mean "restart-persistence" / "persistence time" until validated.

Data model (long DataFrames, one row per judged outcome):
  core     : scene, kind, seed, y, elig         -- base/independent-seed outcomes (A0 comes from these)
  branches : scene, kind, seed, t, m, y, elig, base_y, base_elig [, variant]   -- re-noised branch outcomes vs their own base
`kind` is the outcome family (presence, bind_given_pres, ...). Conditional kinds have elig=0 when the required objects are absent;
a pair contributes only if BOTH members are eligible (D-04), and scenes with no eligible pair at a grid step drop out of that step in
BOTH numerator and denominator. Eligibility counts per step are always returned (selection effect, D-04).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

GRID = (6, 12, 18, 24, 30, 36)


# ----- agreement and chance level -------------------------------------------------------------------------------------
def chance_agreement(core: pd.DataFrame) -> pd.DataFrame:
    """A0 per (scene, kind): mean agreement over all pairs of independent seeds on the canonical prompt (both eligible)."""
    rows = []
    for (s, k), g in core[core.elig == 1].groupby(["scene", "kind"]):
        y = g.sort_values("seed").y.to_numpy()
        n = len(y)
        if n < 2:
            continue
        ones = int(y.sum())
        agree = (ones * (ones - 1) + (n - ones) * (n - ones - 1)) / (n * (n - 1))   # fraction of ordered pairs with equal outcome
        rows.append(dict(scene=s, kind=k, A0=agree, n_seeds=n))
    return pd.DataFrame(rows)


def agreement(branches: pd.DataFrame) -> pd.DataFrame:
    """A per (scene, kind, [variant], t): mean over eligible (base, branch) pairs of 1[y == base_y]."""
    b = branches[(branches.elig == 1) & (branches.base_elig == 1)].copy()
    b["agree"] = (b.y == b.base_y).astype(float)
    keys = ["scene", "kind", "t"] + (["variant"] if "variant" in b else [])
    out = b.groupby(keys).agg(A=("agree", "mean"), n_pairs=("agree", "size")).reset_index()
    return out


def _arrays(adf, a0, kind, scenes, grid, variant=None):
    """num[S,T] = A - A0, den[S,T] = 1 - A0 with zeros where the scene has no eligible pair (or no A0) at that step."""
    a = adf[(adf.kind == kind)]
    if variant is not None:
        a = a[a.variant == variant]
    a = a.merge(a0[a0.kind == kind][["scene", "A0"]], on="scene", how="inner")
    idx = {s: i for i, s in enumerate(scenes)}
    num, den, cnt = (np.zeros((len(scenes), len(grid))) for _ in range(3))
    col = {t: j for j, t in enumerate(grid)}
    for r in a.itertuples():
        if r.t in col and r.scene in idx:
            num[idx[r.scene], col[r.t]] = r.A - r.A0
            den[idx[r.scene], col[r.t]] = 1.0 - r.A0
            cnt[idx[r.scene], col[r.t]] = 1.0
    return num, den, cnt


def curve_from_arrays(num, den):
    s = den.sum(0)
    return np.where(s > 1e-12, num.sum(0) / np.where(s > 1e-12, s, 1), np.nan)


# ----- t50 -------------------------------------------------------------------------------------------------------------------
def t50(curve, grid=GRID, level=0.5):
    """Smallest t with C >= level, linear interpolation between the previous grid point and the crossing point.
    Returns (value, status): status in {'ok', 'left_censored' (C already >= level at the first step: value = grid[0]),
    'right_censored' (never reached: value = inf, report '> grid[-1]'), 'undefined' (NaN at the crossing)}."""
    c = np.asarray(curve, float)
    for j, v in enumerate(c):
        if np.isnan(v):
            continue
        if v >= level:
            if j == 0:
                return float(grid[0]), "left_censored"
            p = c[j - 1]
            if np.isnan(p):
                return float(grid[j]), "undefined"
            return float(grid[j - 1] + (level - p) / (v - p) * (grid[j] - grid[j - 1])), "ok"
    return math.inf, "right_censored"


def nonmonotone(curve, tol=0.05):
    c = np.asarray(curve, float)
    c = c[~np.isnan(c)]
    return bool(np.any(np.diff(c) < -tol))


# ----- curves, bootstrap, comparisons -------------------------------------------------------------------------------
def curves(adf, a0, kinds, grid=GRID, variant=None):
    scenes = sorted(set(adf.scene) | set(a0.scene))
    out = {}
    for k in kinds:
        num, den, cnt = _arrays(adf, a0, k, scenes, grid, variant)
        c = curve_from_arrays(num, den)
        v, st = t50(c, grid)
        out[k] = dict(C=c, t50=v, t50_status=st, nonmonotone=nonmonotone(c), n_scenes=cnt.sum(0), weight=den.sum(0))
    return out, scenes


def _rng(key):
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))


def bootstrap_delta_t50(adf, a0, kind_a, kind_b, grid=GRID, variant=None, n_boot=4000, key="HA"):
    """Scene-level bootstrap of  d = t50(kind_b) - t50(kind_a). Resamples scenes jointly (both kinds on the union of scenes).
    Censored draws (right-censored t50) are counted; if any exist the interval is computed over the finite draws and flagged."""
    scenes = sorted(set(adf.scene) | set(a0.scene))
    na, da, _ = _arrays(adf, a0, kind_a, scenes, grid, variant)
    nb, db, _ = _arrays(adf, a0, kind_b, scenes, grid, variant)
    rng = _rng(f"{key}|{kind_a}|{kind_b}|{variant}")
    S = len(scenes)
    est_a, est_b = t50(curve_from_arrays(na, da), grid), t50(curve_from_arrays(nb, db), grid)
    d_hat = est_b[0] - est_a[0]
    draws, cens = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, S, S)
        ta, _ = t50(curve_from_arrays(na[idx], da[idx]), grid)
        tb, _ = t50(curve_from_arrays(nb[idx], db[idx]), grid)
        if math.isinf(ta) or math.isinf(tb):
            cens += 1
            continue
        draws.append(tb - ta)
    ci = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (math.nan, math.nan)
    return dict(delta_t50=d_hat, ci95=ci, t50_a=est_a, t50_b=est_b, censored_draws=cens, n_boot=n_boot,
                reliable=bool(cens / n_boot < 0.05 and est_a[1] == "ok" and est_b[1] == "ok"))


def classify_HA(res, practical=3.0, null_bound=1.5):
    """Pre-specified HA decision rule (RB 8.1): support / bounded null / inconclusive. Unreliable estimates are inconclusive."""
    lo, hi = res["ci95"]
    if not res["reliable"] or math.isnan(lo):
        return "inconclusive (censored or unreliable t50)"
    if (lo > 0 or hi < 0) and abs(res["delta_t50"]) >= practical:
        return "support"
    if -null_bound < lo and hi < null_bound:
        return "bounded null"
    return "inconclusive"


def sign_agreement(res_fixed, res_equal_ratio):
    """D-02 (3): HA support is conditional on sign agreement of Delta-t50 under the equal-ratio parameterisation."""
    a, b = res_fixed["delta_t50"], res_equal_ratio["delta_t50"]
    return bool(np.isfinite(a) and np.isfinite(b) and np.sign(a) == np.sign(b) and a != 0)


# ----- D-12: windows and onsets from a DEV t50 ------------------------------------------------------------------------
def windows_from_t50_dev(t50_dev, grid=GRID, n_steps=40, window_len=12):
    """tau_pre / tau_post / HC onsets with the undefined-case rules of D-12. t50_dev may be inf ('> 36')."""
    out = dict(t50_dev=t50_dev, notes=[])
    if not np.isfinite(t50_dev):
        out.update(tau_pre=grid[0], tau_pre_label="EXPLORATORY", tau_post=None, onsets=None)
        out["notes"] += ["t50_DEV > grid: tau_pre=6 (EXPLORATORY); tau_post not estimable; HC reduces to D_early, G not estimated"]
        return out
    cand = [g for g in grid if g <= 0.5 * t50_dev]
    if cand:
        out.update(tau_pre=max(cand), tau_pre_label="CONFIRMATORY")
    else:
        out.update(tau_pre=grid[0], tau_pre_label="EXPLORATORY")
        out["notes"].append("t50_DEV < 12: no grid step <= t50/2; tau_pre=6 (EXPLORATORY)")
    post = [g for g in grid if g >= t50_dev + 8]
    out["tau_post"] = min(post) if post else None
    if not post:
        out["notes"].append("t50_DEV > 28: readout-after-commitment not estimable")
    mid = min(t50_dev, n_steps - window_len)
    late = min(math.ceil(t50_dev) + 4, n_steps - window_len, 28)
    out["onsets"] = dict(early=0, mid=mid, late=late)
    out["G_estimable"] = late > mid
    if not out["G_estimable"]:
        out["notes"].append("late onset not later than mid after capping: early/late contrast loses meaning; G not estimated")
    return out
