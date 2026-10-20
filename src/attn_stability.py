"""Attention stabilisation (HF; GRAND_RESEARCH_BLUEPRINT.md section 24.7). Torch-free.
Signals per constraint type (from attn_record maps, one resolution group at a time; 16x16 primary, 32x32 reported separately):
  presence -> the noun's map; colour binding -> the colour word's map;
  size     -> ratio of the two nouns' effective areas (area = pixels needed to hold `mass_q` of the normalised map mass);
  spatial  -> angle of the vector from noun-A centroid to noun-B centroid;   count -> connected components of the noun map above its 90th percentile.
Velocity: v(t) = 1 - cos(a(t), a(t-1)) for maps; |x(t) - x(t-1)| for scalars (angles wrapped to [0, pi]).
t_stab (primary, threshold-free): step at which half of the total path length sum_t v(t) has been travelled (linear interpolation).
t_stab_threshold (secondary): first step after which v stays below 10% of its early (steps 0-5) median.
OPERATIONAL CHOICE TO FREEZE: the blueprint's "area (top-10% mass)" is ambiguous; `mass_q` (default 0.9 = the region holding 90% of the mass)
is a parameter recorded in the protocol and must not be changed after TEST data exist."""
from __future__ import annotations

import math

import numpy as np


def _flat(a):
    return np.asarray(a, dtype=np.float64).reshape(len(a), -1)


def velocity_map(maps):
    """maps [steps, H, W] -> v [steps-1]."""
    x = _flat(maps)
    a, b = x[1:], x[:-1]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cos = np.where((na > 0) & (nb > 0), (a * b).sum(1) / np.maximum(na * nb, 1e-30), 1.0)
    return 1.0 - np.clip(cos, -1.0, 1.0)


def velocity_scalar(x, angle=False):
    x = np.asarray(x, dtype=np.float64)
    d = np.abs(np.diff(x))
    if angle:
        d = np.minimum(d, 2 * math.pi - d)
    return d


def effective_area(m, mass_q=0.9):
    v = np.sort(np.clip(np.asarray(m, dtype=np.float64).ravel(), 0, None))[::-1]
    tot = v.sum()
    if tot <= 0:
        return float("nan")
    return float(np.searchsorted(np.cumsum(v) / tot, mass_q) + 1)


def centroid(m):
    m = np.clip(np.asarray(m, dtype=np.float64), 0, None)
    tot = m.sum()
    if tot <= 0:
        return (float("nan"), float("nan"))
    ys, xs = np.indices(m.shape)
    return float((ys * m).sum() / tot), float((xs * m).sum() / tot)


def n_components(m, q=0.9):
    from scipy import ndimage
    m = np.asarray(m, dtype=np.float64)
    if not np.isfinite(m).all() or m.max() <= m.min():
        return 0                                   # flat or missing map: no component structure
    lab, n = ndimage.label(m >= np.quantile(m, q))
    return int(n)


def size_signal(maps_a, maps_b, mass_q=0.9):
    return np.array([effective_area(a, mass_q) / effective_area(b, mass_q) for a, b in zip(maps_a, maps_b)])


def spatial_signal(maps_a, maps_b):
    out = []
    for a, b in zip(maps_a, maps_b):
        (ya, xa), (yb, xb) = centroid(a), centroid(b)
        out.append(math.atan2(yb - ya, xb - xa))
    return np.array(out)


def count_signal(maps):
    return np.array([n_components(m) for m in maps], dtype=float)


def t_stab_path(v, offset=1):
    """v[i] is the change between map i and map i+1 (map i = state after i+1 completed denoising steps). Returns the interpolated position, in
    completed steps, at which half of the total path length has been travelled; None when the total path length is 0 (degenerate signal)."""
    v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0)
    tot = v.sum()
    if tot <= 0:
        return None
    cum = np.cumsum(v)
    k = int(np.searchsorted(cum, 0.5 * tot))
    prev = cum[k - 1] if k > 0 else 0.0
    return float(k + (0.5 * tot - prev) / v[k] + offset)


def t_stab_threshold(v, early=6, frac=0.10, offset=1):
    v = np.asarray(v, dtype=np.float64)
    base = np.nanmedian(v[:early])
    if not np.isfinite(base) or base <= 0:
        return None
    below = v < frac * base
    for i in range(len(v)):
        if below[i:].all():
            return float(i + offset)
    return None


def signal_velocity(kind, maps_by_word, words, constraint, mass_q=0.9):
    """Velocity series for one constraint of the given type from {word: [steps, H, W]}."""
    if kind == "presence":
        return velocity_map(maps_by_word[constraint["object"]])
    if kind == "binding":
        return velocity_map(maps_by_word[constraint["color"]])
    if kind == "size":
        return velocity_scalar(np.log(size_signal(maps_by_word[constraint["larger"]], maps_by_word[constraint["smaller"]], mass_q)))
    if kind == "spatial":
        return velocity_scalar(spatial_signal(maps_by_word[constraint["left"]], maps_by_word[constraint["right"]]), angle=True)
    if kind == "count":
        return velocity_scalar(count_signal(maps_by_word[constraint["object"]]))
    raise ValueError(kind)


# ---------------------------------------------------------------- HF tests
def concordance(t_stab_by_type, t50_by_type):
    """Fraction of type pairs ordered the same way by t_stab and by t50 (ties in either count as 0.5)."""
    types = sorted(set(t_stab_by_type) & set(t50_by_type))
    agree, n = 0.0, 0
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a, b = types[i], types[j]
            ds = np.sign(t_stab_by_type[a] - t_stab_by_type[b])
            dc = np.sign(t50_by_type[a] - t50_by_type[b])
            agree += 0.5 if ds == 0 or dc == 0 else float(ds == dc)
            n += 1
    return agree / n if n else float("nan"), n


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).correlation)


def hf_decision(conc, conc_ci, spearman_by_type):
    """Pre-specified HF rule. spearman_by_type: {type: (rho, lo, hi)}."""
    pos = [t for t, (r, lo, hi) in spearman_by_type.items() if np.isfinite(lo) and lo > 0]
    null_all = all(np.isfinite(lo) and lo <= 0 <= hi and hi < 0.15 for r, lo, hi in spearman_by_type.values()) if spearman_by_type else False
    if conc >= 0.8 and len(pos) >= 2:
        return "supported"
    if conc <= 0.5 or null_all:
        return "rejected (bounded null)" if null_all else "rejected (ordering concordance <= 0.5)"
    return "inconclusive"
