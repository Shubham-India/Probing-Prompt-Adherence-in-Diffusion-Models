"""Prompt-switch commitment probe (E31; GRAND_RESEARCH_BLUEPRINT.md section 24.8). The prompt-switch operation is Prompt-Mixing's
(Patashnik et al., ICCV 2023); here it is used as a measurement.
A counterfactual (CF) trajectory (CF-C colour swap, CF-R relation reversal, CF-S size swap) is generated with latents saved at the grid
steps; from step t the canonical prompt is used to the end. The question is whether the final image still follows the EARLY (CF) prompt.
Per scene s and grid step t (outcome y_CF = the CF version of the swapped constraint, conditional on presence):
    W_s(t)  = mean over seeds of y_CF(switch at t)                     (eligible images only)
    W0_s    = mean over the canonical prompt's seeds of y_CF(canonical image)   chance that a canonical image satisfies the CF constraint
    W1_s    = mean over the CF trajectory's seeds of y_CF(CF image)            the no-switch ceiling (switch at the very end)
    C_switch(t) = sum_s [W_s(t) - W0_s] / sum_s [W1_s - W0_s]   (ratio of sums, like the re-noising estimator)
    t50_switch  = first grid step with C_switch >= 0.5 (linear interpolation; '>36' if never reached)
The estimator part is torch-free; switch_generate needs a pipeline."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

import commitment as C

GRID = C.GRID


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def scene_tables(switch_rows, canonical_rows, cf_rows):
    """Rows: dicts with scene, seed, y (0/1 or None when not eligible); switch rows also carry t. Returns per-scene W(t), W0, W1."""
    W = defaultdict(lambda: defaultdict(list))
    for r in switch_rows:
        W[r["scene"]][int(r["t"])].append(r["y"])
    W0, W1 = defaultdict(list), defaultdict(list)
    for r in canonical_rows:
        W0[r["scene"]].append(r["y"])
    for r in cf_rows:
        W1[r["scene"]].append(r["y"])
    out = {}
    for s in W:
        w0, w1 = _mean(W0.get(s, [])), _mean(W1.get(s, []))
        out[s] = dict(W={t: _mean(v) for t, v in W[s].items()}, W0=w0, W1=w1)
    return out


def arrays(tables, scenes, grid=GRID):
    num, den = np.zeros((len(scenes), len(grid))), np.zeros((len(scenes), len(grid)))
    for i, s in enumerate(scenes):
        tb = tables[s]
        if tb["W0"] is None or tb["W1"] is None:
            continue
        for j, t in enumerate(grid):
            w = tb["W"].get(t)
            if w is None:
                continue
            num[i, j] = w - tb["W0"]
            den[i, j] = tb["W1"] - tb["W0"]
    return num, den


def curve(tables, grid=GRID):
    scenes = sorted(tables)
    num, den = arrays(tables, scenes, grid)
    c = C.curve_from_arrays(num, den)
    t, status = C.t50(c, grid)
    return dict(C=[None if np.isnan(x) else float(x) for x in c], t50=t if math.isfinite(t) else ">36", t50_status=status, n_scenes=len(scenes),
                margin_sum=float(den[:, 0].sum()))


def bootstrap_t50(tables, n_boot=4000, key="switch", grid=GRID):
    scenes = sorted(tables)
    num, den = arrays(tables, scenes, grid)
    rng = C._rng(key)
    draws, cens = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(scenes), len(scenes))
        t, _ = C.t50(C.curve_from_arrays(num[idx], den[idx]), grid)
        if math.isinf(t):
            cens += 1
        else:
            draws.append(t)
    est = C.t50(C.curve_from_arrays(num, den), grid)
    ci = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (math.nan, math.nan)
    return dict(t50=est[0], status=est[1], ci95=ci, censored_draws=cens, n_boot=n_boot)


def switch_generate(pipe, canonical_embeds, cf_latent_after_t, t, steps=40, cfg=7.5, decode=True):
    """Continue the counterfactual trajectory from its saved latent after t completed steps with the CANONICAL prompt."""
    import branch as B
    return B.denoise_from(pipe, canonical_embeds, cf_latent_after_t, t, steps, cfg, decode)
