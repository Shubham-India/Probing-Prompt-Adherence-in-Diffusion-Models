"""Recompute precision / intervention counts from the DERIVED benchmark counts (D-05). Does not read any image or outcome.
Inputs: data/benchmark_v2/design_counts.json (derived from scene graphs). Assumptions are explicit and quoted, not estimated here:
  ICC = 0.52 (colour-binding outcome, quoted from RB 9.0 'DERIVED FROM PILOT'; reused unchanged for the competitive subset),
  k = 4 seeds, AUC = 0.75 and prevalence 0.236 (pilot all-satisfied rate at 40 steps, outputs/m2_pilot_reanalysis/summary.json).
Single-AUC 95% half-width: Hanley-McNeil (1982) variance on the effective sample size. Sanity anchor: RB states ~0.08 at 140 scenes.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ICC, K, AUC, PREV = 0.52, 4, 0.75, 0.236
Z = 1.96


def n_eff(scenes, k=K, icc=ICC):
    return scenes * k / (1 + (k - 1) * icc)


def hm_halfwidth(n, auc=AUC, prev=PREV):
    npos, nneg = n * prev, n * (1 - prev)
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (npos - 1) * (q1 - auc**2) + (nneg - 1) * (q2 - auc**2)) / (npos * nneg)
    return Z * var**0.5


def main():
    dc = json.loads((ROOT / "data/benchmark_v2/design_counts.json").read_text())
    rows = {}
    for split in ("TEST", "DEV"):
        for label, key in (("binding-bearing (literal D-05)", "binding_bearing"), ("binding-competitive (>=2 coloured objects)", "binding_competitive")):
            n = dc[split][key]
            rows[f"{split} {label}"] = dict(scenes=n, effective_n=round(n_eff(n), 1), auc_halfwidth_95=round(hm_halfwidth(n_eff(n)), 3),
                                            e12_images=(n * 6 * 4 if split == "TEST" else n * 8 * 2))
    anchor = round(hm_halfwidth(n_eff(140)), 3)
    out = dict(assumptions=dict(ICC=ICC, k=K, AUC=AUC, prevalence=PREV, source="ICC quoted from RB 9.0; prevalence from M2 pilot re-analysis; AUC assumed"),
               anchor_check=dict(scenes=140, hanley_mcnea_halfwidth=anchor, rb_stated="~0.08"), rows=rows,
               formulas=dict(e12_TEST="scenes x 6 arms x k=4 (RB: 140 -> 3,360)", e12_DEV_tuning="scenes x 8 x 2 (RB: 70 -> 1,120)",
                             spatial_bearing={s: dc[s]["spatial_bearing"] for s in dc}, rb_spatial_bearing_assumed="~70 TEST (RB 9.0)"))
    (ROOT / "data/benchmark_v2/design_precision.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
