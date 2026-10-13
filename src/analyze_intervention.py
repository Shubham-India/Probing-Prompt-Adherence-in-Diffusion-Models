"""Compares attention re-weighting (rw2, rw4) with its control (rw1, same code path) and with the plain-pipeline baseline.

All conditions use CFG 7.5, 40 steps and the same prompt x seed pairs, so every contrast is paired.
Contrast = intervention - control (positive = intervention better). Holm correction over the two strengths per category.
Usage: python src/analyze_intervention.py --run outputs/intervention --baseline outputs/main
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from analyze import CATEGORIES, cluster_bootstrap, load_table, matrix, paired_contrast

OUTCOMES = {"primary": "primary success", "presence_ok": "presence", "binding_ok": "colour binding",
            "count_ok": "count", "spatial_ok": "spatial", "clip_score": "CLIP score"}


def load_conditions(run, baseline):
    df, emb = load_table(run)
    df["condition"] = "rw" + df.name.str.split("__rw").str[1]
    base, base_emb = load_table(baseline)
    base = base[(base.cfg == 7.5) & (base.steps == 40)].assign(condition="baseline")
    emb.update({n: base_emb[n] for n in base.name})
    return pd.concat([base, df], ignore_index=True), emb


def condition_table(df):
    rows = []
    for cat, cond in product(CATEGORIES + ["all"], ["baseline", "rw1", "rw2", "rw4"]):
        sub = df[(df.condition == cond) & ((df.category == cat) | (cat == "all"))]
        mat = matrix(sub, "primary")
        lo, hi = cluster_bootstrap(mat)
        rows.append({"category": cat, "condition": cond, "n_images": len(sub), "mean": mat.to_numpy().mean(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def contrast_table(df):
    rows = []
    pairs = [("rw1", "baseline"), ("rw2", "rw1"), ("rw4", "rw1"), ("rw4", "rw2")]
    for cat, col in product(CATEGORIES + ["all"], OUTCOMES):
        sub = df if cat == "all" else df[df.category == cat]
        sub = sub.dropna(subset=[col])
        if sub.empty:
            continue
        res = []
        for a, b in pairs:
            if sub[sub.condition == a].empty or sub[sub.condition == b].empty:
                continue
            res.append({"outcome": col, "category": cat, **paired_contrast(sub, col, "condition", a, b)})
        primary_pairs = [r for r in res if r["b"] == "rw1" and r["a"] in ("rw2", "rw4")]
        if len(primary_pairs) == 2:
            adj = multipletests([r["p_wilcoxon"] for r in primary_pairs], method="holm")[1]
            for r, p in zip(primary_pairs, adj):
                r["p_holm"] = p
        rows += res
    return pd.DataFrame(rows)


def diversity_by_condition(df, emb):
    rows = []
    for (pid, cond), g in df.groupby(["prompt_id", "condition"]):
        e = np.array([emb[n] for n in g.name])
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        rows.append({"prompt_id": pid, "condition": cond, "diversity": 1 - (e @ e.T)[np.triu_indices(len(e), 1)].mean()})
    return pd.DataFrame(rows)


def control_pixel_identity(run, baseline):
    """Fraction of control (s=1) images that are pixel-identical to the plain-pipeline baseline image."""
    same = []
    for path in sorted((run / "images").glob("*__rw1.png")):
        base = baseline / "images" / (path.stem.rsplit("__rw", 1)[0] + ".png")
        same.append(bool((np.asarray(Image.open(path)) == np.asarray(Image.open(base))).all()))
    return sum(same), len(same)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "results"
    out.mkdir(exist_ok=True)
    df, emb = load_conditions(run, Path(args.baseline))
    df.to_csv(out / "per_image.csv", index=False)
    cells, contrasts = condition_table(df), contrast_table(df)
    cells.to_csv(out / "intervention_by_condition.csv", index=False)
    contrasts.to_csv(out / "intervention_contrasts.csv", index=False)
    div = diversity_by_condition(df, emb)
    div.to_csv(out / "intervention_diversity_per_prompt.csv", index=False)

    wide = div.pivot(index="prompt_id", columns="condition", values="diversity")
    prompt_level = df.groupby(["prompt_id", "category", "condition"]).primary.mean().unstack()
    prompt_level["rw2_minus_rw1"] = prompt_level.rw2 - prompt_level.rw1
    prompt_level["rw4_minus_rw1"] = prompt_level.rw4 - prompt_level.rw1
    prompt_level.to_csv(out / "intervention_per_prompt.csv")

    N = {}
    for _, r in cells.iterrows():
        N[f"int_{r.category}_{r.condition}"] = f"{r['mean']:.2f}"
        N[f"int_{r.category}_{r.condition}_ci"] = f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
    for _, r in contrasts[contrasts.outcome.isin(["primary", "clip_score", "count_ok", "binding_ok", "presence_ok", "spatial_ok"])].iterrows():
        k = f"intc_{r.outcome}_{r.category}_{r.a}_vs_{r.b}"
        d = 2 if r.outcome != "clip_score" else 2
        N[k] = f"{r['diff']:+.{d}f}"
        N[k + "_ci"] = f"[{r.ci_lo:+.{d}f}, {r.ci_hi:+.{d}f}]"
        N[k + "_p"] = f"{r.p_wilcoxon:.3f}"
        N[k + "_holm"] = f"{r.p_holm:.3f}" if "p_holm" in r and pd.notna(r.p_holm) else "n/a"
        N[k + "_dz"] = f"{r.dz:.2f}" if pd.notna(r.dz) else "n/a"
    for a, b in (("rw2", "rw1"), ("rw4", "rw1")):
        d = (wide[a] - wide[b]).to_numpy()
        N[f"int_div_{a}_vs_{b}"] = f"{d.mean():+.3f}"
        N[f"int_div_{a}_vs_{b}_p"] = f"{wilcoxon(d).pvalue:.3f}" if np.any(d != 0) else "1.000"
    for c in ("baseline", "rw1", "rw2", "rw4"):
        N[f"int_div_{c}"] = f"{wide[c].mean():.3f}"
        N[f"int_clip_{c}"] = f"{df[df.condition == c].clip_score.mean():.1f}"
    same = df[df.condition.isin(["baseline", "rw1"])].pivot_table(index=["prompt_id", "seed"], columns="condition", values="primary")
    N["int_ctrl_identical_outcomes"] = f"{(same.baseline == same.rw1).mean():.3f}"
    identical, total = control_pixel_identity(run, Path(args.baseline))
    N["int_ctrl_pixel_identical"] = f"{identical}/{total}"
    N["int_n_images"] = int((df.condition != "baseline").sum())
    N["int_prompts_improved_rw2"] = int((prompt_level.rw2_minus_rw1 > 0).sum())
    N["int_prompts_worse_rw2"] = int((prompt_level.rw2_minus_rw1 < 0).sum())
    N["int_prompts_improved_rw4"] = int((prompt_level.rw4_minus_rw1 > 0).sum())
    N["int_prompts_worse_rw4"] = int((prompt_level.rw4_minus_rw1 < 0).sum())
    (out / "intervention_numbers.json").write_text(json.dumps(N, indent=1))
    print(contrasts[(contrasts.outcome == "primary") & (contrasts.b == "rw1")][["category", "a", "diff", "ci_lo", "ci_hi", "p_wilcoxon", "p_holm"]].round(3).to_string())


if __name__ == "__main__":
    main()
