"""Statistics on the evaluated images (reads <run>/eval.jsonl + detections.jsonl, writes <run>/results/*).

Design: every (prompt, seed) pair is generated in every CFG x steps cell, so comparisons are paired. Prompts are the
unit of replication (seeds within a prompt are correlated), hence the cluster bootstrap resamples prompts first and
then seeds inside each drawn prompt. The prompt-level paired mean differences feed a Wilcoxon signed-rank test and
Cohen's d_z. Holm correction is applied across the three CFG pairs within each category.

Usage: python src/analyze.py --run outputs/main
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 4000
RNG = np.random.default_rng(2026)
CATEGORIES = ["presence", "binding", "counting", "spatial", "multi"]


def load_table(run):
    prompts = {p["id"]: p for p in json.loads((ROOT / "data" / "prompts.json").read_text())}
    emb = {}
    for line in (run / "detections.jsonl").read_text().splitlines():
        d = json.loads(line)
        emb[d["name"]] = d["clip_image_embedding"]
    rows = []
    for line in (run / "eval.jsonl").read_text().splitlines():
        r = json.loads(line)
        cons = prompts[r["prompt_id"]]["constraints"]
        row = {k: r[k] for k in ("name", "prompt_id", "category", "cfg", "steps", "seed", "clip_score", "all_satisfied")}
        for t in ("presence", "binding", "count", "spatial"):
            vals = [res for c, res in zip(cons, r["constraint_results"]) if c["type"] == t]
            row[f"{t}_ok"] = float(all(vals)) if vals else np.nan
        row["n_constraints"] = len(cons)
        row["frac_constraints"] = float(np.mean(r["constraint_results"]))
        # primary outcome: counting prompts are judged on the count constraint alone (see protocol revisions log)
        row["primary"] = row["count_ok"] if r["category"] == "counting" else row["all_satisfied"]
        rows.append(row)
    return pd.DataFrame(rows), emb


def matrix(df, col):
    """prompts x seeds matrix of `col`, averaged over any other factor present in df."""
    return df.pivot_table(index="prompt_id", columns="seed", values=col, aggfunc="mean")


def cluster_bootstrap(mat, n=N_BOOT):
    values = mat.to_numpy()
    n_p, n_s = values.shape
    means = np.empty(n)
    for i in range(n):
        rows = RNG.integers(0, n_p, n_p)
        cols = RNG.integers(0, n_s, (n_p, n_s))
        means[i] = values[rows[:, None], cols].mean()
    return np.percentile(means, [2.5, 97.5])


def paired_contrast(df, col, factor, a, b):
    """Mean of (level a - level b) over matched prompt x seed pairs."""
    diff = matrix(df[df[factor] == a], col) - matrix(df[df[factor] == b], col)
    per_prompt = diff.mean(axis=1).to_numpy()
    lo, hi = cluster_bootstrap(diff)
    sd = per_prompt.std(ddof=1) if len(per_prompt) > 1 else np.nan
    try:
        p = wilcoxon(per_prompt).pvalue if np.any(per_prompt != 0) else 1.0
    except ValueError:
        p = 1.0
    return {"a": a, "b": b, "diff": diff.to_numpy().mean(), "ci_lo": lo, "ci_hi": hi, "n_prompts": len(per_prompt),
            "n_pairs": int(diff.notna().to_numpy().sum()),
            "dz": per_prompt.mean() / sd if sd and sd > 0 else np.nan, "p_wilcoxon": p}


def cell_table(df, col="primary"):
    rows = []
    for cat in CATEGORIES + ["all"]:
        sub = df if cat == "all" else df[df.category == cat]
        for (cfg, steps), cell in sub.groupby(["cfg", "steps"]):
            mat = matrix(cell, col)
            lo, hi = cluster_bootstrap(mat)
            rows.append({"category": cat, "cfg": cfg, "steps": steps, "n_images": len(cell),
                         "n_prompts": mat.shape[0], "mean": mat.to_numpy().mean(),
                         "sd_across_prompts": mat.mean(axis=1).std(ddof=1), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def contrast_table(df, factor, levels):
    rows = []
    for cat in CATEGORIES + ["all"]:
        sub = df if cat == "all" else df[df.category == cat]
        res = [paired_contrast(sub, "primary", factor, a, b) for a, b in combinations(levels, 2)]
        if len(res) > 1:
            adj = multipletests([r["p_wilcoxon"] for r in res], method="holm")[1]
            for r, p in zip(res, adj):
                r["p_holm"] = p
        else:
            res[0]["p_holm"] = res[0]["p_wilcoxon"]
        rows += [{"category": cat, **r} for r in res]
    return pd.DataFrame(rows)


def constraint_type_table(df):
    rows = []
    for t in ("presence", "binding", "count", "spatial"):
        for (cfg, steps), cell in df.groupby(["cfg", "steps"]):
            vals = cell[f"{t}_ok"].dropna()
            mat = matrix(cell.dropna(subset=[f"{t}_ok"]), f"{t}_ok")
            lo, hi = cluster_bootstrap(mat)
            rows.append({"constraint": t, "cfg": cfg, "steps": steps, "n_images": len(vals), "mean": vals.mean(),
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def interaction_test(df):
    """Prompt-clustered logistic GEE: primary ~ category + cfg * steps; Wald test per term."""
    d = df.assign(cfg=df.cfg.astype(str), steps=df.steps.astype(str))
    model = smf.gee("primary ~ C(category) + C(cfg) * C(steps)", "prompt_id", d, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable())
    res = model.fit()
    terms = res.wald_test_terms(scalar=True).table
    return {"wald_terms": terms.reset_index().to_dict("records"), "n": int(res.nobs)}


def clip_and_diversity(df, emb):
    rows = []
    for (pid, cfg, steps), g in df.groupby(["prompt_id", "cfg", "steps"]):
        e = np.array([emb[n] for n in g.name])
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        sims = (e @ e.T)[np.triu_indices(len(e), 1)]
        rows.append({"prompt_id": pid, "cfg": cfg, "steps": steps, "clip_score": g.clip_score.mean(),
                     "diversity": 1 - sims.mean()})
    per_prompt = pd.DataFrame(rows)
    return per_prompt, per_prompt.groupby(["cfg", "steps"])[["clip_score", "diversity"]].agg(["mean", "std"]).reset_index()


def secondary_contrasts(df, per_prompt, cfgs, steps):
    """CLIP score (prompt x seed pairs) and seed diversity (prompt level) contrasts across CFG and steps."""
    rows = []
    for factor, levels in (("cfg", cfgs), ("steps", steps)):
        for a, b in combinations(levels, 2):
            c = paired_contrast(df, "clip_score", factor, a, b)
            rows.append({"measure": "clip_score", "factor": factor, **c})
            wide = per_prompt.pivot_table(index="prompt_id", columns=factor, values="diversity")
            d = (wide[a] - wide[b]).to_numpy()
            rows.append({"measure": "diversity", "factor": factor, "a": a, "b": b, "diff": d.mean(),
                         "n_prompts": len(d), "dz": d.mean() / d.std(ddof=1) if d.std() > 0 else np.nan,
                         "p_wilcoxon": wilcoxon(d).pvalue if np.any(d != 0) else 1.0})
    return pd.DataFrame(rows)


def write_markdown_tables(out):
    cells = pd.read_csv(out / "adherence_by_cell.csv")
    cfg_c = pd.read_csv(out / "cfg_contrasts.csv")
    st_c = pd.read_csv(out / "steps_contrasts.csv")
    fmt = lambda r: f"{r['mean']:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
    lines = ["## Primary success by category, CFG and steps: mean [95% cluster-bootstrap CI]", "",
             "| category | prompts | " + " | ".join(f"CFG {c:g} / {s:g} st" for c in sorted(cells.cfg.unique()) for s in sorted(cells.steps.unique())) + " |",
             "|---|---|" + "---|" * 6]
    for cat in CATEGORIES + ["all"]:
        sub = cells[cells.category == cat]
        row = [fmt(sub[(sub.cfg == c) & (sub.steps == s)].iloc[0]) for c in sorted(cells.cfg.unique()) for s in sorted(cells.steps.unique())]
        lines.append(f"| {cat} | {int(sub.n_prompts.iloc[0])} | " + " | ".join(row) + " |")
    for title, c in (("CFG contrasts (higher minus lower level, steps pooled)", cfg_c), ("Steps contrasts (40 minus 20 steps, CFG pooled)", st_c)):
        lines += ["", f"## {title}", "", "| category | higher | lower | diff [95% CI] | d_z | p (Wilcoxon) | p (Holm) |", "|---|---|---|---|---|---|---|"]
        for _, r in c.iterrows():
            lines.append(f"| {r.category} | {r.b:g} | {r.a:g} | {-r['diff']:+.3f} [{-r.ci_hi:+.3f}, {-r.ci_lo:+.3f}] | {-r.dz:.2f} | {r.p_wilcoxon:.3f} | {r.p_holm:.3f} |")
    (out / "tables.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    out = run / "results"
    out.mkdir(exist_ok=True)
    df, emb = load_table(run)
    df.to_csv(out / "per_image.csv", index=False)

    cfgs, steps = sorted(df.cfg.unique()), sorted(df.steps.unique())
    cell_table(df).to_csv(out / "adherence_by_cell.csv", index=False)
    cell_table(df, "frac_constraints").to_csv(out / "constraint_fraction_by_cell.csv", index=False)
    contrast_table(df, "cfg", cfgs).to_csv(out / "cfg_contrasts.csv", index=False)
    contrast_table(df, "steps", steps).to_csv(out / "steps_contrasts.csv", index=False)
    constraint_type_table(df).to_csv(out / "constraint_type_by_cell.csv", index=False)
    per_prompt, summary = clip_and_diversity(df, emb)
    per_prompt.to_csv(out / "clip_diversity_per_prompt.csv", index=False)
    secondary_contrasts(df, per_prompt, cfgs, steps).to_csv(out / "clip_diversity_contrasts.csv", index=False)
    summary.to_csv(out / "clip_diversity_summary.csv")
    counts = {"images_evaluated": len(df), "prompts": int(df.prompt_id.nunique()), "seeds": sorted(map(int, df.seed.unique())),
              "cfg": cfgs, "steps": steps, "excluded": 0,
              "images_per_cell_expected": int(df.prompt_id.nunique() * df.seed.nunique()),
              "successes_primary": int(df.primary.sum()), "failures_primary": int((df.primary == 0).sum())}
    (out / "counts.json").write_text(json.dumps(counts, indent=1, default=float))
    try:
        (out / "interaction.json").write_text(json.dumps(interaction_test(df), indent=1, default=float))
    except Exception as e:  # separation or singular covariance in small data
        (out / "interaction.json").write_text(json.dumps({"error": repr(e)}))
    write_markdown_tables(out)
    print(json.dumps(counts, default=float))


if __name__ == "__main__":
    main()
