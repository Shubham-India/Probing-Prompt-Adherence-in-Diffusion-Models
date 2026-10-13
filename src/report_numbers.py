"""Collects every number quoted in the report and slides from the saved result files, so text never carries numbers typed by hand.

`numbers(run)` returns a flat dict; report/build_report.py and report/make_slides.py format their templates with it.
Convention: contrasts are reported as HIGHER level minus LOWER level (the CSVs store a - b with a the lower level).
"""
import json
from pathlib import Path

import pandas as pd

from validate_metrics import compare, load_labels

ROOT = Path(__file__).resolve().parent.parent
CATS = ["presence", "binding", "counting", "spatial", "multi", "all"]


def ptag(x):
    """7.5 -> 7p5 so the value can sit inside a format-string field name."""
    return f"{x:g}".replace(".", "p")


def signed(x, digits=2):
    return f"{x:+.{digits}f}"


def interval(lo, hi, digits=2):
    return f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"


def higher_minus_lower(df, cat, lo, hi, digits=2):
    r = df[(df.category == cat) & (df.a == lo) & (df.b == hi)].iloc[0]
    return {"diff": signed(-r["diff"], digits), "ci": interval(-r.ci_hi, -r.ci_lo, digits), "p": f"{r.p_wilcoxon:.3f}",
            "holm": f"{r.p_holm:.3f}", "dz": f"{-r.dz:.2f}", "raw": -r["diff"], "lo": -r.ci_hi, "hi": -r.ci_lo}


def numbers(run):
    run = Path(run)
    res = run / "results"
    V = {}
    counts = json.loads((res / "counts.json").read_text())
    V.update(n_images=counts["images_evaluated"], n_prompts=counts["prompts"], n_seeds=len(counts["seeds"]),
             n_success=counts["successes_primary"], n_fail=counts["failures_primary"], n_excluded=counts["excluded"])
    samples = [json.loads(l) for l in (run / "samples.jsonl").read_text().splitlines()]
    V["gen_hours"] = f"{sum(s['seconds'] for s in samples) / 3600:.1f}"
    V["n_attention"] = len(list((run / "attention").glob("*.npz")))

    cells = pd.read_csv(res / "adherence_by_cell.csv")
    for _, r in cells.iterrows():
        k = f"cell_{r.category}_{ptag(r.cfg)}_{r.steps:g}"
        V[k] = f"{r['mean']:.2f}"
        V[k + "_ci"] = f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
    V["cell_min_all"] = f"{cells[cells.category == 'all']['mean'].min():.2f}"
    V["cell_max_all"] = f"{cells[cells.category == 'all']['mean'].max():.2f}"
    V["max_binding_multi"] = f"{cells[cells.category.isin(['binding', 'multi'])]['mean'].max():.2f}"
    for cat in CATS:
        V[f"nprompts_{cat}"] = int(cells[cells.category == cat].n_prompts.iloc[0])

    cfg_c = pd.read_csv(res / "cfg_contrasts.csv")
    for cat in CATS:
        for lo, hi, tag in ((2, 7.5, "72"), (7.5, 15, "157"), (2, 15, "152")):
            for k, v in higher_minus_lower(cfg_c, cat, lo, hi).items():
                V[f"cfg{tag}_{cat}_{k}"] = v
    steps_c = pd.read_csv(res / "steps_contrasts.csv")
    for cat in CATS:
        for k, v in higher_minus_lower(steps_c, cat, 20, 40).items():
            V[f"steps_{cat}_{k}"] = v
    terms = {t["index"]: t for t in json.loads((res / "interaction.json").read_text())["wald_terms"]}
    V.update(gee_cfg_p=f"{terms['C(cfg)']['pvalue']:.2f}", gee_steps_p=f"{terms['C(steps)']['pvalue']:.2f}",
             gee_inter_p=f"{terms['C(cfg):C(steps)']['pvalue']:.2f}")

    per_image = pd.read_csv(res / "per_image.csv")
    clip = per_image.groupby("cfg").clip_score.mean()
    div = pd.read_csv(res / "clip_diversity_per_prompt.csv").groupby("cfg").diversity.mean()
    for cfg in clip.index:
        V[f"clip_{ptag(cfg)}"] = f"{clip[cfg]:.1f}"
        V[f"div_{ptag(cfg)}"] = f"{div[cfg]:.2f}"
    cd = pd.read_csv(res / "clip_diversity_contrasts.csv")
    r = cd[(cd.measure == "clip_score") & (cd.factor == "cfg") & (cd.a == 2) & (cd.b == 7.5)].iloc[0]
    V.update(clip72_diff=signed(-r["diff"]), clip72_ci=interval(-r.ci_hi, -r.ci_lo), clip72_dz=f"{-r.dz:.1f}", clip72_p=f"{r.p_wilcoxon:.1e}")
    r = cd[(cd.measure == "diversity") & (cd.factor == "cfg") & (cd.a == 2) & (cd.b == 7.5)].iloc[0]
    V.update(div72_diff=signed(-r["diff"], 3), div72_dz=f"{-r.dz:.1f}", div72_p=f"{r.p_wilcoxon:.1e}")

    eff = pd.read_csv(res / "attention_cfg_effect.csv")
    eff["a"] = eff["a"].astype(str)
    eff["b"] = eff["b"].astype(str)
    for stage in ("early", "middle", "late"):
        for metric in ("top10", "strength"):
            for a, b, tag in (("7.5", "2.0", "72"), ("15.0", "7.5", "157")):
                r = eff[(eff.stage == stage) & (eff.metric == metric) & (eff.a == a) & (eff.b.isin([b, b.replace(".0", "")]))].iloc[0]
                V[f"att_{metric}{tag}_{stage}"] = f"{r['diff']:+.4f}"
                V[f"att_{metric}{tag}_{stage}_ci"] = interval(r.ci_lo, r.ci_hi, 4)
                V[f"att_{metric}{tag}_{stage}_p"] = f"{r.p_wilcoxon:.1e}"
        lvl = eff[(eff.stage == stage) & (eff.metric == "strength") & (eff.b == "level")]
        V[f"att_strength_level2_{stage}"] = f"{lvl[lvl.a.isin(['2.0', '2'])]['diff'].iloc[0]:.4f}"
    auc = pd.read_csv(res / "attention_auc.csv")
    for _, r in auc.iterrows():
        k = f"auc_{r.feature}{'_within' if r.within_prompt else ''}"
        V[k] = f"{r.auc:.2f}"
        V[k + "_ci"] = f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
    V["auc_bind_n"] = int(auc[auc.feature == "middle_bind_margin"].n.iloc[0])
    clf = json.loads((res / "attention_classifier_auc.json").read_text())
    V.update(clf_att=f"{clf['attention_only']:.2f}", clf_base=f"{clf['category_cfg_only']:.2f}", clf_both=f"{clf['attention_plus_category_cfg']:.2f}")

    for t, v in compare(run, load_labels()).items():
        n = sum(v.values())
        V[f"val_{t}_n"] = n
        V[f"val_{t}_acc"] = f"{(v['tp'] + v['tn']) / n:.2f}"
        V[f"val_{t}_recall"] = f"{v['tp'] / max(v['tp'] + v['fn'], 1):.2f}"
        V[f"val_{t}_precision"] = f"{v['tp'] / max(v['tp'] + v['fp'], 1):.2f}"
        V[f"val_{t}_pos"] = v["tp"] + v["fn"]

    intervention = ROOT / "outputs" / "intervention" / "results" / "intervention_numbers.json"
    if intervention.exists():
        V.update(json.loads(intervention.read_text()))
    return V


if __name__ == "__main__":
    import sys
    for k, v in numbers(sys.argv[1]).items():
        print(k, v)
