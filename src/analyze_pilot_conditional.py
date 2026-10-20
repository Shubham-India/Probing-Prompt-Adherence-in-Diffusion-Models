"""M2: pilot re-analysis from SAVED artifacts only (no generation, no model weights).

* Reproduces the legacy per-constraint results by re-scoring outputs/main/detections.jsonl with the unchanged
  evaluate.score() and comparing with outputs/main/eval.jsonl (--check fails otherwise).
* Computes the D-04 outcomes as a standalone function :
    pres_all          every required object (all objects named by any constraint) has a distinct detection >= presence threshold
    bind_given_pres   all binding constraints satisfied, among images with pres_all (eligible = has binding constraints & pres_all)
    joint_bind        pres_all AND all binding constraints satisfied (omission counts as failure)
    spatial_given_pres / joint_spatial   the same for spatial constraints
  Legacy fields (binding_ok, spatial_ok, ...) are carried unchanged next to the new ones.
* Noise-only baseline: how much of the outcome is explained by the initial-noise seed (the only noise information in the pilot),
  and whether early attention features add AUC beyond prompt category + CFG (+ seed). Exploratory; pilot prompts are DEV-only.
Everything here is EXPLORATORY pilot analysis, not confirmatory evidence. Output: outputs/m2_pilot_reanalysis/.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import evaluate as E  # noqa: E402
from evaluate import score  # noqa: E402  (unchanged legacy scorer)
from evaluate import required_objects  # noqa: E402,F401

RUN = ROOT / "outputs" / "main"
OUT = ROOT / "outputs" / "m2_pilot_reanalysis"


def conditional_outcomes(constraints, dets):
    """Thin wrapper over evaluate.conditional_outcomes keeping this script's column names (pres_all, *_legacy, *_eligible)."""
    o = E.conditional_outcomes(constraints, dets)
    res = score(constraints, dets)
    out = {"n_required_objects": o["n_required_objects"], "pres_all": o["objects_present"]}
    for t, key in (("binding", "bind"), ("spatial", "spatial")):
        fam = [r for c, r in zip(constraints, res) if c["type"] == t]
        out[f"{key}_legacy"] = int(all(fam)) if fam else np.nan
        out[f"{key}_eligible"] = o["objects_present"] if fam else np.nan
        out[f"{key}_given_pres"] = np.nan if o[f"{key}_given_pres"] is None else o[f"{key}_given_pres"]
        out[f"joint_{key}"] = np.nan if o[f"joint_{key}"] is None else o[f"joint_{key}"]
    return out


def load():
    prompts = {p["id"]: p for p in json.loads((ROOT / "data" / "prompts.json").read_text())}
    ev = {r["name"]: r for r in map(json.loads, (RUN / "eval.jsonl").read_text().splitlines())}
    rows, mism = [], 0
    for line in (RUN / "detections.jsonl").read_text().splitlines():
        d = json.loads(line)
        e = ev[d["name"]]
        cons = prompts[e["prompt_id"]]["constraints"]
        if score(cons, d["detections"]) != e["constraint_results"]:
            mism += 1
        rows.append({"name": d["name"], "prompt_id": e["prompt_id"], "category": e["category"], "cfg": e["cfg"], "steps": e["steps"],
                     "seed": e["seed"], "all_satisfied_legacy": e["all_satisfied"],
                     "has_binding": any(c["type"] == "binding" for c in cons), **conditional_outcomes(cons, d["detections"])})
    return pd.DataFrame(rows), mism, len(ev)


def boot_ci(df, col, rng_key, n=2000):
    """Mean and 95% percentile CI, resampling prompts (scenes) as clusters; per-call seeded RNG."""
    x = df.dropna(subset=[col])
    if x.empty:
        return np.nan, np.nan, np.nan, 0, 0
    g = x.groupby("prompt_id")[col].agg(["sum", "count"])
    rng = np.random.default_rng(int(hashlib.sha256(rng_key.encode()).hexdigest()[:8], 16))
    idx = rng.integers(0, len(g), (n, len(g)))
    s, c = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    m = s / np.maximum(c, 1)
    return float(g["sum"].sum() / g["count"].sum()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(g["count"].sum()), len(g)


def rates_table(df):
    rows = []
    for (cat, cfg), sub in df.groupby(["category", "cfg"]):
        for col, label in (("pres_all", "P(all required objects present)"), ("bind_given_pres", "P(binding | present)"),
                           ("joint_bind", "P(present & binding)"), ("bind_legacy", "legacy binding_ok"),
                           ("spatial_given_pres", "P(spatial | present)"), ("joint_spatial", "P(present & spatial)"),
                           ("spatial_legacy", "legacy spatial_ok"), ("all_satisfied_legacy", "legacy all_satisfied")):
            m, lo, hi, n, k = boot_ci(sub, col, f"{cat}|{cfg}|{col}")
            if n:
                rows.append(dict(category=cat, cfg=cfg, outcome=label, mean=m, ci_lo=lo, ci_hi=hi, n_images=n, n_prompts=k,
                                 missing_object_rate=1 - sub["pres_all"].mean()))
    return pd.DataFrame(rows)


def eligibility(df):
    b = df[df.has_binding]
    rows = []
    for (cat, cfg), s in b.groupby(["category", "cfg"]):
        rows.append(dict(category=cat, cfg=cfg, n_binding_images=len(s), n_eligible=int(s.pres_all.sum()), eligible_fraction=s.pres_all.mean(),
                         prompts_with_zero_eligible=int((s.groupby("prompt_id").pres_all.sum() == 0).sum()), n_prompts=s.prompt_id.nunique()))
    rows.append(dict(category="ALL binding-bearing", cfg="all", n_binding_images=len(b), n_eligible=int(b.pres_all.sum()),
                     eligible_fraction=b.pres_all.mean(), prompts_with_zero_eligible=int((b.groupby("prompt_id").pres_all.sum() == 0).sum()),
                     n_prompts=b.prompt_id.nunique()))
    return pd.DataFrame(rows)


# ----- noise-only baseline ------------------------------------------------------------------------------------------
def seed_permutation(df, n=5000, seed=0):
    """Within each prompt x cfg x steps cell (same prompt, same settings) only the seed differs. Statistic: sum over seeds of
    (seed-mean of cell-centred outcome)^2. Null: shuffle seeds within cells. Tests a seed main effect shared across prompts."""
    d = df.copy()
    d["y"] = d["all_satisfied_legacy"] - d.groupby(["prompt_id", "cfg", "steps"])["all_satisfied_legacy"].transform("mean")
    d = d[d.groupby(["prompt_id", "cfg", "steps"])["seed"].transform("count") > 1]
    seeds = d.seed.to_numpy()
    cells = d.groupby(["prompt_id", "cfg", "steps"]).ngroup().to_numpy()
    y = d.y.to_numpy()
    useed = np.unique(seeds)

    def stat(sd):
        return sum(y[sd == s].mean() ** 2 for s in useed)
    obs = stat(seeds)
    rng = np.random.default_rng(seed)
    order = np.argsort(cells, kind="stable")
    bounds = np.flatnonzero(np.diff(cells[order])) + 1
    null = []
    for _ in range(n):
        sh = seeds[order].copy()
        for blk in np.split(np.arange(len(order)), bounds):
            sh[blk] = rng.permutation(sh[blk])
        out = np.empty_like(sh)
        out[order] = sh
        null.append(stat(out))
    return dict(statistic=float(obs), p_value=float((np.sum(np.array(null) >= obs) + 1) / (n + 1)), n_images=len(d),
                per_seed_mean_centred={int(s): float(y[seeds == s].mean()) for s in useed})


def nested_auc(df, att, n_boot=1000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    d = df[df.steps == 40].merge(att, on="name", how="inner", suffixes=("", "_a"))
    y = d.all_satisfied_legacy.to_numpy()
    base = pd.get_dummies(d[["category"]].astype(str)).join(pd.get_dummies(d.cfg.astype(str), prefix="cfg")).astype(float)
    seed_oh = pd.get_dummies(d.seed.astype(str), prefix="seed").astype(float)
    cols = [c for c in att.columns if c.startswith(("early_", "middle_", "late_"))]
    A = d[cols].copy()
    ind = A.isna().astype(float).add_suffix("_na")
    A = A.fillna(A.median()).join(ind.loc[:, ind.nunique() > 1])
    sets = {"B_design(category+cfg)": base, "B_noise(+seed id)": base.join(seed_oh), "B_attn(+early/mid/late attention)": base.join(A.astype(float)),
            "B_noise+attn": base.join(seed_oh).join(A.astype(float))}
    groups = d.prompt_id.to_numpy()
    oof = {}
    for k, X in sets.items():
        p = np.zeros(len(d))
        for tr, te in GroupKFold(n_splits=7).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(X.iloc[tr], y[tr])
            p[te] = m.predict_proba(X.iloc[te])[:, 1]
        oof[k] = p
    res = {k: float(roc_auc_score(y, p)) for k, p in oof.items()}
    ug = np.unique(groups)
    rng = np.random.default_rng(1)
    diffs = {"attn_minus_design": ("B_attn(+early/mid/late attention)", "B_design(category+cfg)"),
             "noise_minus_design": ("B_noise(+seed id)", "B_design(category+cfg)"),
             "attn_minus_noise": ("B_noise+attn", "B_noise(+seed id)")}
    boots = {k: [] for k in diffs}
    for _ in range(n_boot):
        pick = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[pick])) < 2:
            continue
        for k, (a, b) in diffs.items():
            boots[k].append(roc_auc_score(y[pick], oof[a][pick]) - roc_auc_score(y[pick], oof[b][pick]))
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    return dict(n=len(d), n_prompts=len(ug), base_rate=float(y.mean()), auc=res,
                delta_auc={k: float(res[a] - res[b]) for k, (a, b) in diffs.items()}, delta_auc_prompt_bootstrap_ci95=ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail unless the legacy results are reproduced exactly")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, mism, n_eval = load()
    assert len(df) == n_eval == 1008
    print(f"reproduction: {len(df) - mism}/{len(df)} images re-score identically to outputs/main/eval.jsonl")
    # legacy per_image.csv fields must agree with the recomputed legacy fields
    old = pd.read_csv(RUN / "results" / "per_image.csv").set_index("name")
    j = df.set_index("name").join(old[["binding_ok", "spatial_ok", "presence_ok", "all_satisfied"]], rsuffix="_old")
    leg_bind = int(((j.bind_legacy.fillna(-1)) != (j.binding_ok.fillna(-1))).sum())
    leg_sp = int(((j.spatial_legacy.fillna(-1)) != (j.spatial_ok.fillna(-1))).sum())
    leg_all = int((j.all_satisfied_legacy != j.all_satisfied).sum())
    print(f"legacy field disagreements vs per_image.csv: binding {leg_bind}, spatial {leg_sp}, all_satisfied {leg_all}")
    if args.check:
        assert mism == 0 and leg_bind == 0 and leg_sp == 0 and leg_all == 0, "legacy results not reproduced"
    df.to_csv(OUT / "per_image_conditional.csv", index=False)
    rt, el = rates_table(df), eligibility(df)
    rt.to_csv(OUT / "conditional_rates.csv", index=False)
    el.to_csv(OUT / "eligibility.csv", index=False)
    b = df[df.has_binding & df.bind_legacy.notna()]
    disc = {"binding_bearing_prompts": int(df[df.has_binding].prompt_id.nunique()),
            "binding_images": int(len(b)),
            "legacy_binding_1_but_not_all_objects_present": int(((b.bind_legacy == 1) & (b.pres_all == 0)).sum()),
            "joint_bind_differs_from_legacy_binding": int((b.joint_bind != b.bind_legacy).sum()),
            "note": "the legacy binding result already scores 0 when an object is missing, so legacy binding equals the joint (present & binding) outcome here; what is new is the conditional rate and its eligibility denominator"}
    perm = seed_permutation(df)
    att = pd.read_csv(RUN / "results" / "attention_features.csv")
    nested = nested_auc(df, att)
    summary = {"status": "EXPLORATORY pilot re-analysis (pilot prompts are DEV-only); not confirmatory",
               "reproduction": {"images": len(df), "rescore_mismatches": mism, "legacy_field_disagreements": {"binding": leg_bind, "spatial": leg_sp, "all_satisfied": leg_all}},
               "binding_bearing": disc, "seed_effect_permutation": perm, "nested_auc_steps40_grouped_cv_by_prompt": nested}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 200)
    print(rt[rt.outcome.isin(["P(all required objects present)", "P(binding | present)", "P(present & binding)", "legacy binding_ok"])]
          .round(3).to_string(index=False))
    print(el.round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("binding_bearing", "seed_effect_permutation", "nested_auc_steps40_grouped_cv_by_prompt")}, indent=1))


if __name__ == "__main__":
    main()
