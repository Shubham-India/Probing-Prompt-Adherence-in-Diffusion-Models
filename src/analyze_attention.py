"""Summarises the stored cross-attention maps and relates them to prompt adherence (all analyses exploratory).

Per image and denoising stage (early = steps 0-12, middle = 13-26, late = 27-39 of 40) and per word:
  strength     mean raw attention probability of the word's tokens over the 32x32 map
  top10        share of the (normalised) map's mass inside its 10% highest pixels  -> localisation
  entropy      normalised Shannon entropy of the normalised map                      -> spread
  box_ratio    share of normalised mass inside the detector box of the object / box area share (>1: grounded)
  bind_margin  corr(colour map, own object map) - corr(colour map, other object map)  (two-object binding prompts)
Usage: python src/analyze_attention.py --run outputs/main
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import RNG, ROOT, cluster_bootstrap, load_table, paired_contrast

STAGES = {"early": slice(0, 13), "middle": slice(13, 27), "late": slice(27, 40)}
RES = 32


def normalise(m):
    return m / (m.sum() + 1e-12)


def top10_share(m):
    p = np.sort(normalise(m).ravel())[::-1]
    return p[: int(0.1 * p.size)].sum()


def entropy(m):
    p = normalise(m).ravel()
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))


def box_mask(box, size=512):
    x0, y0, x1, y1 = [v / size * RES for v in box]
    mask = np.zeros((RES, RES))
    mask[int(max(y0, 0)):int(np.ceil(min(y1, RES))), int(max(x0, 0)):int(np.ceil(min(x1, RES)))] = 1
    return mask


def find_word(words, obj):
    return next((w for w in words if w == obj or w == obj + "s"), None)


def image_features(maps, constraints, dets):
    """maps: {word: [steps, 32, 32]}. Returns per-stage feature dict and the per-step top10 curve."""
    objects = sorted({c["object"] for c in constraints if "object" in c} |
                     {c[k] for c in constraints for k in ("left", "right") if k in c})
    obj_words = {o: find_word(maps, o) for o in objects}
    binding = [c for c in constraints if c["type"] == "binding"]
    feats, curves = {}, []
    for stage, sl in STAGES.items():
        avg = {w: m[sl].astype(float).mean(0) for w, m in maps.items()}
        ow = [w for w in obj_words.values() if w]
        feats[f"{stage}_strength"] = np.mean([avg[w].mean() for w in ow])
        feats[f"{stage}_top10"] = np.mean([top10_share(avg[w]) for w in ow])
        feats[f"{stage}_entropy"] = np.mean([entropy(avg[w]) for w in ow])
        ratios = []
        for o, w in obj_words.items():
            boxes = [d for d in dets.get(o, []) if d["score"] >= 0.10]
            if w and boxes:
                mask = box_mask(boxes[0]["box"])
                ratios.append((normalise(avg[w]) * mask).sum() / max(mask.mean(), 1e-6))
        feats[f"{stage}_box_ratio"] = np.mean(ratios) if ratios else np.nan
        margins = []
        if len(objects) == 2:
            for c in binding:
                cw, own = find_word(maps, c["color"]), obj_words[c["object"]]
                other = obj_words[[o for o in objects if o != c["object"]][0]]
                if cw and own and other:
                    corr = lambda a, b: np.corrcoef(avg[a].ravel(), avg[b].ravel())[0, 1]
                    margins.append(corr(cw, own) - corr(cw, other))
        feats[f"{stage}_bind_margin"] = np.mean(margins) if margins else np.nan
    ow = [w for w in obj_words.values() if w]
    curves = np.mean([[top10_share(m[t].astype(float)) for t in range(m.shape[0])] for w, m in maps.items() if w in ow], axis=0)
    return feats, curves


def auc_with_ci(df, feature, outcome="primary", center_within_prompt=False, n=1000):
    d = df.dropna(subset=[feature, outcome]).copy()
    if center_within_prompt:
        d[feature] = d[feature] - d.groupby("prompt_id")[feature].transform("mean")
    if d[outcome].nunique() < 2:
        return np.nan, np.nan, np.nan, len(d)
    auc = roc_auc_score(d[outcome], d[feature])
    ids = d.prompt_id.unique()
    groups = {i: g for i, g in d.groupby("prompt_id")}
    boots = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in RNG.choice(ids, len(ids))])
        if s[outcome].nunique() == 2:
            boots.append(roc_auc_score(s[outcome], s[feature]))
    return auc, *np.percentile(boots, [2.5, 97.5]), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    out = run / "results"
    prompts = {p["id"]: p for p in json.loads((ROOT / "data" / "prompts.json").read_text())}
    df, _ = load_table(run)
    detections = {json.loads(l)["name"]: json.loads(l)["detections"] for l in (run / "detections.jsonl").read_text().splitlines()}
    rows, curve_rows = [], []
    for f in sorted((run / "attention").glob("*.npz")):
        name = f.stem
        z = np.load(f)
        maps = {w: m for w, m in zip(z["words"], z["maps"])}
        pid = df.loc[df.name == name, "prompt_id"].iat[0]
        feats, curve = image_features(maps, prompts[pid]["constraints"], detections[name])
        rows.append({"name": name, **feats})
        curve_rows += [{"name": name, "step": t, "top10": v} for t, v in enumerate(curve)]
    feat = df.merge(pd.DataFrame(rows), on="name")
    feat.to_csv(out / "attention_features.csv", index=False)
    curves = df[["name", "cfg", "category"]].merge(pd.DataFrame(curve_rows), on="name")
    curves.groupby(["cfg", "step"]).top10.agg(["mean", "sem"]).reset_index().to_csv(out / "attention_top10_by_step.csv", index=False)
    print(f"{len(feat)} images with attention features")

    # 1. does CFG change attention summaries? (paired over prompt x seed, steps fixed at 40)
    cfg_rows = []
    for stage in STAGES:
        for metric in ("strength", "top10", "entropy", "box_ratio"):
            col = f"{stage}_{metric}"
            for a, b in ((7.5, 2), (15, 7.5), (15, 2)):
                cfg_rows.append({"stage": stage, "metric": metric, **paired_contrast(feat.dropna(subset=[col]), col, "cfg", a, b)})
            for cfg, g in feat.groupby("cfg"):
                cfg_rows.append({"stage": stage, "metric": metric, "a": cfg, "b": "level", "diff": g[col].mean()})
    pd.DataFrame(cfg_rows).to_csv(out / "attention_cfg_effect.csv", index=False)

    # 2. do attention summaries separate successes from failures? (AUC, exploratory)
    auc_rows = []
    binding_df = feat[feat.category.isin(["binding", "multi"]) & feat.binding_ok.notna() & feat.middle_bind_margin.notna()]
    for stage in STAGES:
        for metric in ("strength", "top10", "entropy", "box_ratio"):
            col = f"{stage}_{metric}"
            for centered in (False, True):
                auc, lo, hi, n = auc_with_ci(feat, col, "primary", centered)
                auc_rows.append({"feature": col, "outcome": "primary", "within_prompt": centered, "auc": auc, "ci_lo": lo, "ci_hi": hi, "n": n})
        col = f"{stage}_bind_margin"
        for centered in (False, True):
            auc, lo, hi, n = auc_with_ci(binding_df, col, "binding_ok", centered)
            auc_rows.append({"feature": col, "outcome": "binding_ok", "within_prompt": centered, "auc": auc, "ci_lo": lo, "ci_hi": hi, "n": n})
    pd.DataFrame(auc_rows).to_csv(out / "attention_auc.csv", index=False)

    # 3. attention-only classifier with prompt-grouped cross-validation vs the prompt+cfg baseline
    cols = [f"{s}_{m}" for s in STAGES for m in ("strength", "top10", "entropy")]
    d = feat.dropna(subset=cols + ["primary"])
    X_att = d[cols].to_numpy()
    X_base = pd.get_dummies(d[["category"]]).assign(cfg=d.cfg, steps=d.steps).to_numpy(float)
    y = d.primary.to_numpy()
    groups = d.prompt_id.to_numpy()
    summary = {}
    for label, X in (("attention_only", X_att), ("category_cfg_only", X_base), ("attention_plus_category_cfg", np.hstack([X_att, X_base]))):
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            pred[te] = model.predict_proba(X[te])[:, 1]
        summary[label] = roc_auc_score(y, pred)
    summary["n"] = len(y)
    (out / "attention_classifier_auc.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
