"""Nested-model AUC comparison with prompt-grouped CV and scene-level bootstrap (HB machinery, RB 8.2).
`nested_compare` fits each feature set with the same pipeline (standardise + L2 logistic), predicts out-of-fold with GroupKFold over
scene/prompt groups (no group appears in train and test), and bootstraps groups on the frozen out-of-fold predictions."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def oof_predictions(X, y, groups, n_splits=7, C=0.3):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def nested_compare(sets, y, groups, diffs, n_splits=7, C=0.3, n_boot=1000, seed=1):
    """sets: name -> feature DataFrame (same rows); diffs: name -> (set_a, set_b) giving AUC(a) - AUC(b)."""
    y, groups = np.asarray(y), np.asarray(groups)
    oof = {k: oof_predictions(X, y, groups, n_splits, C) for k, X in sets.items()}
    auc = {k: float(roc_auc_score(y, p)) for k, p in oof.items()}
    ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in diffs}
    for _ in range(n_boot):
        pick = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[pick])) < 2:
            continue
        for k, (a, b) in diffs.items():
            boots[k].append(roc_auc_score(y[pick], oof[a][pick]) - roc_auc_score(y[pick], oof[b][pick]))
    return dict(n=len(y), n_groups=len(ug), base_rate=float(y.mean()), auc=auc,
                delta_auc={k: float(auc[a] - auc[b]) for k, (a, b) in diffs.items()},
                delta_auc_ci95={k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}), oof


def permutation_delta(sets, y, groups, diff, n_perm=20, seed=0, **kw):
    """Null distribution of ΔAUC when labels are permuted WITHIN groups (keeps prompt-level base rates, destroys feature-outcome ties)."""
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    a, b = diff
    out = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            yp[idx] = rng.permutation(y[idx])
        pa, pb = (oof_predictions(sets[k], yp, groups, **kw) for k in (a, b))
        out.append(roc_auc_score(yp, pa) - roc_auc_score(yp, pb))
    return np.array(out)
