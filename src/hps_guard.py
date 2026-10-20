"""DEV CAG quality guard (Addendum C; blueprint 24.x "guard: drop <= 0.5 HPS points"; DL-15, DL-42).
Pure computation, no model: given HPS v2 scores of the DEV guard images (A2 at w_lo = 1 and 3, seeds 0-1, DEV canonical prompts) and of the matching
A1 images (constant CFG 7.5, same prompt and seed: the R2a DEV core), it computes the paired mean change in HPS points (score x 100, the HPS v2
paper's scale) and applies the predeclared rule: the guard passes for a w_lo when the mean drop is at most 0.5 points (mean(A2 - A1) >= -0.5).
Decision (Addendum C): w_lo = 1 if its guard passes, otherwise w_lo = 3. If both fail the rule gives no arm and the owner decides."""
import math
from collections import defaultdict

import numpy as np

DROP_POINTS = 0.5
POINTS = 100.0                     # HPS v2 raw score (cosine similarity) x 100


def a1_job(prompt_id, seed):
    return f"{prompt_id}__cfg7.5__st40__seed{int(seed)}"


def pairs(cag_records, scores):
    """{w_lo: [(scene_id, prompt_id, seed, a2_points, a1_points)]}; raises if any image lacks a score (never silently dropped)."""
    out = defaultdict(list)
    missing = []
    for r in cag_records:
        a1 = a1_job(r["prompt_id"], r["seed"])
        if r["job_id"] not in scores or a1 not in scores:
            missing.append(r["job_id"] if r["job_id"] not in scores else a1)
            continue
        out[float(r["w_lo"])].append((r["scene_id"], r["prompt_id"], int(r["seed"]), POINTS * scores[r["job_id"]], POINTS * scores[a1]))
    if missing:
        raise ValueError(f"{len(missing)} images without an HPS score, e.g. {missing[:3]}")
    return dict(out)


def guard(cag_records, scores, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for w, ps in sorted(pairs(cag_records, scores).items()):
        by_scene = defaultdict(list)
        for sc, _, _, a2, a1 in ps:
            by_scene[sc].append(a2 - a1)
        scenes = sorted(by_scene)
        per = np.array([np.mean(by_scene[s]) for s in scenes])
        diffs = np.array([a2 - a1 for *_, a2, a1 in ps])
        mean = float(diffs.mean())
        boots = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
        res[f"{w:g}"] = dict(n_pairs=len(ps), n_scenes=len(scenes), mean_change_points=mean, ci95_points=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                             mean_a2_points=float(np.mean([p[3] for p in ps])), mean_a1_points=float(np.mean([p[4] for p in ps])),
                             passed=bool(mean >= -DROP_POINTS))
    w1, w3 = res.get("1"), res.get("3")
    if w1 is None:
        decision = dict(w_lo=None, note="no w_lo = 1 guard images")
    elif w1["passed"]:
        decision = dict(w_lo=1.0, note="guard passed for w_lo = 1 (mean HPS drop <= 0.5 points)")
    elif w3 is not None and w3["passed"]:
        decision = dict(w_lo=3.0, note="guard failed for w_lo = 1 and passed for w_lo = 3: w_lo = 3 adopted before TEST (Addendum C)")
    else:
        decision = dict(w_lo=None, note="guard failed for w_lo = 1 and w_lo = 3: the predeclared rule gives no arm; owner decision required")
    return dict(rule=f"pass if mean(A2 - A1) >= -{DROP_POINTS} HPS points (x{POINTS:g}), paired by prompt and seed", per_w_lo=res, decision=decision)
