"""Diagnostic of the failed DEV full-resample control (EXPLORATORY; post hoc; does not change any registered estimate).
FULL branches restart from fresh N(0, I) noise, so a branch is an independent draw of the same prompt; under a valid estimator C(t) ~ 0.
The registered gate (max |C| <= 0.15 over kinds and t) failed with 0.568. This script
  1. reproduces the gate value with the frozen functions (r3_analyze.prepare, commitment._arrays / curve_from_arrays);
  2. decomposes presence agreement into: base-vs-FULL (A), base seed pairs (A0), FULL-vs-FULL pairs of the same base, and the marginal success
     rates of base seeds vs FULL branches (a pipeline or selection shift shows up as a rate difference);
  3. checks whether base seed 0 and seed 1 behave differently;
  4. simulates the gate statistic under exchangeable outcomes (each scene's outcomes iid Bernoulli at its pooled rate, same design) to ask whether
     0.568 and the presence offset are within sampling noise.
Usage: python diagnostics/full_control.py --rows <DEV eval_rows.jsonl> [--sims 2000]
Writes reports/diagnostics/full_control.json and paper/figures/diag_full_control.pdf."""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import commitment as C  # noqa: E402
import dedup as D  # noqa: E402
import r3_analyze as RA  # noqa: E402

OUT = ROOT / "reports" / "diagnostics"
KINDS = ("presence", "count", "bind_given_pres", "spatial_given_pres", "joint_bind")
COL = {"presence": "objects_present", "count": "count_ok", "bind_given_pres": "bind_given_pres", "spatial_given_pres": "spatial_given_pres", "joint_bind": "joint_bind"}


def gate_curves(core, full):
    adf, a0 = RA.prepare(core, full, "FULL")
    scenes = sorted(set(adf.scene) | set(a0.scene))
    out = {}
    for k in sorted(set(adf.kind)):
        num, den, cnt = C._arrays(adf, a0, k, scenes, RA.GRID)
        cur = C.curve_from_arrays(num, den)
        out[k] = dict(curve=[None if not math.isfinite(x) else float(x) for x in cur], n_scenes_per_t=[int(x) for x in cnt.sum(0)],
                      mean_A=[float(adf[(adf.kind == k) & (adf.t == t)].A.mean()) for t in RA.GRID],
                      mean_A0=float(a0[a0.kind == k].A0.mean()), n_scenes_A0=int((a0.kind == k).sum()))
    return out


def pair_agree(ys):
    ys = [y for y in ys if y is not None]
    n = len(ys)
    if n < 2:
        return None
    o = sum(ys)
    return (o * (o - 1) + (n - o) * (n - o - 1)) / (n * (n - 1))


def decompose(core, full, col):
    """Per scene: base-seed outcomes (all seeds of the base prompt at 7.5/40), FULL outcomes grouped by base."""
    seeds = defaultdict(list)
    keys = {RA.base_key(r) for r in core if r["job_id"] in {f["base_job"] for f in full}}
    by_job = {r["job_id"]: r for r in core}
    for r in core:
        if RA.base_key(r) in keys and r.get(col) is not None:
            seeds[r["scene_id"]].append((r["seed"], r["stage"], int(r[col])))
    fb = defaultdict(list)
    for f in full:
        if f.get(col) is not None:
            fb[f["base_job"]].append(int(f[col]))
    res = dict(scenes=0, A_full_base=[], A0_seeds=[], A_full_full=[], rate_seeds=[], rate_full=[], A_by_base_seed={0: [], 1: []}, rate_by_seed=defaultdict(list))
    for s, lst in seeds.items():
        ys = [y for _, _, y in lst]
        bases = [b for b in fb if by_job.get(b, {}).get("scene_id") == s]
        fulls = [y for b in bases for y in fb[b]]
        if len(ys) < 2 or not fulls:
            continue
        res["scenes"] += 1
        res["A0_seeds"].append(pair_agree(ys))
        res["rate_seeds"].append(float(np.mean(ys)))
        res["rate_full"].append(float(np.mean(fulls)))
        for seed, _, y in lst:
            res["rate_by_seed"][seed].append(y)
        for b in bases:
            yb = by_job[b].get(col)
            if yb is None:
                continue
            a = float(np.mean([int(y == yb) for y in fb[b]]))
            res["A_full_base"].append(a)
            res["A_by_base_seed"].setdefault(int(by_job[b]["seed"]), []).append(a)
            res["A_full_full"].append(pair_agree(fb[b]))
    return res


def summarise(x):
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return None
    g = np.random.default_rng(7)
    d = [x[g.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return dict(mean=float(x.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], n=int(len(x)))


def simulate(core, full, sims):
    """Null: within a scene, base seeds and FULL branches are exchangeable draws at the scene's pooled rate. Same design (seeds, bases, branch
    counts, eligibility pattern taken from the data), frozen estimator. Returns the distribution of the gate statistic and of presence C."""
    import copy
    g = np.random.default_rng(20261020)
    keys = {RA.base_key(r) for r in core if r["job_id"] in {f["base_job"] for f in full}}
    core_sel = [r for r in core if RA.base_key(r) in keys]
    pooled = defaultdict(lambda: defaultdict(list))
    for r in core_sel + full:
        for k, c in COL.items():
            if r.get(c) is not None:
                pooled[r["scene_id"]][c].append(int(r[c]))
    rate = {s: {c: float(np.mean(v)) for c, v in d.items()} for s, d in pooled.items()}
    gate, pres = [], []
    for _ in range(sims):
        cs, fs = [], []
        for r in core_sel:
            q = dict(r)
            for c in COL.values():
                if r.get(c) is not None:
                    q[c] = int(g.random() < rate[r["scene_id"]][c])
            cs.append(q)
        for r in full:
            q = dict(r)
            for c in COL.values():
                if r.get(c) is not None:
                    q[c] = int(g.random() < rate[r["scene_id"]][c])
            fs.append(q)
        cur = gate_curves(cs, fs)
        vals = [abs(x) for k in cur for x in cur[k]["curve"] if x is not None]
        gate.append(max(vals) if vals else math.nan)
        pc = [x for x in cur.get("presence", {}).get("curve", []) if x is not None]
        pres.append(float(np.mean(pc)) if pc else math.nan)
    return np.asarray(gate), np.asarray(pres)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--sims", type=int, default=500)
    a = ap.parse_args()
    rows, _ = D.dedup(RA.read(Path(a.rows)))
    core = [r for r in rows if r.get("kind", "core") == "core"]
    full = [r for r in rows if r.get("kind") == "branch" and r.get("restart") == "FULL"]
    R = dict(status="EXPLORATORY diagnostic of the DEV full-resample control (post hoc)", n_full=len(full),
             n_bases=len({f["base_job"] for f in full}))
    cur = gate_curves(core, full)
    R["registered_gate_reproduced"] = max(abs(x) for k in cur for x in cur[k]["curve"] if x is not None)
    R["curves"] = cur
    R["argmax"] = max(((k, RA.GRID[i], x, cur[k]["n_scenes_per_t"][i]) for k in cur for i, x in enumerate(cur[k]["curve"]) if x is not None), key=lambda z: abs(z[2]))
    dec = decompose(core, full, "objects_present")
    R["presence_decomposition"] = {k: summarise(dec[k]) for k in ("A_full_base", "A0_seeds", "A_full_full", "rate_seeds", "rate_full")}
    R["presence_decomposition"]["scenes"] = dec["scenes"]
    R["presence_decomposition"]["A_by_base_seed"] = {str(k): summarise(v) for k, v in dec["A_by_base_seed"].items() if v}
    R["presence_decomposition"]["rate_by_seed"] = {str(k): summarise(v) for k, v in sorted(dec["rate_by_seed"].items())}
    # fixed seed panel: every prompt uses initial-noise seeds 0-3, so A0 is agreement within one shared panel of four noise tensors
    import itertools
    panel = defaultdict(dict)
    keys = {RA.base_key(r) for r in core if r["job_id"] in {f["base_job"] for f in full}}
    for r in core:
        if RA.base_key(r) in keys and r.get("objects_present") is not None:
            panel[r["scene_id"]][int(r["seed"])] = int(r["objects_present"])
    fb = defaultdict(list)
    for f in full:
        if f.get("objects_present") is not None:
            fb[f["scene_id"]].append(int(f["objects_present"]))
    S = [s for s in panel if len(panel[s]) == 4 and s in fb]
    Y = np.array([[panel[s][k] for k in range(4)] for s in S])
    g = np.random.default_rng(0)
    obs = Y.mean(0).var()
    perm = [np.array([row[g.permutation(4)] for row in Y]).mean(0).var() for _ in range(5000)]
    R["seed_panel"] = dict(scenes=len(S), seed_rates=Y.mean(0).round(3).tolist(), seed_main_effect_perm_p=float(np.mean(np.array(perm) >= obs)),
                           seed_pair_agreement={f"{a_}-{b_}": float((Y[:, a_] == Y[:, b_]).mean()) for a_, b_ in itertools.combinations(range(4), 2)},
                           full_vs_seed_agreement={str(k): float(np.mean([np.mean([int(y == panel[s][k]) for y in fb[s]]) for s in S])) for k in range(4)},
                           full_full_agreement=float(np.mean([pair_agree(fb[s]) for s in S])))
    dd = np.array([pair_agree(fb[s]) - pair_agree(list(panel[s].values())) for s in S])
    R["seed_panel"]["full_full_minus_panel"] = summarise(dd)
    diff = np.asarray(dec["rate_full"]) - np.asarray(dec["rate_seeds"])
    R["presence_decomposition"]["rate_full_minus_seeds"] = summarise(diff)
    gate, pres = simulate(core, full, a.sims)
    obs_pres = float(np.mean([x for x in cur["presence"]["curve"] if x is not None]))
    R["null_simulation"] = dict(sims=a.sims, gate_quantiles={q: float(np.nanpercentile(gate, q)) for q in (50, 90, 95, 99)},
                                p_gate_ge_observed=float(np.nanmean(gate >= R["registered_gate_reproduced"])),
                                share_null_passing_0_15=float(np.nanmean(gate <= 0.15)),
                                presence_mean_C_observed=obs_pres, presence_mean_C_null_quantiles={q: float(np.nanpercentile(pres, q)) for q in (2.5, 50, 97.5)},
                                p_presence_ge_observed=float(np.nanmean(pres >= obs_pres)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full_control.json").write_text(json.dumps(R, indent=1, default=str))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for k in cur:
        ax[0].plot(RA.GRID, [math.nan if x is None else x for x in cur[k]["curve"]], marker="o", ms=3, label=k.replace("_given_pres", "|pres"))
    ax[0].axhspan(-0.15, 0.15, color="grey", alpha=0.15); ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set_xlabel("restart step t (of 40)"); ax[0].set_ylabel("C(t), FULL control"); ax[0].legend(fontsize=6); ax[0].set_title("(a) DEV full-resample control", fontsize=8)
    ax[1].hist(gate[np.isfinite(gate)], bins=30, color="C7")
    ax[1].axvline(R["registered_gate_reproduced"], color="C3", label="observed"); ax[1].axvline(0.15, color="k", ls="--", label="criterion 0.15")
    ax[1].set_xlabel("max |C| under exchangeable outcomes"); ax[1].set_ylabel("simulations"); ax[1].legend(fontsize=6); ax[1].set_title("(b) gate statistic under the null", fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "paper" / "figures" / "diag_full_control.pdf"); fig.savefig(OUT / "diag_full_control.png", dpi=150); plt.close(fig)
    print(json.dumps({k: R[k] for k in ("registered_gate_reproduced", "argmax", "presence_decomposition", "null_simulation")}, indent=1, default=str))


if __name__ == "__main__":
    main()
