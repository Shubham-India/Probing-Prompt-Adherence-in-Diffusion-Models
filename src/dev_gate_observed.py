"""Data-anchored planning simulation for the HA DEV gate (D-13). Unlike dev_gate_sim.py (synthetic p ~ U(0.3, 0.7), independent eligibility),
this keeps the OBSERVED base outcomes of a core run: per scene and seed, objects_present and bind_given_pres from the first-pass evaluator,
so eligibility, clustering of outcomes within scenes and the chance-agreement weights (1 - A0) are the real ones. Only the BRANCH outcomes are
simulated, from a hard-persistence world with known tau_presence and tau_binding = tau_presence + delta, using the real estimator
(commitment.agreement / chance_agreement / bootstrap_delta_t50). Extra base seeds do NOT add scenes: the bootstrap resamples scenes.
It is a sizing tool, not evidence about the model; judge noise of the unaudited evaluator is an input (--flip).
Usage: python src/dev_gate_observed.py --eval <eval.jsonl> --prompts data/benchmark_v2/prompts_v2.json --bases 2 4 --M 2 --delta 0 6 12 --reps 30"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import commitment as C

GRID = C.GRID


def load_core(eval_path, prompts_path):
    P = {p["id"]: p for p in json.loads(Path(prompts_path).read_text())}
    rows = [json.loads(l) for l in Path(eval_path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if any(c["type"] == "binding" for c in P[r["prompt_id"]]["constraints"])]
    scenes = sorted({r["scene_id"] for r in rows})
    k = max(r["seed"] for r in rows) + 1
    pres = np.zeros((len(scenes), k), int)
    bind = np.full((len(scenes), k), -1, int)           # -1 = not eligible (an object is missing)
    for r in rows:
        i = scenes.index(r["scene_id"])
        pres[i, r["seed"]] = r["objects_present"]
        if r["bind_given_pres"] is not None:
            bind[i, r["seed"]] = r["bind_given_pres"]
    return scenes, pres, bind


def core_frames(pres, bind):
    n, k = pres.shape
    s, sd = np.indices((n, k))
    cp = pd.DataFrame(dict(scene=s.ravel(), kind="presence", seed=sd.ravel(), y=pres.ravel(), elig=1))
    cb = pd.DataFrame(dict(scene=s.ravel(), kind="bind", seed=sd.ravel(), y=np.maximum(bind.ravel(), 0), elig=(bind.ravel() >= 0).astype(int)))
    return pd.concat([cp, cb])


def _posterior_truth(rng, obs, prior, fn, fp):
    """Latent truth of the BASE outcome given the evaluator's observed value: P(Z=1|obs) by Bayes with false-negative rate fn and false-positive rate fp."""
    p1 = np.where(obs == 1, prior * (1 - fn) / (prior * (1 - fn) + (1 - prior) * fp + 1e-12), prior * fn / (prior * fn + (1 - prior) * (1 - fp) + 1e-12))
    return (rng.random(obs.shape) < p1).astype(int)


def _judge(rng, z, fn, fp):
    """Evaluator output for latent truth z: a true 1 is read as 0 with probability fn, a true 0 as 1 with probability fp."""
    u = rng.random(z.shape)
    return np.where(z == 1, (u >= fn).astype(int), (u < fp).astype(int))


def simulate_branches(rng, pres, bind, tau_p, tau_b, n_bases, m, flip=0.0, kmax=1.0, err_p=None, err_b=None, legacy_flip=False):
    """Branch outcomes in a hard-persistence world. Evaluator noise (default model): the BASE observed outcome is the real J1 output; its latent truth is
    drawn from the posterior given the error rates; persistence acts on the latent truth; the branch observation is the evaluator applied to the branch truth,
    with independent error. err_p / err_b = (false-negative rate, false-positive rate); flip=x sets both to (x, x). legacy_flip=True reproduces the earlier
    cruder model (flip applied once to values copied from the observed base) used in the 2026-10-14 sizing note."""
    n, k = pres.shape
    err_p = err_p or (flip, flip)
    err_b = err_b or (flip, flip)
    p_s = (pres.sum(1) + 0.5) / (k + 1.0)                                       # shrunk scene presence rate
    elig_obs = bind >= 0
    g_rate = float(bind[elig_obs].mean()) if elig_obs.any() else 0.5
    q_s = np.where(elig_obs.sum(1) > 0, (np.where(elig_obs, bind, 0).sum(1) + 0.5 * g_rate * 2) / (elig_obs.sum(1) + 1.0), g_rate)
    rows = []
    s, b, mm = np.indices((n, n_bases, m))
    noisy = (max(err_p) > 0 or max(err_b) > 0) and not legacy_flip
    obs_p, obs_b = pres[s, b], np.maximum(bind[s, b], 0)
    zp = _posterior_truth(rng, obs_p, p_s[s], *err_p) if noisy else obs_p                          # latent base truth (== observed when noise-free)
    zb = _posterior_truth(rng, obs_b, q_s[s], *err_b) if noisy else obs_b
    for t in GRID:
        keep_p = kmax * float(t >= tau_p)      # kmax < 1: even after commitment a re-noised branch keeps the base outcome only with this probability
        keep_b = kmax * float(t >= tau_b)
        tp = np.where(rng.random(s.shape) < keep_p, zp, (rng.random(s.shape) < p_s[s]).astype(int))   # branch latent presence
        tb = np.where(rng.random(s.shape) < keep_b, zb, (rng.random(s.shape) < q_s[s]).astype(int))
        if noisy:
            br_p, br_b = _judge(rng, tp, *err_p), _judge(rng, tb, *err_b)
        else:
            br_p, br_b = tp, tb
            if legacy_flip and flip > 0:
                br_p = np.where(rng.random(s.shape) < flip, 1 - br_p, br_p)
                br_b = np.where(rng.random(s.shape) < flip, 1 - br_b, br_b)
        common = dict(scene=s.ravel(), seed=b.ravel(), t=t, m=mm.ravel())
        rows.append(pd.DataFrame(dict(common, kind="presence", y=br_p.ravel(), elig=1, base_y=obs_p.ravel(), base_elig=1)))
        rows.append(pd.DataFrame(dict(common, kind="bind", y=br_b.ravel(), elig=br_p.ravel(),
                                      base_y=obs_b.ravel(), base_elig=(bind[s, b] >= 0).ravel().astype(int))))
    return pd.concat(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--bases", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--M", type=int, nargs="+", default=[2])
    ap.add_argument("--delta", type=float, nargs="+", default=[0, 6, 12])
    ap.add_argument("--flip", type=float, nargs="+", default=[0.0])
    ap.add_argument("--err-presence", type=float, nargs=2, default=None, metavar=("FN", "FP"), help="measured evaluator false-negative / false-positive rate for presence")
    ap.add_argument("--err-bind", type=float, nargs=2, default=None, metavar=("FN", "FP"), help="same for binding given presence")
    ap.add_argument("--legacy-flip", action="store_true")
    ap.add_argument("--kmax", type=float, nargs="+", default=[1.0])
    ap.add_argument("--tau-presence", type=float, default=12)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    scenes, pres, bind = load_core(a.eval, a.prompts)
    core = core_frames(pres, bind)
    a0 = C.chance_agreement(core)
    info = dict(scenes=len(scenes), presence_rate=float(pres.mean()), bind_eligible_rate=float((bind >= 0).mean()),
                scenes_with_ge2_eligible_seeds=int(((bind >= 0).sum(1) >= 2).sum()),
                mean_one_minus_A0={k: float((1 - a0[a0.kind == k].A0).mean()) for k in ("presence", "bind")},
                scenes_with_A0_lt_1={k: int((a0[a0.kind == k].A0 < 1).sum()) for k in ("presence", "bind")})
    print(json.dumps(info), flush=True)
    rng = np.random.default_rng(7)
    table = []
    for nb in a.bases:
        for m in a.M:
            for fl, km in [(f, k_) for f in a.flip for k_ in a.kmax]:
                for d in a.delta:
                    hw, ex, rel, est, usable = [], 0, 0, [], []
                    for r in range(a.reps):
                        br = simulate_branches(rng, pres, bind, a.tau_presence, a.tau_presence + d, nb, m, fl, km, tuple(a.err_presence) if a.err_presence else None, tuple(a.err_bind) if a.err_bind else None, a.legacy_flip)
                        res = C.bootstrap_delta_t50(C.agreement(br), a0, "presence", "bind", n_boot=a.n_boot, key=f"obs{nb}{m}{d}{r}")
                        lo, hi = res["ci95"]
                        rel += int(res["reliable"])
                        if not math.isnan(lo):
                            hw.append((hi - lo) / 2); ex += int(lo > 0 or hi < 0)
                            est.append(res.get("delta", float("nan")))
                    row = dict(base_seeds=nb, M=m, flip=fl, kmax=km, err_presence=a.err_presence, err_bind=a.err_bind, model=('legacy' if a.legacy_flip else 'latent-truth'), true_delta_steps=d, mean_halfwidth=float(np.mean(hw)) if hw else None,
                               median_halfwidth=float(np.median(hw)) if hw else None, share_with_ci=len(hw) / a.reps, share_reliable=rel / a.reps,
                               share_ci_excludes_0=ex / a.reps, share_gate_met=float(np.mean([h <= 4 for h in hw])) if hw else 0.0,
                               images_per_run=len(scenes) * nb * 6 * 3 * m)
                    table.append(row)
                    print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(info=info, reps=a.reps, n_boot=a.n_boot, rows=table), indent=1))


if __name__ == "__main__":
    main()
