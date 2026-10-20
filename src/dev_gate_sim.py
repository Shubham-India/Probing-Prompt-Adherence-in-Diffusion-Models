"""Planning simulation for the HA DEV gate (RB 8.1: 'if the DEV 95% CI half-width of Delta-t50 exceeds 4 steps, raise M or the number of
scenes'). Synthetic world only: hard persistence times tau_presence / tau_binding, scene-specific outcome probabilities, conditional
eligibility. Assumptions are inputs, NOT estimates from data; use the output to size DEV, never as evidence about the model.
Usage: python src/dev_gate_sim.py --scenes 40 60 100 --elig 0.45 0.6 --delta 0 12 --reps 20 --out outputs/dev_gate_sim.json"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import commitment as C

GRID = C.GRID


def simulate(rng, n_scenes, tau, k=4, n_bases=2, m=2, elig=1.0, p_range=(0.3, 0.7), kind="x"):
    p = rng.uniform(*p_range, n_scenes)
    base = rng.random((n_scenes, k)) < p[:, None]
    eb = rng.random((n_scenes, k)) < elig
    core = pd.DataFrame(dict(scene=np.repeat(np.arange(n_scenes), k), kind=kind, seed=np.tile(np.arange(k), n_scenes), y=base.reshape(-1).astype(int), elig=eb.reshape(-1).astype(int)))
    rows = []
    for t in GRID:
        keep = rng.random((n_scenes, n_bases, m)) < float(t >= tau)
        fresh = rng.random((n_scenes, n_bases, m)) < p[:, None, None]
        y = np.where(keep, base[:, :n_bases, None], fresh)
        el = rng.random((n_scenes, n_bases, m)) < elig
        s, b, mm = np.indices((n_scenes, n_bases, m))
        rows.append(pd.DataFrame(dict(scene=s.ravel(), kind=kind, seed=b.ravel(), t=t, m=mm.ravel(), y=y.ravel().astype(int), elig=el.ravel().astype(int),
                                      base_y=base[s.ravel(), b.ravel()].astype(int), base_elig=eb[s.ravel(), b.ravel()].astype(int))))
    return core, pd.concat(rows)


def one(rng, n_scenes, delta, elig, n_boot, tau_a=12):
    ca, ba = simulate(rng, n_scenes, tau_a, kind="presence")
    cb, bb = simulate(rng, n_scenes, tau_a + delta, elig=elig, kind="bind")
    core, br = pd.concat([ca, cb]), pd.concat([ba, bb])
    adf, a0 = C.agreement(br), C.chance_agreement(core)
    return C.bootstrap_delta_t50(adf, a0, "presence", "bind", n_boot=n_boot, key=f"sim{rng.integers(1e9)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", default=[40, 60, 100])
    ap.add_argument("--elig", type=float, nargs="+", default=[0.45, 0.6])
    ap.add_argument("--delta", type=float, nargs="+", default=[0, 12])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    table = []
    for n in a.scenes:
        for e in a.elig:
            for d in a.delta:
                hw, ex, rel = [], 0, 0
                for _ in range(a.reps):
                    r = one(rng, n, d, e, a.n_boot)
                    lo, hi = r["ci95"]
                    if not math.isnan(lo):
                        hw.append((hi - lo) / 2)
                        ex += int(lo > 0 or hi < 0)
                    rel += int(r["reliable"])
                table.append(dict(binding_scenes=n, eligibility=e, true_delta_steps=d, mean_halfwidth=float(np.mean(hw)) if hw else None,
                                  share_ci_excludes_0=ex / a.reps, share_reliable=rel / a.reps, passes_gate_halfwidth_le_4=bool(hw and np.mean(hw) <= 4)))
                print(table[-1], flush=True)
    out = {"assumptions": "synthetic hard persistence (tau_presence=12), scene p~U(0.3,0.7), k=4 seeds, 2 base seeds, M=2, independent eligibility; NOT data",
           "reps": a.reps, "n_boot": a.n_boot, "rows": table}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
