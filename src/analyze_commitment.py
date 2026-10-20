"""HA driver: restart-persistence curves and Delta-t50 from core + branch evaluation rows (jsonl with the D-04 conditional fields).

Inputs
  --core     eval rows of the canonical prompts (scene_id, seed, objects_present, bind_given_pres, joint_bind, ...)
  --branches eval rows of branch images: scene_id, seed, t, m, variant ('D8' fixed jump, 'ER30' equal ratio, 'D4', 'D12', 'FULL'),
             requested_jump, effective_jump, ratio, clipped + the same outcome fields
Outputs (in --out): curves.csv, eligibility.csv, restart_log.csv (requested/effective jump, ratio, clipped per grid step), ha.json
Rules applied: D-02 (clipped t=6 reported separately; equal-ratio sign agreement required for support), D-03 (restart persistence wording),
D-04 (conditional primary + joint companion must agree in sign), D-06 (TEST scenes need a frozen config), D-12 (censoring => inconclusive).
Everything is labelled EXPLORATORY unless --frozen-config points to a config that passes config_schema.check_frozen."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import commitment as C
import config_schema

KINDS = {"presence": ("objects_present", None), "bind_given_pres": ("bind_given_pres", "objects_present"), "joint_bind": ("joint_bind", None),
         "spatial_given_pres": ("spatial_given_pres", "objects_present"), "joint_spatial": ("joint_spatial", None)}


def to_long(rows, with_branch):
    out = []
    for r in rows:
        for kind, (col, elig_col) in KINDS.items():
            v = r.get(col)
            if v is None:
                continue
            elig = 1 if elig_col is None else int(r.get(elig_col, 0))
            rec = dict(scene=r["scene_id"], kind=kind, seed=r["seed"], y=int(v), elig=elig)
            if with_branch:
                rec.update(t=r["t"], m=r["m"], variant=r["variant"])
            out.append(rec)
    return pd.DataFrame(out)


def attach_base(branch_long, core_long):
    base = core_long.rename(columns={"y": "base_y", "elig": "base_elig"})[["scene", "kind", "seed", "base_y", "base_elig"]]
    return branch_long.merge(base, on=["scene", "kind", "seed"], how="inner")


def analyse(core_rows, branch_rows, variant="D8", n_boot=4000):
    core = to_long(core_rows, False)
    br = attach_base(to_long(branch_rows, True), core)
    res = {}
    elig_rows = []
    a0 = C.chance_agreement(core)
    for v in sorted(br.variant.unique()):
        b = br[br.variant == v]
        adf = C.agreement(b.drop(columns="variant")).assign(variant=v)
        for (kind, t), g in b.groupby(["kind", "t"]):
            elig_rows.append(dict(variant=v, kind=kind, t=t, n_pairs=len(g), n_eligible_pairs=int(((g.elig == 1) & (g.base_elig == 1)).sum()),
                                  eligible_fraction=float(((g.elig == 1) & (g.base_elig == 1)).mean())))
        kinds = [k for k in KINDS if k in set(b.kind)]
        cur, scenes = C.curves(adf, a0, kinds, variant=v)
        res[v] = dict(curves={k: [None if np.isnan(x) else float(x) for x in cur[k]["C"]] for k in kinds},
                      t50={k: (cur[k]["t50"] if math.isfinite(cur[k]["t50"]) else ">36", cur[k]["t50_status"]) for k in kinds},
                      nonmonotone={k: cur[k]["nonmonotone"] for k in kinds}, n_scenes={k: cur[k]["n_scenes"].tolist() for k in kinds})
        pairs = [("presence", "bind_given_pres"), ("presence", "joint_bind")]
        res[v]["delta_t50"] = {f"{kb}-{ka}": C.bootstrap_delta_t50(adf, a0, ka, kb, variant=v, n_boot=n_boot, key="HA") for ka, kb in pairs if ka in kinds and kb in kinds}
    out = dict(variants=res)
    if variant in res and "presence-bind_given_pres" not in res[variant]["delta_t50"] and "bind_given_pres-presence" in res[variant]["delta_t50"]:
        prim, comp = res[variant]["delta_t50"]["bind_given_pres-presence"], res[variant]["delta_t50"].get("joint_bind-presence")
        decision = C.classify_HA(prim)
        notes = []
        if comp is not None and not (np.sign(prim["delta_t50"]) == np.sign(comp["delta_t50"])):
            decision, notes = "inconclusive (conditional and joint disagree in sign, D-04)", notes + ["D-04 companion disagreement"]
        er = res.get("ER30", {}).get("delta_t50", {}).get("bind_given_pres-presence")
        if decision == "support":
            if er is None:
                decision, notes = "support NOT QUALIFIED (no equal-ratio sensitivity run, D-02)", notes + ["missing ER30"]
            elif not C.sign_agreement(prim, er):
                decision, notes = "inconclusive (sign differs under equal-ratio restart, D-02)", notes + ["ER30 sign disagreement"]
        out["HA"] = dict(primary=prim, companion_joint=comp, equal_ratio=er, decision=decision, notes=notes)
    return out, pd.DataFrame(elig_rows)


def restart_log(branch_rows):
    df = pd.DataFrame(branch_rows)
    cols = [c for c in ("variant", "t", "requested_jump", "effective_jump", "ratio", "clipped") if c in df]
    return df[cols].drop_duplicates().sort_values(cols[:2]) if cols else pd.DataFrame()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--branches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen-config", default=None)
    ap.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args()
    core = [json.loads(l) for l in Path(a.core).read_text().splitlines()]
    br = [json.loads(l) for l in Path(a.branches).read_text().splitlines()]
    prompts_test = any(str(r.get("split", "")).upper() == "TEST" for r in core + br)
    if a.frozen_config:
        config_schema.check_frozen(json.loads(Path(a.frozen_config).read_text()))
        status = "CONFIRMATORY (freeze verified)"
    else:
        if prompts_test:
            sys.exit("TEST rows present but no verified frozen config (D-06)")
        status = "EXPLORATORY (no freeze verification)"
    out, elig = analyse(core, br, n_boot=a.n_boot)
    out["status"] = status
    Path(a.out).mkdir(parents=True, exist_ok=True)
    elig.to_csv(Path(a.out) / "eligibility.csv", index=False)
    restart_log(br).to_csv(Path(a.out) / "restart_log.csv", index=False)
    (Path(a.out) / "ha.json").write_text(json.dumps(out, indent=1, default=float))
    print(status, json.dumps(out.get("HA", {}).get("decision")))
