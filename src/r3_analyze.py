"""Revision 3 analyses from judged rows (src/r3_evaluate.py output; R2a/legacy core rows are accepted when they carry the same fields).
  dev        : DEV constants (t50 per outcome kind at w = 7.5, restart D8) + DEV gates            -> dev_constants.json (input to the freeze)
  confirm    : HA, HD, HE, HF with the pre-specified rules + Holm over the four                   -> confirmatory.json  (refuses TEST rows unless
               protocol/protocol_freeze.json verifies)
All intervals are scene-level bootstrap intervals; one resample of scenes is shared across CFG levels (HD), across arms (HE) and across kinds.
Numbers in reports must come from these JSON files (src/build_draft.py), never be typed by hand.
Usage: python src/r3_analyze.py dev --rows eval_dev.jsonl [--rows2 r2a_core_eval.jsonl] --out dev_constants.json
       python src/r3_analyze.py confirm --rows eval_test.jsonl --attention-dir DIR [--embeddings emb.npz] --out confirmatory.json [--protocol FILE]"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import commitment as C   # noqa: E402

KINDS = {"presence": ("objects_present", None), "count": ("count_ok", None), "bind_given_pres": ("bind_given_pres", "objects_present"),
         "spatial_given_pres": ("spatial_given_pres", "objects_present"), "size_given_pres": ("size_given_pres", "objects_present"),
         "joint_bind": ("joint_bind", None)}
PLAN = json.loads((ROOT / "configs" / "analysis_plan_r3.json").read_text())
GRID = C.GRID


def long(rows, branch):
    out = []
    for r in rows:
        for kind, (col, ec) in KINDS.items():
            v = r.get(col)
            if v is None:
                continue
            rec = dict(scene=r["scene_id"], kind=kind, seed=r["seed"], y=int(v), elig=1 if ec is None else int(r.get(ec) or 0),
                       base=r["base_job"] if branch else (r.get("job_id") or r["name"]))
            if branch:
                rec.update(t=r["t"], m=r["m"], variant=r.get("restart") or r.get("variant"))
            out.append(rec)
    return pd.DataFrame(out)


def attach(br, core_rows):
    base = long(core_rows, False).rename(columns={"y": "base_y", "elig": "base_elig"})[["base", "kind", "base_y", "base_elig"]]
    return br.merge(base, on=["base", "kind"], how="inner")


def base_key(r):
    return (r.get("prompt_id"), float(r["cfg"]), int(r["steps"]))


def prepare(core_rows, branch_rows, variant):
    """Chance agreement A0 uses only the independent seeds of the branches' own BASE prompts at the base CFG and steps (Addendum C: "A0 the
    agreement of independent seeds"; commitment.chance_agreement: "pairs of independent seeds on the canonical prompt"). Fix 2026-10-20 (DL-40):
    every core row of a scene was pooled before (other variants, CFG levels and step counts), which lowered A0 and biased C upwards."""
    sel = [r for r in branch_rows if (r.get("restart") or r.get("variant")) == variant]
    bases = {r["base_job"] for r in sel}
    keys = {base_key(r) for r in core_rows if (r.get("job_id") or r.get("name")) in bases}
    core = long([r for r in core_rows if base_key(r) in keys], False).drop(columns=["base"])
    br = attach(long(sel, True), core_rows)
    a0 = C.chance_agreement(core)
    adf = C.agreement(br.drop(columns=["variant", "base"]))
    return adf, a0


def arrays_for(adf, a0, kinds, scenes):
    return {k: C._arrays(adf, a0, k, scenes, GRID)[:2] for k in kinds}


def t50_of(num, den):
    v, st = C.t50(C.curve_from_arrays(num, den), GRID)
    return v, st


def boot_p_two_sided(draws, null=0.0):
    d = np.asarray([x for x in draws if math.isfinite(x)])
    if len(d) == 0:
        return float("nan")
    return float(min(1.0, 2 * min((d >= null).mean(), (d <= null).mean())))


def classify(d, ci, practical, bound, sign=None):
    lo, hi = ci
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "inconclusive (interval not estimable)"
    if bound and -bound < lo and hi < bound:
        return "bounded null"
    if lo > 0 or hi < 0:
        if sign is not None and np.sign(d) != sign:
            return "contradicted"
        return "supported" if abs(d) >= practical else "inconclusive (below the practical threshold)"
    return "inconclusive"


# ---------------------------------------------------------------- HA
def analyse_ha(core_rows, branch_rows, n_boot, key="HA"):
    res = {}
    for variant in ("D8", "ER30"):
        if not any((r.get("restart") or r.get("variant")) == variant for r in branch_rows):
            continue
        adf, a0 = prepare(core_rows, branch_rows, variant)
        scenes = sorted(set(adf.scene) | set(a0.scene))
        kinds = [k for k in KINDS if k in set(adf.kind)]
        arr = arrays_for(adf, a0, kinds, scenes)
        t50s = {k: t50_of(*arr[k]) for k in kinds}
        out = dict(t50={k: (v if math.isfinite(v) else ">36", st) for k, (v, st) in t50s.items()},
                   curves={k: [None if np.isnan(x) else float(x) for x in C.curve_from_arrays(*arr[k])] for k in kinds}, n_scenes=len(scenes))
        if "presence" in kinds and "bind_given_pres" in kinds:
            rng = C._rng(f"{key}|{variant}")
            draws, comp = [], []
            for _ in range(n_boot):
                idx = rng.integers(0, len(scenes), len(scenes))
                tp, _ = t50_of(arr["presence"][0][idx], arr["presence"][1][idx])
                tb, _ = t50_of(arr["bind_given_pres"][0][idx], arr["bind_given_pres"][1][idx])
                draws.append(tb - tp)
            fin = [x for x in draws if math.isfinite(x)]
            d = t50s["bind_given_pres"][0] - t50s["presence"][0]
            ci = (float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))) if fin else (math.nan, math.nan)
            out["delta_t50"] = dict(estimate=d, ci95=ci, p=boot_p_two_sided(draws), censored=len(draws) - len(fin), halfwidth=(ci[1] - ci[0]) / 2 if fin else None)
            if "joint_bind" in kinds:
                out["delta_joint"] = t50s["joint_bind"][0] - t50s["presence"][0]
        res[variant] = out
    p = PLAN["HA"]
    dec = "not estimable"
    if "D8" in res and "delta_t50" in res["D8"]:
        dd = res["D8"]["delta_t50"]
        dec = classify(dd["estimate"], dd["ci95"], p["practical_steps"], p["null_bound"])
        if dec == "supported":
            if not np.sign(res["D8"].get("delta_joint", dd["estimate"])) == np.sign(dd["estimate"]):
                dec = "inconclusive (joint companion disagrees in sign, D-04)"
            er = res.get("ER30", {}).get("delta_t50")
            if er is None or np.sign(er["estimate"]) != np.sign(dd["estimate"]):
                dec = "inconclusive (equal-ratio restart missing or disagrees in sign, D-02)"
    return dict(variants=res, decision=dec, p=res.get("D8", {}).get("delta_t50", {}).get("p"))


# ---------------------------------------------------------------- HD
def analyse_hd(core_rows, branch_rows, n_boot, levels=(2, 7.5, 15)):
    per = {}
    for w in levels:
        cr = [r for r in core_rows if float(r["cfg"]) == float(w) and r["steps"] == 40]
        brr = [r for r in branch_rows if float(r["cfg"]) == float(w) and (r.get("restart") or r.get("variant")) == "D8"]
        if cr and brr:
            per[w] = prepare(cr, brr, "D8")
    if not {levels[0], levels[-1]} <= set(per):
        return dict(decision="not estimable (missing CFG levels)", p=None)
    scenes = sorted(set().union(*[set(a.scene) | set(b.scene) for a, b in per.values()]))
    arr = {w: arrays_for(*per[w], ["presence", "bind_given_pres"], scenes) for w in per}
    out, ps = {}, []
    rng = C._rng("HD")
    idx_all = [rng.integers(0, len(scenes), len(scenes)) for _ in range(n_boot)]
    for k in ("presence", "bind_given_pres"):
        lo_w, hi_w = levels[0], levels[-1]
        est = t50_of(*arr[hi_w][k])[0] - t50_of(*arr[lo_w][k])[0]
        draws = [t50_of(arr[hi_w][k][0][i], arr[hi_w][k][1][i])[0] - t50_of(arr[lo_w][k][0][i], arr[lo_w][k][1][i])[0] for i in idx_all]
        fin = [x for x in draws if math.isfinite(x)]
        ci = (float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))) if fin else (math.nan, math.nan)
        p = boot_p_two_sided(draws)
        ps.append(p)
        out[k] = dict(t50_by_w={str(w): (lambda v: v if math.isfinite(v) else ">36")(t50_of(*arr[w][k])[0]) for w in arr}, delta=est, ci95=ci, p=p,
                      decision=classify(est, ci, PLAN["HD"]["practical_steps"], PLAN["HD"]["null_bound"], sign=PLAN["HD"]["predicted_sign"]))
    p_hd = min(1.0, 2 * min(x for x in ps if math.isfinite(x))) if any(math.isfinite(x) for x in ps) else None
    decs = [out[k]["decision"] for k in out]
    dec = "supported" if "supported" in decs else "contradicted" if "contradicted" in decs else "bounded null" if all(d == "bounded null" for d in decs) else "inconclusive"
    return dict(estimands=out, p=p_hd, decision=dec, note="A0 is computed separately at each w; raw agreement is reported by the curves module")


# ---------------------------------------------------------------- HE
FAMS = ("objects_present", "count_ok", "bind_given_pres", "spatial_given_pres", "size_given_pres")


def F_of(r):
    vals = [r.get(f) for f in FAMS if r.get(f) is not None]
    return float(np.mean(vals)) if vals else None


def analyse_he(core_rows, cag_rows, embeddings, n_boot):
    import diversity as DV
    a1 = [dict(r, arm="A1") for r in core_rows if r.get("variant") == "CAN" and float(r["cfg"]) == 7.5 and r["steps"] == 40]
    rows = a1 + cag_rows
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        f = F_of(r)
        if f is not None:
            by[r["scene_id"]][r["arm"]].append(f)
    emb = {}
    if embeddings is not None:
        ids, E = embeddings
        emb = dict(zip(ids, E))
    div = defaultdict(dict)
    for arm_rows in (a1, cag_rows):
        g = defaultdict(list)
        for r in arm_rows:
            if r["job_id"] in emb:
                g[(r["scene_id"], r["arm"])].append(emb[r["job_id"]])
        for (s, arm), v in g.items():
            div[s][arm] = DV.mean_pairwise_cosine_distance(np.asarray(v))
    scenes = sorted(by)
    def paired(arm_a, arm_b, metric):
        xs = []
        for s in scenes:
            src = by[s] if metric == "F" else div.get(s, {})
            a, b = src.get(arm_a), src.get(arm_b)
            if a is None or b is None or (metric == "F" and (not a or not b)):
                continue
            xs.append((np.mean(a) - np.mean(b)) if metric == "F" else (a - b))
        return np.asarray(xs, float)
    rng = C._rng("HE")
    def boot(x, level):
        if len(x) == 0:
            return math.nan, (math.nan, math.nan), []
        d = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
        a = (1 - level) / 2
        return float(x.mean()), (float(np.percentile(d, 100 * a)), float(np.percentile(d, 100 * (1 - a)))), d
    ni_est, ni_ci, ni_d = boot(paired("A2", "A1", "F"), PLAN["HE"]["non_inferiority"]["ci"])
    dv_est, dv_ci, dv_d = boot(paired("A2", "A1", "div"), PLAN["HE"]["diversity"]["ci"])
    tm_est, tm_ci, tm_d = boot(paired("A2", "A4", "F"), PLAN["HE"]["timing"]["ci"])
    margin = PLAN["HE"]["non_inferiority"]["margin"]
    p_ni = float(np.mean(np.asarray(ni_d) <= margin)) if ni_d else float("nan")
    p_dv = float(np.mean(np.asarray(dv_d) <= 0)) if dv_d else float("nan")
    p_tm = float(np.mean(np.asarray(tm_d) <= 0)) if tm_d else float("nan")
    conds = dict(non_inferior=bool(math.isfinite(ni_ci[0]) and ni_ci[0] > margin), more_diverse=bool(math.isfinite(dv_ci[0]) and dv_ci[0] > 0),
                 beats_reversed=bool(math.isfinite(tm_ci[0]) and tm_ci[0] > 0))
    ps = [p for p in (p_ni, p_dv, p_tm) if math.isfinite(p)]
    calls = defaultdict(list)
    for r in rows:
        if r.get("unet_calls_measured"):
            calls[r["arm"]].append(r["unet_calls_measured"])
    return dict(non_inferiority=dict(estimate=ni_est, ci90=ni_ci, margin=margin, p=p_ni), diversity=dict(estimate=dv_est, ci95=dv_ci, p=p_dv),
                timing=dict(estimate=tm_est, ci95=tm_ci, p=p_tm), conditions=conds, p=max(ps) if len(ps) == 3 else None,
                decision=("not estimable (" + ", ".join(k for k, e in (("non_inferior", ni_est), ("more_diverse", dv_est), ("beats_reversed", tm_est)) if not math.isfinite(e)) + " without paired data)")
                if not all(math.isfinite(e) for e in (ni_est, dv_est, tm_est))
                else "supported" if all(conds.values()) else "not supported (" + ", ".join(k for k, v in conds.items() if not v) + " failed)",
                secondary=dict(A3_minus_A1=boot(paired("A3", "A1", "F"), 0.95)[:2], A6_minus_A1=boot(paired("A6", "A1", "F"), 0.95)[:2],
                               mean_unet_calls={k: float(np.mean(v)) for k, v in calls.items()}), n_scenes=len(scenes))


# ---------------------------------------------------------------- HF
SIG_KIND = {"presence": "presence", "binding": "bind_given_pres", "size": "size_given_pres", "spatial": "spatial_given_pres", "count": "count"}


def t_stab_table(att_rows, prompts, res=16, mass_q=0.9):
    """att_rows: dicts with job_id, scene_id, prompt_id, seed, path (npz). Returns DataFrame scene, kind, seed, t_stab."""
    import attn_record as AR
    import attn_stability as AS
    out = []
    for r in att_rows:
        maps, words, meta = AR.load_npz(r["path"])
        if res not in maps:
            continue
        m = maps[res].astype(np.float64)
        by_word = {w: m[:, i] for i, w in enumerate(words)}
        for c in prompts[r["prompt_id"]]["constraints"]:
            sig = {"presence": "presence", "count": "count", "binding": "binding", "size": "size", "spatial": "spatial"}[c["type"]]
            try:
                v = AS.signal_velocity(sig, by_word, words, c, mass_q)
            except KeyError:
                continue
            ts = AS.t_stab_path(v)
            if ts is not None:
                out.append(dict(scene=r["scene_id"], kind=SIG_KIND[sig], seed=r["seed"], t_stab=ts))
    return pd.DataFrame(out)


def analyse_hf(core_rows, branch_rows, tstab, n_boot):
    import attn_stability as AS
    if tstab is None or tstab.empty:
        return dict(decision="not estimable (no attention signals at the primary resolution)", p=None)
    adf, a0 = prepare(core_rows, branch_rows, "D8")
    ts = tstab.groupby(["scene", "kind"]).t_stab.mean().reset_index()
    kinds = sorted(set(ts.kind) & set(adf.kind))
    scenes = sorted(set(adf.scene) | set(a0.scene) | set(ts.scene))
    arr = arrays_for(adf, a0, kinds, scenes)
    T = {k: np.array([ts[(ts.kind == k) & (ts.scene == s)].t_stab.mean() if ((ts.kind == k) & (ts.scene == s)).any() else np.nan for s in scenes]) for k in kinds}

    def type_means(idx):
        t50 = {k: t50_of(arr[k][0][idx], arr[k][1][idx])[0] for k in kinds}
        tsm = {k: np.nanmean(T[k][idx]) for k in kinds}
        ok = [k for k in kinds if math.isfinite(t50[k]) and math.isfinite(tsm[k])]
        return {k: tsm[k] for k in ok}, {k: t50[k] for k in ok}
    full = np.arange(len(scenes))
    conc, npairs = AS.concordance(*type_means(full))
    rng = C._rng("HF")
    draws = [AS.concordance(*type_means(rng.integers(0, len(scenes), len(scenes))))[0] for _ in range(n_boot)]
    d = np.asarray([x for x in draws if math.isfinite(x)])
    ci = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (math.nan, math.nan)
    p = float((d <= PLAN["HF"]["concordance_reject"]).mean()) if len(d) else None
    # scene level: per-scene area under the commitment curve vs per-scene t_stab
    sp = {}
    for k in kinds:
        num, den = arr[k]
        auc = np.where(den.sum(1) > 0, np.nanmean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), axis=1), np.nan)
        x, y = T[k], auc
        rho = AS.spearman(x, y)
        bs = []
        for _ in range(min(n_boot, 2000)):
            i = rng.integers(0, len(scenes), len(scenes))
            bs.append(AS.spearman(x[i], y[i]))
        bs = np.asarray([b for b in bs if math.isfinite(b)])
        sp[k] = (rho, float(np.percentile(bs, 2.5)) if len(bs) else math.nan, float(np.percentile(bs, 97.5)) if len(bs) else math.nan)
    return dict(concordance=conc, pairs=npairs, ci95=ci, p=p, spearman=sp, decision=AS.hf_decision(conc, ci, sp), n_scenes=len(scenes))


# ---------------------------------------------------------------- Holm
def holm(pvals, alpha=0.05):
    """Holm over the WHOLE registered family: a hypothesis without a p-value (not estimable) counts as p = 1, so m is always the family size
    (DL-48; previously m shrank to the number of available p-values)."""
    items = sorted(((p if p is not None and math.isfinite(p) else 1.0), k) for k, p in pvals.items())
    adj, prev = {}, 0.0
    m = len(items)
    for i, (p, k) in enumerate(items):
        prev = max(prev, min(1.0, (m - i) * p))
        adj[k] = prev
    return adj


def finalise(results):
    """Holm over the registered family, then the classifications fixed before TEST (analysis_plan_r3.json `pre_test_status`, Addendum C C5):
    an EXPLORATORY hypothesis is never reported SUPPORTED, and HD is supportable only through the estimands listed there."""
    fam = PLAN["multiplicity"]["family"]
    adj = holm({k: (results.get(k) or {}).get("p") for k in fam}, PLAN["multiplicity"]["alpha"])
    st = PLAN.get("pre_test_status", {})
    for k, v in results.items():
        v["p_holm"] = None if v.get("p") is None else adj.get(k)
        dec = v.get("decision", "")
        if dec == "supported" and (v["p_holm"] is None or v["p_holm"] >= PLAN["multiplicity"]["alpha"]):
            final = "inconclusive (rule passed but Holm-adjusted p >= 0.05)"
        else:
            final = dec
        sup_est = st.get("supportable_estimands", {}).get(k)
        if final == "supported" and sup_est is not None and isinstance(v.get("estimands"), dict):
            if not any((v["estimands"].get(e) or {}).get("decision") == "supported" for e in sup_est):
                final = f"EXPLORATORY: rule passed only through estimands outside {sup_est} (judge criterion not met)"
        if k in st.get("exploratory", {}):
            final = f"EXPLORATORY ({st['exploratory'][k]}); rule outcome: {final}"
        v["final"] = final
        v["pre_test_status"] = st.get("exploratory", {}).get(k) or st.get("disclosure", {}).get(k)
    return results


# ---------------------------------------------------------------- DEV constants and gates
def dev_constants(core_rows, branch_rows, n_boot):
    ha = analyse_ha(core_rows, [r for r in branch_rows if float(r["cfg"]) == 7.5], n_boot, key="HA-DEV")
    t50 = ha["variants"].get("D8", {}).get("t50", {})
    import cag as CAG
    t50_num = {k: v[0] for k, v in t50.items()}
    gates = {}
    dd = ha["variants"].get("D8", {}).get("delta_t50")
    gates["HA_dev_halfwidth_le_4"] = dict(value=dd and dd["halfwidth"], passed=bool(dd and dd["halfwidth"] is not None and dd["halfwidth"] <= PLAN["HA"]["dev_gate_halfwidth_steps"]))
    if any((r.get("restart") or r.get("variant")) == "FULL" for r in branch_rows):
        adf, a0 = prepare(core_rows, branch_rows, "FULL")
        scenes = sorted(set(adf.scene) | set(a0.scene))
        cs = [abs(x) for k in set(adf.kind) for x in C.curve_from_arrays(*C._arrays(adf, a0, k, scenes, GRID)[:2]) if math.isfinite(x)]
        gates["full_resample_control_max_abs_C_le_0.15"] = dict(value=max(cs) if cs else None, passed=bool(cs and max(cs) <= PLAN["HA"]["full_resample_control_max_C"]))
    for w in (2, 15):
        base = [r for r in core_rows if float(r["cfg"]) == w and r["variant"] == "CAN"]
        n = len({r["scene_id"] for r in base if r.get("objects_present") == 1})
        gates[f"HD_presence_gate_w{w:g}"] = dict(value=n, passed=n >= 40)
    gates["CAG_quality_guard"] = dict(value=None, passed=None, note="PENDING: needs a preference scorer (HPS v2) on DEV A2 w_lo 1 vs 3 vs A1; owner decides before the freeze")
    return dict(status="DEV-derived constants (EXPLORATORY until frozen)", constants={"t50_dev_w7.5": t50_num, "w_lo": 1.0,
                "s_star_global": CAG.s_star_global(t50_num, 2)}, gates=gates, ha_dev=ha)


# ---------------------------------------------------------------- S2 (supporting, exploratory): variant-level outcome rates
def analyse_s2(core_rows):
    """TEST variant-level outcome rates (CAN, PAR, ORD, CF-*) at CFG 7.5 / 40 steps, after the Q2 rules: generations counted once (dedup R1/R2, applied
    by main) and TEST rows whose prompt text also occurs in DEV excluded (dedup R3). Descriptive; outside the Holm family."""
    import dedup as D
    rows = [r for r in core_rows if r.get("split") == "TEST" and float(r["cfg"]) == 7.5 and int(r["steps"]) == 40]
    rows, excluded = D.exclude_dev_overlap(rows)
    by = defaultdict(lambda: defaultdict(list))
    scenes = defaultdict(set)
    for r in rows:
        scenes[r.get("variant")].add(r.get("scene_id"))
        for f in FAMS:
            if r.get(f) is not None:
                by[r.get("variant")][f].append(float(r[f]))
    rates = {v: dict(n_rows=sum(1 for r in rows if r.get("variant") == v), n_scenes=len(scenes[v]),
                     **{f: dict(mean=round(float(np.mean(x)), 4), n=len(x)) for f, x in fams.items()}) for v, fams in sorted(by.items())}
    return dict(status="SUPPORTING (S2, exploratory)", excluded_dev_overlap=len(excluded), excluded_job_ids=excluded[:200], rates=rates)


def clean(o):
    """JSON-safe: non-finite floats become strings ('inf', 'nan'); numpy scalars become Python numbers; tuples become lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "confirm"])
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attention-dir", default=None)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--n-boot", type=int, default=PLAN["bootstrap"]["draws"])
    ap.add_argument("--allow-fake", action="store_true")
    a = ap.parse_args()
    rows = [r for p in a.rows for r in read(p)]
    if not a.allow_fake and any(r.get("fake_pipeline") or r.get("fake_judge") for r in rows):
        sys.exit("fake rows present (dry-run output); refusing to analyse without --allow-fake")
    import dedup as D
    rows, dlog = D.dedup(rows)                 # Q2 (DL-33): one generation counts once; duplicate canonical scenes count once
    dedup_summary = dict(dropped_scenes=dlog["dropped_scenes"], n_dropped_generations=len(dlog["dropped_generations"]),
                         n_dropped_dependents=len(dlog["dropped_dependents"]), dropped_generations=dlog["dropped_generations"][:200])
    core = [r for r in rows if r.get("kind", "core") == "core"]
    branches = [r for r in rows if r.get("kind") == "branch"]
    if a.mode == "dev":
        if any(r.get("split") == "TEST" for r in rows):
            sys.exit("TEST rows are not allowed in the DEV analysis")
        out = dev_constants(core, branches, a.n_boot)
        out["dedup"] = dedup_summary
    else:
        if any(r.get("split") == "TEST" for r in rows):
            import protocol_r3
            probs = protocol_r3.verify(a.protocol) if a.protocol else ["no --protocol"]
            if probs:
                sys.exit("confirmatory analysis refused: " + "; ".join(probs[:5]))
        att = []
        if a.attention_dir:
            for r in core:
                p = Path(a.attention_dir) / f"{r['job_id']}.npz"
                if p.exists():
                    att.append(dict(r, path=str(p)))
        cfg = json.loads((ROOT / "configs" / "experiment_r3.json").read_text())
        prompts = {p["id"]: p for f in (cfg["benchmark"]["prompts"], cfg["benchmark"]["ps_prompts"]) for p in json.loads((ROOT / f).read_text())}
        emb = None
        if a.embeddings:
            z = np.load(a.embeddings)
            emb = ([str(x) for x in z["ids"]], z["emb"])
        res = {"HA": analyse_ha([r for r in core if float(r["cfg"]) == 7.5], [r for r in branches if float(r["cfg"]) == 7.5], a.n_boot),
               "HD": analyse_hd(core, branches, a.n_boot),
               "HE": analyse_he(core, [r for r in rows if r.get("kind") == "cag"], emb, a.n_boot),
               "HF": analyse_hf([r for r in core if float(r["cfg"]) == 7.5], [r for r in branches if float(r["cfg"]) == 7.5], t_stab_table(att, prompts), a.n_boot)
               if att else dict(decision="not estimable (no attention maps)", p=None)}
        fake = any(r.get("fake_pipeline") or r.get("fake_judge") for r in rows)
        status = ("DRY-RUN (fake pipeline/judge; not a result)" if fake
                  else "CONFIRMATORY" if any(r.get("split") == "TEST" for r in rows) else "EXPLORATORY (DEV rows)")
        out = dict(results=finalise(res), status=status, dedup=dedup_summary, supporting=dict(S2=analyse_s2(core)))
    Path(a.out).write_text(json.dumps(clean(out), indent=1))
    print(json.dumps({k: (v.get("final", v.get("decision")) if isinstance(v, dict) else v) for k, v in out.get("results", out.get("gates", {})).items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
