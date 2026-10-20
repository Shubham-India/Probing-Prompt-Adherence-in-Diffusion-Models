"""Robustness and sensitivity analyses of the PARTIAL TEST rows (EXPLORATORY; not part of the registered analysis plan).
Reuses the frozen functions of src/r3_analyze.py, src/commitment.py and src/dedup.py on the saved judged rows; no image is generated or re-judged.
  1. HD dose: t50(presence) at each guidance level with shared scene-bootstrap CIs, mean chance agreement A0 per level, C(t) and raw agreement curves.
  2. HD judge-error sensitivity: under symmetric, non-differential misclassification at rate e, observed agreement is A_obs = k^2 A + c with
     k = 1 - 2e and c = 2e(1 - e), for A and A0 alike, so the error-corrected persistence is (A_obs - A0_obs) / (1 - c - A0_obs). The HD contrast is
     recomputed with that denominator for e in {0, 0.05, 0.10, 1 - presence audit accuracy}.
  3. HD by TEST shard and by scene category (heterogeneity).
  4. CFG dose-response of constraint satisfaction on canonical prompts (w in {2, 5, 7.5, 10, 15}, 20 and 40 steps).
  5. HE per-arm contrasts against A1 on F, on presence only, and on DINO diversity; mean applied guidance per arm.
  6. Cross-hardware reproducibility of the judge (optional second eval file).
Usage: python partial/robustness.py --final <folder with eval_rows.jsonl, dino_embeddings.npz> [--records all_records.jsonl] [--compare eval_rows.jsonl]
Writes reports/partial/robustness.json, paper/tables/partial_robust_*.tex, paper/tables/partial_robust_numbers.tex, paper/figures/partial_{hd_dose,cfg_rates,he_arms}.pdf"""
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

TAB, FIG, REP = ROOT / "paper" / "tables", ROOT / "paper" / "figures", ROOT / "reports" / "partial"
LEVELS = (2, 7.5, 15)
NB = 2000
FAMS = RA.FAMS
FAM_TEX = {"objects_present": "presence", "count_ok": "count", "bind_given_pres": "binding$\\mid$pres.", "spatial_given_pres": "spatial$\\mid$pres.",
           "size_given_pres": "size$\\mid$pres.", "all_satisfied": "all satisfied"}


def rng():
    return np.random.default_rng(20261020)


def pct(d, lo=2.5, hi=97.5):
    d = np.asarray([x for x in d if math.isfinite(x)])
    return (float(np.percentile(d, lo)), float(np.percentile(d, hi))) if len(d) else (math.nan, math.nan)


def t50(num, den):
    return RA.t50_of(num, den)[0]


def corrected(num, den, cnt, e):
    """Error-corrected arrays for the pooled estimator C = sum(A - A0) / sum(1 - A0): the denominator of every observed cell becomes 1 - c - A0_obs
    (c = 2e(1 - e)); numerators are unchanged. e = 0 reproduces the registered estimator exactly."""
    c = 2 * e * (1 - e)
    return num, np.where(cnt > 0, den - c, 0.0)


def hd_block(core, br, scene_filter=None, errors=(0.0,), kinds=("presence",)):
    per = {}
    for w in LEVELS:
        cr = [r for r in core if float(r["cfg"]) == w and r["steps"] == 40 and (scene_filter is None or r["scene_id"] in scene_filter)]
        brr = [r for r in br if float(r["cfg"]) == w and r.get("restart") == "D8" and (scene_filter is None or r["scene_id"] in scene_filter)]
        if not cr or not brr:
            return None
        per[w] = RA.prepare(cr, brr, "D8")
    scenes = sorted(set().union(*[set(a.scene) | set(b.scene) for a, b in per.values()]))
    arr = {w: {k: C._arrays(per[w][0], per[w][1], k, scenes, RA.GRID) for k in kinds} for w in per}
    g = rng()
    idx = [g.integers(0, len(scenes), len(scenes)) for _ in range(NB)]
    both = {s for s in scenes if all(any(r.scene == s for r in per[w][0][["scene"]].itertuples()) for w in (2, 15))}
    out = dict(n_scenes=len(scenes), n_informative=len(both), by_error={})
    for e in errors:
        res = {}
        for k in kinds:
            a = {w: corrected(*arr[w][k], e) for w in arr}
            est = t50(*a[15]) - t50(*a[2])
            draws = [t50(a[15][0][i], a[15][1][i]) - t50(a[2][0][i], a[2][1][i]) for i in idx]
            ci = pct(draws)
            fin = [x for x in draws if math.isfinite(x)]
            res[k] = dict(delta=est, ci95=ci, p=RA.boot_p_two_sided(draws), finite_share=len(fin) / len(draws),
                          t50_by_w={str(w): t50(*a[w]) for w in a},
                          t50_ci={str(w): pct([t50(a[w][0][i], a[w][1][i]) for i in idx]) for w in a})
        out["by_error"][str(e)] = res
    out["A0_mean"] = {str(w): {k: float(per[w][1][per[w][1].kind == k].A0.mean()) for k in kinds} for w in per}
    out["curves"] = {str(w): {k: [None if not math.isfinite(x) else float(x) for x in C.curve_from_arrays(*arr[w][k][:2])] for k in kinds} for w in arr}
    out["raw_agreement"] = {str(w): {k: [float(per[w][0][(per[w][0].kind == k) & (per[w][0].t == t)].A.mean()) for t in RA.GRID] for k in kinds} for w in per}
    return out


def scene_boot_mean(vals_by_scene, g):
    s = [v for v in vals_by_scene.values() if v]
    if not s:
        return math.nan, (math.nan, math.nan), 0
    m = np.array([np.mean(v) for v in s])
    d = [m[g.integers(0, len(m), len(m))].mean() for _ in range(NB)]
    return float(m.mean()), pct(d), len(m)


def cfg_rates(core):
    """Scene means on the scenes generated at every (w, steps) cell, so that all points share one scene set."""
    out = {}
    cells = defaultdict(set)
    for r in core:
        if r.get("variant") == "CAN":
            cells[(float(r["cfg"]), r["steps"])].add(r["scene_id"])
    common = set.intersection(*cells.values()) if cells else set()
    for steps in (20, 40):
        for w in (2, 5, 7.5, 10, 15):
            rs = [r for r in core if r.get("variant") == "CAN" and float(r["cfg"]) == w and r["steps"] == steps and r["scene_id"] in common]
            if not rs:
                continue
            row = {}
            for f in list(FAMS) + ["all_satisfied"]:
                by = defaultdict(list)
                for r in rs:
                    if r.get(f) is not None:
                        by[r["scene_id"]].append(float(r[f]))
                row[f] = scene_boot_mean(by, rng())
            out[f"{w}|{steps}"] = row
    return out


def he_arms(core, cag, emb):
    import diversity as DV
    a1 = [dict(r, arm="A1") for r in core if r.get("variant") == "CAN" and float(r["cfg"]) == 7.5 and r["steps"] == 40]
    rows = a1 + cag
    vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        f = RA.F_of(r)
        if f is not None:
            vals["F"][r["scene_id"]][r["arm"]].append(f)
        if r.get("objects_present") is not None:
            vals["presence"][r["scene_id"]][r["arm"]].append(float(r["objects_present"]))
    g = defaultdict(list)
    for r in rows:
        if r["job_id"] in emb:
            g[(r["scene_id"], r["arm"])].append(emb[r["job_id"]])
    div = defaultdict(dict)
    for (s, arm), v in g.items():
        if len(v) >= 2:
            div[s][arm] = DV.mean_pairwise_cosine_distance(np.asarray(v))
    guid = defaultdict(list)
    for r in cag:
        if r.get("mean_guidance") is not None:
            guid[r["arm"]].append(float(r["mean_guidance"]))
    def contrast(arm, ref, metric):
        xs = []
        for s in sorted(div if metric == "div" else vals[metric]):
            if metric == "div":
                x, y = div[s].get(arm), div[s].get(ref)
                if x is not None and y is not None:
                    xs.append(x - y)
            else:
                x, y = vals[metric][s].get(arm), vals[metric][s].get(ref)
                if x and y:
                    xs.append(np.mean(x) - np.mean(y))
        x = np.asarray(xs, float)
        gg = rng()
        return dict(estimate=float(x.mean()) if len(x) else math.nan, n=len(x),
                    ci95=pct([x[gg.integers(0, len(x), len(x))].mean() for _ in range(NB)]) if len(x) else (math.nan, math.nan))
    pairs = [(arm, "A1") for arm in sorted({r["arm"] for r in cag})] + [("A2", "A6"), ("A2", "A4")]
    out = {}
    for arm, ref in pairs:
        key = arm if ref == "A1" else f"{arm}-{ref}"
        out[key] = dict(mean_guidance=float(np.mean(guid[arm])) if guid[arm] else None, ref=ref,
                        mean_guidance_ref=float(np.mean(guid[ref])) if guid.get(ref) else 7.5, **{m: contrast(arm, ref, m) for m in ("F", "presence", "div")})
    return out


def reproducibility(rows_a, rows_b):
    b = {r["job_id"]: r for r in rows_b}
    fields = list(FAMS) + ["all_satisfied"]
    same = {f: [0, 0] for f in fields}
    clip = []
    for r in rows_a:
        o = b.get(r["job_id"])
        if o is None:
            continue
        for f in fields:
            if r.get(f) is not None or o.get(f) is not None:
                same[f][1] += 1
                same[f][0] += int(r.get(f) == o.get(f))
        if r.get("clip_score") is not None and o.get("clip_score") is not None:
            clip.append(abs(float(r["clip_score"]) - float(o["clip_score"])))
    rows_identical = sum(all(r.get(f) == b[r["job_id"]].get(f) for f in fields) for r in rows_a if r["job_id"] in b)
    return dict(matched=sum(1 for r in rows_a if r["job_id"] in b), rows_identical=rows_identical, by_field=same,
                clip_max_abs_diff=max(clip) if clip else None)


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def ci_s(ci, nd=2):
    return f"[{fmt(ci[0], nd)}, {fmt(ci[1], nd)}]"


def write_table(name, caption, label, head, rows, spec, note=None, fit=False):
    L = [f"% generated by partial/robustness.py", r"\begin{table}[t]\centering\small", rf"\caption{{{caption}}}\label{{{label}}}",
         (r"\resizebox{\linewidth}{!}{" if fit else "") + rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    L += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}" + ("}" if fit else "")] +([rf"\par\smallskip\footnotesize {note}"] if note else []) + [r"\end{table}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--records")
    ap.add_argument("--compare")
    a = ap.parse_args()
    d = Path(a.final)
    rows, dlog = D.dedup(RA.read(d / "eval_rows.jsonl"))
    core = [r for r in rows if r.get("kind", "core") == "core"]
    br = [r for r in rows if r.get("kind") == "branch"]
    cag = [r for r in rows if r.get("kind") == "cag"]
    z = np.load(d / "dino_embeddings.npz")
    emb = dict(zip([str(x) for x in z["ids"]], z["emb"]))
    au = json.loads((ROOT / "docs" / "evidence" / "audit_20261019" / "audit_verdicts.json").read_text())
    e_pres = round(1 - au["presence"]["weighted_accuracy"], 3)
    R = dict(status="EXPLORATORY robustness analyses of the PARTIAL TEST rows (not in the registered analysis plan)", source=str(d.name),
             n_rows=len(rows), dedup=dlog if isinstance(dlog, dict) else str(dlog), presence_error_rate=e_pres)

    R["hd"] = hd_block(core, br, errors=(0.0, 0.05, 0.10, e_pres), kinds=("presence", "bind_given_pres"))
    inf = {r["scene_id"] for r in br if float(r["cfg"]) == 2 and r.get("restart") == "D8"} & {r["scene_id"] for r in br if float(r["cfg"]) == 15 and r.get("restart") == "D8"}
    b = hd_block(core, br, scene_filter=inf)
    R["hd_informative"] = dict(n_scenes=b["n_scenes"], **b["by_error"]["0.0"]["presence"])
    shard_of = {}
    if a.records:
        for line in open(a.records, encoding="utf-8"):
            j = json.loads(line)
            if j.get("stage") == "TEST_CORE":
                shard_of[j["scene_id"]] = j.get("shard")
    R["hd_by_shard"] = {}
    for sh in sorted(set(shard_of.values())):
        b = hd_block(core, br, scene_filter={s for s, v in shard_of.items() if v == sh})
        if b:
            R["hd_by_shard"][sh] = dict(n_scenes=b["n_scenes"], **b["by_error"]["0.0"]["presence"])
    cat = {r["scene_id"]: r.get("category") for r in core}
    R["hd_by_category"] = {}
    for c in sorted({v for v in cat.values() if v}):
        b = hd_block(core, br, scene_filter={s for s, v in cat.items() if v == c})
        if b:
            R["hd_by_category"][c] = dict(n_scenes=b["n_scenes"], **b["by_error"]["0.0"]["presence"])
    R["cfg_rates"] = cfg_rates(core)
    R["he_arms"] = he_arms(core, cag, emb)
    if a.compare:
        R["reproducibility"] = reproducibility(rows, D.dedup(RA.read(Path(a.compare)))[0])
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "robustness.json").write_text(json.dumps(RA.clean(R), indent=1, default=str))

    # ---------------- tables
    hd, pres = R["hd"], R["hd"]["by_error"]
    rows_t = [[f"registered estimand ($e = 0$; resampling all {hd['n_scenes']} scenes)", hd["n_informative"], fmt(pres["0.0"]["presence"]["delta"]), ci_s(pres["0.0"]["presence"]["ci95"])],
              ["resampling only the scenes with $w = 2$ and $w = 15$ branches", R["hd_informative"]["n_scenes"], fmt(R["hd_informative"]["delta"]), ci_s(R["hd_informative"]["ci95"])]]
    for e in ("0.05", "0.1", str(e_pres)):
        r = pres[e]["presence"]
        rows_t.append([f"judge error $e = {float(e):.3f}$" + (" (audit)" if float(e) == e_pres else ""), hd["n_informative"], fmt(r["delta"]), ci_s(r["ci95"])])
    for sh, r in R["hd_by_shard"].items():
        rows_t.append([f"TEST\\_CORE shard {sh.split('/')[0]} only", r["n_scenes"], fmt(r["delta"]), ci_s(r["ci95"])])
    for c, r in R["hd_by_category"].items():
        rows_t.append([f"category: {c}", r["n_scenes"], fmt(r["delta"]), ci_s(r["ci95"])])
    write_table("partial_robust_hd", r"Robustness of the HD presence contrast $\Delta t_{50} = t_{50}(w{=}15) - t_{50}(w{=}2)$ (steps; scene-bootstrap 95\% CI). "
                r"Judge-error rows correct $C(t)$ for symmetric, non-differential misclassification at rate $e$ (Eq.~\ref{eq:err-corr}). Partial TEST, exploratory.",
                "tab:partial-robust-hd", ["Analysis", "scenes", r"$\Delta t_{50}$", "95\\% CI"], rows_t, "lrrr",
                note="Subgroup rows have few scenes; an interval that includes 0 there is a statement about power, not about the effect.")
    t_rows = []
    for w in LEVELS:
        k = str(w)
        t_rows.append([f"{w:g}", fmt(pres["0.0"]["presence"]["t50_by_w"][k], 1), ci_s(pres["0.0"]["presence"]["t50_ci"][k], 1), fmt(hd["A0_mean"][k]["presence"], 3),
                       fmt(pres["0.0"]["bind_given_pres"]["t50_by_w"][k], 1)])
    write_table("partial_t50_by_w", r"Commitment time by guidance scale on canonical prompts (restart D8, 40 steps; partial TEST, exploratory). $\bar A_0$: mean chance agreement of independent seeds.",
                "tab:partial-t50-w", ["$w$", r"$t_{50}$ presence", "95\\% CI", r"$\bar A_0$ presence", r"$t_{50}$ binding$\mid$pres."], t_rows, "rrrrr",
                note=r"``--'' marks a curve that never reaches $C = 0.5$ on the grid (right-censored, $t_{50} > 36$).")
    arm_name = {"A2": "A2 (CAG) $-$ A1", "A3": "A3 (global $s^\star$) $-$ A1", "A4": "A4 (reversed) $-$ A1", "A6": "A6 (constant at A2 mean) $-$ A1",
                "A2-A6": "A2 $-$ A6 (timing at equal dose)", "A2-A4": "A2 $-$ A4 (registered timing contrast)"}
    a_rows = []
    for arm, r in R["he_arms"].items():
        a_rows.append([arm_name.get(arm, arm), f"{fmt(r['mean_guidance'])} vs {fmt(r['mean_guidance_ref'])}", r["F"]["n"], f"{fmt(r['F']['estimate'], 3)} {ci_s(r['F']['ci95'], 3)}",
                       f"{fmt(r['presence']['estimate'], 3)} {ci_s(r['presence']['ci95'], 3)}", f"{fmt(r['div']['estimate'], 3)} {ci_s(r['div']['ci95'], 3)}"])
    write_table("partial_he_arms", r"Guidance-schedule contrasts paired by scene (difference, 95\% CI); A1 is constant $w = 7.5$. $\bar w$: mean applied guidance of the two arms. "
                r"Presence is the only judged family that met the audit criterion. Partial TEST, exploratory.",
                "tab:partial-he-arms", ["Contrast", r"$\bar w$", "scenes", r"$\Delta F$", r"$\Delta$ presence", r"$\Delta$ diversity"], a_rows, "lrrlll", fit=True)

    # ---------------- DEV vs partial TEST commitment times (D8 and ER30, w = 7.5)
    dev = json.loads((ROOT / "docs" / "evidence" / "dev_analysis_20261020" / "dev_constants_final.json").read_text())["ha_dev"]["variants"]
    tst = json.loads((d / "partial_results.json").read_text())["results"]["HA"]["variants"]
    lab = {"presence": "presence", "count": "count", "bind_given_pres": r"binding$\mid$presence", "spatial_given_pres": r"spatial$\mid$presence",
           "size_given_pres": r"size$\mid$presence", "joint_bind": "joint binding"}

    def tv(x):
        if not x:
            return "--"
        v, st = x
        v = math.inf if v in (None, "inf") else float(v)
        return r"$>36$" if not math.isfinite(v) else (rf"$\le{v:.0f}$" if st == "left_censored" else f"{v:.1f}")
    x_rows = [[lab[k]] + [tv(src[v]["t50"].get(k)) for v in ("D8", "ER30") for src in (dev, tst)] for k in lab]
    x_rows.append(["scenes"] + [str(src[v]["n_scenes"]) for v in ("D8", "ER30") for src in (dev, tst)])
    write_table("partial_dev_vs_test", r"Commitment time $t_{50}$ (steps of 40, $w = 7.5$) on DEV (the values that set $s^\star$) and on the partial TEST split. "
                r"$\le6$: left-censored at the first grid step. Only presence has a validated judge. Exploratory.",
                "tab:dev-vs-test", ["Type", "DEV D8", "TEST D8", "DEV ER30", "TEST ER30"], x_rows, "lrrrr",
                note=r"DEV sets the frozen switch steps $s^\star = 12$ (presence, binding, spatial prompts) and $s^\star = 21$ (count prompts).")
    R["dev_vs_test"] = {v: {k: dict(dev=dev[v]["t50"].get(k), test=tst[v]["t50"].get(k)) for k in lab} for v in ("D8", "ER30")}
    (REP / "robustness.json").write_text(json.dumps(RA.clean(R), indent=1, default=str))

    # ---------------- macros
    def mac(name, val):
        return rf"\newcommand{{\{name}}}{{{val}}}"
    p0, pe = pres["0.0"]["presence"], pres[str(e_pres)]["presence"]
    he2 = R["he_arms"].get("A2", {})
    M = ["% generated by partial/robustness.py from reports/partial/robustness.json",
         mac("rbErrPres", f"{e_pres:.3f}"), mac("rbHdErr", fmt(pe["delta"])), mac("rbHdErrLo", fmt(pe["ci95"][0])), mac("rbHdErrHi", fmt(pe["ci95"][1])),
         mac("rbTwoPres", fmt(p0["t50_by_w"]["2"], 1)), mac("rbSevenPres", fmt(p0["t50_by_w"]["7.5"], 1)), mac("rbFifteenPres", fmt(p0["t50_by_w"]["15"], 1)),
         mac("rbAzeroTwo", fmt(hd["A0_mean"]["2"]["presence"], 3)), mac("rbAzeroFifteen", fmt(hd["A0_mean"]["15"]["presence"], 3)),
         mac("rbHdScenes", hd["n_informative"]), mac("rbHdUnion", hd["n_scenes"]),
         mac("rbHdInf", fmt(R["hd_informative"]["delta"])), mac("rbHdInfLo", fmt(R["hd_informative"]["ci95"][0])), mac("rbHdInfHi", fmt(R["hd_informative"]["ci95"][1])),
         mac("rbHdFive", fmt(pres["0.05"]["presence"]["delta"])), mac("rbHdFiveLo", fmt(pres["0.05"]["presence"]["ci95"][0])), mac("rbHdFiveHi", fmt(pres["0.05"]["presence"]["ci95"][1])),
         mac("rbHdTen", fmt(pres["0.1"]["presence"]["delta"])), mac("rbHdTenLo", fmt(pres["0.1"]["presence"]["ci95"][0])), mac("rbHdTenHi", fmt(pres["0.1"]["presence"]["ci95"][1])),
         mac("rbDivAsix", fmt(R["he_arms"]["A6"]["div"]["estimate"], 3)), mac("rbDivTwoSix", fmt(R["he_arms"]["A2-A6"]["div"]["estimate"], 3)),
         mac("rbDivTwoSixLo", fmt(R["he_arms"]["A2-A6"]["div"]["ci95"][0], 3)), mac("rbDivTwoSixHi", fmt(R["he_arms"]["A2-A6"]["div"]["ci95"][1], 3)),
         mac("rbFTwoSix", fmt(R["he_arms"]["A2-A6"]["F"]["estimate"], 3)), mac("rbFTwoSixLo", fmt(R["he_arms"]["A2-A6"]["F"]["ci95"][0], 3)), mac("rbFTwoSixHi", fmt(R["he_arms"]["A2-A6"]["F"]["ci95"][1], 3)),
         mac("rbMeanWAtwo", fmt(R["he_arms"]["A2"]["mean_guidance"])),
         mac("rbRawTwo", fmt(hd["raw_agreement"]["2"]["presence"][2], 2)), mac("rbRawFifteen", fmt(hd["raw_agreement"]["15"]["presence"][2], 2)),
         mac("rbHdCount", ("+" if R["hd_by_category"].get("counting", {}).get("delta", 0) > 0 else "") + fmt(R["hd_by_category"].get("counting", {}).get("delta"), 1)),
         mac("rbHdCountN", R["hd_by_category"].get("counting", {}).get("n_scenes", "--")),
         mac("rbDevJoint", fmt(dev["D8"]["t50"]["joint_bind"][0], 1)),
         mac("rbTestPres", fmt(tst["D8"]["t50"]["presence"][0], 1)), mac("rbTestBind", fmt(tst["D8"]["t50"]["bind_given_pres"][0], 1)),
         mac("rbTestCount", fmt(tst["D8"]["t50"]["count"][0], 1)), mac("rbTestJoint", fmt(tst["D8"]["t50"]["joint_bind"][0], 1)),
         mac("rbDevPres", fmt(dev["D8"]["t50"]["presence"][0], 1)), mac("rbDevCount", fmt(dev["D8"]["t50"]["count"][0], 1)),
         mac("rbDivAtwoFour", fmt(R["he_arms"]["A2-A4"]["div"]["estimate"], 3)),
         mac("rbDivAfour", fmt(R["he_arms"]["A4"]["div"]["estimate"], 3)), mac("rbFAthree", fmt(R["he_arms"]["A3"]["F"]["estimate"], 3)),
         mac("rbFAthreeLo", fmt(R["he_arms"]["A3"]["F"]["ci95"][0], 3)), mac("rbFAthreeHi", fmt(R["he_arms"]["A3"]["F"]["ci95"][1], 3)),
         mac("rbDivAtwo", fmt(he2.get("div", {}).get("estimate"), 3)), mac("rbPresAtwo", fmt(he2.get("presence", {}).get("estimate"), 3)),
         mac("rbPresAtwoLo", fmt(he2.get("presence", {}).get("ci95", [math.nan])[0], 3)), mac("rbPresAtwoHi", fmt(he2.get("presence", {}).get("ci95", [0, math.nan])[1], 3))]
    cr = R["cfg_rates"]
    for w, nm in ((2, "Two"), (7.5, "Seven"), (15, "Fifteen")):
        r = cr.get(f"{w}|40", {})
        M += [mac(f"rbRatePres{nm}", fmt(r.get("objects_present", (math.nan,))[0], 2)), mac(f"rbRateAll{nm}", fmt(r.get("all_satisfied", (math.nan,))[0], 2))]
    if "reproducibility" in R:
        rp = R["reproducibility"]
        M += [mac("rbRepMatched", f"{rp['matched']:,}"), mac("rbRepIdentical", f"{rp['rows_identical']:,}"),
              mac("rbRepShare", f"{100 * rp['rows_identical'] / max(1, rp['matched']):.2f}")]
    PR = json.loads((d / "partial_results.json").read_text())["results"]
    hdp = PR["HD"]["estimands"]["presence"]
    M += [mac("rbHdReg", fmt(hdp["delta"])), mac("rbHdRegLo", fmt(hdp["ci95"][0])), mac("rbHdRegHi", fmt(hdp["ci95"][1])),
          mac("rbHdHolm", fmt(PR["HD"]["p_holm"], 3)), mac("rbHfConc", fmt(PR["HF"]["concordance"])),
          mac("rbHfLo", fmt(PR["HF"]["ci95"][0])), mac("rbHfHi", fmt(PR["HF"]["ci95"][1])),
          mac("rbHeNi", fmt(PR["HE"]["non_inferiority"]["estimate"], 3)), mac("rbHeNiLo", fmt(PR["HE"]["non_inferiority"]["ci90"][0], 3)),
          mac("rbHeNiHi", fmt(PR["HE"]["non_inferiority"]["ci90"][1], 3)), mac("rbHaD", fmt(PR["HA"]["variants"]["D8"]["delta_t50"]["estimate"])),
          mac("rbHaDLo", fmt(PR["HA"]["variants"]["D8"]["delta_t50"]["ci95"][0])), mac("rbHaDHi", fmt(PR["HA"]["variants"]["D8"]["delta_t50"]["ci95"][1]))]
    (TAB / "partial_robust_numbers.tex").write_text("\n".join(M) + "\n", encoding="utf-8")

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for w in LEVELS:
        cv = hd["curves"][str(w)]["presence"]
        ax[0].plot(RA.GRID[:len(cv)], [math.nan if x is None else x for x in cv], marker="o", ms=3, label=f"w = {w:g}")
    ax[0].axhline(0.5, color="grey", ls="--", lw=0.8); ax[0].set_xlabel("restart step t (of 40)"); ax[0].set_ylabel("persistence C(t), presence"); ax[0].legend(fontsize=7)
    ax[0].set_title("(a) presence commitment curves", fontsize=8)
    for i, e in enumerate(("0.0", "0.05", "0.1", str(e_pres))):
        r = pres[e]["presence"]
        ax[1].errorbar([i], [r["delta"]], yerr=[[r["delta"] - r["ci95"][0]], [r["ci95"][1] - r["delta"]]], fmt="o", capsize=3, color="C0")
    ax[1].axhline(0, color="grey", lw=0.8); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["e=0", "0.05", "0.10", f"{e_pres:.3f}\n(audit)"])
    ax[1].set_ylabel(r"$\Delta t_{50}$ (w=15 $-$ w=2), steps"); ax[1].set_title("(b) HD under judge error", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "partial_hd_dose.pdf"); fig.savefig(REP / "partial_hd_dose.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.3), sharex=True)
    for ax, f in zip(axs, ("objects_present", "bind_given_pres", "all_satisfied")):
        for steps, ls in ((40, "-"), (20, "--")):
            ws = [w for w in (2, 5, 7.5, 10, 15) if f"{w}|{steps}" in cr]
            m = [cr[f"{w}|{steps}"][f][0] for w in ws]
            lo = [cr[f"{w}|{steps}"][f][1][0] for w in ws]
            hi = [cr[f"{w}|{steps}"][f][1][1] for w in ws]
            ax.plot(ws, m, ls, marker="o", ms=3, label=f"{steps} steps"); ax.fill_between(ws, lo, hi, alpha=0.15)
        ax.set_title(FAM_TEX[f].replace("$\\mid$", "|"), fontsize=8); ax.set_xlabel("guidance w")
    axs[0].set_ylabel("rate (scene mean)"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "partial_cfg_rates.pdf"); fig.savefig(REP / "partial_cfg_rates.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=True)
    arms = list(R["he_arms"])
    for ax, (metric, title) in zip(axs, (("F", "composite F"), ("presence", "presence"), ("div", "DINO diversity"))):
        for i, arm in enumerate(arms):
            r = R["he_arms"][arm][metric]
            if math.isfinite(r["estimate"]):
                ax.errorbar([r["estimate"]], [i], xerr=[[r["estimate"] - r["ci95"][0]], [r["ci95"][1] - r["estimate"]]], fmt="o", capsize=3, color="C0" if arm == "A2" else "C7")
        ax.axvline(0, color="grey", lw=0.8)
        if metric == "F":
            ax.axvline(-0.03, color="C3", ls=":", lw=0.8)
        ax.set_title(title, fontsize=8)
    axs[0].set_yticks(range(len(arms))); axs[0].set_yticklabels([a.replace("-", " - ") if "-" in a else f"{a} - A1" for a in arms])
    fig.tight_layout(); fig.savefig(FIG / "partial_he_arms.pdf"); fig.savefig(REP / "partial_he_arms.png", dpi=150); plt.close(fig)
    print(json.dumps({"hd": {e: {k: (round(v["delta"], 2), [round(x, 2) for x in v["ci95"]]) for k, v in r.items()} for e, r in pres.items()},
                      "A0": hd["A0_mean"], "by_shard": {k: (round(v["delta"], 2), v["ci95"], v["n_scenes"]) for k, v in R["hd_by_shard"].items()},
                      "by_cat": {k: (round(v["delta"], 2), v["ci95"], v["n_scenes"]) for k, v in R["hd_by_category"].items()},
                      "he": R["he_arms"], "repro": R.get("reproducibility")}, indent=1, default=str))


if __name__ == "__main__":
    main()
