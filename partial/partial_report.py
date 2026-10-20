"""Report from the PARTIAL TEST analysis (partial/nb06_partial_test_analysis.py). Every number is read from partial_results.json; every statement is
labelled PARTIAL TEST / EXPLORATORY; missing shards are listed, never imputed. It does not touch the frozen protocol or the confirmatory pipeline.
Input: the output folder of notebook 06 (final_r3-test-partial-*, containing partial_results.json) or its archive folder (ARCHIVE_MANIFEST.json + parts).
Writes: reports/partial/{partial_report.md, partial_results.json, hypothesis_summary.csv}; paper/tables/partial_*.tex; paper/figures/partial_*.pdf;
paper/sections/results_partial.tex (inserted after the DEV section of paper/main.tex).
Usage: python partial/partial_report.py --final <folder>"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "kaggle")]
TABLES, FIGS, SECS, REP = ROOT / "paper" / "tables", ROOT / "paper" / "figures", ROOT / "paper" / "sections", ROOT / "reports" / "partial"
KINDS = {"presence": "presence", "count": "count", "bind_given_pres": r"binding $\mid$ presence", "spatial_given_pres": r"spatial $\mid$ presence",
         "size_given_pres": r"size $\mid$ presence", "joint_bind": "joint binding"}
LABEL = "PARTIAL TEST, EXPLORATORY"


def f(x, d=2):
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(v, d) for v in x) + "]"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return "--" if isinstance(x, float) and not math.isfinite(x) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def tex(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace(">", r"$>$").replace("<", r"$<$")


def table(name, caption, label, head, rows, spec):
    L = [r"\begin{table}[t]\centering\small", rf"\caption{{{caption}}}\label{{{label}}}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    L += [" & ".join(str(c) for c in r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / name).write_text("\n".join(L) + "\n", encoding="utf-8")


def locate(p):
    p = Path(p)
    if (p / "partial_results.json").exists():
        return p, None
    if (p / "ARCHIVE_MANIFEST.json").exists():
        import nbtools
        am = json.loads((p / "ARCHIVE_MANIFEST.json").read_text())
        if am.get("stage") != "PARTIAL_TEST" or am.get("verified") is not True or am.get("dry_run"):
            sys.exit(f"not a verified PARTIAL_TEST archive (stage={am.get('stage')}, verified={am.get('verified')}, dry_run={am.get('dry_run')})")
        dest, probs = nbtools.extract_and_verify(p, ROOT / "local_audit" / "partial_x" / am["run_id"])
        if probs:
            sys.exit(f"archive verification failed: {probs[:5]}")
        return dest, am
    sys.exit("give notebook 06's output folder or archive folder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    a = ap.parse_args()
    d, am = locate(a.final)
    P = json.loads((d / "partial_results.json").read_text())
    if not str(P.get("status", "")).startswith("PARTIAL TEST"):
        sys.exit(f"unexpected status {P.get('status')!r}")
    R, C = P["results"], P["coverage"]
    envf = d / "env_bootstrap.json"
    ENV = {}
    if envf.exists():
        v = (json.loads(envf.read_text()).get("versions") or {})
        ENV = dict(torch=v.get("torch"), gpus=v.get("gpu_count"))
    prov = (f"archive {am['run_id']}" if am else f"output folder {d.name}") + (f"; judged with torch {ENV.get('torch')} on {ENV.get('gpus')} GPU(s)" if ENV else "")
    for x in (TABLES, FIGS, SECS, REP):
        x.mkdir(parents=True, exist_ok=True)

    cov_rows = [[tex(st), f(C[st].get("present")), f(C[st].get("missing")), f"{C[st].get('merged')}/{C[st].get('manifest_jobs')}", C[st].get("scenes", "--")]
                for st in ("TEST_CORE", "TEST_COMMIT", "TEST_CAG_SIZE")]
    table("partial_coverage.tex", rf"TEST coverage of this {LABEL} analysis. Shards are disjoint scene groups; missing shards are not imputed.", "tab:partial-cov",
          ["Stage", "Shards present", "Missing", "Jobs merged", "Scenes"], cov_rows, "llllr")

    def est(h, r):
        if h == "HA":
            dt = r.get("variants", {}).get("D8", {}).get("delta_t50", {}) or {}
            return dt.get("estimate"), dt.get("ci95"), r"$\Delta t_{50}$ binding$-$presence (steps)"
        if h == "HD":
            e = (r.get("estimands", {}) or {}).get("presence", {}) or {}
            return e.get("delta"), e.get("ci95"), r"$\Delta t_{50}$ presence, $w$15$-$2 (steps)"
        if h == "HE":
            e = r.get("non_inferiority", {}) or {}
            return e.get("estimate"), e.get("ci90"), r"A2$-$A1 on $F$ (90\% CI; margin $-0.03$)"
        return r.get("concordance"), r.get("ci95"), "type-order concordance"
    hyp, csv_rows = [], []
    for h in ("HA", "HD", "HE", "HF"):
        r = R.get(h, {})
        e, ci, what = est(h, r)
        rule = str(r.get("final", "")).replace(f"{LABEL} (not the registered confirmatory test): ", "")
        nd = 3 if h == "HE" else 2
        hyp.append([h, what, f(e, nd), f(ci, nd), f(r.get("p"), 4), f(r.get("p_holm"), 4), tex(rule).replace("$<$=", r"$\le$")])
        csv_rows.append(dict(hypothesis=h, estimand=what, estimate=e, ci=ci, p=r.get("p"), p_holm=r.get("p_holm"), rule_outcome=rule, label=LABEL))
    table("partial_hypotheses.tex", rf"Hypotheses on the available TEST shards ({LABEL}). $p_{{\mathrm{{Holm}}}}$ over the registered family is shown for completeness only; "
          "no result here is a confirmatory finding.", "tab:partial-hyp", ["Hyp.", "Estimand", "Estimate", "CI", "$p$", "$p_{\\mathrm{Holm}}$", "Rule outcome on partial data"],
          hyp, "llrlrrp{4.2cm}")

    ha = R.get("HA", {}).get("variants", {})
    rows = [[n] + [f((ha.get(v, {}).get("t50", {}).get(k) or ["--"])[0], 1) + ("$^\\dagger$" if (ha.get(v, {}).get("t50", {}).get(k) or ["", ""])[1] == "left_censored" else "")
                   for v in ("D8", "ER30")] for k, n in KINDS.items()]
    rows.append([r"$\Delta t_{50}$ (binding$-$presence) [95\% CI]"] + [f"{f(ha.get(v, {}).get('delta_t50', {}).get('estimate'))} {f(ha.get(v, {}).get('delta_t50', {}).get('ci95'))}" for v in ("D8", "ER30")])
    rows.append(["scenes"] + [f(ha.get(v, {}).get("n_scenes")) for v in ("D8", "ER30")])
    table("partial_ha.tex", rf"Persistence steps $t_{{50}}$ at $w=7.5$ on TEST shards present ({LABEL}; HA is exploratory by its DEV gate). $^\dagger$left-censored.",
          "tab:partial-ha", ["Kind", "D8", "ER30"], rows, "lrr")

    hd = (R.get("HD", {}).get("estimands") or {})
    table("partial_hd.tex", rf"HD on the available shards ({LABEL}): $t_{{50}}$ by guidance scale; binding estimand exploratory by the audit.", "tab:partial-hd",
          ["Estimand", "$t_{50}$ ($w$=2 / 7.5 / 15)", r"$\Delta t_{50}$ (15$-$2)", "95\\% CI", "$p$", "rule"],
          [[KINDS.get(k, k), " / ".join(f(v.get("t50_by_w", {}).get(w), 1) for w in ("2", "7.5", "15")), f(v.get("delta")), f(v.get("ci95")), f(v.get("p"), 4), tex(v.get("decision"))]
           for k, v in hd.items()], "llrlrl")

    he = R.get("HE", {})
    cond, sec = he.get("conditions") or {}, he.get("secondary") or {}
    table("partial_he.tex", rf"HE: CAG ($w_{{lo}}=3$) vs constant CFG 7.5 (A1) and the reversed schedule A4, on scenes with both A1 and CAG images ({LABEL}; "
          rf"{C.get('hypotheses', {}).get('HE_scenes_with_A1_and_CAG', '--')} scenes).", "tab:partial-he", ["Contrast", "Estimate", "CI", "Requirement", "Met"],
          [["A2$-$A1 on $F$", f((he.get("non_inferiority") or {}).get("estimate"), 3), f((he.get("non_inferiority") or {}).get("ci90"), 3), r"90\% CI lower $>-0.03$", f(cond.get("non_inferior"))],
           ["A2$-$A1 diversity", f((he.get("diversity") or {}).get("estimate"), 4), f((he.get("diversity") or {}).get("ci95"), 4), r"95\% CI $>0$", f(cond.get("more_diverse"))],
           ["A2$-$A4 on $F$", f((he.get("timing") or {}).get("estimate"), 3), f((he.get("timing") or {}).get("ci95"), 3), r"95\% CI $>0$", f(cond.get("beats_reversed"))],
           ["A3$-$A1 on $F$", f((sec.get("A3_minus_A1") or [None])[0], 3), f((sec.get("A3_minus_A1") or [None, None])[1], 3), "secondary", "--"],
           ["A6$-$A1 on $F$", f((sec.get("A6_minus_A1") or [None])[0], 3), f((sec.get("A6_minus_A1") or [None, None])[1], 3), "secondary", "--"]], "lrlll")
    calls = sec.get("mean_unet_calls") or {}
    if calls:
        table("partial_he_cost.tex", rf"Mean UNet evaluations per image by arm ({LABEL}).", "tab:partial-cost", ["Arm", "UNet calls"], [[k, f(v, 1)] for k, v in sorted(calls.items())], "lr")

    hf = R.get("HF", {})
    table("partial_hf.tex", rf"HF: attention stabilisation vs persistence ({LABEL}).", "tab:partial-hf", ["Quantity", "Value"],
          [["concordance", f(hf.get("concordance"))], ["95\\% CI", f(hf.get("ci95"))], ["pairs", f(hf.get("pairs"))],
           *[[f"Spearman $\\rho$, {KINDS.get(k, tex(k))}", f"{f(v[0])} [{f(v[1])}, {f(v[2])}]"] for k, v in sorted((hf.get("spearman") or {}).items())],
           ["rule", tex(hf.get("decision")).replace("$<$=", r"$\le$")]], "ll")

    s2 = (P.get("supporting") or {}).get("S2") or {}
    fams = ["objects_present", "count_ok", "bind_given_pres", "spatial_given_pres", "size_given_pres"]
    table("partial_s2.tex", rf"S2 (EXPLORATORY): outcome rates by prompt variant at $w=7.5$ on the available TEST shards; {s2.get('excluded_dev_overlap', 0)} rows whose prompt occurs in DEV excluded.",
          "tab:partial-s2", ["Variant", "rows"] + [tex(x) for x in fams],
          [[v, x.get("n_rows")] + [f((x.get(k) or {}).get("mean"), 3) for k in fams] for v, x in sorted((s2.get("rates") or {}).items())], "lr" + "r" * len(fams))

    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grid = [6, 12, 18, 24, 30, 36]
        for v in ("D8", "ER30"):
            cur = ha.get(v, {}).get("curves") or {}
            if not cur:
                continue
            fig, ax = plt.subplots(figsize=(5, 3.2))
            for k, c in cur.items():
                ax.plot(grid[:len(c)], [math.nan if x is None else x for x in c], marker="o", label=KINDS.get(k, k).replace("$\\mid$", "|"))
            ax.axhline(0.5, color="grey", lw=0.8, ls="--"); ax.set_ylim(-0.3, 1.05)
            ax.set_xlabel("restart step t (of 40)"); ax.set_ylabel("persistence C(t)"); ax.set_title(f"{v}, w = 7.5 (partial TEST, exploratory)", fontsize=9)
            ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGS / f"partial_curves_{v}.pdf"); fig.savefig(REP / f"partial_curves_{v}.png", dpi=150); plt.close(fig)
            figs.append(f"partial_curves_{v}.pdf")
        cons = [("A2-A1 F", he.get("non_inferiority") or {}, "ci90"), ("A2-A4 F", he.get("timing") or {}, "ci95")]
        cons = [(n, e, k) for n, e, k in cons if e.get("estimate") is not None and e.get(k)]
        if cons:
            fig, ax = plt.subplots(figsize=(4, 3))
            for i, (n, e, k) in enumerate(cons):
                ax.errorbar([i], [e["estimate"]], yerr=[[e["estimate"] - e[k][0]], [e[k][1] - e["estimate"]]], fmt="o", capsize=4)
            ax.axhline(0, color="grey", lw=0.8); ax.axhline(-0.03, color="red", lw=0.8, ls=":")
            ax.set_xticks(range(len(cons))); ax.set_xticklabels([n for n, _, _ in cons]); ax.set_ylabel("difference in F"); ax.set_title("HE contrasts (partial TEST)", fontsize=9)
            fig.tight_layout(); fig.savefig(FIGS / "partial_he.pdf"); fig.savefig(REP / "partial_he.png", dpi=150); plt.close(fig); figs.append("partial_he.pdf")
        if s2.get("rates"):
            vs = sorted(s2["rates"]); fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.bar(range(len(vs)), [(s2["rates"][v].get("objects_present") or {}).get("mean") or 0 for v in vs])
            ax.set_xticks(range(len(vs))); ax.set_xticklabels(vs, fontsize=7); ax.set_ylabel("presence rate"); ax.set_title("S2: presence by variant (partial TEST)", fontsize=9)
            fig.tight_layout(); fig.savefig(FIGS / "partial_s2_presence.pdf"); fig.savefig(REP / "partial_s2_presence.png", dpi=150); plt.close(fig); figs.append("partial_s2_presence.pdf")
    except Exception as e:
        (REP / "figure_errors.txt").write_text(repr(e))

    missing = {st: C[st].get("missing") for st in ("TEST_CORE", "TEST_COMMIT", "TEST_CAG_SIZE") if C[st].get("missing")}
    lim = [f"Missing TEST shards (not imputed): {'; '.join(f'{st} shard {chr(44).join(map(str, v))}' for st, v in missing.items())}; estimates cover only the scene groups of the present shards.",
           "This is not the registered confirmatory analysis: that requires all nine shards and notebook 04; Holm-adjusted p-values are shown for completeness only.",
           "HA is exploratory by its failed DEV precision gate; the full-resample control failed on DEV (Section on DEV results).",
           "Only the presence judge met the audit criterion (single annotator, no valid double annotation); count, binding, spatial and size outcomes are exploratory.",
           f"HE uses only scenes with both A1 and CAG images ({C.get('hypotheses', {}).get('HE_scenes_with_A1_and_CAG')} scenes)."]
    sec = [r"\section{Partial TEST results (exploratory)}\label{sec:resPARTIAL}",
           "Generated by \\texttt{partial/partial\\_report.py} from the verified TEST shards that exist (" + tex(prov) + "). "
           "Every quantity below is \\textbf{exploratory}: it covers a subset of the TEST scenes and is not the registered confirmatory test.",
           r"\input{tables/partial_coverage}", r"\input{tables/partial_hypotheses}", r"\input{tables/partial_ha}", r"\input{tables/partial_hd}", r"\input{tables/partial_he}"]
    if calls:
        sec.append(r"\input{tables/partial_he_cost}")
    sec += [r"\input{tables/partial_hf}", r"\input{tables/partial_s2}"]
    caps = {"partial_curves_D8.pdf": "Chance-corrected persistence $C(t)$ by prompt type under the D8 restart probe ($w=7.5$); dashed line marks $C=0.5$, whose crossing defines $t_{50}$.",
            "partial_curves_ER30.pdf": "Chance-corrected persistence $C(t)$ by prompt type under the ER30 restart probe ($w=7.5$); dashed line marks $C=0.5$.",
            "partial_he.pdf": "HE contrasts on the composite $F$: A2$-$A1 (90\\% CI, non-inferiority; dotted line is the $-0.03$ margin) and A2$-$A4 (95\\% CI, timing versus the reversed schedule).",
            "partial_s2_presence.pdf": "S2: object-presence rate by prompt variant."}
    for fg in figs:
        sec += [r"\begin{figure}[t]\centering", rf"\includegraphics[width=0.75\linewidth]{{figures/{fg}}}",
                rf"\caption{{{caps.get(fg, tex(fg[:-4]))} Partial TEST, exploratory.}}\label{{fig:{fg[:-4].replace('_', '-')}}}", r"\end{figure}"]
    sec += [r"\paragraph{Limitations of this analysis.}", r"\begin{itemize}"] + [rf"\item {tex(x)}" for x in lim] + [r"\end{itemize}"]
    (SECS / "results_partial.tex").write_text("\n".join(sec) + "\n", encoding="utf-8")
    m = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    if "sections/results_partial" not in m:
        m = m.replace(r"\input{sections/results_dev}", "\\input{sections/results_dev}\n\\input{sections/results_partial}", 1)
        (ROOT / "paper" / "main.tex").write_text(m, encoding="utf-8")

    md = [f"# Guide Before Commitment R3 — partial TEST report ({LABEL})", "", f"Source: `{d}` ({prov})", "", "## Coverage", "| Stage | present | missing | jobs merged | scenes |", "|---|---|---|---|---|"]
    md += [f"| {st} | {C[st].get('present')} | {C[st].get('missing')} | {C[st].get('merged')}/{C[st].get('manifest_jobs')} | {C[st].get('scenes', '--')} |" for st in ("TEST_CORE", "TEST_COMMIT", "TEST_CAG_SIZE")]
    md += ["", "## Hypotheses (exploratory on partial data)", "| H | estimand | estimate | CI | p | Holm p | rule outcome |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['hypothesis']} | {r['estimand']} | {f(r['estimate'])} | {f(r['ci'])} | {f(r['p'], 4)} | {f(r['p_holm'], 4)} | {r['rule_outcome']} |" for r in csv_rows]
    md += ["", "## Limitations"] + [f"- {x}" for x in lim] + ["", f"Figures: {figs}", "LaTeX: paper/sections/results_partial.tex, paper/tables/partial_*.tex"]
    (REP / "partial_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REP / "partial_results.json").write_text(json.dumps(P, indent=1, default=str))
    with open(REP / "hypothesis_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
