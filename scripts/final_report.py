"""Final report from the COMPLETED TEST analysis (notebook 04, PHASE = "TEST"). Every number is read from the verified outputs; nothing is typed by hand.
Input: the notebook-04 TEST archive folder (ARCHIVE_MANIFEST.json + parts; verified, stage FINAL_TEST) or an already extracted final_* folder.
Refuses: unverified or dry-run archives, a confirmatory.json whose status is not CONFIRMATORY, a protocol that does not verify.
Writes: paper/tables/test_hypotheses.tex, test_ha.tex, test_hd.tex, test_he.tex, test_hf.tex, test_s2.tex; paper/figures/test_commitment_curves.pdf,
test_he_arms.pdf; paper/sections/results_test.tex (generated, replaces the PENDING results sections in paper/main.tex); reports/final_results.md;
reports/final_results.json (all values used). Then runs paper/check_paper.py.
Usage: python scripts/final_report.py --final <04 TEST archive folder or extracted final_* folder>"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "kaggle"), str(ROOT / "src")]
TABLES, FIGS, SECS, REPORTS = ROOT / "paper" / "tables", ROOT / "paper" / "figures", ROOT / "paper" / "sections", ROOT / "reports"
NAMES = {"presence": "presence", "count": "count", "bind_given_pres": r"binding $\mid$ presence", "spatial_given_pres": r"spatial $\mid$ presence",
         "size_given_pres": r"size $\mid$ presence", "joint_bind": "joint binding"}


def f(x, d=2):
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(v, d) for v in x) + "]"
    return "--" if isinstance(x, float) and not math.isfinite(x) else f"{x:.{d}f}"


def tex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace(">", r"$>$").replace("<", r"$<$")


def table(name, caption, label, head, rows, spec):
    L = [r"\begin{table}[t]\centering\small", rf"\caption{{{caption}}}\label{{{label}}}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         " & ".join(head) + r" \\", r"\midrule"] + [" & ".join(str(c) for c in r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / name).write_text("\n".join(L) + "\n", encoding="utf-8")


def locate(path):
    import nbtools
    p = Path(path)
    if (p / "ARCHIVE_MANIFEST.json").exists():
        am = json.loads((p / "ARCHIVE_MANIFEST.json").read_text())
        if am.get("stage") != "FINAL_TEST" or not nbtools.archive_is_verified(am):
            sys.exit(f"not a verified FINAL_TEST archive (stage={am.get('stage')}, verified={am.get('verified')}, dry_run={am.get('dry_run')})")
        dest, probs = nbtools.extract_and_verify(p, ROOT / "local_audit" / "final_test" / am["run_id"])
        if probs:
            sys.exit(f"archive verification failed: {probs[:5]}")
        return dest, am
    if not (p / "confirmatory.json").exists():
        sys.exit("give the notebook-04 TEST archive folder or its extracted final_* folder")
    return p, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    a = ap.parse_args()
    d, am = locate(a.final)
    res = json.loads((d / "confirmatory.json").read_text())
    if res.get("status") != "CONFIRMATORY":
        sys.exit(f"confirmatory.json status is {res.get('status')!r}, not CONFIRMATORY: no confirmatory report is written")
    import protocol_r3
    prot = d / "protocol" / "protocol_freeze.json"
    probs = protocol_r3.verify(prot if prot.exists() else ROOT / "protocol" / "protocol_freeze.json")
    if probs:
        sys.exit("protocol does not verify: " + "; ".join(probs[:3]))
    R = res["results"]
    for p in (TABLES, FIGS, SECS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)

    # hypothesis summary
    est = {"HA": lambda r: (r.get("variants", {}).get("D8", {}).get("delta_t50", {}) or {}),
           "HD": lambda r: (r.get("estimands", {}).get("presence", {}) or {}),
           "HE": lambda r: (r.get("non_inferiority", {}) or {}),
           "HF": lambda r: dict(estimate=r.get("concordance"), ci95=r.get("ci95"))}
    rows = []
    for h in ("HA", "HD", "HE", "HF"):
        r = R.get(h, {})
        e = est[h](r)
        rows.append([h, f(e.get("estimate", e.get("delta"))), f(e.get("ci95", e.get("ci90"))), f(r.get("p"), 4), f(r.get("p_holm"), 4), tex(r.get("final", "--"))])
    table("test_hypotheses.tex", r"Confirmatory family on TEST (Holm over HA, HD, HE, HF). SUPPORTED requires the registered rule and Holm-adjusted $p<0.05$; "
          r"classifications fixed before TEST are applied (HA exploratory; HD supportable only through presence).", "tab:test-hyp",
          ["Hyp.", "Estimate", "CI", "$p$", "$p_{\\mathrm{Holm}}$", "Reported result"], rows, "lrlrrp{5.2cm}")

    ha = R.get("HA", {}).get("variants", {})
    trows = []
    for k, n in NAMES.items():
        trows.append([n] + [f((ha.get(v, {}).get("t50", {}).get(k) or ["--"])[0], 1) for v in ("D8", "ER30")])
    trows.append([r"$\Delta t_{50}$ (binding $-$ presence)"] + [f"{f(ha.get(v, {}).get('delta_t50', {}).get('estimate'))} {f(ha.get(v, {}).get('delta_t50', {}).get('ci95'))}" for v in ("D8", "ER30")])
    table("test_ha.tex", r"TEST persistence steps $t_{50}$ at $w=7.5$ (HA; EXPLORATORY by the DEV gate).", "tab:test-ha", ["Kind", "D8", "ER30"], trows, "lrr")

    hd = R.get("HD", {}).get("estimands", {})
    table("test_hd.tex", r"HD: $t_{50}$ by guidance scale and $\Delta t_{50}=t_{50}(w{=}15)-t_{50}(w{=}2)$. The binding estimand is EXPLORATORY.", "tab:test-hd",
          ["Estimand", "$t_{50}$ by $w$", r"$\Delta t_{50}$", "95\\% CI", "Rule"],
          [[NAMES.get(k, k), tex(json.dumps(v.get("t50_by_w"))), f(v.get("delta")), f(v.get("ci95")), tex(v.get("decision"))] for k, v in hd.items()], "llrll")

    he = R.get("HE", {})
    sec = he.get("secondary", {})
    table("test_he.tex", r"HE: Commitment-Aligned Guidance ($w_{lo}=3$, DEV quality guard) vs constant CFG 7.5 (A1) and the reversed schedule (A4).", "tab:test-he",
          ["Contrast", "Estimate", "CI", "Requirement", "Met"],
          [["A2$-$A1 on $F$ (non-inferiority)", f((he.get("non_inferiority") or {}).get("estimate"), 3), f((he.get("non_inferiority") or {}).get("ci90"), 3), r"90\% CI lower $>-0.03$", (he.get("conditions") or {}).get("non_inferior")],
           ["A2$-$A1 diversity", f((he.get("diversity") or {}).get("estimate"), 4), f((he.get("diversity") or {}).get("ci95"), 4), r"95\% CI $>0$", (he.get("conditions") or {}).get("more_diverse")],
           ["A2$-$A4 on $F$ (timing)", f((he.get("timing") or {}).get("estimate"), 3), f((he.get("timing") or {}).get("ci95"), 3), r"95\% CI $>0$", (he.get("conditions") or {}).get("beats_reversed")],
           ["A3$-$A1 on $F$ (secondary)", f((sec.get("A3_minus_A1") or [None])[0], 3), f((sec.get("A3_minus_A1") or [None, None])[1], 3), "--", "--"],
           ["A6$-$A1 on $F$ (secondary)", f((sec.get("A6_minus_A1") or [None])[0], 3), f((sec.get("A6_minus_A1") or [None, None])[1], 3), "--", "--"]], "lrlll")

    hf = R.get("HF", {})
    table("test_hf.tex", r"HF: attention stabilisation vs commitment.", "tab:test-hf", ["Quantity", "Value"],
          [["type-order concordance", f(hf.get("concordance"))], ["95\\% CI", f(hf.get("ci95"))], ["pairs", f(hf.get("pairs"), 0)],
           ["within-type Spearman", tex(json.dumps(hf.get("spearman"), default=str)[:120])], ["rule", tex(hf.get("decision"))]], "ll")

    s2 = (res.get("supporting") or {}).get("S2", {})
    fams = ["objects_present", "count_ok", "bind_given_pres", "spatial_given_pres", "size_given_pres"]
    table("test_s2.tex", rf"Supporting S2 (EXPLORATORY): TEST outcome rates by prompt variant at $w=7.5$; {s2.get('excluded_dev_overlap', 0)} rows whose prompt also occurs in DEV excluded.",
          "tab:test-s2", ["Variant", "rows"] + [tex(x) for x in fams],
          [[v, x.get("n_rows")] + [f((x.get(k) or {}).get("mean"), 3) for k in fams] for v, x in sorted((s2.get("rates") or {}).items())], "lr" + "r" * len(fams))

    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grid = [6, 12, 18, 24, 30, 36]
        curves = ha.get("D8", {}).get("curves", {})
        if curves:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            for k, c in curves.items():
                ax.plot(grid[:len(c)], [math.nan if v is None else v for v in c], marker="o", label=NAMES.get(k, k).replace("$\\mid$", "|"))
            ax.axhline(0.5, color="grey", lw=0.8, ls="--")
            ax.set_xlabel("restart step t (of 40)"); ax.set_ylabel("persistence C(t)"); ax.legend(fontsize=7); fig.tight_layout()
            fig.savefig(FIGS / "test_commitment_curves.pdf"); plt.close(fig); figs.append("test_commitment_curves.pdf")
        arms = {k: v for k, v in (("A2-A1 (F)", (he.get("non_inferiority") or {})), ("A2-A4 (F)", (he.get("timing") or {}))) if v.get("estimate") is not None}
        if arms:
            fig, ax = plt.subplots(figsize=(4, 3))
            for i, (k, v) in enumerate(arms.items()):
                ci = v.get("ci90") or v.get("ci95")
                ax.errorbar([i], [v["estimate"]], yerr=[[v["estimate"] - ci[0]], [ci[1] - v["estimate"]]], fmt="o", capsize=4)
            ax.axhline(0, color="grey", lw=0.8); ax.axhline(-0.03, color="red", lw=0.8, ls=":")
            ax.set_xticks(range(len(arms))); ax.set_xticklabels(list(arms)); ax.set_ylabel("difference in F"); fig.tight_layout()
            fig.savefig(FIGS / "test_he_arms.pdf"); plt.close(fig); figs.append("test_he_arms.pdf")
    except Exception as e:                                     # figures are optional; the tables carry every number
        (REPORTS / "figure_errors.txt").write_text(repr(e))

    lines = [r"\section{TEST results}\label{sec:resTEST}",
             "This section is generated by \\texttt{scripts/final_report.py} from the verified TEST analysis"
             + (f" (archive \\texttt{{{tex(am['run_id'])}}}, source \\texttt{{{tex(str(am.get('source_commit'))[:7])}}})" if am else "") + ". "
             "Reported results follow the frozen protocol; classifications fixed before TEST are applied (Section~\\ref{sec:resDEV}).",
             r"\input{tables/test_hypotheses}"]
    for h, t in (("HA", "test_ha"), ("HD", "test_hd"), ("HE", "test_he"), ("HF", "test_hf")):
        r = R.get(h, {})
        lines.append(rf"\paragraph{{{h}.}} Reported result: {tex(r.get('final', '--'))}; raw $p={f(r.get('p'), 4)}$, Holm-adjusted $p={f(r.get('p_holm'), 4)}$."
                     + (f" {tex(r['pre_test_status'])}." if r.get("pre_test_status") else ""))
        lines.append(rf"\input{{tables/{t}}}")
    for fg, cap in (("test_commitment_curves.pdf", "TEST persistence curves C(t) (D8, w = 7.5)."), ("test_he_arms.pdf", "HE contrasts with confidence intervals; dotted line: non-inferiority margin.")):
        if fg in figs:
            lines += [r"\begin{figure}[t]\centering", rf"\includegraphics[width=0.8\linewidth]{{figures/{fg}}}", rf"\caption{{{cap}}}", r"\end{figure}"]
    lines += [r"\paragraph{Supporting S2 (exploratory).}", r"\input{tables/test_s2}"]
    (SECS / "results_test.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    main_tex = ROOT / "paper" / "main.tex"
    m = main_tex.read_text(encoding="utf-8")
    if "sections/results_test" not in m:
        for old in (r"\input{sections/results_HA}", r"\input{sections/results_HB}", r"\input{sections/results_HC}"):
            m = m.replace(old + "\n", "")
        m = m.replace(r"\input{sections/results_dev}", "\\input{sections/results_dev}\n\\input{sections/results_test}", 1)
        main_tex.write_text(m, encoding="utf-8")

    md = ["# Final TEST results (generated)", f"Source: {d}", ""] + [f"- **{h}**: {R.get(h, {}).get('final')} (p = {f(R.get(h, {}).get('p'), 4)}, Holm p = {f(R.get(h, {}).get('p_holm'), 4)})" for h in ("HA", "HD", "HE", "HF")]
    md += ["", f"Figures: {figs}", "Tables: paper/tables/test_*.tex"]
    (REPORTS / "final_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPORTS / "final_results.json").write_text(json.dumps(dict(source=str(d), archive=am, results=R, supporting=res.get("supporting"), dedup=res.get("dedup")), indent=1, default=str))
    chk = subprocess.run([sys.executable, str(ROOT / "paper" / "check_paper.py")], capture_output=True, text=True)
    print("\n".join(md)); print(chk.stdout[-800:])
    sys.exit(chk.returncode)


if __name__ == "__main__":
    main()
