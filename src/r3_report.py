"""Render tables and figures from r3_analyze outputs (dev_constants.json / confirmatory.json). Nothing is typed by hand; missing values print as
PENDING / NOT ESTIMABLE. Usage: python src/r3_report.py --results confirmatory.json --out DIR [--dev dev_constants.json]
Writes DIR/results_tables.md, DIR/results_tables.tex and DIR/fig_*.pdf (only for quantities that exist)."""
import argparse
import json
import math
from pathlib import Path


def fmt(x, nd=2):
    if x is None:
        return "PENDING"
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return f"[{fmt(x[0], nd)}, {fmt(x[1], nd)}]"
    return f"{float(x):.{nd}f}" if math.isfinite(float(x)) else str(x)


def rows_from(results):
    r = results.get("results", {})
    ha, hd, he, hf = (r.get(k, {}) for k in ("HA", "HD", "HE", "HF"))
    d8 = ha.get("variants", {}).get("D8", {}).get("delta_t50", {})
    pres = hd.get("estimands", {}).get("presence", {})
    bind = hd.get("estimands", {}).get("bind_given_pres", {})
    return [("HA", "Delta t50 = t50(bind|pres) - t50(presence), steps", fmt(d8.get("estimate")), fmt(d8.get("ci95")), fmt(ha.get("p_holm"), 3), ha.get("final", "PENDING")),
            ("HD", "t50(w=15) - t50(w=2): presence / binding|pres", f"{fmt(pres.get('delta'))} / {fmt(bind.get('delta'))}", f"{fmt(pres.get('ci95'))} / {fmt(bind.get('ci95'))}",
             fmt(hd.get("p_holm"), 3), hd.get("final", "PENDING")),
            ("HE", "A2-A1 F (90% CI); A2-A1 diversity; A2-A4 F", "; ".join(fmt(he.get(k, {}).get("estimate"), 3) for k in ("non_inferiority", "diversity", "timing")),
             "; ".join(fmt(he.get(k, {}).get(c)) for k, c in (("non_inferiority", "ci90"), ("diversity", "ci95"), ("timing", "ci95"))), fmt(he.get("p_holm"), 3), he.get("final", "PENDING")),
            ("HF", "type-order concordance of t_stab vs t50", fmt(hf.get("concordance")), fmt(hf.get("ci95")), fmt(hf.get("p_holm"), 3), hf.get("final", "PENDING"))]


def write(results, out, dev=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = rows_from(results)
    md = ["| Hypothesis | Estimand | Estimate | 95% CI | Holm p | Verdict |", "|---|---|---|---|---|---|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    md.insert(0, f"Status: {results.get('status', 'PENDING')}\n")
    (out / "results_tables.md").write_text("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("|", r"$\mid$").replace("&", r"\&")
    tex = [r"\begin{tabular}{llllll}", r"\toprule", "Hyp. & Estimand & Estimate & 95\\% CI & Holm $p$ & Verdict \\\\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
    (out / "results_tables.tex").write_text("\n".join(tex) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ha = results.get("results", {}).get("HA", {}).get("variants", {}).get("D8", {})
        if ha.get("curves"):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            for k, c in ha["curves"].items():
                ax.plot([6, 12, 18, 24, 30, 36], [float(x) if x not in (None, "nan") else float("nan") for x in c], marker="o", label=k)
            ax.axhline(0.5, ls="--", c="grey", lw=0.8)
            ax.set_xlabel("restart step t (of 40)"); ax.set_ylabel("chance-corrected persistence C(t)"); ax.legend(fontsize=6)
            fig.tight_layout(); fig.savefig(out / "fig_commitment_curves.pdf"); plt.close(fig)
    except Exception as e:
        (out / "figure_errors.txt").write_text(repr(e))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default=None)
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())
    for r in write(res, a.out, json.loads(Path(a.dev).read_text()) if a.dev else None):
        print(r)
