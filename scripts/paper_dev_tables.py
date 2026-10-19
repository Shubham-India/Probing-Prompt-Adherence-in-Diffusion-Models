"""LaTeX tables for the manuscript's DEV-stage section, generated ONLY from committed, verified evidence (no demonstration values):
docs/evidence/dev_analysis_20261020/dev_constants_A0fix.json (corrected DEV analysis), docs/evidence/audit_20261019/audit_verdicts.json (DEV evaluator
and size audit, annotator A), docs/evidence/hps_guard_20261020/hps_guard.json (CAG quality guard). Writes paper/tables/dev_*.tex.
Usage: python scripts/paper_dev_tables.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "docs" / "evidence"
OUT = ROOT / "paper" / "tables"


def f(x, d=2):
    return "--" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))


def table(name, caption, label, head, rows, spec):
    L = [r"\begin{table}[t]\centering\small", rf"\caption{{{caption}}}\label{{{label}}}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         " & ".join(head) + r" \\", r"\midrule"] + [" & ".join(map(str, r)) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / name).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    dc = json.loads((EV / "dev_analysis_20261020" / "dev_constants_A0fix.json").read_text())
    au = json.loads((EV / "audit_20261019" / "audit_verdicts.json").read_text())
    hp = json.loads((EV / "hps_guard_20261020" / "hps_guard.json").read_text())
    g = dc["gates"]
    table("dev_gates.tex", r"DEV gates (corrected estimator). Failed gates are reported as failed; HA is therefore exploratory.", "tab:dev-gates",
          ["Gate", "Criterion", "DEV value", "Result"],
          [["HA precision", r"CI half-width $\le 4$ steps", f(g["HA_dev_halfwidth_le_4"]["value"]), "failed"],
           ["Full-resample control", r"$\max|C| \le 0.15$", f(g["full_resample_control_max_abs_C_le_0.15"]["value"]), "failed"],
           [r"HD presence, $w=2$", r"$\ge 40$ scenes", g["HD_presence_gate_w2"]["value"], "passed"],
           [r"HD presence, $w=15$", r"$\ge 40$ scenes", g["HD_presence_gate_w15"]["value"], "passed"],
           ["CAG quality (HPS v2), $w_{lo}=1$", r"mean drop $\le 0.5$", f(hp["per_w_lo"]["1"]["mean_change_points"]), "failed"],
           ["CAG quality (HPS v2), $w_{lo}=3$", r"mean drop $\le 0.5$", f(hp["per_w_lo"]["3"]["mean_change_points"]), "passed"]], "llrl")
    t50 = dc["ha_dev"]["variants"]
    names = {"presence": "presence", "count": "count", "bind_given_pres": "binding $\\mid$ presence", "spatial_given_pres": "spatial $\\mid$ presence", "joint_bind": "joint binding"}
    rows = []
    for k, n in names.items():
        cells = [n]
        for v in ("D8", "ER30"):
            val, st = t50[v]["t50"].get(k, (None, ""))
            cells.append(f(val, 1) + ("$^{\\dagger}$" if st == "left_censored" else ""))
        rows.append(cells)
    rows.append([r"$\Delta t_{50}$ (binding $-$ presence)"] + [f"{f(t50[v]['delta_t50']['estimate'], 2)} [{f(t50[v]['delta_t50']['ci95'][0], 2)}, {f(t50[v]['delta_t50']['ci95'][1], 2)}]" for v in ("D8", "ER30")])
    table("dev_t50.tex", r"DEV persistence steps $t_{50}$ at $w=7.5$ (EXPLORATORY; 163 scenes). $^{\dagger}$left-censored: already above 0.5 at the first grid step.",
          "tab:dev-t50", ["Kind", "D8", "ER30"], rows, "lrr")
    lab = {"presence": "presence", "count": "count", "binding": "binding", "spatial": "spatial", "size": "relative size"}
    table("dev_audit.tex", r"Human audit of the first-pass judge on DEV images (annotator A only; no valid double annotation). Criterion: weighted accuracy $\ge 0.85$.",
          "tab:dev-audit", ["Type", "$n$", "Weighted accuracy [95\\% CI]", "Status"],
          [[lab[t], au[t]["n_used"], f"{f(au[t]['weighted_accuracy'], 3)} [{f(au[t]['ci95'][0], 3)}, {f(au[t]['ci95'][1], 3)}]", "met" if au[t]["criterion_met"] else "not met"]
           for t in lab], "lrll")
    print("wrote", [p.name for p in sorted(OUT.glob("dev_*.tex"))])


if __name__ == "__main__":
    main()
