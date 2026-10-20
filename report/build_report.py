"""Builds report/report.md from report/report_template.md + report/intervention_template.md and the saved results.

Every number is filled from src/report_numbers.py. Directional claims in the text are asserted against the data first,
so the build fails if the data contradicts the wording.
Usage: python report/build_report.py
"""
import json
import re
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from report_numbers import numbers  # noqa: E402

RUN = ROOT / "outputs" / "main"
INTERVENTION = ROOT / "outputs" / "intervention"


def check_claims(V):
    res = RUN / "results"
    cats = ["presence", "binding", "counting", "spatial", "multi"]
    assert all(V[f"cfg72_{c}_raw"] > 0 for c in cats), "H1 wording: every category should be higher at CFG 7.5 than at 2"
    assert all(V[f"cfg157_{c}_raw"] <= 0.05 and V[f"cfg157_{c}_raw"] < V[f"cfg72_{c}_raw"] for c in ["binding", "counting", "multi"]), "H1 wording: 7.5->15 gain"
    assert all(V[f"cfg72_{c}_lo"] <= 0 <= V[f"cfg72_{c}_hi"] for c in cats), "per-category CIs should include 0"
    assert V["cfg72_all_lo"] <= 0 <= V["cfg72_all_hi"], "pooled CFG 7.5 vs 2 CI should include 0"
    contrasts = pd.read_csv(res / "cfg_contrasts.csv")
    assert contrasts[contrasts.category != "all"].p_holm.min() >= 0.05, "a per-category Holm p < 0.05 exists; wording must change"
    steps = pd.read_csv(res / "steps_contrasts.csv")
    assert ((steps.ci_lo <= 0) & (steps.ci_hi >= 0)).all(), "a steps CI excludes 0; wording must change"
    cells = pd.read_csv(res / "adherence_by_cell.csv")
    sp = cells[cells.category == "spatial"].set_index(["cfg", "steps"])["mean"]
    assert sp.idxmin() == (15.0, 40.0), "spatial lowest cell claim"
    eff = pd.read_csv(res / "attention_cfg_effect.csv")
    eff["a"], eff["b"] = eff["a"].astype(str), eff["b"].astype(str)
    t72 = eff[(eff.metric == "top10") & (eff.a == "7.5") & (eff.b.isin(["2", "2.0"]))]
    assert (t72.ci_lo > 0).all(), "localisation CFG 7.5 vs 2 should exclude 0 at all stages"
    t157 = eff[(eff.metric == "top10") & (eff.a == "15.0") & (eff.b == "7.5") & (eff.stage != "early")]
    assert (t157.ci_hi < 0).all(), "localisation CFG 15 vs 7.5 should be lower at middle/late"
    auc = pd.read_csv(res / "attention_auc.csv").set_index(["feature", "within_prompt"])
    assert auc.loc[("middle_bind_margin", False)].ci_lo > 0.5 and auc.loc[("late_bind_margin", False)].ci_lo > 0.5
    clf = json.loads((res / "attention_classifier_auc.json").read_text())
    assert clf["attention_only"] < 0.75 and clf["attention_only"] <= clf["category_cfg_only"], "H3c wording"


def check_intervention_claims():
    res = INTERVENTION / "results"
    c = pd.read_csv(res / "intervention_contrasts.csv")
    allc = c[(c.category == "all") & (c.b == "rw1")]
    p = allc[(allc.outcome == "primary") & (allc.a == "rw2")].iloc[0]
    assert p["diff"] < 0.05 and p.ci_lo <= 0, "H4 'not supported' wording: primary contrast must be < 0.05 with CI including 0"
    pr = allc[(allc.outcome == "presence_ok") & (allc.a == "rw2")].iloc[0]
    assert pr.ci_lo > 0 and pr.p_holm < 0.05, "presence 'excludes zero' wording"
    others = allc[(allc.outcome.isin(["binding_ok", "count_ok", "spatial_ok"])) & (allc.a == "rw2")]
    assert ((others.ci_lo <= 0) & (others.ci_hi >= 0)).all(), "'only presence excludes zero' wording"
    percat = c[(c.category != "all") & (c.outcome == "primary") & (c.a.isin(["rw2", "rw4"])) & (c.b == "rw1")]
    assert percat.p_holm.min() >= 0.05, "per-category intervention wording"
    clip = allc[(allc.outcome == "clip_score")]
    assert (clip["diff"].abs() < 0.5).all(), "CLIP 'no change' wording"
    n = json.loads((res / "intervention_numbers.json").read_text())
    assert n["int_ctrl_pixel_identical"].split("/")[0] == n["int_ctrl_pixel_identical"].split("/")[1], "control must be pixel-identical"


def last_line(path):
    lines = [l for l in path.read_text(errors="ignore").splitlines() if l.strip()]
    return lines[-1]


def split_sections(text):
    parts = re.split(r"^@@(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def pilot_versions():
    """Library versions that PRODUCED the pilot (not the versions installed where the report happens to be rebuilt):
    torch from the first samples.jsonl record, the others from the pinned requirements.txt."""
    pins = dict(l.strip().split("==", 1) for l in (ROOT / "requirements.txt").read_text().splitlines() if "==" in l)
    first = json.loads((RUN / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return dict(diffusers_version=pins["diffusers"], torch_version=first["torch"], transformers_version=pins["transformers"])


def main():
    V = numbers(RUN)
    V.update(**pilot_versions())
    check_claims(V)
    check_intervention_claims()
    tables = (RUN / "results" / "tables.md").read_text().split("## ")
    V["table_cells"] = tables[1].split("\n", 1)[1].strip()
    sections = split_sections((ROOT / "report" / "intervention_template.md").read_text(encoding="utf-8"))
    for key, text in sections.items():
        V["INT_" + key] = text.format_map(V)
    V["test_summary"] = last_line(ROOT / "outputs" / "test_results.txt").strip("= ")
    e2e = (ROOT / "outputs" / "e2e.log").read_text(errors="ignore") if (ROOT / "outputs" / "e2e.log").exists() else ""
    V["e2e_status"] = "completed: E2E-DONE" if "E2E-DONE" in e2e else "NOT completed"
    for template, target in (("report/report_template.md", "report/report.md"), ("report/defense_template.md", "report/defense.md"),
                             ("report/PILOT_README_template.md", "report/PILOT_README.md"), ("report/validation_template.md", "report/validation.md")):
        out = (ROOT / template).read_text(encoding="utf-8").format_map(V)
        (ROOT / target).write_text(out, encoding="utf-8")
        print(target, "written;", len(out.split()), "words")


if __name__ == "__main__":
    main()
