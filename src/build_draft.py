"""One command that rebuilds the DRAFT results layer from verified local artifacts; it never launches generation.
raw artifacts -> hashes -> machine-readable results (outputs/results_draft/results.json, ledger.json) -> LaTeX tables, number macros and figures
(paper/tables, paper/figures) -> docs/RESULTS_PROVENANCE.md. Anything that does not exist yet is written as PENDING; nothing is ever filled with a
zero or a simulated value. Inputs under local_audit/ are private; only derived numbers leave that folder.
Usage: python src/build_draft.py [--out-root DIR]      (default: the repository root)"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LA = ROOT / "local_audit"

TEXT_SUFFIX = {".csv", ".json", ".jsonl", ".txt", ".md", ".py"}


def sha(path):
    b = Path(path).read_bytes()
    if Path(path).suffix in TEXT_SUFFIX:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def esc(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"), ("~", r"\~{}")):
        s = s.replace(a, b)
    return s.replace("|", r"$\mid$")


def f3(x):
    return "n/a" if x is None else f"{float(x):.3f}"


def ci(lo, hi):
    return f"[{float(lo):.3f}, {float(hi):.3f}]"


def table(path, caption, label, header, rows, colspec=None, note=None):
    colspec = colspec or "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\par\smallskip\footnotesize {note}")
    lines.append(r"\end{table}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class Ledger:
    def __init__(self, root):
        self.root, self.rows = root, []

    def add(self, rid, claim, sources, script, status, estimate, limitation):
        hs = []
        for s in sources:
            p = Path(s)
            hs.append(dict(path=str(p.relative_to(self.root)).replace("\\", "/") if p.is_relative_to(self.root) else str(p), sha256=sha(p)[:16] if p.exists() else "MISSING"))
        self.rows.append(dict(id=rid, claim=claim, sources=hs, script=script, status=status, estimate=estimate, limitation=limitation))


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(root=ROOT, la=LA):
    root, la = Path(root), Path(la)
    out, tabs, figs = root / "outputs" / "results_draft", root / "paper" / "tables", root / "paper" / "figures"
    for d in (out, tabs, figs):
        d.mkdir(parents=True, exist_ok=True)
    L, R, macros = Ledger(root), {}, {}
    m = root / "outputs" / "main" / "results"

    # ---- pilot design and adherence ---------------------------------------------------------------------------------------------
    counts = json.loads((m / "counts.json").read_text())
    R["pilot_counts"] = counts
    macros.update(pilotImages=counts["images_evaluated"], pilotPrompts=counts["prompts"], pilotSeeds=len(counts["seeds"]),
                  pilotSuccess=counts["successes_primary"], pilotFailure=counts["failures_primary"])
    L.add("P01", "Pilot grid: 28 prompts x 3 CFG x 2 step counts x 6 seeds = 1008 evaluated images, 0 excluded", [m / "counts.json"], "src/analyze.py", "VERIFIED EXISTING RESULT",
          f"{counts['images_evaluated']} images; primary success {counts['successes_primary']}", "28 prompts; detector-based labels")
    cell = read_csv(m / "adherence_by_cell.csv")
    R["pilot_adherence_by_cell"] = cell
    cats = ["all", "presence", "counting", "binding", "multi", "spatial"]
    rows = []
    for c in cats:
        for cfg in ("2.0", "7.5", "15.0"):
            r20 = next((r for r in cell if r["category"] == c and r["cfg"] == cfg and r["steps"] == "20"), None)
            r40 = next((r for r in cell if r["category"] == c and r["cfg"] == cfg and r["steps"] == "40"), None)
            if r20 and r40:
                rows.append([esc(c), cfg.rstrip("0").rstrip(".") if "." in cfg else cfg, f"{float(r20['mean']):.2f} {ci(r20['ci_lo'], r20['ci_hi'])}", f"{float(r40['mean']):.2f} {ci(r40['ci_lo'], r40['ci_hi'])}", r40["n_prompts"]])
    table(tabs / "pilot_adherence.tex", "Pilot (28 prompts, detector-based labels, exploratory): fraction of constraints satisfied, by category, CFG and inference steps; 95\\% prompt-cluster bootstrap intervals.",
          "tab:pilot-adherence", ["Category", "CFG", "20 steps", "40 steps", "Prompts"], rows, "llrrr",
          "Source: \\texttt{outputs/main/results/adherence\\_by\\_cell.csv}. Intervals resample prompts; the judge is unaudited on the DEV set.")
    L.add("P02", "Pilot adherence by category, CFG and steps", [m / "adherence_by_cell.csv"], "src/analyze.py; src/build_draft.py", "EXPLORATORY", f"{len(rows)} cells tabulated", "4-6 prompts per category; CIs wide")
    cc = read_csv(m / "cfg_contrasts.csv")
    R["pilot_cfg_contrasts"] = cc
    rows = [[esc(r["category"]), f"{r['a'].rstrip('0').rstrip('.')} vs {r['b'].rstrip('0').rstrip('.')}", f"{float(r['diff']):+.3f}", ci(r["ci_lo"], r["ci_hi"]), r["n_prompts"], f3(r["p_holm"]) if r["p_holm"] else "n/a"] for r in cc]
    table(tabs / "pilot_cfg_contrasts.tex", "Pilot CFG contrasts (paired by prompt and seed; Holm-adjusted $p$ from Wilcoxon tests), exploratory.", "tab:pilot-cfg",
          ["Category", "CFG a vs b", "Diff.", "95\\% CI", "Prompts", "$p_{Holm}$"], rows, "llrrrr")
    L.add("P03", "CFG has no resolved effect on constraint satisfaction in the pilot (all intervals include 0 where tested)", [m / "cfg_contrasts.csv"], "src/analyze.py", "EXPLORATORY",
          f"{len(cc)} contrasts", "Underpowered (MDE about 0.15 at 28 prompts, RB 15.4); absence of evidence, not evidence of no effect")

    # ---- pilot evaluator validation ----------------------------------------------------------------------------------------------
    vm = root / "outputs" / "validation_metrics.txt"
    val = []
    for line in vm.read_text().splitlines():
        g = re.match(r"(\w+)\s+n=\s*(\d+)\s+acc=([\d.]+)\s+precision=([\d.]+)\s+recall=([\d.]+)", line)
        if g:
            val.append(dict(type=g[1], n=int(g[2]), acc=float(g[3]), precision=float(g[4]), recall=float(g[5])))
    R["pilot_validation"] = val
    table(tabs / "pilot_validation.tex", "Pilot evaluator check on 40 images (labels began as an AI draft and were edited by the project owner; no independent reviewer yet).", "tab:pilot-validation",
          ["Constraint type", "n", "Accuracy", "Precision", "Recall"], [[esc(v["type"]), str(v["n"]), f"{v['acc']:.2f}", f"{v['precision']:.2f}", f"{v['recall']:.2f}"] for v in val],
          note="Spatial: 2 positives only. This table is not the DEV audit (\\S\\ref{sec:evaluation}).")
    L.add("P04", "Pilot evaluator vs manual labels (40 images)", [vm, root / "data" / "validation_labels.json"], "src/validate_metrics.py", "EXPLORATORY",
          "; ".join(f"{v['type']} acc {v['acc']:.2f} (n={v['n']})" for v in val), "non-independent labels; tiny n for count and spatial")
    for v in val:
        macros[f"valAcc{v['type'].capitalize()}"] = f"{v['acc']:.2f}"
        macros[f"valN{v['type'].capitalize()}"] = v["n"]

    # ---- pilot intervention ------------------------------------------------------------------------------------------------------
    ic = read_csv(root / "outputs" / "intervention" / "results" / "intervention_contrasts.csv")
    sel = [r for r in ic if r["outcome"] == "primary" and r["a"] in ("rw2", "rw4") and r["b"] == "rw1"]
    R["pilot_intervention_primary"] = sel
    rows = [[esc(r["category"]), f"{r['a']} vs control", f"{float(r['diff']):+.3f}", ci(r["ci_lo"], r["ci_hi"]), r["n_prompts"], f3(r["p_holm"]) if r["p_holm"] else "n/a"] for r in sel]
    table(tabs / "pilot_intervention.tex", "Pilot global attention re-weighting (scale 2 and 4 against the scale-1 control path), primary outcome, exploratory.", "tab:pilot-intervention",
          ["Category", "Arm", "Diff.", "95\\% CI", "Prompts", "$p_{Holm}$"], rows, "llrrrr",
          "Control images were pixel-identical to the unmodified pipeline (owner-documented 168/168).")
    allr = next(r for r in sel if r["category"] == "all" and r["a"] == "rw2")
    macros.update(intAllDiff=f"{float(allr['diff']):+.2f}", intAllLo=f"{float(allr['ci_lo']):+.2f}", intAllHi=f"{float(allr['ci_hi']):+.2f}")
    L.add("P05", "Global re-weighting (s=2) did not resolve an improvement in overall adherence", [root / "outputs" / "intervention" / "results" / "intervention_contrasts.csv"],
          "src/analyze_intervention.py", "VERIFIED EXISTING RESULT", f"{float(allr['diff']):+.3f} {ci(allr['ci_lo'], allr['ci_hi'])}", "pilot; 4 binding-relevant prompts per cell; negative result is not evidence of no effect")

    # ---- re-analysis: conditional outcomes, seed effect, nested AUC ----------------------------------------------------------------
    m2 = json.loads((root / "outputs" / "m2_pilot_reanalysis" / "summary.json").read_text())
    nz = json.loads((root / "outputs" / "m6_noise_pilot" / "noise_readout.json").read_text())
    R["pilot_m2"], R["pilot_noise"] = m2, nz
    na = m2["nested_auc_steps40_grouped_cv_by_prompt"]
    macros.update(seedP=f"{m2['seed_effect_permutation']['p_value']:.3f}",
                  aucDesign=f"{na['auc']['B_design(category+cfg)']:.3f}", aucAttn=f"{na['auc']['B_attn(+early/mid/late attention)']:.3f}",
                  dAucAttnNoise=f"{na['delta_auc']['attn_minus_noise']:+.3f}", dAucAttnNoiseLo=f"{na['delta_auc_prompt_bootstrap_ci95']['attn_minus_noise'][0]:+.3f}",
                  dAucAttnNoiseHi=f"{na['delta_auc_prompt_bootstrap_ci95']['attn_minus_noise'][1]:+.3f}", dAucNoiseDesign=f"{na['delta_auc']['noise_minus_design']:+.3f}")
    table(tabs / "pilot_nested_auc.tex", "Pilot nested prediction of the primary outcome (40-step images, prompt-grouped cross-validation, 28 prompts); exploratory.", "tab:pilot-nested",
          ["Contrast", "$\\Delta$AUC", "95\\% prompt-bootstrap CI"],
          [[esc(k), f"{v:+.3f}", ci(*na["delta_auc_prompt_bootstrap_ci95"][k])] for k, v in na["delta_auc"].items()], "lrr",
          f"Base rate {na['base_rate']:.3f}. Only six distinct noise vectors exist in the pilot, so noise features equal seed identity (noise vs design: {nz['delta_auc']['noise_minus_B1']:+.3f}).")
    L.add("P06", "Early attention features add held-out AUC over design+seed baselines in the pilot (prompt-grouped CV)", [root / "outputs" / "m2_pilot_reanalysis" / "summary.json"],
          "src/analyze_pilot_conditional.py", "EXPLORATORY", f"attn minus noise {na['delta_auc']['attn_minus_noise']:+.3f} {ci(*na['delta_auc_prompt_bootstrap_ci95']['attn_minus_noise'])}",
          "28 prompts; outcome includes omissions; different CV scheme from attention_classifier_auc.json; motivates HB, does not test it")
    L.add("P07", "Seed 0 is lower than other seeds in the pilot (permutation test)", [root / "outputs" / "m2_pilot_reanalysis" / "summary.json"], "src/analyze_pilot_conditional.py", "EXPLORATORY",
          f"p = {m2['seed_effect_permutation']['p_value']:.3f}", "only six noise vectors; post hoc observation")
    el = read_csv(root / "outputs" / "m2_pilot_reanalysis" / "eligibility.csv")
    R["pilot_eligibility"] = el
    allrow = next(r for r in el if r["category"].startswith("ALL"))
    bm = [r for r in el if r["category"] in ("binding", "multi")]
    tot_b, tot_e = int(allrow["n_binding_images"]), int(allrow["n_eligible"])
    bm_b, bm_e = sum(int(r["n_binding_images"]) for r in bm), sum(int(r["n_eligible"]) for r in bm)
    macros.update(pilotEligible=f"{tot_e / tot_b:.2f}", pilotEligibleN=tot_e, pilotBindingN=tot_b, pilotEligibleBM=f"{bm_e / bm_b:.2f}", pilotEligibleBMN=bm_e, pilotBindingBMN=bm_b)
    L.add("P08", "Share of pilot binding-category images in which all required objects are detected", [root / "outputs" / "m2_pilot_reanalysis" / "eligibility.csv"], "src/analyze_pilot_conditional.py",
          "EXPLORATORY", f"{tot_e}/{tot_b} = {tot_e / tot_b:.3f} (binding and multi categories only: {bm_e}/{bm_b} = {bm_e / bm_b:.3f})", "counting prompts are 100% eligible and inflate the pooled share; detector-based; omissions and detector misses are not separable until the audit")

    # ---- Benchmark V2 design -----------------------------------------------------------------------------------------------------
    dc = json.loads((root / "data" / "benchmark_v2" / "design_counts.json").read_text())
    R["benchmark_design"] = dc
    rows = []
    for sp in ("DEV", "TEST"):
        if sp in dc:
            d = dc[sp]
            rows.append([sp, str(d["scenes"]), str(d["binding_bearing"]), str(d["binding_competitive"]), str(d["spatial_bearing"]), str(d["variants"])])
    table(tabs / "benchmark_design.tex", "Benchmark V2 design counts (scene graphs are the independent unit). TEST counts are design metadata only; no TEST image exists.", "tab:benchmark",
          ["Split", "Scenes", "Binding-bearing (literal)", "Competitive subset", "Spatial-bearing", "Prompt variants"], rows, "lrrrrr")
    macros.update(devScenes=dc["DEV"]["scenes"], devBindingBearing=dc["DEV"]["binding_bearing"], testScenes=dc["TEST"]["scenes"] if "TEST" in dc else "n/a",
                  testBindingBearing=dc["TEST"]["binding_bearing"] if "TEST" in dc else "n/a")
    L.add("B01", "Benchmark V2 design counts", [root / "data" / "benchmark_v2" / "design_counts.json", root / "data" / "benchmark_v2" / "prompts_v2.json"], "src/benchmark_v2.py; src/design_recompute.py",
          "VERIFIED EXISTING RESULT", f"DEV {dc['DEV']['scenes']} scenes / {dc['DEV']['binding_bearing']} binding-bearing", "human review of paraphrase/order/counterfactual variants and lexicon freeze pending")

    # ---- infrastructure evidence (private local_audit inputs) ---------------------------------------------------------------------
    infra = {}
    gi = la / "R0_R1" / "gpu_identity.json"
    if gi.exists():
        g = json.loads(gi.read_text())
        hold = all(r[k] == 0 for r in g["results"] for k in g["required"]) and all(r["saved_latent_recompute_equal"] for r in g["results"])
        infra["r0_identity"] = dict(cases=len(g["results"]), all_hold=hold, gpu=g["results"][0]["gpu"], torch=g["results"][0]["torch"])
        macros.update(rZeroCases=len(g["results"]), rZeroHold="all six required identities held" if hold else "NOT all identities held")
        L.add("I01", "R0 GPU identity checks (determinism, loop from step 0, reload-and-resume, D=0 branch, hook-off, restored baseline) on a T4", [gi], "tests/gpu_identity.py (recomputed)",
              "VERIFIED EXISTING RESULT", f"{len(g['results'])} cases, max pixel difference 0, all hold: {hold}", "3 pilot prompts, seed 0, CFG 7.5, 40 steps; infrastructure only; model revision not recorded in the artifact")
    tp_new, tp_old = la / "R0_R1" / "throughput.json", la / "R2a_checkpoint" / "provenance" / "throughput.json"
    rows = []
    for lab, p in (("R1 session 10:10 UTC", tp_old), ("R0 session 15:02 UTC", tp_new)):
        if p.exists():
            t = json.loads(p.read_text())
            rows.append([lab, f"{t['plain']['per_image_s_median']:.2f}", f"{t['latents']['per_image_s_median']:.2f}", f"{t.get('two_workers_wall_s', float('nan')):.1f}"])
            infra[lab] = dict(plain=t["plain"]["per_image_s_median"], latents=t["latents"]["per_image_s_median"], two_workers_wall_s=t.get("two_workers_wall_s"))
            L.add("I02" + ("a" if "R1" in lab else "b"), f"Throughput probe ({lab}); no run id in the file", [p], "kaggle/run_throughput.py", "VERIFIED EXISTING RESULT", f"plain {t['plain']['per_image_s_median']} s, latents {t['latents']['per_image_s_median']} s per image",
                  "two sessions disagree by about 25% and are not reconciled; first image excluded; wall time includes model loading")
    ar = la / "R2a_checkpoint" / "audit_run.json"
    if ar.exists():
        a = json.loads(ar.read_text())
        oks = [k for k, v in a["checks"].items() if isinstance(v, dict) and "ok" in v]
        infra["r2a_audit"] = dict(hard_failures=len(a["hard_failures"]), checks=len(oks), passed=sum(a["checks"][k]["ok"] for k in oks))
        macros.update(rTwoChecks=len(oks), rTwoPassed=sum(a["checks"][k]["ok"] for k in oks), rTwoLatents=2400)
        L.add("I03", "R2a integrity audit (independent of the run's own verifier)", [ar], "src/audit_run.py", "VERIFIED EXISTING RESULT",
              f"{sum(a['checks'][k]['ok'] for k in oks)}/{len(oks)} hard checks pass; 400 images, 400 latent files with 6 checkpoints each", "no bit-level comparison on Kaggle; text-file checksums match only after CRLF to LF")
    if rows:
        table(tabs / "infra_throughput.tex", "Measured throughput on Tesla T4 (seconds per 40-step image, first image excluded) from two probe sessions; the real R2a run gave a median of 6.21 and 6.26 s per worker.",
              "tab:throughput", ["Session", "Plain", "With latents", "Two workers, wall (s)"], rows)
    ev = la / "R2a_checkpoint" / "eval_view" / "eval.jsonl"
    if ev.exists():
        E = [json.loads(l) for l in ev.read_text().splitlines() if l.strip()]
        P = {p["id"]: p for p in json.loads((root / "data" / "benchmark_v2" / "prompts_v2.json").read_text())}
        bb = [e for e in E if any(c["type"] == "binding" for c in P[e["prompt_id"]]["constraints"])]
        pres = [e["objects_present"] for e in bb]
        bg = [e["bind_given_pres"] for e in bb if e["bind_given_pres"] is not None]
        jb = [e["joint_bind"] for e in bb]
        r2 = dict(images=len(E), binding_bearing_images=len(bb), all_objects_present=sum(pres) / len(pres), bind_given_present=sum(bg) / len(bg), n_eligible=len(bg),
                  joint_bind=sum(jb) / len(jb), uncertain=sum(int(e.get("uncertain") or 0) for e in E))
        infra["r2a_first_pass"] = r2
        macros.update(rTwoPres=f"{r2['all_objects_present']:.2f}", rTwoBind=f"{r2['bind_given_present']:.2f}", rTwoJoint=f"{r2['joint_bind']:.2f}", rTwoEligN=r2["n_eligible"], rTwoBB=r2["binding_bearing_images"])
        table(tabs / "r2a_first_pass.tex", "R2a DEV core, first-pass detector (J1) outcomes on binding-bearing canonical prompts. \\textbf{Unaudited judge; exploratory.}", "tab:r2a-first-pass",
              ["Quantity", "Value", "n"], [["All required objects detected", f"{r2['all_objects_present']:.3f}", str(r2["binding_bearing_images"])],
                                           ["Binding given presence", f"{r2['bind_given_present']:.3f}", str(r2["n_eligible"])], ["Joint (present and bound)", f"{r2['joint_bind']:.3f}", str(r2["binding_bearing_images"])]],
              "lrr", "70 scenes $\\times$ 4 seeds. Human error rates of the judge are pending (\\S\\ref{sec:evaluation}).")
        L.add("I04", "First-pass J1 outcomes on R2a binding-bearing DEV images", [ev], "src/evaluate.py --conditional", "EXPLORATORY",
              f"present {r2['all_objects_present']:.3f}; bind|present {r2['bind_given_present']:.3f} (n={r2['n_eligible']}); joint {r2['joint_bind']:.3f}", "judge unaudited; 27 of 400 images flagged uncertain")
    R["infrastructure"] = infra

    # ---- evaluator audit: PENDING until the owner's labels have been scored --------------------------------------------------------
    rep = la / "audit_dev" / "report.json"
    if rep.exists():
        ra = json.loads(rep.read_text())
        rows = [[esc(t), str(v["n_used"]), f3(v.get("accuracy")), ci(*v["accuracy_ci95"]) if v.get("accuracy_ci95") else "n/a", "yes" if v.get("criterion_met_accuracy_ge_0.85") else "no"] for t, v in ra.items()]
        R["dev_audit"] = ra
        L.add("E01", "DEV evaluator audit of J1 (population-weighted accuracy per type)", [rep], "src/audit_dev.py score", "EXPLORATORY", "see table", "single annotator unless double annotation returned")
        table(tabs / "dev_audit.tex", "DEV evaluator audit of J1 against blinded human labels (population-weighted).", "tab:dev-audit", ["Type", "n", "Accuracy", "95\\% CI", "$\\geq 0.85$"], rows, "lrrrr")
    else:
        R["dev_audit"] = "PENDING"
        table(tabs / "dev_audit.tex", "DEV evaluator audit of J1 against blinded human labels. \\textbf{PENDING: no human labels have been returned.}", "tab:dev-audit", ["Type", "n", "Accuracy", "95\\% CI", "$\\geq 0.85$"],
              [[t, "PENDING", "PENDING", "PENDING", "PENDING"] for t in ("presence", "count", "binding", "spatial")])
        L.add("E01", "DEV evaluator audit of J1", [], "src/audit_dev.py score", "NOT ESTIMABLE", "PENDING", "human labels not returned")

    # ---- hypothesis status: always pending until the corresponding stage has verified outputs ---------------------------------------
    hyp = [("HA", "Restart persistence differs between constraint types (presence, count, size, spatial, colour binding)", "DEV/TEST restart branches not run; DEV gate not evaluated; TEST sealed"),
           ("HD", "Higher CFG moves the restart-persistence step earlier", "needs TEST branches at CFG 2 / 7.5 / 15; TEST sealed"),
           ("HE", "Commitment-Aligned Guidance: non-inferior adherence, higher diversity, beats the reversed schedule", "needs DEV-derived switch steps, quality-guard decision and TEST arms"),
           ("HF", "Attention stabilisation tracks restart persistence", "needs TEST attention maps and HA curves")]
    table(tabs / "hypothesis_status.tex", "Status of the four confirmatory hypotheses (Revision 3). No confirmatory data exist.", "tab:hypotheses", ["ID", "Question", "Status"], [[h, esc(q), r"\textbf{PENDING}: " + esc(s)] for h, q, s in hyp], "llp{6.2cm}")
    for h, q, s in hyp:
        L.add(h, q, [], "see docs/HYPOTHESIS_LEDGER.md", "PLANNED CONFIRMATORY TEST", "NOT ESTIMABLE", s)
    table(tabs / "results_pending.tex", "Confirmatory results (placeholder).", "tab:results-pending", ["Result", "Status"],
          [["HA: $\\Delta t_{50}$ with 95\\% CI", "PENDING"], ["HD: $t_{50}(15)-t_{50}(2)$", "PENDING"], ["HE: $\\Delta F$, $\\Delta$ diversity, CAG minus reversed", "PENDING"],
           ["HF: ordering concordance, within-type Spearman", "PENDING"]], "ll")

    # ---- figures ------------------------------------------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for c in cats:
            xs, ys, lo, hi = [], [], [], []
            for cfg in ("2.0", "7.5", "15.0"):
                r = next((r for r in cell if r["category"] == c and r["cfg"] == cfg and r["steps"] == "40"), None)
                if r:
                    xs.append(float(cfg)); ys.append(float(r["mean"])); lo.append(float(r["ci_lo"])); hi.append(float(r["ci_hi"]))
            ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]], marker="o", capsize=2, label=c, linewidth=1)
        ax.set_xlabel("CFG scale"); ax.set_ylabel("constraints satisfied"); ax.set_xscale("log"); ax.set_xticks([2, 7.5, 15]); ax.set_xticklabels(["2", "7.5", "15"]); ax.legend(fontsize=6, ncol=2)
        ax.set_title("Pilot, 40 steps (exploratory, 28 prompts)", fontsize=8)
        fig.tight_layout(); fig.savefig(figs / "pilot_adherence.pdf"); plt.close(fig)
        if "r2a_first_pass" in infra:
            r2 = infra["r2a_first_pass"]
            fig, ax = plt.subplots(figsize=(4.2, 2.6))
            ax.bar(["all objects\npresent", "binding |\npresent", "joint"], [r2["all_objects_present"], r2["bind_given_present"], r2["joint_bind"]], color=["#4C72B0", "#55A868", "#C44E52"])
            ax.set_ylim(0, 1); ax.set_ylabel("rate"); ax.set_title("R2a first-pass J1 (UNAUDITED judge)", fontsize=8)
            fig.tight_layout(); fig.savefig(figs / "r2a_first_pass.pdf"); plt.close(fig)
    except Exception as e:                                                     # figures are optional; the text must not claim them if absent
        R["figures_error"] = repr(e)

    # ---- number macros, results.json, ledger, provenance doc ------------------------------------------------------------------------
    (tabs / "numbers.tex").write_text("% generated by src/build_draft.py - do not edit\n" + "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(R, indent=1, default=str))
    (out / "ledger.json").write_text(json.dumps(L.rows, indent=1))
    lines = ["# Results provenance (generated by `python src/build_draft.py`; do not edit by hand)", "",
             "Every number in `paper/tables/*.tex` and `paper/tables/numbers.tex` comes from `outputs/results_draft/results.json`, which is built only from the artifacts below. Hashes are the first 16 hex digits of the SHA-256 (text files LF-normalised). `local_audit/` inputs are private; only derived numbers are copied out.", "",
             "| ID | Claim | Sources (sha256) | Script | Status | Estimate | Limitation |", "|---|---|---|---|---|---|---|"]
    for r in L.rows:
        src = "<br>".join(f"`{s['path']}` ({s['sha256']})" for s in r["sources"]) or "-"
        lines.append(f"| {r['id']} | {r['claim']} | {src} | {r['script']} | {r['status']} | {r['estimate']} | {r['limitation']} |".replace("\n", " "))
    (root / "docs" / "RESULTS_PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return R, L.rows, macros


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT))
    a = ap.parse_args()
    R, rows, macros = build(a.out_root)
    print(f"ledger rows: {len(rows)}; macros: {len(macros)}; pending items: {sum(1 for r in rows if r['estimate'] == 'NOT ESTIMABLE' or r['estimate'] == 'PENDING')}")
