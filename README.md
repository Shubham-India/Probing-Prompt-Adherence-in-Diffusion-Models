# Guide Before Commitment

**When does a diffusion model decide what it draws, and what does classifier-free guidance change?**

A pre-registered study of *when* compositional prompt constraints (object presence, count, colour binding, spatial relation, relative size) become fixed along the denoising trajectory of Stable Diffusion 1.5, how classifier-free guidance (CFG) changes that timing, what cross-attention reveals about it, and whether a guidance schedule timed by the measurement helps.

> **Status.** The development split is complete. **6 of 9 test shards exist, so every test-split result in this repository is partial and exploratory.** The registered confirmatory analysis has **not** been run; it needs the missing shards (TEST_CORE shard 2, TEST_COMMIT shard 2, TEST_CAG_SIZE shard 0).

- Paper: [`paper/main.pdf`](paper/main.pdf), with LaTeX source in [`paper/main.tex`](paper/main.tex).
- Every number and its evidence file: [`PAPER_STATUS.md`](PAPER_STATUS.md).

---

## Research question and approach

Prompt adherence is usually reported as one score per image. Compositional failures are constraint-specific, though. If different constraints are settled at different moments of sampling, then guidance strength, early diagnostics and pooled scores can all mislead. We ask four questions:

1. **When** does each constraint type stop depending on the remaining randomness of the trajectory?
2. **What does CFG change** about that timing, once the higher seed-to-seed agreement it causes is accounted for?
3. **Can attention tell?** Does the step at which cross-attention stops changing track the interventional measurement?
4. **Is the timing usable?** Does a guidance schedule whose switch step is read off development-set measurements keep adherence while raising diversity?

**Measurement: restart persistence.** Each image is generated with 40 DDIM steps. The latent is saved at steps t ∈ {6, 12, …, 36}, re-noised to an earlier step with fresh noise, and denoised again. Agreement between branch and base outcomes is corrected for the agreement of independent seeds:

C(t) = Σ_s (A_s(t) − A0_s) / Σ_s (1 − A0_s)

The step at which C first crosses 0.5 is t50. Two re-noising probes are used:
- a fixed eight-step jump (D8);
- an equal-noise-ratio probe (ER30).

A full resample from pure noise serves as a validity control.

**Benchmark.** Machine-generated scene graphs, split by scene into 100 DEV and 200 TEST scenes, plus a factorial colour × size × spatial family on 24 object pairs. Each scene has a canonical prompt, meaning-preserving paraphrases and single-field counterfactuals. Design: [`data/benchmark_v2/`](data/benchmark_v2), [`docs/ADDENDUM_C.md`](docs/ADDENDUM_C.md).

**Judge and audit.** The automatic judge combines OWL-ViT detection with CLIP colour classification at pinned revisions. A blinded, pre-declared human audit decided which constraint types may carry claims. **Only presence met the 0.85 accuracy criterion** (0.856); count, binding, spatial relation and size did not. There was a single annotator.

**Intervention: Commitment-Aligned Guidance (CAG).** Use w = 7.5 until a prompt-specific switch step s\* = ⌈max over the prompt's types of DEV t50⌉ + 2, then w_lo = 3; w_lo was chosen by a pre-declared HPS v2 quality guard. CAG is tested against:
- constant CFG 7.5 (A1);
- the reversed schedule with equal mean guidance (A4);
- constant guidance at CAG's mean (A6);
- a single global switch step (A3).

**Pre-registration.** Four hypotheses form one Holm family:

| Hypothesis | Claim |
|---|---|
| HA | Types differ in t50 |
| HD | Higher CFG moves t50 earlier |
| HE | CAG is non-inferior, more diverse and beats its reverse |
| HF | Attention stabilisation tracks t50 |

The decision rules, development-set constants and test manifests were frozen in a signed protocol ([`protocol/protocol_freeze.json`](protocol/protocol_freeze.json), [`configs/analysis_plan_r3.json`](configs/analysis_plan_r3.json)) before any test image was generated. Test workers refuse to run unless the protocol verifies.

## Findings (partial test split; exploratory unless stated)

| | Result | Caveat |
|---|---|---|
| **HD: CFG and timing** | Presence becomes persistent earlier with stronger guidance: t50 = 22.5 / 16.2 / 13.8 steps at w = 2 / 7.5 / 15; registered contrast −8.67 [95% CI −16.34, −4.72], Holm p = 0.046. The only registered rule that passed. | Not a chance-correction artefact, and survives subtracting the full-resample offset. But under an independent-judge-error stress test at the audited error rate it is not identifiable, and it reverses in the 10 counting scenes. Informed by 67 scenes. |
| **Attention (post hoc)** | On identical initial noise, higher CFG separates object-noun attention and moves attention stabilisation about 9.6 steps earlier for presence. Within a scene, objects the judge later misses already have less localised, more overlapping attention from step 1 (226 pairs, 159 scenes). | Associations, not interventions; maps exist only for canonical 40-step images. |
| **HF: attention vs persistence** | Rejected by the registered rule (type-order concordance 0.30). The binding signal (colour-word map) follows object layout and stabilises early, while binding persistence comes late. | Only presence has a validated judge. |
| **HA: type differences** | Inconclusive; exploratory because its DEV precision gate failed. DEV commitment times did **not** transfer to the test scenes (binding ≤ 6 on DEV vs 19.8 on TEST). | |
| **HE: CAG** | **Not supported**: not non-inferior to constant CFG (lower 90% bound −0.035 vs margin −0.03), and not better than its reverse. Matched-mean controls attribute its diversity gain to lower average guidance, not timing. | CAG used DEV switch steps that are mis-timed for binding on TEST. |
| **Full-resample control (DEV)** | Gate failed, but the registered gate statistic is unpassable at this sample size; null simulation median max\|C\| = 1.0. A real presence offset of +0.16 (p = 0.008) remains **unexplained**: fixed seed panel or base/branch generation difference. | Absolute t50 values are probe- and panel-relative. |

Other observations:
- From w = 2 to 15, whole-prompt satisfaction rises from 0.09 to 0.17, CLIP score rises, and DINOv2 seed diversity falls from 0.67 to 0.48.
- 20 and 40 steps do not differ in adherence.
- Satisfaction falls from 0.33 for 2-constraint prompts to at most 0.07 for 4 or more.

## Repository structure

```text
paper/                 LaTeX manuscript (main.tex, sections/, tables/, figures/, references.bib), compiled main.pdf,
                       make_paper_assets.py (DEV figures/tables/macros), make_overleaf_zip.py, check_paper.py
protocol/              signed protocol freeze (hashes of 103 bound files, frozen constants, TEST manifest hashes)
configs/               experiment and analysis-plan configuration (analysis_plan_r3.json = registered rules)
data/benchmark_v2/     scene-graph benchmark (prompts, lexicon, splits); data/validation_* = pilot labels
src/                   generation, restart/switch probes, judge, attention recording, CAG, frozen analysis (r3_analyze.py)
kaggle/                Kaggle notebook scripts (nb00 preflight ... nb05 HPS guard) and environment bootstrap
scripts/               manifests, merging, verification, protocol finalisation, confirmatory report (final_report.py)
partial/               analysis of the partial TEST split: notebook 06, partial_report.py, robustness.py (post hoc)
diagnostics/           post-hoc diagnostics: full_control.py (DEV full-resample control), attention.py
reports/               compact verified results: partial/ (partial_results.json, robustness.json), diagnostics/
docs/                  protocol addenda, decision log (R3_DECISION_LOG.md), runbooks, audits, evidence/ (DEV constants,
                       audit verdicts, HPS guard, baseline hashes)
tests/                 unit and integration tests (CPU), GPU identity script
GRAND_RESEARCH_BLUEPRINT.md   full research design and its revision history
report/, outputs/      Revision 1 pilot (28 prompts) report and outputs; see report/PILOT_README.md
```

Files under `src/`, `kaggle/`, `scripts/`, `configs/`, `data/` and `docs/ADDENDUM_C.md` are bound by the signed protocol. Do not edit them unless you are starting a new protocol revision; `python src/protocol_r3.py --verify` checks them.

## Setup

Python 3.13. PyTorch is pinned to the CUDA 12.4 build.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

On Linux or macOS use `.venv/bin/python`. Image generation and judging require a GPU and were run on Kaggle (2 × T4); see [`docs/KAGGLE_MASTER_RUNBOOK.md`](docs/KAGGLE_MASTER_RUNBOOK.md). Credentials are read from Kaggle Secrets or the environment and never stored in the repository.

## Reproducing results

**From the repository alone (no GPU):**

```bash
python src/protocol_r3.py --verify          # checks the signed protocol against the bound files
python -m pytest -q tests                   # unit/integration tests (about 5 minutes on CPU)
python paper/make_paper_assets.py           # DEV figures, tables and macros from docs/evidence/
python paper/check_paper.py                 # static checks: citations, labels, macros, inputs
python paper/make_overleaf_zip.py           # regenerates assets, checks, writes paper/paper_overleaf.zip
```

Upload `paper/paper_overleaf.zip` to Overleaf (pdfLaTeX + BibTeX, main file `main.tex`), or compile locally with `latexmk -pdf main.tex` or `tectonic main.tex` inside `paper/`.

The test suite currently reports 5 failures, in `test_fail_closed.py` (2), `test_stage_guard.py` (2) and `test_variant_review.py` (1). These tests encode the pre-freeze state, in which owner gates were false and the protocol unsigned. They fail because the protocol has since been signed and the variant-review gate set. The tests and the files they read are unchanged since the freeze commit `40ade4d`.

**From the run archives (not in the repository):** the generation and analysis archives are verified, checksummed Kaggle outputs kept as private datasets. With them:

```bash
python partial/partial_report.py --final <partial-analysis archive dir>          # registered analysis on present shards
python partial/robustness.py --final <meta dir> --records <merged/all_records.jsonl> --compare <second eval_rows.jsonl>
python diagnostics/attention.py --attention <attention dirs> --rows <eval_rows.jsonl> --partial <partial_results.json> --images-zip <TEST_CORE image zips>
python diagnostics/full_control.py --rows <DEV eval_rows.jsonl> --sims 1000
```

The compact outputs of these runs are committed under [`reports/`](reports). Once all nine test shards exist, the registered confirmatory analysis runs through `kaggle/nb04_merge_analyze_export.py` (PHASE = TEST) and `python scripts/final_report.py --final <archive>`, which replaces the confirmatory section of the paper without changing estimators, rules or constants.

## Documentation

| Topic | Document |
|---|---|
| Research design and revisions | [`GRAND_RESEARCH_BLUEPRINT.md`](GRAND_RESEARCH_BLUEPRINT.md) |
| Registered protocol addendum | [`docs/ADDENDUM_C.md`](docs/ADDENDUM_C.md) |
| Decisions and deviations, dated | [`docs/R3_DECISION_LOG.md`](docs/R3_DECISION_LOG.md), [`docs/DECISION_REGISTER.md`](docs/DECISION_REGISTER.md) |
| Judge audit protocol | [`docs/AUDIT_PROTOCOL_DEV.md`](docs/AUDIT_PROTOCOL_DEV.md) |
| Running on Kaggle | [`docs/KAGGLE_MASTER_RUNBOOK.md`](docs/KAGGLE_MASTER_RUNBOOK.md), [`docs/EXPERIMENT_RUNBOOK.md`](docs/EXPERIMENT_RUNBOOK.md) |
| Claim-to-evidence map and paper status | [`PAPER_STATUS.md`](PAPER_STATUS.md) |
| Earlier pilot (Revision 1) | [`report/PILOT_README.md`](report/PILOT_README.md), [`report/report.md`](report/report.md) |

## Limitations

- **Incomplete test split.** No confirmatory claim is made. The HE contrasts against constant guidance rest on 67 scenes from one CAG shard.
- **Judge.** Only presence is validated. HD depends on judge errors not being independent across seeds, which the audit did not measure.
- **Estimator.** The full-resample presence offset is unexplained. Commitment times did not replicate from DEV to TEST.
- **Scope.** One model (SD 1.5), one sampler, synthetic prompts. Size-family scenes share 11 object pairs.
- **Post-hoc analyses.** These are labelled as such throughout and not corrected for multiplicity.
- **Data not distributed.** Generated images, latents, attention maps and raw human labels are not in the repository.
- **AI assistance.** Code, analyses and manuscript drafts were produced with AI assistance under the author's direction. The DEV audit labels were produced without AI assistance.
