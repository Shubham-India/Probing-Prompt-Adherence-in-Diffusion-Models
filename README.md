# Guide Before Commitment

**When does a diffusion model decide what it draws, and what does classifier-free guidance change?**

A pre-registered study of *when* compositional prompt constraints (object presence, count, colour binding, spatial relation, relative size) become fixed along the denoising trajectory of Stable Diffusion 1.5. It asks how classifier-free guidance (CFG) changes that timing, what cross-attention reveals about it, and whether a guidance schedule timed by the measurement helps.

> ## Status (read this first)
> - The **development (DEV) split is complete**.
> - **6 of 9 test shards exist**, so **every test-split result here is partial and exploratory**.
> - The registered **confirmatory analysis has not been run**. It needs the 3 missing shards: TEST_CORE shard 2, TEST_COMMIT shard 2, TEST_CAG_SIZE shard 0.
> - Only **object presence** has a judge that passed the human audit. Count, binding, spatial and size claims are exploratory.
> - Of four registered hypotheses, **one rule passed (H2, CFG moves presence commitment earlier)**. H1 is inconclusive, **H3 (our guidance schedule, CAG) is not supported**, and **H4 (attention tracks commitment) is rejected**.

---

## 1. Judge quick-start (2 minutes)

| I want to see… | Go to |
|---|---|
| The paper | [`paper/main.pdf`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/paper/main.pdf) (LaTeX: [`paper/main.tex`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/paper/main.tex)) |
| Where every number in the paper comes from | [`PAPER_STATUS.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/PAPER_STATUS.md) |
| The four hypotheses and their frozen decision rules | [`configs/analysis_plan_r3.json`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/configs/analysis_plan_r3.json), [`protocol/protocol_freeze.json`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/protocol/protocol_freeze.json) |
| The headline results table | [`reports/partial/hypothesis_summary.csv`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/reports/partial/hypothesis_summary.csv), [`reports/partial/partial_report.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/reports/partial/partial_report.md) |
| Every decision and why it was made | [`docs/R3_DECISION_LOG.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/R3_DECISION_LOG.md) |
| The full research design | [`GRAND_RESEARCH_BLUEPRINT.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/GRAND_RESEARCH_BLUEPRINT.md) |
| The core algorithms | [`src/commitment.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/commitment.py), [`src/branch.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/branch.py), [`src/cag.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/cag.py), [`src/r3_analyze.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/r3_analyze.py) |

---

## 2. Research question

Prompt adherence is usually reported as one score per image. Compositional failures are constraint-specific, though. If different constraints are settled at different moments of sampling, then guidance strength, early diagnostics and pooled scores can all mislead. We ask four questions, each tied to one hypothesis:

| # | Question | Hypothesis |
|---|---|---|
| 1 | **When** does each constraint type stop depending on the remaining randomness of the trajectory? | **H1**: binding and presence differ in commitment step t50 |
| 2 | **What does CFG change** about that timing, once the higher seed-to-seed agreement it causes is accounted for? | **H2**: t50(presence, w=15) − t50(presence, w=2) < 0, by at least 3 of 40 steps with a CI excluding 0 |
| 3 | **Is the timing usable?** Does a guidance schedule switched at a measured step keep adherence while raising diversity? | **H3**: CAG is non-inferior to constant CFG, more diverse, and better than its reverse |
| 4 | **Can attention tell?** Does the step at which cross-attention stops changing track the interventional measurement? | **H4**: attention stabilisation ordering matches the t50 ordering |

Why it matters: the guidance scale and its schedule are the simplest inference-time controls for trading adherence against diversity. Knowing when guidance acts shows where it can be reduced safely.

**Short answers (partial test split; see caveats in Section 6):**
1. H1: inconclusive. Its DEV precision gate failed, so it is exploratory.
2. H2: supported. Stronger CFG makes object presence persistent about 8.7 steps earlier (Section 6).
3. H3: not supported. CAG is not non-inferior, and its diversity gain is explained by lower mean guidance.
4. H4: rejected by its registered rule. Attention moves in the same direction as persistence under CFG, but it does not reproduce the type ordering.

---

## 3. Method

### 3.1 Generation setting
Stable Diffusion 1.5 (pinned revision `f03de327`), DDIM with η = 0, 40 steps, fp16, 512×512.
CFG: ε̂ = ε(x, ∅) + w · (ε(x, c) − ε(x, ∅)).

### 3.2 Restart persistence (the measurement)
Each base image is generated once. The latent is saved at steps t ∈ {6, 12, 18, 24, 30, 36}, re-noised to an earlier step j with fresh noise, and denoised again (2 branches per base):

x′_j = √r · x_t + √(1 − r) · ε, with r = ᾱ(j) / ᾱ(t)

which has exactly the forward marginal at step j. A branch is therefore a valid state of the sampler and only the injected noise is new.
- **D8**: fixed eight-step jump, j = max(t − 8, 0).
- **ER30**: equal-noise-ratio probe, r ≈ 0.30.
- **FULL**: a full resample from pure noise, used as a validity control.

For a constraint outcome (satisfied or not), agreement A between a base and its branches is corrected for **chance agreement** of independent seeds of the same prompt:

A0 = [o(o−1) + (n−o)(n−o−1)] / [n(n−1)]  (o of n seeds satisfy the constraint)

C(t) = Σ (A − A0) / Σ (1 − A0), pooled over scenes
t50 is the first step at which C(t) crosses 0.5, found by interpolation with left/right censoring.

Chance correction is needed because raw agreement is inflated whenever seeds already agree, and CFG makes them agree more.

### 3.3 Attention stabilisation (H4)
Cross-attention maps are recorded for object and attribute nouns. The stabilisation step t_stab is where half of the cumulative cosine-velocity path of the map is reached. We also compute localisation (top-10% mass) and noun-to-noun overlap (histogram intersection).

### 3.4 Commitment-Aligned Guidance (CAG)
Use w_hi = 7.5 until a prompt-specific switch step, then w_lo = 3:

s\* = min(⌈max over the prompt's constraint types of DEV t50⌉ + 2, 40)

This gives s\* = 12 for presence, binding and spatial prompts and s\* = 21 for count prompts. w_lo was chosen by a pre-declared HPS v2 quality guard. Arms:

| Arm | Schedule |
|---|---|
| A1 | constant w = 7.5 |
| A2 | CAG |
| A3 | single global switch step |
| A4 | reversed schedule, equal mean guidance |
| A6 | constant guidance at CAG's mean |

### 3.5 Judge and human audit
- **Judge:** OWL-ViT detection (presence threshold 0.10, count 0.15, NMS 0.3, distinct IoU 0.5), CLIP colour classification over 10 colours, size ratio 1.5, spatial relation by box centre. Quality metrics: DINOv2 diversity, CLIP score, HPS v2.
- **Audit:** blinded, stratified and population-weighted, with a pre-declared 0.85 point-accuracy threshold and a single annotator. **Only presence passed (0.856).** Count, binding, spatial relation and size did not, so claims about them are exploratory.
- **Judge-error stress test:** if the judge flips outcomes independently with error e, A_obs = (1 − 2e)² A + 2e(1 − e). We invert this to get C^(e) and report H2 under assumed e.

### 3.6 Statistics
- Scene-level bootstrap (10,000 draws, fixed keys), because images of one scene are correlated.
- Holm correction over the 4 hypotheses (a missing p-value counts as 1).
- H3 non-inferiority margin −0.03 on a 90% CI, with an intersection–union test for the combined rule.
- Two-sided p for H2 is min(1, 2 · min p).

### 3.7 Benchmark
Machine-generated scene graphs, split by scene into DEV and TEST, plus a factorial colour × size × spatial family on 24 object pairs. Each scene has a canonical prompt, meaning-preserving paraphrases and single-field counterfactuals, so a failure can be attributed to one field.
Data: [`data/benchmark_v2/`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/tree/main/data/benchmark_v2). Design: [`docs/ADDENDUM_C.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/ADDENDUM_C.md).

### 3.8 Pre-registration (internal, hash-signed)
Decision rules, DEV constants and TEST manifests were frozen in a signed protocol before any test image was generated. It binds 103 files by hash, and every TEST worker refuses to run unless the protocol verifies. This is an internal freeze, **not** an external registry such as OSF.

---

## 4. Pipeline

```text
nb00 preflight + DEV freeze  ->  DEV generation + judge  ->  human audit  ->  nb05 HPS guard
        ->  protocol freeze (signed)  ->  TEST workers nb01 / nb02 / nb03  ->  nb04 merge + analyze + export
        ->  scripts/final_report.py (confirmatory)   |   partial/ (what we can run now)
```

Manifests are split into shards by scene group. Workers are resumable, archives are checksummed, and validators fail closed. Runs were executed on Kaggle (2 × T4); see [`docs/KAGGLE_MASTER_RUNBOOK.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/KAGGLE_MASTER_RUNBOOK.md).

---

## 5. Results at a glance (partial test split, exploratory)

Source: [`reports/partial/hypothesis_summary.csv`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/reports/partial/hypothesis_summary.csv).

| Hypothesis | Estimand | Estimate | Interval | p / Holm p | Registered-rule outcome |
|---|---|---|---|---|---|
| H1 | Δt50, binding − presence (steps) | +3.59 | 95% [−10.43, 13.06] | 0.565 / 1.0 | **Inconclusive** (exploratory: DEV precision gate failed, half-width 7.94 > 4) |
| H2 | Δt50, presence, w=15 − w=2 (steps) | **−8.67** | 95% [−16.34, −4.72] | 0.0116 / **0.046** | **Supported** |
| H3 | A2 − A1 adherence (margin −0.03) | −0.0155 | **90%** [−0.0354, +0.0031] | 0.4825 / 1.0 | **Not supported** (fails non-inferiority and beats-reversed) |
| H4 | type-order concordance | 0.30 | [0.20, 0.75] | 0.804 / 1.0 | **Rejected** (concordance ≤ 0.5) |

---

## 6. Findings, with the caveats that go with each

| Finding | Evidence | Caveat |
|---|---|---|
| **CFG moves presence commitment earlier (H2).** t50 = 22.5 / 16.2 / 13.8 steps at w = 2 / 7.5 / 15. | Registered contrast −8.67 [95% CI −16.34, −4.72], Holm p = 0.046. The only registered rule that passed. | Not a chance-correction artefact, and it survives subtracting the full-resample offset. But under the independent-judge-error stress test at the audited error rate it is **not identifiable**, and it **reverses in the 10 counting scenes**. It rests on 67 scenes. |
| **Equal-noise-ratio probe agrees in kind.** ER30 presence t50 = 13.3 (w = 7.5) vs 16.2 under D8. | `reports/partial/robustness.json` | Post hoc. Same order of magnitude with the expected shift. |
| **Attention responds to CFG (post hoc).** On identical initial noise, higher CFG separates the two object nouns' attention and makes stabilisation about 9.6 steps earlier for presence. Within a scene, objects the judge later misses already have less localised, more overlapping attention from step 1 (226 pairs, 159 scenes). | `reports/diagnostics/attention.json` | Associations, not interventions. Maps exist only for canonical 40-step images (221 scenes in total; 72 with maps at both w = 2 and w = 15). |
| **H4 rejected.** The colour-word attention map follows object layout and stabilises early, while binding persistence comes late. | type-order concordance 0.30 | Only presence has a validated judge. |
| **H1 inconclusive.** DEV commitment times did **not** transfer to TEST (binding ≤ 6 on DEV vs 19.8 on TEST). | `hypothesis_summary.csv` | Exploratory by its own DEV gate. |
| **CAG fails (H3).** Not non-inferior (lower 90% bound −0.035 vs margin −0.03) and not better than its reverse (A4 − A2 ≈ +0.001, CI [−0.025, 0.027]). A single global switch step (A3) lowers adherence (−0.031, 95% CI [−0.056, −0.006]). | Matched-mean control A6 shows the diversity gain is a **dose effect** of lower average guidance, not timing. | CAG used DEV switch steps that are mis-timed for binding on TEST. |
| **Full-resample control (DEV): gate failed, and the gate itself is unpassable.** Null simulation median max\|C\| = 1.0, with 0 of 1,000 null runs passing. | `reports/diagnostics/full_control.json` | A real presence offset of +0.16 (p = 0.008) remains **unexplained**: fixed seed panel or a base/branch generation difference. Absolute t50 values are probe- and panel-relative. |

Other observations:
- From w = 2 to 15, presence rises by about 0.11, whole-prompt satisfaction by about 0.08 (0.09 → 0.17), and DINOv2 seed diversity falls from 0.67 to 0.48. CFG trades diversity for adherence.
- 20 and 40 steps do not differ measurably in adherence.
- Whole-prompt satisfaction falls from 0.33 for 2-constraint prompts to at most 0.07 for 4 or more constraints.

### What we do NOT claim
- No confirmatory result. The registered analysis needs 3 more shards.
- No claim about count, binding, spatial or size timing beyond exploration. Their judges failed the audit.
- No claim that CAG, or any guidance schedule, improves adherence.
- No claim that attention maps explain commitment.
- The 163-vs-100 DEV scene count and the 85 / 111 / 67 / 222 / 134 scene counts are reconciled in `PAPER_STATUS.md`.
- The audit images come from the first DEV run at w = 7.5 only, and there was a single annotator.

---

## 7. Related work and literature

The exact bibliography is [`paper/references.bib`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/paper/references.bib), with the survey notes in `research_Papers.txt`. The most relevant works:

| Area | Work | How it relates |
|---|---|---|
| Guidance | Ho & Salimans, *Classifier-Free Diffusion Guidance* (2022) | The mechanism we vary (H2, H3) |
| Sampler | Song et al., *DDIM* (2021); Rombach et al., *Latent Diffusion* (2022) | Deterministic sampler and model family used |
| Restart / re-noising | Xu et al., *Restart Sampling* (2023) | Motivates our noise-injection probe; our probe measures outcome persistence, not sample quality |
| Stages of generation | Patashnik et al., *Localizing Object-level Shape Variations with Text-to-Image Diffusion Models* (2023); Zhang et al., *Prospect* (2023) | Layout and early structure form before details (basis of the H1 and H2 predictions) |
| Critical windows | Li et al., *Critical Windows* (2024) | Features settle in feature-specific windows |
| Attention control | Hertz et al., *Prompt-to-Prompt* (2023); Chefer et al., *Attend-and-Excite* (2023) | Cross-attention as a control and diagnostic signal (H4) |
| Compositional evaluation | Huang et al., *T2I-CompBench* (2023) | Compositional failure types behind our constraint taxonomy |
| Detector | Minderer et al., *OWL-ViT* (2022) | Object-presence and count judge |
| Image-text and features | Radford et al., *CLIP* (2021); Oquab et al., *DINOv2* (2024) | Colour classification, CLIP score, diversity |
| Quality | Wu et al., *HPS v2* (2023) | Pre-declared quality guard for w_lo |

To our knowledge no prior study combines an interventional per-constraint timing measurement, a test of how CFG moves it, an attention comparison, and a measurement-derived schedule with matched controls. This is a gap statement from our own search, not an exhaustive survey. Full texts of the closest references were not all read in full (see `PAPER_STATUS.md`).

---

## 8. Repository map

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
reports/               compact verified results: partial/ and diagnostics/
docs/                  protocol addenda, decision log, runbooks, audits, evidence/ (DEV constants, audit verdicts, HPS guard)
tests/                 unit and integration tests (CPU), GPU identity script
report/, outputs/      Revision 1 pilot (28 prompts) report; see report/PILOT_README.md
GRAND_RESEARCH_BLUEPRINT.md   full research design and revision history
PAPER_STATUS.md               number-to-evidence map for the paper
```

### Key files, linked

| Purpose | File |
|---|---|
| Commitment and chance correction | [`src/commitment.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/commitment.py) |
| Restart branches | [`src/branch.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/branch.py) |
| Guidance schedules (CAG, A1–A6) | [`src/cag.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/cag.py), [`src/switch.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/switch.py) |
| Judge | [`src/evaluate.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/evaluate.py) |
| Attention recording / stability | [`src/attn_record.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/attn_record.py), [`src/attn_stability.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/attn_stability.py) |
| Benchmark and manifests | [`src/benchmark_v2.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/benchmark_v2.py), [`src/manifests.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/manifests.py) |
| Frozen protocol and verification | [`src/protocol_r3.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/protocol_r3.py) |
| Frozen analysis (bootstrap, Holm) | [`src/r3_analyze.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/src/r3_analyze.py) |
| Confirmatory report (needs all shards) | [`scripts/final_report.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/scripts/final_report.py) |
| Partial analysis | [`partial/nb06_partial_test_analysis.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/partial/nb06_partial_test_analysis.py), [`partial/partial_report.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/partial/partial_report.py) |
| Post-hoc robustness (stress test, offset, quality) | [`partial/robustness.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/partial/robustness.py), [`tests/test_posthoc_sensitivity.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/tests/test_posthoc_sensitivity.py) |
| Diagnostics | [`diagnostics/full_control.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/diagnostics/full_control.py), [`diagnostics/attention.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/diagnostics/attention.py) |
| Kaggle notebooks | [`kaggle/nb00_preflight_dev_freeze.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/kaggle/nb00_preflight_dev_freeze.py) … [`kaggle/nb05_dev_hps_guard.py`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/kaggle/nb05_dev_hps_guard.py) |

### Documentation index

| Document | What it contains |
|---|---|
| [`docs/R3_DECISION_LOG.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/R3_DECISION_LOG.md) | Numbered decisions (DL-01…DL-48), including the A0 estimator bug fix made before the test split |
| [`docs/DECISION_REGISTER.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/DECISION_REGISTER.md) | Decision register |
| [`docs/HYPOTHESIS_LEDGER.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/HYPOTHESIS_LEDGER.md) | Hypotheses and their status |
| [`docs/ADDENDUM_C.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/ADDENDUM_C.md) | Benchmark v2 design (bound by the protocol) |
| [`docs/AUDIT_PROTOCOL_DEV.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/AUDIT_PROTOCOL_DEV.md) | Human audit protocol |
| [`docs/EXPERIMENT_RUNBOOK.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/EXPERIMENT_RUNBOOK.md), [`docs/KAGGLE_MASTER_RUNBOOK.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/KAGGLE_MASTER_RUNBOOK.md) | How the runs were executed |
| [`docs/evidence/`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/tree/main/docs/evidence) | DEV constants, audit verdicts, HPS guard, baseline hashes |
| [`docs/R0_R1_AUDIT.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/R0_R1_AUDIT.md), [`docs/R2A_AUDIT_REPORT.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/docs/R2A_AUDIT_REPORT.md) | Earlier revision audits |
| [`report/PILOT_README.md`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/report/PILOT_README.md) | The Revision 1 pilot (28 prompts), kept for history |

Files under `src/`, `kaggle/`, `scripts/`, `configs/`, `data/` and `docs/ADDENDUM_C.md` are bound by the signed protocol. Do not edit them unless you are starting a new protocol revision; `python src/protocol_r3.py --verify` checks them.

---

## 9. Setup and reproduction

Python 3.13. PyTorch is pinned to the CUDA 12.4 build.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

On Linux or macOS use `.venv/bin/python`. Image generation and judging need a GPU (Kaggle 2 × T4 was used). Credentials are read from Kaggle Secrets or the environment and are never stored in the repository.

**From the repository alone (no GPU):**

```bash
python src/protocol_r3.py --verify
python partial/robustness.py
python diagnostics/attention.py
python diagnostics/full_control.py
python paper/make_paper_assets.py
python paper/check_paper.py
python -m pytest tests -q
```

**Not reproducible from this repository alone:** the generated images, latents, model weights and human-label files. They are large or private and are deliberately excluded from version control. The compact verified outputs are in `reports/`.

**Completing the confirmatory analysis:** run the 3 missing TEST shards through `kaggle/nb01…nb03` with the frozen protocol, then `kaggle/nb04_merge_analyze_export.py` and `scripts/final_report.py`.

---

## 10. Limitations

- Partial test split (6 of 9 shards): no confirmatory claim.
- Only presence has an audit-validated judge, and the audit had a single annotator.
- The H2 contrast is sensitive to correlated judge error: it is not identifiable at the audited error rate and reverses in counting scenes.
- An unexplained +0.16 presence offset in the DEV full-resample control, and the registered gate for that control is unpassable at this sample size.
- DEV-derived switch steps do not transfer to TEST for binding.
- One model (Stable Diffusion 1.5), one sampler (DDIM), 40 steps.
- The pre-registration is an internal hash-signed freeze, not an external registry.

## 11. Use of AI assistance

An AI coding assistant was used during this project for code, analysis scripts, and drafting, including producing machine-generated first-pass labels for a small pilot. All experimental design decisions, the freeze, the runs and the interpretation of results were directed and checked by the Me. Every reported number traces to a file in `reports/` through `PAPER_STATUS.md`. The human audit labels that decide which claims are allowed are human labels.

## 12. Papers in this repository

- **Main paper:** [`paper/main.pdf`](https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models/blob/main/paper/main.pdf), the full paper with all tables and figures.
- A short conference-format version exists separately and is not part of this README's claims. Both versions state the same partial-test caveat.

## 13. Citation

If you use the code or the protocol design, cite the paper above. Citation metadata will be added once author information is final.
