# GRAND RESEARCH BLUEPRINT
## Where Does the Prompt Go? — a research programme for text-to-image prompt adherence
**Revision 2 (2026-10-14).** An independent research-discovery phase (§6A) re-examined the headline of Revision 1. The recommended central paper changed from *early attention diagnostics* to *commitment timing of compositional constraints* (hypotheses HA, HB, HC; §1, §7, §8). Revision 1 analyses that still apply are kept and marked; superseded text is labelled.

**Contents**
1. Executive research decision (Revision 2)
2. Repository audit and verified baseline
3. Official problem-statement compliance checklist
4. Literature review and prior-art matrix
5. Novelty audit and unresolved research gaps
6. Ranked candidate research directions (Revision 1 set)
6A. Independent research-discovery phase (Revision 2)
7. Recommended central paper and formal contributions (Revision 2)
8. Primary hypotheses and preregistration plan (Revision 2)
9. Comprehensive experiment catalogue
10. Observation and interpretation matrices
11. Benchmark V2: schema and generation plan
12. Evaluation and human-validation system
13. Method design and feasibility (the interventional probe, not a claimed method)
14. Baselines, controls and ablations
15. Sample-size and statistical-analysis plan
16. Two-GPU compute and runtime budget
17. File-by-file implementation plan
18. Figure and table design
19. Result-interpretation handbook
20. Titles, contribution statements and the paper outline
21. Risks, limitations, alternative outcomes and stop conditions
22. Prioritised execution roadmap (phases and gates; no calendar times are asserted)
23. Verified references and official code repositories


*Planning document. Written 2026-10-14 (Revision 2 the same day). No new experiment was run to write it, no source file was modified, and **no future experimental result appears in it as fact**. Every number below is either (a) read from a saved file in this repository, (b) computed from those saved files by a read-only script (stated where it happens), or (c) a design parameter I propose.*

### Label legend (used throughout)
| Label | Meaning |
|---|---|
| **VERIFIED EXISTING RESULT** | Read from a saved output file in this repository (path given) |
| **VERIFIED PRIOR WORK (FT)** | Statement checked in the full text of a paper PDF in `Researcha-thon Project/` |
| **VERIFIED PRIOR WORK (ABS)** | Statement checked only on an arXiv/venue abstract page or search snippet, 2026-10-14. Full text not read |
| **PROPOSED HYPOTHESIS** | A prediction I recommend testing; not evidence |
| **PLANNED CONFIRMATORY TEST** | A test whose analysis is frozen before the data exist |
| **POSSIBLE FUTURE OUTCOME** | Something an experiment *might* show; never stated in the past tense |
| **EXPLORATORY RESULT** | Any analysis not pre-specified; cannot support a headline claim |
| **UNVERIFIED CLAIM** | Could not be confirmed with the sources available |
| **DERIVED FROM PILOT** | Computed from the existing 28-prompt pilot by formula; valid only if the stated assumption holds |

---

## 1. Executive research decision

### 1.1 The decision (Revision 2, 2026-10-14)
> **Revision 1** (earlier today) recommended *early cross-attention diagnostics* (Direction A) as the headline. An **independent discovery phase (§6A)** then examined 15 additional directions, mined the pilot data for unexplained patterns, and re-checked prior art. **Revision 2 replaces the headline.** The old text of §7–§8 is superseded where the two conflict; the Revision 1 analyses survive as supporting analyses.

**Recommended central paper (Revision 2):** *When, and from what, is the fate of a compositional constraint decided during denoising?* The paper is a **mechanistic and evaluation study** (not a new-algorithm paper) built on three links of evidence:
1. **Causal timing** (new, primary): per-constraint **commitment curves** measured by *interventional re-noising* of the sampling trajectory (restart-style branching), independent of any attention readout (§6A ND01, E23).
2. **Readout vs forecast** (reframed Rev. 1 H1): *when* do attention features, prompt structure and the initial noise carry information about the final outcome — **before** or only **after** the commitment point? (HB, E05 + E25)
3. **Time-gated causal handle** (reframed Rev. 1 H3): does a targeted attention edit work only when applied **before** the constraint's commitment point, relative to matched random edits? (HC, E12)

Benchmark V2, the evaluator audit and the controlled CFG × steps × complexity measurement from Revision 1 are **kept unchanged** (they are required by the problem statement and are prerequisites for the new questions). Supporting mechanisms (layer/head knockouts, guidance attribution, fragility) are P2 (§6A, E24, E26, E27).

### 1.2 Why Revision 1's headline was replaced (verified or derived reasons)
1. **Label flaw found in the pilot data (EXPLORATORY, read-only analysis).** The Rev. 1 outcome "colour-binding correct" is `binding_ok`, which `evaluate.py: score()` sets to **false whenever the object is not detected**. In the 216 binding-category pilot images, **61% of the 195 failures contain at least one undetected requested object**; only 39% are both-present-but-wrong-colour; P(binding correct | both objects detected) = 0.21 versus 0.10 unconditional. The pilot attention margin separates the *mixed* label with AUC 0.77, but **0.61 for object presence alone and 0.72 for binding among images with both objects present (n = 89, 12 prompts)** (§2.7). A diagnostic for "binding" that is partly a diagnostic for omission would be mis-described.
2. **The nearest published work covers the headline.** *Diffusion Probe* (arXiv 2602.23783; ABS) predicts final quality from early attention; InitNO (CVPR 2024; ABS) scores initial noise by attention; the constraint-level gap is **unverified** (§5).
3. **The pilot shows strong seed determinism**, so any attention "forecast" may be reading information already in the initial noise: the per-(prompt, seed) outcome correlates **0.79** between CFG 7.5 and 15 (**0.72** after removing prompt means) and 20- vs 40-step outcomes agree in **87.5%** of cases (§2.7). A noise-only baseline is therefore mandatory — absent from Revision 1.
4. **The timing question is unanswered at constraint level and can be tested causally.** Critical-window theory exists (ICML 2024, ABS), early-layout and late-binding claims conflict (R&B vs SD-as-feature-extractor, ABS; CoBind's stage assumption, ABS), but I found no interventional, constraint-type-specific commitment measurement on a text-to-image model (narrow search; **UNVERIFIED gap**).
5. **Statistical fit.** Commitment curves are measured within scenes with cheap local re-denoising rather than by predicting a rare binary label across scenes, and they do not depend on the unknown SE of a nested-model ΔAUC.
6. **Title fit.** "Where does the prompt go?" becomes *where in the trajectory does a prompt constraint get decided*.

### 1.3 What will *not* be claimed
No new attention-correction algorithm name; no state-of-the-art claim; no cross-architecture claim without replication; no causal claim from correlation; no claim that commitment curves characterise models other than SD 1.5 unless E16 replicates them; no claim of novelty beyond what §5/§6A verifies.

### 1.4 Ranked directions (Revision 2 top five; the full analysis is in §6A)
| Rank | Direction | Role in the paper |
|---|---|---|
| 1 | **ND01** Constraint commitment curves by interventional re-noising | **Primary (HA)** |
| 2 | **ND10** Outcome fragility / decision margin under micro-perturbation | Supporting (E24) |
| 3 | **ND03** Causal layer/head localisation of constraint types | Supporting (E26) |
| 4 | **ND02** Initial-noise determinism and noise-only readout | Supporting; mandatory baseline for HB (E25) |
| 5 | **ND04** Guidance attribution by constraint tokens | Supporting (E27) |
| — | Rev. 1 Direction A (attention diagnostics) | Reframed as HB (readout vs forecast) |
| — | Rev. 1 Direction D (targeted edit) | Reframed as HC (time-gated handle) |

### 1.5 What changed relative to Revision 1
New primary hypotheses **HA, HB, HC** (§8); new experiments **E23–E28** (§9); H1/H2/H3 retained as **supporting** hypotheses with a corrected outcome definition (conditional binding) and a noise baseline; E12 gains **onset-time arms**; compute matrices updated (§16.5b); new files (§17.1); new figures (§18.1); roadmap revised (§22b); references added (§23.1b).

### 1.6 Revision 1 decision (superseded; kept for traceability)
*Where this text conflicts with §1.1–1.5, §1.1–1.5 prevails. The Revision 1 analyses of the Benchmark V2, evaluator, statistics and compute remain in force.*

#### 1.6.1 Revision 1 decision paragraph
**Recommended central paper:** *Do early cross-attention trajectories carry information about constraint-level compositional failure that prompt structure alone does not — and is that information causally usable?* The paper is a **mechanistic and evaluation study**, not a new-algorithm paper. It rests on three things: (i) a **Benchmark V2** of machine-readable scene graphs with counterfactual and meaning-preserving prompt variants, split at scene-graph level; (ii) a **validated, multi-judge evaluator** with a blinded human audit; (iii) a **three-step evidence ladder** — association (does attention predict failure?), generalisation (on held-out scene graphs, beyond strong prompt-only baselines), and an **interventional probe** (does a targeted, matched-control attention edit move binding?). Intervention is used as a *causal probe*, **not** as a claimed new method, because the novelty audit (§5) does not establish that a conflict-triggered local attention correction is distinct from existing work.

#### 1.6.2 Revision 1 rationale
1. **The pilot's strongest signal is diagnostic, not interventional.** **VERIFIED EXISTING RESULT**: colour/object attention margin separates colour-binding success from failure (middle-stage AUC 0.77, 95% CI [0.68, 0.87]; late 0.79 [0.73, 0.87]; within-prompt 0.76 and 0.80; `outputs/main/results/attention_auc.csv`) while the fixed global re-weighting did not improve overall success (+0.02, CI [−0.04, +0.09]; `outputs/intervention/results/intervention_numbers.json`).
2. **But the pilot cannot yet support a diagnostic claim.** That AUC comes from **216 images of only 12 prompts** (6 binding + 6 multi prompts that contain two objects; `analyze_attention.py`, `binding_df`), which are at most 12 independent scene constructions. Binding outcomes inside a prompt are strongly correlated (**DERIVED FROM PILOT**: one-way ICC of `binding_ok` across prompts = 0.52, `per_image.csv`), so effective sample size is far below 216. The diagnostic needs *many more scene graphs and a held-out test*.
3. **The competing "new method" route is crowded.** At least a dozen published training-free methods act on attention, text embeddings or latents to fix binding (§4–§5). A new-algorithm claim would need a fair comparison against several of them on a common benchmark and model — beyond a student programme's compute and reproduction budget.
4. **The competing "early prediction" route has a close neighbour, but a plausible gap.** *Diffusion Probe* (arXiv 2602.23783, CVPR 2026 per its arXiv page) predicts overall final-image quality from early cross-attention statistics (**ABS**). Its abstract page does not mention constraint-level failures (binding, counting, spatial). The gap — *constraint-specific* prediction, *interpretable* features, *scene-graph-held-out* evaluation against *prompt-only baselines*, plus a counterfactual and interventional follow-up — is a **candidate gap, UNVERIFIED until the Diffusion Probe full text is read** (the PDF is not in the project folder).
5. **The study still yields a publishable contribution if every method-like part fails.** A rigorous negative result ("attention adds nothing beyond prompt structure and CFG on held-out scenes"), plus Benchmark V2 and an evaluator audit, is a defensible contribution on its own (§21.3).

#### 1.6.3 Revision 1 non-claims
No new attention-correction algorithm name; no state-of-the-art claim; no cross-architecture claim without a replication; no causal claim from correlation; no claim that a benchmark is novel merely because it is larger.

#### 1.6.4 Revision 1 ranked directions (§6)
| Rank | Direction | Decision |
|---|---|---|
| 1 | **A** Temporal failure prediction (with **G** evaluation reliability as P0 enabler and **B**'s counterfactual benchmark as the data engine) | **Primary** |
| 2 | **B** Counterfactual / meaning-preserving prompt robustness | Supporting hypothesis (H2); partial overlap with MetaLogic, HRS-Bench |
| 3 | **C** Constraint-complexity interactions; **F** guidance/steps/compute | Required by the problem statement (CFG + second factor); run on the same data |
| 4 | **D** Failure-triggered intervention, **E** token/stage/layer ablations | Interventional probe (H3) only; algorithm novelty **not** claimed |
| 5 | **H** Generalisation | SDXL replication P2; SD3 generation+evaluation only, deferred |

---

## 2. Repository audit and verified baseline

### 2.1 What exists (actual paths inspected)
Project root `D:\prompt_Project`. There is **no git repository** (`git status` fails), so version history is not available; file timestamps are the only provenance.

| Area | Paths (inspected) | Notes |
|---|---|---|
| Problem statement | `Research-a-thon2.0 PS.pdf` | 8 pages; requirements listed in §3 |
| Papers | `research_Papers.txt` (21 lines); `Researcha-thon Project/*.pdf` (28 PDFs) | Folder contains papers **not** in the list (EDM, Universal Guidance, Gather-and-Bind, Unified Compositional Controller, AnchorSteer, robustness paper, regional prompting, prompt engineering) and **lacks PDFs** for list items 20–21 (ImageReward, HPS v2, DrawBench, PartiPrompts). Their abstracts were verified online (ABS) |
| Config / data | `configs/experiment.json`; `data/prompts.json`; `data/validation_subset.json`; `data/validation_labels.json`; `data/validation_subsets_images/` (40 PNG); `data/data.md` | Benchmark built by `src/build_benchmark.py` |
| Source | `src/generate.py`, `evaluate.py`, `attention.py`, `intervene.py`, `analyze.py`, `analyze_attention.py`, `analyze_intervention.py`, `validate_metrics.py`, `report_numbers.py`, `figures.py`, `build_benchmark.py` | Read in full |
| Tests | `tests/test_logic.py` (6), `tests/test_attention_gpu.py` (1) | `outputs/test_results.txt`: `7 passed` |
| Outputs | `outputs/main` (1008 images, `samples/detections/eval.jsonl`, 504 attention `.npz`, `results/`, `figures/`), `outputs/intervention` (504 images), `smoke/`, `intervention_smoke/`, `e2e/`, `e2e_intervention/`, logs | `outputs/outputs.md`, `outputs/intervention/intervention.md` are documentation |
| Report layer | `report/protocol.md`, `report_template.md`, `report.md`, `validation_template.md`, `validation.md`, `intervention_template.md`, `defense_template.md`, `defense.md`, `speaker_notes.md`, `presentation.pptx`, `audit.md`, `demo_cheatsheet.md`, `build_report.py`, `make_slides.py`; `README.md`, `README_template.md` | See freshness table |
| Environment | `requirements.txt` | torch 2.6.0+cu124, diffusers 0.41.0, transformers 5.19.0 |

### 2.2 Documents that disagree with saved outputs or with each other (**VERIFIED**, by timestamp and content comparison)
| File | Problem | Evidence |
|---|---|---|
| `report/report.md`, `report/defense.md`, `report/speaker_notes.md`, `report/presentation.pptx` | **Stale.** Generated 2026-10-10/11, *before* the final validation labels were merged on 2026-10-12. They still quote the old validation figures (e.g. binding n = 35/36, accuracy 0.83/0.81) | `report.md` line "binding (colour) \| 35 \| 0.81 \| 0.83 \| 0.89"; final labels give binding n = 37, accuracy 0.89 (`outputs/validation_metrics.txt`) |
| `report/validation.md` | **Stale** hand-written version (timestamp 10-10 02:09, n = 59/36/11/10) predating `validation_template.md` (10-12) | content vs `outputs/validation_metrics.txt` |
| `README.md` | Diverges from `README_template.md`; first line is a stray paste from `data.md`; states the final end-to-end run is **"Not completed"** although `outputs/e2e.log` contains `E2E-DONE` (1 match) and 0 `Traceback`; quotes a 28.78 s test run while `outputs/test_results.txt` records 22.87 s | `grep` of `e2e.log`; `test_results.txt` |
| `report/presentation [Autosaved].pptx` | PowerPoint autosave artefact | file listing |
| `outputs/pilot_detections_noembed.jsonl.bak`, `outputs/pilot_eval/` (empty) | Obsolete / empty | file listing |
| `data/validation_labels.json` | The final labels differ from the first draft at images 2, 9, 29, 35 (second pass) and 26, 32, 38 (owner revisions); provenance is recorded only in `report/validation_template.md`, not in the data file | label diff |

**Action (documentation hygiene, not research):** re-run `report/build_report.py` and `report/make_slides.py` after the next result change; delete autosave/backup artefacts; correct `README.md` e2e line.

### 2.3 Verified baseline (every number checked against a saved file)
| Quantity | Value | Source |
|---|---|---|
| Design | SD 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, fp16), DDIM eta 0, 512×512, empty negative prompt; CFG {2, 7.5, 15}; steps {20, 40}; seeds 0–5; 28 prompts (presence 4, binding 6, counting 6, spatial 6, multi 6) | `configs/experiment.json`, `data/prompts.json`, `samples.jsonl` |
| Images | **1008** main; **504** attention maps (all 40-step images); **504** intervention images (3 conditions × 28 × 6) | `results/counts.json`; `attention/*.npz` count; `intervention/samples.jsonl` |
| Primary success (all cells) | 275 / 1008 images succeed; cell means 0.22–0.32 overall; per-category cell maximum for binding+multi = 0.17 | `counts.json`, `adherence_by_cell.csv` |
| CFG 7.5 vs 2 (all prompts, steps pooled) | +0.07, 95% CI [−0.01, +0.15]; Wilcoxon p = 0.058; Holm p = 0.174; d_z = 0.40 | `cfg_contrasts.csv` |
| CFG 15 vs 7.5 | −0.01 [−0.07, +0.04], p = 0.873 | `cfg_contrasts.csv` |
| Steps 40 vs 20 | −0.02 [−0.07, +0.02], p = 0.428; CFG×steps interaction p = 0.37 (GEE) | `steps_contrasts.csv`, `interaction.json` |
| CLIP score | CFG 2 → 7.5: +1.88 [+1.21, +2.54], d_z 1.2; seed diversity 0.21 → 0.14 → 0.12 (CFG 2/7.5/15) | `clip_diversity_contrasts.csv`, `clip_diversity_per_prompt.csv` |
| Attention localisation (top-10% share) | CFG 7.5 vs 2: +0.013 / +0.014 / +0.014 (early/middle/late; CIs exclude 0); CFG 15 vs 7.5: +0.000 / −0.008 / −0.007 (middle and late CIs exclude 0) | `attention_cfg_effect.csv` |
| Attention vs success | bind-margin AUC 0.74 / 0.77 / 0.79 (early/mid/late); within-prompt 0.71 / 0.76 / 0.80; **n = 216 images, 12 prompts**; attention-only classifier AUC 0.71 vs category+CFG 0.74 vs both 0.75 (n = 504) | `attention_auc.csv`, `attention_classifier_auc.json` |
| Intervention (CFG 7.5, 40 steps) | control s=1 pixel-identical to baseline in 168/168; primary s=2 vs control +0.02 [−0.04, +0.09] (Holm 0.757); s=4 +0.02 [−0.05, +0.10]; presence-only outcome +0.09 [+0.01, +0.17] (Holm 0.021, **one of six outcomes, uncorrected across outcomes → exploratory**); CLIP s=2 +0.07 [−0.28, +0.47] | `intervention_numbers.json` |
| Evaluator validation (final labels, 40 images) | presence n 58, acc 0.93, recall 0.91; binding n 37, acc 0.89, recall 0.81; count n 10, acc 1.00; spatial n 9, acc 0.89, recall 0.50, **2 labelled positives** | `outputs/validation_metrics.txt` |
| Recorded generation time | 4.7 h summed over 1008 images (RTX 3050 Laptop 6 GB, partly shared with evaluation) | `samples.jsonl` `seconds` field |

### 2.4 Which claims are backed by what
| Claim in the current report | Backed by | Status |
|---|---|---|
| CFG 2→7.5 increases CLIP similarity and lowers diversity | 1008 generated images + CLIP computation | **VERIFIED EXISTING RESULT** (secondary outcomes, 28 prompts) |
| Adherence increases with CFG 2→7.5 | same | **Inconclusive** (CI includes 0) |
| Attention localisation rises 2→7.5 | 504 saved attention maps | **VERIFIED EXISTING RESULT** (correlational; attention averaged over layers/heads) |
| Colour/object overlap predicts binding success | 216 attention images | **EXPLORATORY RESULT** (12 prompts) |
| Global re-weighting does not raise overall success | 504 intervention images | **VERIFIED EXISTING RESULT** (one CFG, one step count, one rule) |
| Evaluator is 81–93% accurate | 40 manually labelled images | **Weak**: one non-independent annotator, one image sample from pilot seeds 0–1, thresholds swept on the same sample, spatial n = 9 |

### 2.5 Code and metric review findings (read from source)
| # | Finding | File / function | Severity |
|---|---|---|---|
| 1 | Spatial rule is **box-centre x only** (left iff centre_x(left) < centre_x(right)); no offset-vs-vertical test, no overlap/IoU test. T2I-CompBench's UniDet rule (per the T2I-CompBench++ text) is stricter, so spatial success is probably over-credited | `evaluate.py: score()` | High (metric validity) |
| 2 | Presence uses OWL-ViT score ≥ 0.10; count ≥ 0.15 after NMS 0.3; these were fixed a priori and a sweep on the 40-image validation sample showed insensitivity — but the sweep used the same sample that later set the labels | `evaluate.py: THRESHOLDS` | Medium |
| 3 | Colour binding = CLIP-B/32 top-1 among 10 colours on the **box crop**; background, lighting and multi-colour objects are uncontrolled; CLIP-B/32 is also the similarity metric, so both share biases | `evaluate.py: colour_probs()` | High |
| 4 | `box_ratio` in the attention analysis takes the **top-scoring box** (`boxes[0]`) while scoring uses the **distinct-box assignment** (`distinct_best_boxes`) — inconsistent object grounding | `analyze_attention.py: image_features()` | Medium |
| 5 | `bind_margin` exists only for prompts with exactly two objects (`len(objects) == 2`), so H1-type claims are restricted to 12 prompts | `analyze_attention.py` | High (scope) |
| 6 | Primary outcome differs by category (counting = count constraint only; others = all constraints); the **"all" row pools 28 prompts of arbitrary composition**, so it is not a population parameter | `analyze.py: load_table()` | Medium |
| 7 | Bootstrap uses one module-level RNG, so CIs shift in the last digit when the call order changes (observed: CLIP CI [1.19, 2.56] → [1.21, 2.54] after an unrelated bug fix) | `analyze.py: RNG` | Low |
| 8 | Wilcoxon on 4–6 prompt-level differences is approximate with ties; minimum attainable p is ~0.03–0.06 | `analyze.py: paired_contrast()` | Low (disclosed) |
| 9 | Attention maps include the 8×8 mid-block layers, bilinear-resized to 32×32 and equally weighted over layers/heads; the DAAM repository (per its `trace.py` summary, **ABS-level**) skips the 8×8 factor and resizes to 64×64 with bicubic interpolation — numbers are therefore not comparable to DAAM | `attention.py: generate_with_attention()` | Medium |
| 10 | Generation resume has **no file locking**: two processes writing one folder duplicated metadata lines once (fixed manually) | `generate.py: done_names()` | Medium (matters for two-GPU workers) |
| 11 | Hook-vs-plain bit-identity was shown for SD 1.5 SDPA only; it is **not** established for SDXL (different attention layout) or SD3 (joint attention) | `tests/test_attention_gpu.py` | Scope |
| 12 | Validation images come only from pilot seeds 0–1; all 28 prompts are in both the "exploration" and the "validation" role | `data/validation_subset.json` | Medium |

### 2.6 What can be reused and what needs independent confirmation
| Reusable as-is | Needs independent (new-scene) confirmation |
|---|---|
| Code skeleton: resume logic, paired seeds, hook that leaves images bit-identical, cluster-bootstrap and paired-contrast code, report-from-data build chain | Any claim about attention diagnostics (12 prompts only) |
| The 1008-image grid and 504 attention maps as the **pilot/dev** set | Benchmark-level claims (28 prompts, 4–6 per category, categories confounded with objects) |
| The control-path equivalence (rw1 ≡ baseline) as a unit test | The intervention null result (one rule) — cannot be generalised |
| CLIP/diversity CFG effect (strong, d_z 1.2) as a secondary-outcome anchor | Evaluator accuracy (needs a larger, independent, blinded audit) |

### 2.7 New verified pilot findings from the independent discovery phase (Revision 2)
*Read-only analyses of `outputs/main/results/per_image.csv`, `detections.jsonl`, `attention_features.csv`, `samples.jsonl`. All are **EXPLORATORY**: 28 prompts, detector-based labels with the §2.5 limitations. They motivate hypotheses; they are not evidence for them.*

| ID | Finding | Numbers | Caveat |
|---|---|---|---|
| F1 | **Outcome persistence across settings** | Mean pairwise outcome agreement across the six cells of the same (prompt, seed) = 0.821 (0.714 expected if cells were independent given the prompt rate). Per-(prompt, seed) success correlation: CFG 7.5 vs 15 = **0.79** (0.72 after removing prompt means); CFG 2 vs 7.5 = 0.53 (0.39); steps 20 vs 40 agree in 87.5% (corr 0.69) | Pooled over 28 prompts |
| F2 | **Failure composition (binding category, 216 images, 195 failures)** | 61% of failures have ≥ 1 requested object undetected; 39% have both present but wrong colour; P(both present) = 0.45; P(binding correct \| both present) = 0.21 vs 0.10 unconditional. Multi category: P(presence) 0.38, P(binding \| presence) 0.29, P(spatial \| presence) 0.62 | Detector misses count as omissions (recall 0.91 for presence in the audit) |
| F3 | **Misbinding mode (both objects detected, n = 98)** | both objects get the same colour 34%; one correct + a third colour 24%; both correct 21%; swapped only 12%; neither requested colour 8%. First-mentioned object has its colour 62% vs second-mentioned 38%. Among same-colour cases (n = 33) the first-mentioned colour wins 58%, second 39%, third 3% | CLIP-B/32 crop colour; small n |
| F4 | **Relation words and layout (spatial, 116 of 216 images with both objects detected)** | P(relation satisfied \| both detected) = 0.53; P(first-mentioned object on the left) = 0.53. P(A left \| prompt says A left) vs P(A left \| prompt says A right): car/bicycle 0.43 (n = 30) vs 0.50 (n = 26); cup/book 0.67 (n = 24) vs 0.64 (n = 22); cat/dog 0.62 (n = 8) vs 0.00 (n = 6) | Box-centre rule; cat/dog n tiny |
| F5 | **Counting compresses toward ≈ 3 (apple/ball prompts)** | Mean detected count for requested 2 / 3 / 4 = 2.82 / 3.12 / 3.75; exactly right 0.49 / 0.47 / 0.21; over-count 0.43 / 0.22 / 0.31; under-count 0.08 / 0.31 / 0.49 | OWL-ViT count at score ≥ 0.15 |
| F6 | **CLIP score and adherence** | Within-prompt correlation of CLIP score with success = 0.28 (25 prompts with variation); mean CLIP by CFG 30.42 / 32.30 / 32.55 | Weak association, not a proxy |
| F7 | **What the pilot attention margin predicts** | AUC of middle-stage bind margin: vs `binding_ok` 0.77; vs both-objects-present 0.61; **restricted to both-present images, vs `binding_ok` 0.72 (n = 89, 12 prompts)** | 12 prompts |
| F8 | **Pairing efficiency (common random numbers)** | For CFG 7.5 vs 15, pairing by (prompt, seed) removes 79% of the variance of the cell-level difference (ρ = 0.79) | Standard statistics, quantified here |

---

## 3. Official problem-statement compliance checklist
Source: `Research-a-thon2.0 PS.pdf` (text extracted). Status = current repository; "V2" = what this blueprint adds.

| # | Requirement in the PS | Current status | In this plan |
|---|---|---|---|
| 1 | Specific, testable hypothesis, mechanism, support/contradiction criteria — **before** main experiments | Present for the pilot (`report/protocol.md`) | H1–H3 + new addendum frozen before confirmatory data (§8) |
| 2 | Controlled prompt benchmark; justify categories, what each tests, how complexity is defined and controlled; systematic generation | 28 template prompts; complexity = constraint count | Benchmark V2 scene graphs, grammar-generated, with controlled variants (§11) |
| 3 | Vary CFG **and** ≥1 other inference-time parameter/mechanism, rest controlled | CFG × steps; attention re-weighting | CFG × steps retained (E11); intervention arms (E12) |
| 4 | Multiple seeds; mean, variance/CI, effect sizes, appropriate test; conclusion from statistical evidence across generations | Done with prompt-cluster bootstrap, Wilcoxon, GEE | Scene-graph-level inference; mixed/GEE models; power-based sample size (§15) |
| 5 | Metrics per constraint type: presence, counting, binding, spatial, overall | Implemented; weakly validated | Multi-judge + human audit (§12) |
| 6 | Generation quality (image-text similarity, aesthetic where appropriate), diversity, mechanistic metrics | CLIP score, seed diversity; attention metrics; **no aesthetic metric** | Add HPS v2 / ImageReward as quality proxies (ABS-verified existence) |
| 7 | Cross-attention maps; tokens vs timesteps; success vs failure; attention vs CFG | Done on 504 images | Done at scale on held-out scenes (E05–E07) |
| 8 | Expected outcome 4: simple inference-time intervention without retraining | Done; **not supported** | Targeted/gated arm vs controls as causal probe (E12) |
| 9 | Distinguish what results establish vs suggest vs cannot establish | Done in report | Result-interpretation handbook (§19) |

---

## 4. Literature review and prior-art matrix
*Search date 2026-10-14. FT = full text read from the provided PDF; ABS = abstract/search snippet only; CODE = repository page summarised.*

### 4.1 Foundations (FT; venue taken from the PDF header where it is printed, otherwise from the online check of 2026-10-10 noted in the row)
| Paper | Venue / ID | Role here |
|---|---|---|
| DDPM (Ho et al.) | NeurIPS 2020, arXiv 2006.11239 | Diffusion formulation |
| Improved DDPM (Nichol & Dhariwal) | arXiv 2102.09672 (venue not printed in the PDF; **UNVERIFIED here**) | Schedules, learned variance |
| DDIM (Song et al.) | ICLR 2021, arXiv 2010.02502 | Deterministic sampler used here |
| Latent Diffusion (Rombach et al.) | CVPR 2022 (CVF listing, checked 2026-10-10), arXiv 2112.10752 | SD architecture, cross-attention |
| Classifier-Free Guidance (Ho & Salimans) | arXiv 2207.12598 (short version NeurIPS 2021 workshop) | The guidance factor |
| EDM (Karras et al.) | NeurIPS 2022, arXiv 2206.00364 | Design-space view of samplers/guidance |
| Attention Is All You Need | NIPS 2017, arXiv 1706.03762 | Attention definition |
| CLIP (Radford et al.) | arXiv 2103.00020; ICML 2021 / PMLR 139 (checked 2026-10-10) | Colour classifier and similarity |
| Universal Guidance (Bansal et al.) | ICLR 2024 | Guidance by external detectors — related to detector-guided sampling |

### 4.2 Attention-based control, binding and compositional methods
| Paper (year, venue) | Model | Training / gradient? | What is modified | Trigger / timing | Benchmark & metrics | Closest similarity to this project | Remaining gap |
|---|---|---|---|---|---|---|---|
| **Prompt-to-Prompt** (Hertz et al., ICLR 2023) — FT + CODE | SD 1.4, LDM | No | Controller edits attention **after softmax** before ×V; Replace/Refine/**Reweight** (equaliser multiplies a token's attention) | `cross_replace_steps` fraction; optional local blend | Editing demos | My fixed global re-weighting is a variant (mine adds `log s` **before** softmax, so it re-normalises) | My rule is not P2P's exactly; no new claim |
| **Attend-and-Excite** (Chefer et al., TOG 42(4) 2023) — FT + CODE | SD 1.4/2.1 | **Gradient on latents**; no weights | Loss `max(0, 1 − max attn)` per subject token, max over tokens; 16×16 maps; Gaussian smoothing; `scale_factor 20`, thresholds {0:0.05, 10:0.5, 20:0.8}, steps < 25 | Iterative refinement at fixed steps if loss > threshold | Own 276 prompts; CLIP / BLIP-caption similarity | Shares "attention is the diagnostic" idea | Always-on, hand-set thresholds, no failure prediction study |
| **Divide & Bind** (Li et al., BMVC 2023) — FT | SD | Gradient on latents | Total-variation **attendance** loss + JSD **binding** loss; `t_end = 25`; λ = 1 when attributes given | Always-on to `t_end` | Multi-Object, COCO-Subject/Attribute; text-text sim, TIFA | Binding as attention-overlap alignment — **same objective family as my proposed correction** | Gradient-based; no gating |
| **Gather-and-Bind** (Fu & Cheng, *Computers & Graphics* 125, 2024, 104118) — FT | SD | Gradient on latents | **Entropy loss** (reduce max entropy over concept maps) + **JSD attribute-binding loss**; `L = L_ent + 3·L_bind`; 16×16 maps; updates 0–25 of 50 steps; spaCy parser | Always-on to `t_end` | A&E prompts + TIFA + FID; CLIP/BLIP | **Highest overlap** with a conflict-driven binding correction; also uses entropy of attention as the "scatter" signal that I analyse | Latent-gradient, always-on, 16×16; no trigger/mask/ablation of trigger |
| **SynGen** (Rassin et al., NeurIPS 2023) — ABS | SD | Gradient on latents | Syntax-parsed entity/modifier attention-overlap loss, first 25 of 50 steps | Always-on | Own sets; human eval | Same objective | Not read in full |
| **Token Merging / ToMe** (Hu et al., NeurIPS 2024) — ABS | not named | Optimises a composite token early | Merge object+attribute tokens; end-token substitution; entropy + binding loss | Early stages | T2I-CompBench, GPT-4o benchmark | Binding via tokens | Embedding-level |
| **Attention Modulation** (Wu et al., arXiv 2404.13899) — ABS | diffusion models | Training-free | Temperature in early **self**-attention; **object-focused masking** in cross-attention; **phase-wise dynamic weights** | Phase-wise | Not read | **Local masked cross-attention edit with phase schedule — close to a masked logit edit** | Only abstract read; overlap possibly large |
| **Magnet** (NeurIPS 2024) — ABS | SD | Training-free | Binding vector in text-embedding space | n/a | not read | embedding route | — |
| **StructureDiffusion** (Feng et al., ICLR 2023) — ABS | SD | Training-free | Sentence-structure-guided cross-attention; ABC-6K | n/a | ABC-6K | early baseline | — |
| **BoxDiff** (Xie et al., arXiv 2307.10816; venue not printed in the PDF) — FT abstract | SD | Training-free (latent optimisation) | Inner-box / outer-box / corner constraints on cross-attention | Per step | COCO-based | Spatial attention constraints | needs boxes |
| **MultiDiffusion** (Bar-Tal et al., arXiv 2302.08113; venue not printed in the PDF) — FT abstract | pre-trained diffusion | Training-free | Fuses multiple diffusion paths with region constraints | Per step | Own | Regional control | needs regions |
| **Regional Prompting for DiT** (Chen et al., arXiv 2411.02395) — FT abstract | **FLUX.1** | Training-free | Attention manipulation with prompt–mask pairs | not specified on abstract page | Not in abstract | Regional attention masks | DiT; needs masks (user/LLM) |
| **Unified Compositional Controller (UCC)** (Liu, Xu & Zhang, *Information Sciences* 745 (2026) 123380) — **FT** | **FLUX.1-dev** (DiT) | Training-free, LLM in loop | **DAPS** (LLM → layout + structured prompt), **SSF** (embedding-level fusion: anchor token `e' = α Σ e_subject + β Σ e_attr`, others zeroed), **TACA-TB** (temperature-sharpened cross-attention + text bridge) | Proactive (pre-denoising) + deeper-layer sharpening | COCO491, COCO-MIG | Attention sharpening ≈ my global re-weighting; "reactive vs proactive" framing | DiT, LLM parsing, different benchmarks; **not** a UNet failure-triggered method |
| **AnchorSteer** (Wang et al., arXiv 2607.26647; ICLR 2026 per PDF header) — **FT (abstract, intro)** | latent diffusion | Training-free | Semantic anchoring of initial noise (CLIP prior + LP-SDS); **Think–Erase–Retouch**: **VLM diagnosis mid-generation**, latent rollback, guided refinement | Triggered by VLM-detected deviation | GenEval, T2I-CompBench++ | **Mid-generation diagnose-then-correct** | VLM-based, latent rollback, not attention-conflict-based |
| **CoBind** (Chen et al., arXiv 2607.16307, July 2026) — ABS | multiple | Training-free | Composition graph; stage-aware schedule; contrastive cross-entity optimisation; **adaptive strength by current constraint satisfaction** | Stage-aware | T2I-CompBench++, GenEval | **Stage-aware + adaptive strength — closest to a "trigger"** | Only abstract read; mechanism details unknown |
| **EPIC** (Mun et al., arXiv 2605.11722, May 2026) — ABS | base generator + editor | Training-free | Predicate-guided search; failed predicates trigger local edit or resample | Image level | GenEval2: 34.16% → 71.46% prompt accuracy (abstract) | Verification-triggered correction | image-level, MLLM-heavy |

### 4.3 Diagnostics, analysis and guidance
| Paper | Status | Relevance |
|---|---|---|
| **DAAM** (Tang et al., ACL 2023) — FT | Upscale+aggregate cross-attention into word heat maps; also finds adjectives attend broadly | Basis of my extraction |
| **Diffusion Probe** (Cui et al., arXiv 2602.23783; "CVPR 2026" per arXiv page) — ABS | Predicts final image **overall quality** from early cross-attention statistics with a lightweight predictor (PCC > 0.7, AUC > 0.9 stated); multiple models; failures show diffuse/fragmented early attention | **Closest to Direction A**; constraint-level prediction not mentioned on the abstract page |
| **Text self-attention maps** (Kim et al., arXiv 2411.15236) — ABS | Test-time optimisation transferring syntactic relations from text self-attention into cross-attention; a search snippet reports a step-10 similarity association with presence/binding | Prompt-side signal I must use as a **competing baseline** |
| **Towards Understanding the Working Mechanism of T2I Diffusion** (arXiv 2405.15330) — ABS | Cross-attention maps take semantic shape early | Supports early-stage diagnostics |
| **SD as Task-Aware Feature Extractor** (arXiv 2507.07106) — ABS | Reports **later** timesteps capture attribute binding more precisely | A counter-hypothesis I must test (late may beat early) |
| **Guidance in a Limited Interval** (Kynkäänniemi et al., NeurIPS 2024) — ABS | Guidance harmful at high noise, helpful mid-range | CFG × timing |
| **CFG Weight Schedulers** (Wang et al., TMLR 2024) — ABS | Increasing schedules beat static | CFG as a schedule, not a scalar |
| **Rethinking spatial inconsistency in CFG** (CVPR 2024) — ABS (title only) | CFG and spatial layout | Spatial-CFG link |
| **When Do Diffusion Models Learn to Generate Multiple Objects?** (arXiv 2605.00273) — ABS | Reports difficulty order Attribution < Counting < Spatial (training-data coverage) | Context for category difficulty |

### 4.4 Benchmarks and evaluators
| Item | What it is (verified) | Limit relevant to this project |
|---|---|---|
| **GenEval** (Ghosh et al., NeurIPS 2023 D&B) — FT | 553 prompts, 6 tasks; detector + CLIP colour; per-image binary "all elements correct"; SD 1.5 overall 0.43 (position 0.04, attribute binding 0.06) | Detection-based — same family as my evaluator |
| **GenEval 2** (Kamath et al., arXiv 2512.16853, Dec 2025) — FT | Shows **benchmark drift**: GenEval deviates from human judgement by up to 17.7% absolute for current models (23,226 human annotations); introduces Soft-TIFA (AUROC 94.5% with Qwen3-VL-8B); code `facebookresearch/GenEval2` | **Direct warning that a GenEval-style evaluator can drift from human judgement** |
| **T2I-CompBench / ++** (Huang et al., NeurIPS 2023 D&B) — FT | 6,000 prompts (8,000 in ++); B-VQA (soft, multiplicative), UniDet spatial rule (offset + IoU conditions), CLIPScore, "3-in-1"; SD v1.4 colour 0.3765, spatial 0.1246 | Scores are soft; not directly comparable to binary success |
| **TIFA** (Hu et al., 2023) — FT | 4K prompts, 25K questions, 12 categories; QG + VQA | Needs an LLM + VQA |
| **DSG** (Cho et al., ICLR 2024) — ABS | Dependency-aware question graph | Reliability of QG/A |
| **GenAI-Bench / VQAScore** (Lin et al., ECCV 2024) — ABS | 1,600 prompts, 15,000+ human ratings; VQAScore | Holistic judge; drift-prone per GenEval 2 |
| **DALL-Eval** (Cho et al., ICCV 2023) — FT | PaintSkills: object recognition, counting, spatial | Early skill-based evaluation |
| **HRS-Bench** (Bakr et al., arXiv 2304.05390; venue not printed in the PDF) — FT | 13 skills, 50 scenarios, 9 models; human agreement 95% on average | Includes a robustness category (details not verified) |
| **Robustness of Prompt Adherence** (Vemishetty et al., arXiv 2507.08039) — FT abstract/intro | SD3 M, SD3.5 L, SD3.5 L Turbo, Janus Pro 1B/7B; **GPT-4o description round-trip**; simple binary images with shape + location; guidance 9.0 | Closest "robustness" paper; method is description-based, not scene-graph minimal pairs |
| **MetaLogic** (Shen et al., arXiv 2510.00796, ICFEM 2025) — ABS | **Metamorphic testing with logically equivalent prompts**; misalignment 59% (Flux.dev), 71% (DALL-E 3) | **Direct overlap with meaning-preserving wording tests** (Direction B) |
| **Prompt Engineering in Diffusion Models** (Witteveen & Andrews, arXiv 2211.15462) — FT | Measuring the effect of specific words/phrases with fixed seed and scheduler; art-oriented | Methodological precedent for seed-controlled word probing |
| **ImageReward** (arXiv 2304.05977; 137k expert comparisons) / **HPS v2** (arXiv 2306.09341; HPD v2 ≈ 798k preferences) — ABS | Human-preference scorers | Quality proxies (not adherence) |
| **DrawBench** (Saharia et al., 2022) / **PartiPrompts** (Yu et al., 2022; > 1,600 prompts) — ABS | General prompt suites | Not compositional-controlled |

### 4.5 Papers I could not read in full (ask the owner to add the PDFs)
Diffusion Probe, MetaLogic, CoBind, EPIC, SynGen, ToMe, Attention Modulation (Wu 2024), Kim 2024 (text self-attention), StructureDiffusion, VQAScore/GenAI-Bench, DSG, T2I-CompBench++ (journal version). **The novelty verdict in §5 stays provisional until at least Diffusion Probe, MetaLogic, CoBind and Attention Modulation are read at full-text level.**

---

## 5. Novelty audit and unresolved research gaps

### 5.1 Method for the audit
For each candidate claim I compare: **mathematical operation, trigger rule, spatial selection, timing, optimisation objective, experimental claim**. A claim is "supported as distinct" only if at least one of these differs *and* the difference changes what is tested.

### 5.2 Candidate claims
| ID | Candidate claim | Nearest prior art | Comparison | Verdict |
|---|---|---|---|---|
| C1 | Colour–object attention overlap is associated with binding correctness | A&E (max-attention), D&B / G&B / SynGen (overlap used as **loss**), Kim 2024 (similarity association) | Association is already the premise of those losses | **Not novel as an observation.** Contribution can only be rigour (held-out, baselines) |
| C2 | Early attention features predict **constraint-level** failures (binding, count, spatial, presence) on **held-out scene graphs**, beyond prompt-only baselines | Diffusion Probe (overall quality; ABS) | Different target granularity, interpretable features, held-out design, prompt-side baselines, counterfactual follow-up | **Candidate gap; UNVERIFIED** (full text unread) |
| C3 | Counterfactual **minimal-pair** benchmark with a declared edit target and unchanged-constraint preservation scoring | MetaLogic (equivalence), HRS-Bench robustness, Robustness paper (description round-trip) | They test equivalence/robustness, not **semantic edits with a known expected change** | **Candidate gap**, but my search was narrow; must be re-searched before claiming |
| C4 | **Conflict-triggered, spatially local attention-logit correction** (no gradients) | G&B/D&B/SynGen (same objective, latent gradient, always-on), Attention Modulation (masking + phase weights), CoBind (adaptive strength, stage-aware), UCC TACA (attention temperature), AnchorSteer/EPIC (triggered correction, other operator) | Operator, trigger and locality differ individually, but each *combination of two* appears in at least one paper | **Not established.** Treat as a probe, never as "new method" |
| C5 | CFG × constraint-complexity interaction with controlled scenes | GenEval/T2I-CompBench report per-category scores; my search found no direct CFG-scale sweep on these tasks | Descriptive | Modest, defensible as a measurement result |
| C6 | Detection-based evaluator disagreement can change apparent intervention gains | GenEval 2 (drift of GenEval vs humans) | I would measure the effect on *my intervention comparison* with a blinded audit | Supporting contribution |

### 5.3 Does a conflict-gated local correction add information, or re-implement a binding loss?
Gather-and-Bind's `L_bind` (JSD between attribute and noun maps) and my proposed conflict score (negative bind-margin) quantify the **same thing** (attribute/noun overlap). What differs is the *operator*: they take a gradient step on the latent; I would add a bounded logit bias to attention. That is an **implementation difference with unknown scientific consequence**; it is interesting only as a test of *whether the attention maps themselves are a causal handle* (§13). If the audit's full-text reading shows Attention Modulation or CoBind already uses a mask-based, schedule-dependent attention edit, even the operator difference disappears. **Therefore the plan proposes the correction strictly as an interventional probe with matched controls.**

### 5.4 Unresolved gaps worth the programme's effort
1. Is there **constraint-specific** information in early attention beyond prompt structure? (C2)
2. Do **semantic counterfactual edits** produce the intended and *only* the intended change? (C3)
3. How much of an apparent method gain is **evaluator-dependent**? (C6)
4. How does the **CFG effect depend on constraint count** under controlled scenes? (C5)

### 5.5 Rule applied
No algorithm name is proposed. If the full-text audit confirms overlap for C2, the paper falls back to **C3 + C5 + C6** (a benchmark/evaluation/mechanism study), which §21.3 shows is still a valid contribution.

### 5.6 Revision 2 addition to the novelty audit
The discovery phase (§6A) extends the audit to fifteen further directions, each with its closest papers and the verification level of each source. The headline claim of Revision 2 (constraint-type-specific interventional commitment curves, ND01) is an **UNVERIFIED gap**: it rests on a narrow web search and on abstract-level reading of the critical-windows, Restart-sampling, SDEdit and negative-prompt-timing papers (§23.1b). The full-text reading listed in §22 (Phase 0) must precede any novelty statement in the paper.

---

## 6. Ranked candidate research directions

*Scores are my judgement (1 = poor, 5 = strong), not measurements. "GPU cost" is expressed in **image counts**; wall-clock is deliberately not guessed (§16).*

### 6.1 Per-direction analysis

#### Direction A — Temporal failure prediction
| Item | Content |
|---|---|
| Research gap | Early cross-attention is known to predict **overall quality** (Diffusion Probe, ABS) and attention is used as a **loss** for binding (D&B, G&B, SynGen, FT/ABS). It is not established (UNVERIFIED) whether early attention predicts **constraint-specific** failure on **held-out scene graphs** beyond prompt-only features |
| Research question | RQ-A: After conditioning on prompt structure and CFG/steps, does early-stage cross-attention add held-out predictive information about colour-binding, counting, spatial and presence outcomes? How early? |
| Scientific value | High if positive (an early-stop/re-roll signal; evidence on where binding is decided). Informative if negative (attention is no better than prompt structure) |
| Novelty concerns | Diffusion Probe (closest), Kim 2024 (prompt-side similarity), SD-as-feature-extractor (late-timestep binding). Must read full texts |
| Feasibility | High: attention hook exists and is image-neutral; needs more scenes, not new methods |
| GPU cost | Core generation with hook: `S × V × k` images (§16); no extra pass because the hook is image-neutral |
| Evaluation difficulty | Medium: needs a trustworthy binding label (§12) |
| Sample size | ≈ 140 held-out binding-bearing scene graphs give a **single**-AUC 95% half-width of ≈ 0.08 (DERIVED FROM PILOT, §15.4); the nested-model difference needs a simulation |
| Risk of inconclusive | Medium–high: pilot AUC is from 12 prompts; baseline may absorb the signal |
| Minimum viable experiment | E21 (reuse existing 504 maps) + E05 on a 40-scene DEV set with leave-scene-out CV |
| Stronger extension | Add count/spatial constraints; layer/head subsets; x0-pred detector baseline; SDXL transfer |
| Success / failure / stop | Success: ΔAUC ≥ 0.05 with CI excluding 0 on TEST. Failure: upper CI < 0.03. **Stop** (do not run TEST) if DEV simulation power < 0.5 for ΔAUC = 0.05 |

#### Direction B — Counterfactual and meaning-preserving prompt robustness
| Item | Content |
|---|---|
| Gap | MetaLogic (logically equivalent prompts, ABS), HRS-Bench robustness, and Vemishetty et al. (description round-trip, FT abstract) test robustness. A scene-graph benchmark with **declared edit targets** and **preservation scoring** for *both* semantic-preserving and counterfactual edits was not found (narrow search) |
| RQ | RQ-B: Do wording changes that keep the scene graph fixed change constraint success beyond seed noise, and do one-property edits change exactly the target property? |
| Value | Medium–high; gives the benchmark its distinct feature |
| Novelty concerns | MetaLogic, HRS-Bench — need full texts |
| Feasibility | High (text-side generation is cheap; image cost is `S × V × k`) |
| Evaluation difficulty | **High**: pair-level fidelity needs per-constraint judges and a human audit |
| Sample size | `S` scene graphs × `V` variants; unit = scene graph |
| Inconclusive risk | Low–medium |
| MVE | 40 scene graphs × canonical + 1 rewrite + 1 counterfactual, k = 4 |
| Extension | More edit types; cross-model |
| Success / failure / stop | Success: excess RMS wording effect ≥ 0.10 for binding/spatial (H2). Failure: CI upper < 0.05. Stop: if evaluator disagreement with humans on the edited constraint exceeds 20% in audit (then pair scoring is unreliable) |

#### Direction C — Complexity and constraint interactions
| Item | Content |
|---|---|
| Gap | Complexity is usually category-confounded. Pilot confounds category with object identity (§2.5) |
| RQ | RQ-C: How does success decline with constraint count and attribute count when object identity is balanced? Does CFG's effect depend on complexity? |
| Value | Medium (descriptive); required by PS |
| Feasibility | High; reuses core data |
| GPU | Same images as core + E11 |
| Eval difficulty | Low–medium |
| Sample size | Balanced cells (≥ 20 scenes per complexity level) |
| Inconclusive risk | Medium (floor effect: pilot pooled success for binding/multi median prompts = 0.06; 3 of 28 prompts have **zero** successes) |
| MVE | Fit success ~ complexity on DEV |
| Success / stop | Stop adding complexity levels once a level has < 5% baseline success (floor) |

#### Direction D — Failure-triggered intervention
| Item | Content |
|---|---|
| Gap | See §5.3: not established as distinct |
| RQ | RQ-D: Does reducing measured colour/object conflict *specifically* improve binding, relative to matched random and global edits? |
| Value | Medium as a **causal probe**, low as an algorithm claim |
| Novelty concerns | **High** (G&B, D&B, SynGen, ToMe, CoBind, Attention Modulation, AnchorSteer, EPIC) |
| Feasibility | Medium: needs per-pixel attention bias; SD 1.5 only |
| GPU | `arms × S_test × k` |
| Eval difficulty | Medium (binding label) |
| Inconclusive risk | High (pilot global edit: +0.02) |
| MVE | One conflict score, one threshold on DEV, 3 arms |
| Extension | Trigger/stage/locality ablations |
| Success / failure / stop | Success: targeted − random ≥ 0.08 on binding-correct rate (CI > 0). Stop if no arm beats control by ≥ 0.03 on DEV |

#### Direction E — Token, stage and layer ablations
Valuable **only inside D** (what makes an intervention work). Without D it is a parameter sweep. Sample: reuse D's test arms plus 6 ablation arms (§14). Priority P2–P3.

#### Direction F — Guidance, quality and compute trade-offs
| Item | Content |
|---|---|
| Gap | CFG interacts with timing (Kynkäänniemi 2024; Wang 2024; ABS) but not with **constraint count** in controlled scenes (search found no such study) |
| RQ | RQ-F: How do CFG, steps and any correction trade adherence against CLIP similarity, diversity, latency, memory? |
| Value | Medium; PS-required |
| Feasibility | High |
| GPU | CFG 5 levels × steps 2 on canonical subset (§16) |
| Eval | Low |
| Inconclusive risk | Medium |
| MVE | The existing grid on the V2 DEV set |
| Success | A reportable, CI-bounded CFG × complexity interaction (or a bounded null) |

#### Direction G — Evaluation reliability
| Item | Content |
|---|---|
| Gap | GenEval 2 (FT) shows detector-based benchmarks can drift from human judgement by up to 17.7%. Whether **my** evaluator's error changes the **conclusion** of my own comparisons is unmeasured |
| RQ | RQ-G: Per constraint type, where do OWL-ViT+CLIP, a VQA-style judge and blinded humans disagree, and does the disagreement change any primary conclusion? |
| Value | High as a supporting contribution; P0 prerequisite |
| Feasibility | Medium (annotation effort) |
| Sample size | ≥ ~140 constraint judgements per constraint type for a ±0.05 half-width at accuracy ≈ 0.9 (formula §12.4) |
| Success | A calibrated judge with documented per-type error |

#### Direction H — Generalisation
Held-out scene graphs are built into the design (§11). A second **model** (SDXL: UNet, so extraction mostly carries over; **UNVERIFIED** until tested) is P2. SD3 uses joint attention, so the extraction code does not transfer; run generation + evaluation only (P3/Defer).

### 6.2 Ranking matrix (judgement)
| Direction | Scientific value | Novelty risk (5 = low risk) | Feasibility | Cost (5 = cheap) | Eval difficulty (5 = easy) | Inconclusive risk (5 = low) | **Total /30** |
|---|---|---|---|---|---|---|---|
| A + G (with B data) | 5 | 3 | 4 | 4 | 3 | 3 | **22** |
| B | 4 | 3 | 4 | 4 | 2 | 4 | 21 |
| C + F | 3 | 4 | 5 | 4 | 4 | 3 | 23 |
| D (+E) | 3 | 1 | 3 | 3 | 3 | 2 | 15 |
| H | 3 | 4 | 3 | 2 | 3 | 3 | 18 |

**Decision.** C+F scores highest on feasibility but is descriptive and already required by the PS; it is *folded into every experiment as design factors* instead of being the headline. A+G (with B's benchmark) is the highest-value *research* direction after novelty risk is accounted for. D is retained only as H3, the causal probe.

---

## 6A. Independent research-discovery phase (Revision 2, 2026-10-14)

### 6A.1 Method
Directions were generated from three independent sources and **not** restricted to the Revision 1 list: (a) the official problem statement; (b) read-only mining of the saved pilot outputs for unexplained patterns (§2.7); (c) new literature searches on 2026-10-14 (arXiv/venue pages and search snippets; levels marked ABS unless a provided PDF was read). Each direction was then tested against four rejection rules: **(R1)** a renamed existing method; **(R2)** novelty claim not distinguishable in operation, trigger, locality, timing, objective or claim; **(R3)** data generation without a scientific question; **(R4)** needs resources beyond two 16 GB GPUs or cannot be tested with the available independent scene graphs.
*Limit:* searches were targeted, not exhaustive. A "gap" below means "not found in these searches" (**UNVERIFIED**), never "does not exist".

### 6A.2 Fifteen directions, their closest work and verdicts
| ID | Direction and question | Origin | Closest papers (verification level) | What would distinguish it | Verdict |
|---|---|---|---|---|---|
| **ND01** | **Constraint commitment curves.** At which denoising step is each constraint's outcome (presence, count, colour binding, spatial relation) *decided*, measured causally by re-noising the trajectory and resampling? | F1 (seed persistence), PS ("changes across timesteps") | Critical windows (Li & Chen, ICML 2024; theory, Gaussian-mixture bounds, preliminary SD experiments; ABS); Restart sampling (Xu et al., NeurIPS 2023; add-noise-then-deterministic-denoise operation; ABS); SDEdit (Meng et al., ICLR 2022; partial noising; ABS); negative-prompt timing (arXiv 2406.02965: negative prompts take effect later than positive ones; ABS); R&B (layout in first steps; ABS); SD as Task-Aware Feature Extractor (binding captured later; ABS); CoBind (early entity layout, later binding, ABS) | **Constraint-type-specific, chance-corrected, interventional commitment curves** on controlled scene graphs, compared across constraint types, then used to predict when diagnostics and edits can work (HA, HB, HC). Not an attention study, not a new sampler | **SURVIVES — primary** |
| **ND02** | **Failure in the noise.** How much of a constraint outcome is fixed by the initial latent, and is it linearly readable from it? | F1 | InitNO (CVPR 2024; valid/invalid noise via attention scores; ABS); noise refinement for composition (arXiv 2503.06506; ABS); inference-time scaling by noise search (Ma et al., CVPR 2025; ABS); reproducibility of diffusion outputs (Zhang et al., NeurIPS 2023; CIFAR-scale; ABS); Diffusion Probe (ABS) | Variance decomposition into noise / prompt / CFG and a *noise-only readout* used as a **mandatory baseline** for attention claims; a measurement, not a noise-search method | **SURVIVES — supporting**; novelty weak (R2 risk) |
| **ND03** | **Causal layer/head localisation.** Which cross-attention layer groups (and heads) are *necessary* for each constraint type? | PS ("which tokens influence…"), F7 | Per-layer ablation across the 16 cross-attention layers for **editing** (arXiv 2403.03431; ABS); head-level token-removal attribution with ablation (ACL 2026 Findings, 1265; **snippet only, full text not retrievable**); CoBind's resolution assumption (low-res = layout, mid-res = binding; ABS); DAAM (FT) | Knockout by layer group × constraint type on controlled scenes with a **matched random-token control**, linked to commitment time | **SURVIVES — supporting**; read the ACL 2026 paper first (overlap risk) |
| **ND04** | **Guidance attribution.** Which constraint tokens does CFG actually amplify, and does that explain why CFG raises similarity more than adherence? | F6 + pilot CFG result | Token-removal counterfactual attribution (ACL 2026; ABS); Condition-Degradation Guidance (CVPR 2026; content vs context tokens; ABS); token-merging that uses per-token CFG magnitude (search snippet; arXiv ID not confirmed); Composable Diffusion (not searched this session) | Tie the **scale effect of CFG** on adherence to per-constraint-token guidance contribution along fixed trajectories | **SURVIVES — supporting** |
| ND05 | **Relation words vs layout prior.** Do "left/right" words move the layout, or do object-pair priors and the seed decide? | F4 | VISOR (Gokhale et al., arXiv 2212.10015: first-mentioned-object bias; inconsistency between equivalent relations; conditional metric; ABS); "Rethinking spatial inconsistency in CFG" (CVPR 2024; title only); InfSplign (ABS) | Seed-matched effect-size decomposition (word vs pair prior vs seed) | **Partial overlap with VISOR** → folded into E08/E09/E23, **not** a separate direction |
| ND06 | **Numerosity compression.** Does requested count regress toward ≈ 3? | F5 | Make It Count / CountGen (arXiv 2406.10210; CVPR 2025 per NVIDIA page; ABS); GenEval/TIFA counting (FT) | Documenting compression by CFG/timing | **Rejected as headline** (R2, weak); descriptive in E19 |
| ND07 | **Mention-order asymmetry and attribute collapse** (causal-mask leakage in the text encoder). | F3 | Structure Diffusion (ICLR 2023; states that causal masks blend earlier tokens into later ones; ABS); Magnet; "Understanding and Mitigating Compositional Issues" (arXiv 2406.07844; ABS); ToMe (end-token substitution; ABS); VISOR first-object bias; "Follow the Flow" (arXiv 2504.01137; ABS) | Matched-seed mention-order tests; collapse-vs-swap geometry | **Rejected as headline** (mechanism already hypothesised, R1/R2); exploratory inside E08/E09 |
| ND08 | **Negative-prompt timing for binding repair.** Does a negative prompt naming the wrong binding help, and when? | PS (inference-time mechanisms) | arXiv 2406.02965 (negative prompts delayed; "reverse activation"; ABS) | Binding-specific negative prompts × step windows × random-string control | **SURVIVES as P3** (E28); reverse-activation risk |
| ND09 | **Label-free edit-equivariance evaluation** for counterfactual pairs. | Evaluation principle | MetaLogic (prompt-side invariance; ABS); ACTesting (metamorphic relations; text–image consistency; ABS); 2026 vision-language Equivariance-Consistency Score (figure reading; ABS) | Edit-side equivariance for T2I without labels | **Deferred** — validity unproven, still needs a detector/human audit (R3 risk) |
| **ND10** | **Outcome fragility / decision margin.** How often does a constraint outcome flip under micro-perturbations (tiny latent noise, fp16 vs fp32, hardware)? | PS (stochasticity), E00 | Reproducibility results with same noise + deterministic solver (Zhang et al., NeurIPS 2023; ABS); no T2I constraint-level fragility study found (**UNVERIFIED gap**) | Flip-rate as a *margin* estimator, related to commitment and to failure | **SURVIVES — supporting** |
| ND11 | **Common-random-number evaluation.** | F8 | Standard variance-reduction statistics | None | **Rejected as contribution**; **adopted as design rule** (pair seeds; 79% variance removed in the pilot) |
| ND12 | **Test-time compute economics** (early abort, best-of-N). | Downstream use | Ma et al. inference-time scaling (ABS); GenAI-Bench best-of-N (ABS); Diffusion Probe early quality-aware decisions (ABS); EPIC (ABS) | Constraint-level early-abort rules | **Rejected** (crowded, R2); a *use* of ND01/ND02 |
| ND13 | **Adherence–diversity coupling.** | Pilot (diversity 0.21 → 0.14) | CFG paper (FT); guidance in a limited interval (ABS) | None beyond E14 | **Rejected** (R2) |
| ND14 | **Distractor-token dilution.** | PS ("conditioning") | Oriyad et al., TMLR 2025 (attention overlap → entity missing; ABS); arXiv 2406.07844; Attention Regulation (arXiv 2403.06381; ABS) | None | **Rejected** (covered); kept only as the CF-O variant |
| ND15 | **Conditional (presence-aware) binding as an evaluation rule.** | F2 | VISOR conditional metric (ABS); GenEval (FT); DSG dependency questions (ABS) | None | **Rejected as contribution; adopted as a measurement rule** — it corrects a Rev. 1 label flaw |

### 6A.3 Ideas rejected as renamed or weakly distinct (R1/R2)
| Idea | Why rejected |
|---|---|
| Attention sharpening / temperature gating | UCC's TACA-TB (FT) and Attention Modulation's temperature control (ABS) |
| Entropy- or overlap-triggered correction | Gather-and-Bind, Divide & Bind, SynGen (FT/ABS) already use entropy and JSD as losses |
| Conflict-triggered local logit bias | See §5.3: partial overlap with Attention Modulation and CoBind (ABS) |
| Attention-based noise selection | InitNO (ABS) |
| Merging attribute and noun tokens | ToMe (ABS) |
| "More CFG sweeps", "more counterfactual prompt testing" without a mechanism question | R3: generate data without answering a question |

### 6A.4 Ranking of the five strongest surviving directions
*Scores 1–5 are my judgement (5 = best; compute 5 = cheapest); distinction scores are limited by what the searches verified.*
| Rank | Direction | Research value | Verified distinction | Testability | Compute | Publication potential | Total /25 |
|---|---|---|---|---|---|---|---|
| 1 | **ND01** Commitment curves | 5 | 4 | 5 | 4 | 4 | **22** |
| 2 | **ND10** Fragility / margin | 3 | 4 | 4 | 5 | 3 | **19** |
| 3 | **ND03** Layer/head causal localisation | 4 | 3 | 4 | 4 | 3 | **18** |
| 3 | **ND02** Noise determinism and readout | 3 | 2 | 5 | 5 | 3 | **18** |
| 5 | **ND04** Guidance attribution | 4 | 3 | 4 | 3 | 3 | **17** |
| (6) | ND05, ND08 | 3 | 2–3 | 4–5 | 4–5 | 2–3 | 17 |
| (—) | Rev. 1 Direction A (attention diagnostics, as originally framed) | 4 | 2 | 3 | 4 | 3 | 16 |
*Tie-break rule for fifth place (ND04 vs ND05/ND08):* relevance to the PS's required CFG factor, then verified distinction → ND04.

### 6A.5 Comparison with the Revision 1 blueprint and the decision
| Criterion | Rev. 1 headline (early-attention diagnostics) | Rev. 2 headline (ND01 commitment programme) |
|---|---|---|
| Outcome validity | Flawed: `binding_ok` includes omissions (61% of pilot binding failures) | Conditional definitions; presence, count, binding-given-presence, spatial-given-presence reported separately |
| Overlap with prior work | High with Diffusion Probe, InitNO, G&B/D&B/SynGen (ABS/FT) | Lower: no interventional constraint-level commitment measurement found (UNVERIFIED; narrow search) |
| Dependence on attention | Total | None for the primary claim; attention enters as HB, a test |
| Statistical efficiency | Needs ≈ 140 binding scenes for ΔAUC half-width ≈ 0.08; nested-model SE unknown | Within-scene, within-seed comparisons; seed pairing removes ≈ 79% of variance (F8); resolution set by the step grid |
| Causal content | Correlational (H1); causal only through H3 | Causal by design (interventional re-noising) |
| Compute | Cheap (no extra generation) | Moderate: ≈ 1,600 image-equivalents for the primary experiment (§16.5b) |
| Fit to PS | CFG + steps + attention | CFG + steps + attention + a new inference-time probe (re-noising) and timing across denoising steps (explicit PS question) |
| Failure mode if results are null | Weak paper | A bounded null (no type-specific commitment difference of 1.5 steps or more) is itself informative |
**Decision: replace the Revision 1 headline.** Rev. 1's benchmark, evaluator audit and CFG × steps × complexity components are kept. Old H1 and H3 become the reframed **HB** and **HC**; old H2 (wording sensitivity) and the E09 counterfactual-fidelity analysis move to **supporting**.

### 6A.6 The coherent programme (one story, three links)
```
Link 1 (causal timing)  HA: commitment curve t50 per constraint type        <- E23 (+E24 fragility)
          |  predicts WHEN information can exist and WHEN edits can act
          v
Link 2 (readout)        HB: attention / prompt / noise features before vs after t50  <- E05, E25
          |  tests whether attention is forecast or readout
          v
Link 3 (handle)         HC: targeted edit effect gated by onset time vs t50           <- E12 (+E26, E27, E28)
```
Support for each link is evaluated separately (§19); a failure of one link does not invalidate the others (§21.3, §21.6).

### 6A.7 Caveats of the discovery phase
Searches were limited; "closest papers" were read at ABS level unless stated; the ACL 2026 head-level paper's contents are unknown beyond snippets; ND10's gap and ND01's gap are **UNVERIFIED**; the pilot findings in §2.7 are exploratory; no new experiment was run and no existing result was changed.

---

## 7. Recommended central paper and formal contributions (Revision 2; supersedes Revision 1)

**Working title:** *When Is a Prompt Constraint Decided? Commitment Curves, Readout and a Time-Gated Test of Attention Edits in Text-to-Image Diffusion* (alternatives in §20.1b).

**Central question.** At which denoising step is each compositional constraint (presence, count, colour binding, spatial relation) decided; is what attention, prompt structure or the initial noise reveals a *forecast* (before commitment) or a *readout* (after it); and can an attention edit change the outcome only before commitment?

**Primary contribution (C-main).** A causal, constraint-type-specific measurement of **commitment timing** by interventional re-noising on controlled scene graphs, and a test of whether attention-based information and attention edits are time-gated by it.

**Supporting contributions.**
1. **Benchmark V2** (scene-graph prompts, counterfactual and meaning-preserving variants, scene-level splits, leakage policy) — unchanged from Revision 1.
2. **Evaluator audit** with conditional (presence-aware) outcome definitions, multi-judge evaluation and a blinded human audit.
3. **Controlled CFG × steps × complexity** measurement (PS requirement).
4. **Mechanism probes** (P2): outcome fragility, noise-only readout, causal layer-group knockouts, guidance attribution.
5. **Negative-result record:** the fixed global re-weighting (verified pilot) and any failed edit.

**Direct competitors / closest work.** Critical windows (Li & Chen; theory and preliminary SD experiments), Restart sampling (operation only), negative-prompt timing (arXiv 2406.02965), Diffusion Probe, InitNO, Gather-and-Bind / Divide & Bind / SynGen, Attention Modulation, CoBind, MetaLogic (all ABS/FT as in §4).

**Evidence required for C-main.** HA supported on sealed TEST (§8.1), with commitment curves robust to the restart magnitude Δ (E23b), not contradicted by the judge-sensitivity analysis (E18).
**Evidence that would weaken it.** Commitment curves for different constraint types statistically indistinguishable *and* with an interval narrow enough to rule out a 3-step difference (a bounded null, which is itself publishable); strong dependence of t50 on Δ; evaluator-confounded outcomes; branch outcomes dominated by judge noise.

---

## 8. Primary hypotheses and preregistration plan (Revision 2; supersedes Revision 1)

*At most three confirmatory hypotheses (HA, HB, HC), Holm-corrected as a family. All are **PROPOSED HYPOTHESES** until frozen. Revision 1's H1–H3 are retained in §8.5 as **supporting** hypotheses outside the Holm family.*

### 8.0 Common definitions (new in Revision 2)
- **Conditional outcomes.** `Y_pres` = every requested object is detected. `Y_count` = detected instances equal `n`. **`Y_bind|pres`** = every requested colour is assigned to its intended object, **defined only for images in which both (all) objects of the constraint are detected**. **`Y_spat|pres`** analogously. The pilot's `binding_ok` (which counts an undetected object as a binding failure) is *not* used for primary claims.
- **Commitment (restart) operation.** For a base trajectory (scene `s`, seed `j`, core condition: CFG 7.5, 40 steps, DDIM eta 0) whose latents `x_t` were saved at grid steps `t ∈ {6, 12, 18, 24, 30, 36}`, construct a branch at `t` by **noising back by Δ steps** (design value Δ = 8) with fresh noise and re-denoising deterministically with the same prompt, CFG and sampler:
  `x_{t−Δ}' = sqrt(ᾱ_{t−Δ}/ᾱ_t) · x_t + sqrt(1 − ᾱ_{t−Δ}/ᾱ_t) · ε`, `ε ~ N(0, I)` from a branch-specific seeded generator (index clipped at 0, i.e. the full-noise level). *Correctness test:* if `x_t ~ N(sqrt(ᾱ_t)·x_0, (1 − ᾱ_t)I)` then `x_{t−Δ}' ~ N(sqrt(ᾱ_{t−Δ})·x_0, (1 − ᾱ_{t−Δ})I)` (checked numerically in the unit test, §17.1). This is the add-noise step of Restart sampling (ABS); here it is used **as a measurement**, not as a sampler.
- **Agreement and commitment.** `A_{κs}(t)` = fraction of the `M = 2` branches of scene `s` (averaged over seeds) whose outcome for constraint type `κ` equals the base outcome (for conditional outcomes, only pairs where the required objects are present in both images). `A0_{κs}` = agreement between outcomes of **independent seeds** on the same canonical prompt (from the core data, `k = 4` seeds → 6 pairs): the chance level with no shared information. The **chance-corrected commitment** is the ratio estimator
  `C_κ(t) = Σ_s [A_{κs}(t) − A0_{κs}] / Σ_s [1 − A0_{κs}]`  (scenes are not selected on outcomes; scenes with `A0 ≈ 1` simply carry little weight).
  **`t50_κ`** = smallest `t` (linear interpolation on the grid) with `C_κ(t) ≥ 0.5`; "> 36" if never reached.

### 8.1 HA — Commitment timing differs by constraint type (**PLANNED CONFIRMATORY TEST**)
| Field | Specification |
|---|---|
| Statement | The step at which a constraint's outcome is committed differs between presence and colour binding (given presence): `Δt50 = t50(bind\|pres) − t50(presence) ≠ 0` |
| Rationale | Conflicting reports: layout forms in the first steps (R&B, ABS) but binding may be captured later (SD as feature extractor, ABS); critical-window theory (ICML 2024, ABS) predicts feature-specific windows. Prior work did not measure this causally per constraint type on SD with controlled scenes (**UNVERIFIED gap**) |
| Operational definition | §8.0 with Δ = 8, `M = 2`, grid of six steps; judge = primary judge per type (§9.0); uses TEST scenes (≈ 100 canonical scenes: presence all, binding ≈ 70) with `k = 2` base seeds |
| Direction | **Two-sided**; no sign is pre-asserted |
| Independent variable | Constraint type (and, for E23b, Δ) |
| Primary outcome | `Δt50` in denoising steps |
| Unit of analysis | Scene graph |
| Minimum practically meaningful effect | 3 steps (7.5% of the trajectory; ≈ one half of the 6-step grid spacing) |
| Sample size | Not derivable from the pilot (no commitment data exist). **Gate:** run E23 on 40 DEV scenes; if the DEV 95% CI half-width of `Δt50` exceeds 4 steps, raise `M` or the number of scenes before TEST |
| Interval / test | 95% scene-level bootstrap of `Δt50` (resample scenes, recompute ratio estimators) ; Holm |
| Support | CI excludes 0 **and** `abs(Δt50) ≥ 3` |
| Reject (bounded null) | CI ⊂ (−1.5, +1.5) |
| Inconclusive | otherwise |
| Secondary (EXPLORATORY) | ordering of `t50` among presence, count, spatial\|pres, binding\|pres; dependence on Δ ∈ {4, 12} (E23b); dependence on CFG |
| Threats | Judge noise attenuates both `A` and `A0` (ratio estimator limits but does not remove this; E18); Δ defines "local" re-noising so `t50` is Δ-relative (reported, not hidden); restart noise may interact with the prompt differently by constraint type; conditional outcomes shrink the usable sample |

### 8.2 HB — Is attention a forecast or a readout? (**PLANNED CONFIRMATORY TEST**; reframes Revision 1 H1)
| Field | Specification |
|---|---|
| Statement | Relative to **prompt-only and noise-only** baselines, attention features carry information about the final `Y_bind\|pres` that is available **before** the binding commitment point (forecast), or only **after** it (readout), or never |
| Rationale | Pilot F7: bind-margin AUC 0.72 among both-present images (12 prompts, EXPLORATORY); F1: seed determinism suggests the noise itself may carry information; Diffusion Probe (ABS) reports early attention predicting overall quality |
| Operational definition | Frozen logistic models on TEST binding-bearing scenes with `Y_bind\|pres` (both-present images only). **B1** = CFG, steps, n_constraints, prompt-structure features (colour pair, attribute–noun distance, mention order, text-embedding similarity of the noun phrases, detector-prior difficulty from DEV). **B_noise** = B1 + features of the initial latent `x_T` (reproducible from the seed: channel moments, pooled-grid statistics, low-frequency spectrum). **M_att** = B_noise + attention-conflict features (bind margin, JSD(attr, own noun), JSD(attr, other noun), top-10% share, entropy, box ratio) computed from steps `0…τ` |
| Windows (set from DEV, frozen) | `τ_pre` = the last grid step ≤ 0.5 · `t50_DEV(bind\|pres)`; `τ_post` = the first grid step ≥ `t50_DEV(bind\|pres) + 8` |
| Primary outcomes | `ΔAUC_pre = AUC(M_att@τ_pre) − AUC(B_noise)` and `ΔAUC_post` analogously |
| Unit | Scene graph (k = 4 seeds nested) |
| Minimum meaningful effect | 0.05 AUC (judgement) |
| Sample size | ≈ 140 binding-bearing scenes give a single-AUC half-width ≈ 0.08 (DERIVED FROM PILOT); the nested SE is unknown → DEV simulation gate: if projected CI half-width for ΔAUC exceeds 0.04, HB is downgraded to exploratory *before* TEST is opened |
| Interval / test | Scene-level bootstrap on frozen predictions; Holm |
| Outcomes (pre-specified) | **Forecast** supported: `ΔAUC_pre ≥ 0.05` with CI > 0. **Readout only**: `ΔAUC_pre` CI upper < 0.03 **and** `ΔAUC_post ≥ 0.05` with CI > 0. **None**: both CI upper bounds < 0.03. Otherwise inconclusive |
| Threats | Label noise; feature leakage (shuffled-attention control, E07); HB depends on HA's DEV estimate being stable |

### 8.3 HC — The effect of a targeted attention edit is time-gated by commitment (**PLANNED CONFIRMATORY TEST**; reframes Revision 1 H3)
| Field | Specification |
|---|---|
| Statement | The targeted edit of §13 improves `Y_bind\|pres` relative to an equal-magnitude random-pair edit when applied **before** the binding commitment point, and does so more than when applied **after** it |
| Rationale | If commitment is real and causal, edits after it cannot change the outcome; this is a **falsifiable consistency test of HA's commitment measure** and of attention as a handle. Pilot global edit: +0.02 [−0.04, +0.09] (VERIFIED), i.e. no evidence yet |
| Operational definition | TEST binding-bearing scenes, canonical prompts, CFG 7.5, 40 steps, `k = 4`. Edit window length fixed at **L = 12 steps**. Onsets (from DEV): early = 0; mid = `t50_DEV`; late = `ceil(t50_DEV) + 4` (capped at 28 so the window ends by step 40). Arms: **targeted@early, random@early, targeted@mid (dose–response, EXPLORATORY), targeted@late, random@late, global s = 2@0**. Control = core images (bit-identical hook-off path) |
| Primary outcomes | `D_early = targeted@early − random@early` and `G = D_early − D_late` with `D_late = targeted@late − random@late` (binding-correct given presence; presence rate reported as a guard) |
| Unit | Scene graph; pairing by (scene, seed) |
| Minimum meaningful effect | `D_early ≥ 0.08` (≈ MDE for ≈ 140 scenes at k = 4: 0.07–0.08, DERIVED FROM PILOT) and `G ≥ 0.10` (a difference of two paired contrasts has ≈ √2 × the MDE of one; smaller effects are labelled "unresolvable at this N") |
| Interval / test | Paired scene bootstrap CI for `D_early` and `G`; Holm |
| Support | `D_early` CI > 0 with point ≥ 0.08, **and** `G` CI > 0 with point ≥ 0.10, guards respected (CLIP drop ≤ 0.5, duplicate-object rate increase ≤ 0.03) |
| Reject | `D_early` CI upper < 0.04 (no handle), or `G` CI inside (−0.05, +0.05) while `D_early` is supported (a handle that is **not** time-gated, contradicting HA's interpretation) |
| Threats | Mask errors; edit strength confounded with onset (matched β and mask area); evaluator-dependent gain (E18); HA misestimated on DEV |

### 8.4 Preregistration protocol (to be frozen **before** any TEST generation)
1. Append **Addendum B** to `report/protocol.md`: HA–HC verbatim; §8.0 definitions; the grid, Δ, `M`, `L`; feature definitions; model classes and hyperparameter grid; the primary-judge rule; thresholds; exclusion rules; split manifest; analysis scripts; DEV-derived `t50_DEV`, `τ_pre`, `τ_post`, onsets (recorded as numbers when DEV analysis ends).
2. Record **SHA-256** hashes of generator, manifest, analysis scripts and `protocol.md` with a timestamp (recommended: `git init` first).
3. **Seal TEST:** no generation, scoring or inspection of TEST images before the freeze; TEST labels computed once.
4. **Order of work:** DEV commitment curves (E23-DEV) → freeze `t50_DEV` → generate TEST core + branches + onset arms → analyse HA, HB, HC once each.
5. **Amendments** are appended with date and reason and re-label affected analyses EXPLORATORY.
6. **DEV may** tune: features, τ, thresholds, mask quantile q, bias β, trigger τ_c, Δ sensitivity. **TEST may** only evaluate frozen choices.

### 8.5 Supporting hypotheses (Revision 1 H1–H3, retained outside the Holm family)
| Name (kept for continuity with E05/E08/E12) | Change in Revision 2 | Status |
|---|---|---|
| **H1** — Early attention features add held-out predictive information about binding | **Outcome redefined** to `Y_bind\|pres` (conditional) and **noise-only baseline added**; subsumed by HB for the confirmatory claim; the original single-time ΔAUC at τ = 12 is reported as EXPLORATORY | Supporting (E05) |
| **H2** — Meaning-preserving rewrites shift scene-level binding/spatial success beyond seed noise (within-seed permutation test; excess RMS ≥ 0.10) | Unchanged in definition; now supporting because MetaLogic/HRS-Bench overlap (ABS) and the commitment programme is the headline | Supporting (E08) |
| **H3** — Targeted edit beats a matched random edit at onset 0 | Folded into HC's `D_early`; no separate test | Folded into HC (E12) |

### 8.6 Revision 1 specifications of H1–H3 (retained for reference)
*These specifications are superseded by §8.5 where they differ (outcome = conditional binding; noise-only baseline; H3 folded into HC). Cards E05, E08 and E12 still cite them by name.*

#### 8.6.1 H1 — Early attention diagnostics (**PLANNED CONFIRMATORY TEST**)
| Field | Specification |
|---|---|
| Statement | On held-out scene graphs, a logistic model using early-stage attention-conflict features **plus** baseline covariates predicts colour-binding correctness better than the baseline model alone |
| Rationale | Pilot (EXPLORATORY): bind-margin AUC 0.74–0.79 on 12 prompts. Mechanism: if binding is largely settled early (R&B, ABS) the colour token's map should already overlap the wrong object in doomed samples |
| Operational definition | `Y = 1` if all colour-binding constraints of the prompt are satisfied under the **primary judge** (§12). Features from steps `0…τ`, `τ ∈ {12, 20, 39}` of 40. **Attention model** `M_att = B1 + {bind_margin, JSD(p_attr, p_noun), JSD(p_attr, p_other), top-10% share, entropy, box ratio}` (aggregated over layers; 16×16 and 32×32 reported separately) |
| Baselines (frozen) | **B0** = CFG, steps, n_constraints. **B1** = B0 + prompt-structure features (colour pair, attribute–noun token distance, mention order, CLIP text-embedding similarity of the two noun phrases [prompt-side analogue of Kim 2024], detector-prior difficulty of each object estimated on DEV only). Optional **B3** = B1 + step-12 x0-prediction detector score (a generic early signal) |
| Independent variable | Feature set (B1 vs B1 + attention) |
| Primary outcome | ΔAUC = AUC(M_att) − AUC(B1) on TEST, `τ = 12` (early) |
| Unit of analysis | **Scene graph** (seeds nested; k = 4 per scene) |
| Minimum practically meaningful effect | ΔAUC = 0.05 (judgement: below this the diagnostic would not change an early re-roll decision) |
| Sample size | N_test = 140 binding-bearing scene graphs (of 200 TEST scenes), k = 4: single-AUC 95% half-width ≈ 0.08 (DERIVED FROM PILOT; §15.4). The nested-model difference has a smaller SE that is **unknown** → fix by simulation on DEV; if projected power < 0.8 for ΔAUC = 0.05, H1 is downgraded to exploratory *before* TEST is touched |
| Interval / test | 95% CI by **scene-level bootstrap** (resample scenes, refit nothing: model frozen) ; Holm-adjusted p from the bootstrap distribution |
| Multiple testing | Holm over {H1, H2, H3}; secondary families FDR-BH, labelled EXPLORATORY |
| Support | CI lower bound > 0 **and** ΔAUC ≥ 0.05 |
| Reject (bounded null) | CI upper bound < 0.03 |
| Inconclusive | otherwise |
| Threats | Evaluator label noise (attenuates AUC); attention features encoding object identity (control: E15); scene leakage across splits (control: split policy §11.6); hyperparameter overfit (frozen after DEV) |

#### 8.6.2 H2 — Wording sensitivity (**PLANNED CONFIRMATORY TEST**)
| Field | Specification |
|---|---|
| Statement | Meaning-preserving rewrites (clause order, mention order, synonym paraphrase) shift **scene-level** success rates of colour-binding and spatial constraints by more than seed-to-seed noise explains; presence and counting shifts are smaller |
| Operational definition | For constraint type κ and scene `s`: `p̂_s^can`, `p̂_s^rw` over the same k seeds. Statistic `T_κ = mean_s (p̂_s^can − p̂_s^rw)²`. Null: wording is irrelevant, so within each seed the pair `(Y_can, Y_rw)` is exchangeable → **within-seed label-swap permutation** distribution of `T_κ`. Effect size: **excess RMS shift** `sqrt(max(T_obs − mean(T_perm), 0))` |
| Primary outcome | Excess RMS shift for binding and spatial (two estimands) |
| Unit | Scene graph |
| Minimum meaningful effect | RMS shift ≥ 0.10 (10 percentage points) |
| Sample size | N_test = 200 scenes (≈ 140 binding-bearing, ≈ 70 spatial-bearing) × (1 canonical + 3 rewrites) × k = 4 |
| Interval | 95% CI by scene bootstrap |
| Support / reject | Support: CI lower > 0 and RMS ≥ 0.10 for binding or spatial. Reject: CI upper < 0.05 for both |
| Threats | The judge may be less reliable on rewritten prompts (audit stratified by variant); rewrites that subtly change meaning (human review of every rewrite) |

#### 8.6.3 H3 — Specific causal handle (**PLANNED CONFIRMATORY TEST**)
| Field | Specification |
|---|---|
| Statement | Reducing measured colour/object attention conflict through a targeted, bounded attention-logit edit improves binding correctness **more than** (i) an equal-magnitude edit on a random token pair and (ii) global re-weighting |
| Rationale | Separates "attention is a symptom" from "attention is a handle". If H1 holds and H3 fails, attention is diagnostic but not causal under this edit |
| Operational definition | Arms on TEST binding-bearing scenes (canonical prompts, CFG 7.5, 40 steps, k = 4): **control** (same code path, s = 1, bit-identical to baseline), **global** (s = 2, existing `intervene.py`), **targeted** (§13), **random-pair** (same bias magnitude and mask area on a random token pair). Primary contrast: `targeted − random-pair` on binding-correct rate |
| Unit | Scene graph; pairing by (scene, seed) |
| Minimum meaningful effect | +0.08 absolute binding-correct rate (≈ the MDE for N ≈ 140 binding-bearing scenes, k = 4: 0.07–0.08, DERIVED FROM PILOT; smaller effects are labelled "unresolvable at this N") |
| Secondary guards | CLIP score drop ≤ 0.5; duplicate-object rate increase ≤ 0.03 |
| Interval / test | 95% scene-bootstrap CI on the paired difference; Holm |
| Support | CI lower > 0, point ≥ 0.08, guards respected |
| Reject | CI upper < 0.04 |
| Threats | Evaluator-dependent gain (E18); mask error; implementation fault (unit tests §17) |
---

## 9. Comprehensive experiment catalogue

### 9.0 Shared definitions (referenced by every card)
| Symbol | Definition |
|---|---|
| **PILOT** | The existing 28 prompts, 1008 + 504 images (SD 1.5, RTX 3050). Already used for exploration; **may not serve as TEST**. May be reported as a pilot and used as extra DEV data (labelled) |
| **DEV** | 100 new scene graphs (binding 30, multi 25, counting 15, spatial 15, presence 15; ≈ 70 binding-bearing). Used to fit, tune, calibrate, simulate power |
| **TEST** | 200 new scene graphs (binding 60, multi 50, counting 30, spatial 30, presence 30; ≈ 140 binding-bearing, ≈ 70 spatial-bearing). **Sealed** until freeze (§8.4) |
| **Variants** | CAN canonical; PAR synonym paraphrase; ORD clause/mention-order change; CF-C colour swap; CF-R relation reversal; CF-N count change; CF-O object deletion/insertion; CF-A attribute insertion/removal. Avg V = 5 variants per scene (CAN + 1 PAR + 1 ORD + ≈ 2 applicable CF) |
| **Core condition** | SD 1.5 fp16, DDIM eta 0, CFG 7.5, 40 steps, 512², negative prompt "", **attention hook on** (image-neutral for SD 1.5; re-verify per hardware) |
| **k** | Seeds per condition. Design uses **k = 4** (justification: derived ICC of the colour-binding outcome is 0.52, so extra seeds beyond 4 add little: for 120 scenes the effective n rises only 187 → 199 from k = 4 → 6; DERIVED FROM PILOT) and k = 2 for DEV tuning |
| **Judges** | **J1** = existing OWL-ViT + CLIP-B/32; **J2** = atomic-VQA judge (Soft-TIFA-style; VLM chosen by VRAM, §12.5); **J3** = blinded human audit. **Primary judge per constraint type** = the one of J1/J2 with higher accuracy against J3 on the DEV audit, **chosen before TEST**; if neither reaches 0.85 for a type, that type is EXPLORATORY only |
| **Latent grid (Revision 2)** | For the ≈ 100 commitment scenes × 2 base seeds, the core generation also saves `x_t` at steps `{6, 12, 18, 24, 30, 36}` (fp16). **Commitment scenes** = canonical prompts of TEST (and 40 of DEV) |
| **Splits policy** | Split at **scene-graph level**; all variants of a scene stay together (§11.6) |

*Image counts below use S/V/k as in §16 and are formulas, not timings.*

### 9.1 Experiment index
| ID | Title | Priority | Needs new generation? | Hypothesis |
|---|---|---|---|---|
| E00 | Reproduce and re-analyse the pilot baseline; cross-hardware determinism check | **P0** | small | — |
| E01 | Evaluator validation with a blinded human audit | **P0** | no (uses E04 images) | RQ-G |
| E02 | Benchmark V2 construction and QA | **P0** | no | — |
| E03 | Throughput, memory and difficulty-calibration pilot (Kaggle) | **P0** | small (DEV only) | — |
| E04 | Core generation with attention hook (DEV + TEST) | **P0** | yes | data engine |
| E21 | Feasibility re-analysis of saved pilot attention (go/no-go gate; Revision 2: also conditional-label and noise-only checks) | **P0** | no | HB gate |
| E05 | Early-attention failure prediction on held-out scenes (Revision 2: forecast-vs-readout analysis with noise-only baseline, conditional outcome) | **P1** | via E04/E11 | **HB** (H1 supporting) |
| E08 | Wording sensitivity under meaning-preserving rewrites | P2 (Revision 2) | via E04 | H2 (supporting) |
| E11 | CFG × steps × complexity | **P1** | yes | PS; supports H1 |
| E12 | Time-gated causal probe: onset arms, targeted vs random vs global vs control | **P1** | yes | **HC** |
| E18 | Evaluator sensitivity of every primary conclusion | **P1** | no | RQ-G |
| E06 | Stage/timing of the diagnostic | P2 | no | H1 support |
| E07 | Feature families, layers, resolutions vs baselines | P2 | no | H1 support |
| E09 | Counterfactual edit fidelity | P2 | via E04 | RQ-B |
| E10 | Constraint-complexity and interaction effects | P2 | via E04/E11 | RQ-C |
| E14 | Adherence–quality–diversity–compute trade-off | P2 | via E11/E12 | RQ-F |
| E15 | Object-identity robustness (leave-object-pair-out) | P2 | no | H1 validity |
| E16 | Cross-model replication (SDXL) | P2 | yes (400 images) | generalisation |
| E19 | Error analysis and failure taxonomy per category | P2 | no | — |
| E20 | Seed-sufficiency and variance decomposition | P2 | no | design check |
| E22 | Synthetic ground-truth evaluator probe | P2 | no T2I | RQ-G |
| E13 | Intervention component ablations | P3 | yes | H3 follow-up |
| E17 | SD3 Medium generation + evaluation only | P3 / Defer | yes | generalisation |

**Revision 2:** new experiments E23–E28 and the changes to E00, E01, E03, E04, E05, E08, E12, E18, E21 are specified in §9.3. Wherever a Revision 1 card below names H1, H2 or H3, read it with §8.5 (H1 supporting / HB; H2 supporting; H3 folded into HC).

**Deferred (not recommended):** training or fine-tuning any model; DiT attention extraction code for SD3/FLUX (new project); per-head pruning studies; user-study of aesthetics; LLM-based prompt rewriting as an intervention (outside the PS focus and overlaps UCC).

### 9.2 Experiment cards
*Field order follows the specification: 1 question, 2 motivation, 3 hypothesis, 4 mechanism, 5 IV, 6 controls, 7 DV/primary metric, 8 dataset, 9 prompt construction, 10 scenes/variants, 11 seeds, 12 images/extractions, 13 baselines/controls, 14 comparisons, 15 statistics, 16 runtime/memory method, 17 code changes, 18 outputs/logs, 19 figures/tables, 20 plausible patterns, 21 interpretation, 22 alternatives, 23 falsifier, 24 priority. (Title and ID are in the heading.)*

#### E00 — Reproduce and re-analyse the pilot baseline (P0)
| Field | Content |
|---|---|
| 1 | Do the saved pilot numbers regenerate from saved data, and do images reproduce across hardware? |
| 2 | Report/README disagree with outputs (§2.2); Kaggle GPUs differ from the RTX 3050; attention hook identity is untested off SD 1.5/3050 |
| 3 | (no hypothesis) Re-running `analyze*.py` reproduces every number in §2.3 within bootstrap noise |
| 4 | Deterministic sampling is device-independent only up to kernel numerics (unverified across GPUs) |
| 5 | Hardware (RTX 3050 vs Kaggle GPU) |
| 6 | Prompt, seed, scheduler, CFG, steps, model revision, dtype, library versions |
| 7 | Mean absolute pixel difference and outcome agreement between 3050 and Kaggle images for 12 pilot (prompt, seed) pairs; recomputed table values |
| 8 | PILOT |
| 9 | Existing |
| 10 | 12 existing prompt/seed pairs (2 per category + 2) |
| 11 | Existing seeds |
| 12 | ≈ 12 images (regeneration) + 0 extraction |
| 13 | The saved 3050 images |
| 14 | Pixel-identical? primary outcome identical? |
| 15 | Descriptive; fraction identical |
| 16 | Trivial; `torch.cuda.max_memory_allocated()` logged per image |
| 17 | `generate.py`: log GPU name, torch/diffusers versions, `max_memory_allocated`; `report_numbers.py` unchanged |
| 18 | `outputs/e00/hardware_check.jsonl` |
| 19 | One reproducibility table |
| 20 | (a) pixel-identical; (b) tiny differences, same outcomes; (c) different images, different outcomes |
| 21 | (a) pilot images may be pooled with Kaggle images; (b) pool but add a hardware covariate; (c) **never pool** — treat 3050 data as pilot only |
| 22 | Non-deterministic kernels, TF32/fp16 behaviour, different cuDNN |
| 23 | n/a |
| 24 | P0 |

#### E01 — Evaluator validation with a blinded human audit (P0)
| Field | Content |
|---|---|
| 1 | Per constraint type, how accurate are J1 and J2 against blinded humans, and what is the inter-annotator agreement? |
| 2 | Current validation: 40 images, one non-independent annotator, spatial n = 9 with 2 positives (§2.3); GenEval 2 shows detector benchmarks drift |
| 3 | PROPOSED: J1 accuracy ≥ 0.85 for presence/count, lower for binding and spatial |
| 4 | Detector thresholds, CLIP colour on crops and a box-centre rule are coarse proxies |
| 5 | Judge (J1, J2, J3) × constraint type × variant × model |
| 6 | Images, constraints, rendering of the question |
| 7 | Accuracy, precision, recall, Cohen's κ between annotators and between judges and consensus |
| 8 | Audit sample drawn from DEV + TEST images **after** generation, stratified |
| 9 | n/a (images) |
| 10 | ≥ 300 images stratified by type × variant × CFG; ≈ 3 constraints each → ≈ 900 constraint judgements |
| 11 | Seeds as generated |
| 12 | 300 images, 0 extraction |
| 13 | Human consensus |
| 14 | J1 vs J3, J2 vs J3, J1 vs J2 per type |
| 15 | Wilson 95% CI per type; sample-size rule `n ≥ 1.96²·p(1−p)/h²` (§12.4); κ with bootstrap CI over images; double annotation of 100 images |
| 16 | Human time: ≈ 900 judgements + 300 duplicate judgements; no GPU beyond J2 inference |
| 17 | New `src/audit_sample.py` (draw sample, export blind CSV + image folder), `src/validate_human.py` (agreement stats); `validate_metrics.py` refactored to read the audit file |
| 18 | `data/audit/audit_items.csv`, `annotator_A.csv`, `annotator_B.csv`, `audit_results.json` |
| 19 | Table: judge × type accuracy/recall with CIs; confusion matrices; disagreement gallery |
| 20 | (a) J1 ≥ 0.9 on all types; (b) J1 good for presence/count, weak for binding/spatial; (c) both J1 and J2 weak for a type; (d) annotators disagree strongly on a type |
| 21 | (a) J1 usable as the primary judge; (b) use J2 or humans as primary for weak types; (c) mark that type EXPLORATORY and do not base a primary hypothesis on it; (d) the constraint definition is ambiguous → rewrite the benchmark item |
| 22 | Annotator fatigue; blinding leakage; sampling not representative |
| 23 | n/a (measurement) |
| 24 | P0 |

#### E02 — Benchmark V2 construction and QA (P0)
| Field | Content |
|---|---|
| 1 | Can the scene-graph grammar generate balanced, unambiguous, non-duplicated prompts with controlled variants? |
| 2 | Pilot: 4–6 prompts per category, object–category confounding, reused wording (§2.5) |
| 3 | (construction task; QA gates only) |
| 4 | Template grammar over a scene-graph schema (§11) |
| 5 | n/a |
| 6 | Object inventory, colour inventory, relation inventory |
| 7 | QA metrics: balance (object × category), duplicate rate, ambiguity flags, rewrite validity (human) |
| 8 | New `data/benchmark_v2/` |
| 9 | §11.2 |
| 10 | S = 300; V ≈ 5 → ≈ 1,500 prompts (counts per category fixed in §9.0) |
| 11 | n/a |
| 12 | 0 |
| 13 | n/a |
| 14 | Pilot benchmark vs V2 on balance metrics |
| 15 | Descriptive; chi-square on object × category balance; inter-reviewer agreement on rewrites |
| 16 | CPU only |
| 17 | New `src/benchmark_v2.py` (grammar, validators), `data/benchmark_v2/{scene_graphs.json, prompts_v2.json, splits.json, DATASHEET.md}`; tests |
| 18 | `splits.json` with hashes; QA report |
| 19 | Dataset-composition table (balance, complexity, variants) |
| 20 | (a) balance met; (b) some objects too hard/easy; (c) many ambiguous items |
| 21 | (a) proceed; (b) re-balance using DEV calibration only; (c) rewrite grammar |
| 22 | Grammar artefacts |
| 23 | n/a |
| 24 | P0 |

#### E03 — Throughput, memory and difficulty-calibration pilot on Kaggle (P0)
| Field | Content |
|---|---|
| 1 | What are the **measured** seconds per image, VRAM, hook overhead, failure rate, and two-worker scaling; which scene templates are at floor/ceiling? |
| 2 | Previous timings are from an RTX 3050 only; the plan must not extrapolate |
| 3 | (none) |
| 4 | — |
| 5 | Workers (1 vs 2), model (SD 1.5, SDXL), hook on/off |
| 6 | Seeds, prompts, settings |
| 7 | Mean/95th-percentile seconds per image, peak VRAM, hook overhead %, crash rate; DEV pooled success per template |
| 8 | 20 DEV scenes × CAN × k = 2 per model/arm |
| 9 | §11 |
| 10 | 20 scenes |
| 11 | 2 |
| 12 | ≈ 40–160 images depending on the number of arms (SD 1.5 hook on/off, 1 vs 2 workers, SDXL) |
| 13 | Hook off |
| 14 | Throughput table |
| 15 | Descriptive with bootstrap CI over images |
| 16 | The measurement itself |
| 17 | `generate.py`: `--shard i/n`, per-worker output files, timing/VRAM fields |
| 18 | `outputs/e03/throughput.csv` |
| 19 | Throughput table |
| 20 | Any |
| 21 | Decide FULL vs LIMITED matrix (§16.5) and drop templates with DEV pooled success < 3% or > 97% (DEV only) |
| 22 | Kaggle contention; session restarts |
| 23 | n/a |
| 24 | P0 |

#### E04 — Core generation with attention hook (P0)
| Field | Content |
|---|---|
| 1 | Produce the paired image + attention dataset for all variants of DEV and TEST scenes |
| 2 | The pilot hook is image-neutral for SD 1.5, so one pass yields images **and** attention |
| 3 | (data collection) |
| 4 | Hook stores per-step word maps (existing `attention.py` design) |
| 5 | Scene × variant × seed |
| 6 | Core condition |
| 7 | Image, constraint outcomes, attention features |
| 8 | DEV + TEST (TEST generated **after** freeze) |
| 9 | §11.2 variants |
| 10 | 300 scenes × V ≈ 5 |
| 11 | k = 4 (DEV tuning arms k = 2) |
| 12 | FULL: `S·V·k = 6,000` images and 6,000 extractions; LIMITED: 2,560 |
| 13 | Hook-off equality subsample |
| 14 | n/a |
| 15 | n/a |
| 16 | Storage: current npz ≈ 0.3 MB/image (147 MB / 504) → ≈ 1.8 GB for 6,000; VRAM per §16 |
| 17 | `generate.py`: call hooked generation by default; `attention.py`: store per-step trajectories at 16×16 and 32×32, save per-layer-group summaries |
| 18 | `samples_gpu{0,1}.jsonl` merged by `merge_samples.py`; hash of every PNG |
| 19 | Manifest table |
| 20 | n/a |
| 21 | n/a |
| 22 | n/a |
| 23 | n/a |
| 24 | P0 |

#### E21 — Feasibility re-analysis of saved pilot attention (P0 gate, no generation)
| Field | Content |
|---|---|
| 1 | Using only the 504 saved maps, does a scene-grouped model show any gain of attention features over prompt-only features, and are the features stable across layers/resolutions? |
| 2 | Pilot AUC is from 12 prompts (§1.2) |
| 3 | PROPOSED: leave-prompt-out ΔAUC > 0 |
| 4 | As H1 |
| 5 | Feature set |
| 6 | Same images |
| 7 | Leave-one-prompt-out ΔAUC and Brier |
| 8 | PILOT (12 binding/multi prompts, 216 images) |
| 9 | Existing |
| 10 | 12 scenes |
| 11 | Existing 6 seeds × 3 CFG |
| 12 | 0 new |
| 13 | B0, B1 baselines |
| 14 | M_att vs B1 |
| 15 | Leave-one-prompt-out CV; scene bootstrap; **EXPLORATORY** label mandatory |
| 16 | CPU, minutes |
| 17 | New `src/predict_failure.py` (shared with E05) |
| 18 | `outputs/main/results/feasibility_h1.csv` |
| 19 | Table of ΔAUC with CI |
| 20 | (a) ΔAUC clearly > 0; (b) ≈ 0 with wide CI; (c) < 0 |
| 21 | (a) go; (b) go but flag power concerns, rely on DEV simulation; (c) reconsider the primary direction (fall back to C3+C5+C6, §21.3) |
| 22 | 12-prompt noise; object-identity leakage |
| 23 | CI upper bound < 0 with leave-prompt-out |
| 24 | P0 (gate) |

#### E05 — Early-attention failure prediction on held-out scenes (P1, H1)
| Field | Content |
|---|---|
| 1 | Does early cross-attention add held-out predictive information about colour-binding correctness beyond B1? |
| 2 | Pilot exploratory AUC (§2.3); Diffusion Probe (quality, ABS); Kim 2024 (prompt-side signal) |
| 3 | H1 (§8.1) |
| 4 | If binding is largely decided early, the colour token's map overlaps the wrong object in doomed samples |
| 5 | Feature set; step fraction τ ∈ {12, 20, 39}; CFG {2, 4.5, 7.5, 11, 15} as covariate |
| 6 | Model class (L2 logistic regression), regularisation grid, split manifest |
| 7 | ΔAUC (primary), ΔBrier, Δlog-loss |
| 8 | DEV for fitting; TEST sealed |
| 9 | CAN prompts of binding-bearing scenes |
| 10 | DEV ≈ 70 scenes, TEST ≈ 140 scenes |
| 11 | k = 4 (shared with E11) |
| 12 | `140 scenes × 5 CFG × 4 seeds = 2,800` TEST predictions (images from E04/E11) |
| 13 | B0, B1 (and B3 if implemented) |
| 14 | M_att vs B1 at τ = 12; also vs B3 |
| 15 | Scene-level bootstrap CI (§8.1); Holm; calibration plot |
| 16 | CPU |
| 17 | `src/features.py` (feature extraction from npz), `src/predict_failure.py`, tests for feature invariants |
| 18 | `outputs/h1/{model.pkl, predictions.csv, delta_auc.json}` |
| 19 | Fig. ΔAUC with CI; calibration; feature-importance (EXPLORATORY) |
| 20 | See §10 matrix M-E05 |
| 21 | See §10 |
| 22 | Features proxy object identity, CFG or detectability |
| 23 | ΔAUC CI upper bound < 0.03 |
| 24 | P1 |

#### E08 — Wording sensitivity under meaning-preserving rewrites (P1, H2)
| Field | Content |
|---|---|
| 1 | Do PAR/ORD rewrites shift scene-level success beyond seed noise, by constraint type? |
| 2 | MetaLogic (equivalence), HRS-Bench robustness, Vemishetty et al. (ABS/FT abstract) |
| 3 | H2 (§8.2) |
| 4 | Text encoder maps rewrites to different embeddings, changing attention allocation |
| 5 | Variant (CAN, PAR, ORD) × constraint type |
| 6 | Seed (same seed within pair), CFG, steps |
| 7 | Excess RMS scene-level shift (§8.2) |
| 8 | TEST (and DEV for protocol tuning) |
| 9 | §11.2 rewrite rules + **human validation of every rewrite** |
| 10 | 200 TEST scenes × (CAN + 3 rewrites) |
| 11 | k = 4 |
| 12 | `200 × 4 × 4 = 3,200` TEST images (inside the E04 count) |
| 13 | Seed-label-swap permutation null |
| 14 | `T_κ` observed vs permutation null; between-type differences |
| 15 | Within-seed permutation test; scene bootstrap CI; Holm |
| 16 | CPU |
| 17 | New `src/analyze_wording.py` |
| 18 | `outputs/h2/{wsi.json, permutations.csv}` |
| 19 | Fig. excess RMS shift by type with CI |
| 20 | §10 matrix M-E08 |
| 21 | §10 |
| 22 | Judge less accurate on rewritten prompts; rewrite not truly meaning-preserving |
| 23 | CI upper < 0.05 for binding and spatial |
| 24 | P1 |

#### E11 — CFG × steps × complexity (P1, PS requirement)
| Field | Content |
|---|---|
| 1 | How does the CFG effect depend on constraint count and constraint type; do steps interact? |
| 2 | Pilot: CFG 2→7.5 effect +0.07 [−0.01, +0.15] (VERIFIED); CFG schedules and intervals matter in the literature (ABS) |
| 3 | PROPOSED (exploratory): gain from CFG 2→7.5 is smaller for ≥ 3-constraint scenes than for 2-constraint scenes |
| 4 | Guidance trades diversity for prompt fidelity; multi-constraint prompts need more simultaneous alignment |
| 5 | CFG {2, 4.5, 7.5, 11, 15}; steps {20, 40}; complexity level (constraint count 2–5, balanced) |
| 6 | Scene, seed, sampler, resolution |
| 7 | Constraint success and soft fraction; CLIP score; diversity |
| 8 | 150 canonical scenes (50 DEV + 100 TEST) |
| 9 | CAN |
| 10 | 150 scenes |
| 11 | k = 4 |
| 12 | `150 × (5×2 − 1) × 4 = 5,400` images beyond the shared core cell (FULL); LIMITED: `100 × (3×2 − 1) × 4 = 2,000`; attention on the 40-step cells |
| 13 | CFG 7.5 / 40 steps core cell |
| 14 | Paired contrasts across CFG; interaction CFG × complexity |
| 15 | Scene-clustered logistic GEE with interaction terms; effect sizes as risk differences; CI by scene bootstrap |
| 16 | Per §16 |
| 17 | `configs/experiment_v2.json` (cfg list, steps list, scene subset), `analyze.py` extended with a scene-level key |
| 18 | `outputs/e11/…` |
| 19 | Fig. adherence vs CFG by complexity; interaction table |
| 20 | §10 matrix M-E11 |
| 21 | §10 |
| 22 | Floor effects; object-difficulty confounding |
| 23 | Interaction CI includes 0 and the CI is narrow enough to exclude an interaction of absolute size ≥ 0.05 |
| 24 | P1 |

#### E12 — Causal probe: targeted vs random vs global vs control (P1, H3)
| Field | Content |
|---|---|
| 1 | Does a targeted attention edit improve binding more than random and global edits? |
| 2 | Pilot: global edit +0.02 (VERIFIED); H1-dependent |
| 3 | H3 (§8.3) |
| 4 | Attribute-token logit bias on the noun's region (and negative bias on the other noun's region), gated by a conflict score (§13) |
| 5 | Arm {control, global s = 2, targeted, random-pair} |
| 6 | Scene, seed, CFG 7.5, 40 steps, bias magnitude β and mask area (matched between targeted and random) |
| 7 | Binding-correct rate; CLIP score; duplicate-object rate |
| 8 | DEV (tune: 8 configs, k = 2) then TEST (frozen) |
| 9 | CAN prompts of binding-bearing scenes |
| 10 | DEV ≈ 70, TEST ≈ 140 |
| 11 | k = 4 (TEST) |
| 12 | TEST: `140 × 3 new arms × 4 = 1,680`; DEV tuning: `70 × 8 × 2 = 1,120` (control reuses core images — they are bit-identical to hook-off) |
| 13 | Control, global (existing `intervene.py`), random-pair, shuffled-mask |
| 14 | targeted − random (primary), targeted − control, targeted − global |
| 15 | Paired scene-bootstrap CI; Holm |
| 16 | Per-step extra compute logged |
| 17 | `intervene.py`: add `--mode {global,targeted,random_pair,shuffled_mask}`, conflict score, mask builder, per-pixel bias; unit tests (§17) |
| 18 | `samples.jsonl` fields `arm`, `beta`, `tau_c`, `q`, `n_edits`, `skipped_reason` |
| 19 | Fig. arm comparison with CI; guard-metric table |
| 20 | §10 matrix M-E12 |
| 21 | §10 |
| 22 | Mask error; evaluator-dependent gain; implementation fault |
| 23 | CI upper bound of targeted − random < 0.04 |
| 24 | P1 |

#### E18 — Evaluator sensitivity of every primary conclusion (P1)
| Field | Content |
|---|---|
| 1 | Does any primary conclusion change if J1 is replaced by J2 or by audit-based calibration? |
| 2 | GenEval 2: evaluator drift can change rankings |
| 3 | PROPOSED: the sign and rough size of H1–H3 estimates are stable across judges |
| 4 | Judge errors are not uniform across conditions |
| 5 | Judge (J1, J2, J-calibrated) |
| 6 | Images, constraints |
| 7 | Change in each primary estimate and CI |
| 8 | All TEST images; audit subset for calibration |
| 9 | n/a |
| 10 | TEST |
| 11 | n/a |
| 12 | `images × atoms` VLM calls for J2 (assumption ≈ 6 atoms per prompt; measure in E03) |
| 13 | J1 |
| 14 | Estimate under each judge |
| 15 | Same as the primary analysis per judge; regression calibration with the audit as a sensitivity analysis |
| 16 | VLM inference (VRAM-limited) |
| 17 | `evaluate.py`: judge interface; new `src/judge_vqa.py` |
| 18 | per-judge `eval_J1/J2.jsonl` |
| 19 | Fig. estimate by judge |
| 20 | Stable / sign flip / size change |
| 21 | See §10 matrix M-E18 |
| 22 | J2 bias; calibration model misspecified |
| 23 | n/a |
| 24 | P1 |

#### E06 — Stage/timing of the diagnostic (P2)
| Field | Content |
|---|---|
| 1 | At what step fraction does the attention signal appear, and does a late signal beat an early one? |
| 2 | R&B (layout early; ABS) vs SD-as-feature-extractor (binding later; ABS) |
| 3 | PROPOSED: ΔAUC at τ = 12 ≥ ΔAUC at τ = 39 − 0.02 (early not much worse) |
| 4 | Binding ownership may be settled mid-trajectory |
| 5 | τ ∈ {5, 8, 12, 16, 20, 26, 33, 39} |
| 6 | Everything else |
| 7 | ΔAUC(τ), time-to-detect curve |
| 8 | DEV (curve), TEST (single frozen τ) |
| 9 | CAN |
| 10 | As E05 |
| 11 | 4 |
| 12 | 0 new |
| 13 | B1 at each τ |
| 14 | ΔAUC across τ |
| 15 | Scene bootstrap; monotone-trend description; **EXPLORATORY** |
| 16 | CPU |
| 17 | `features.py` parameter τ |
| 18 | `h1/tau_curve.csv` |
| 19 | Fig. ΔAUC vs step |
| 20 | §10 M-E06 |
| 21 | §10 |
| 22 | Features become more informative simply because outcome leaks into later maps |
| 23 | n/a |
| 24 | P2 |

#### E07 — Feature families, layers and resolutions vs baselines (P2)
| Field | Content |
|---|---|
| 1 | Which attention feature family or resolution carries the signal, relative to simple baselines? |
| 2 | Pilot averaged all layers; DAAM uses different aggregation (§2.5 #9) |
| 3 | PROPOSED: no single family accounts for the gain; 16×16 ≥ 32×32 |
| 4 | Layer resolution tracks layout vs detail |
| 5 | Family (overlap, entropy, localisation, box ratio), resolution (16, 32, all), layer group (down/mid/up) |
| 6 | Model class |
| 7 | ΔAUC per family |
| 8 | DEV; TEST for the frozen choice |
| 9 | CAN |
| 10 | As E05 |
| 11 | 4 |
| 12 | 0 new (needs per-layer storage in E04) |
| 13 | B0, B1, B3, and a **shuffled-attention** negative control (features from another scene's maps) |
| 14 | Family ablations |
| 15 | Scene bootstrap; FDR-BH; EXPLORATORY |
| 16 | CPU |
| 17 | `attention.py`: per-layer-group storage; `features.py` families |
| 18 | `h1/family_ablation.csv` |
| 19 | Table of family ablations |
| 20 | §10 M-E06/E07 |
| 21 | §10 |
| 22 | Collinearity among features |
| 23 | n/a |
| 24 | P2 |

#### E09 — Counterfactual edit fidelity (P2)
| Field | Content |
|---|---|
| 1 | After a one-property edit, does the target property change correctly and do the others stay the same? |
| 2 | Gap C3 (§5) |
| 3 | PROPOSED: pair-level fidelity is lower than the product of the two single-image success rates under independence |
| 4 | Entanglement of attributes through shared text context (DAAM: adjectives attend broadly; FT) |
| 5 | Edit type CF-C, CF-R, CF-N, CF-O, CF-A |
| 6 | Seed (same seed in pair), CFG, steps |
| 7 | Target-change success; preservation of unchanged constraints; pair-level edit fidelity |
| 8 | TEST pairs |
| 9 | §11.2 |
| 10 | ≈ 200 scenes × ≈ 2 applicable edits |
| 11 | 4 |
| 12 | Inside E04 |
| 13 | Independence expectation |
| 14 | Observed vs expected |
| 15 | Scene bootstrap; per-edit McNemar-style paired test |
| 16 | CPU |
| 17 | `analyze_counterfactual.py` |
| 18 | `outputs/e09/…` |
| 19 | Fig. fidelity by edit type |
| 20 | §10 M-E09 |
| 21 | §10 |
| 22 | Evaluator weaker on edits |
| 23 | n/a |
| 24 | P2 |

#### E10 — Constraint complexity and interactions (P2)
| Field | Content |
|---|---|
| 1 | How do success and soft score fall with constraint count, attribute count and relation count, with objects balanced? |
| 2 | Pilot confounds complexity with category/object |
| 3 | PROPOSED: success declines roughly multiplicatively with the number of independent constraints |
| 4 | Conjunction of independent per-constraint probabilities |
| 5 | Constraint count; relation count; attribute count |
| 6 | Objects balanced across levels |
| 7 | Success; soft fraction |
| 8 | DEV + TEST CAN |
| 9 | Grammar-controlled |
| 10 | ≥ 20 scenes per level |
| 11 | 4 |
| 12 | Inside E04 |
| 13 | Independence model `Π p_i` |
| 14 | Observed vs independence prediction |
| 15 | GEE with scene clustering; deviation CI |
| 16 | CPU |
| 17 | `analyze.py` complexity factors |
| 18 | `outputs/e10/…` |
| 19 | Fig. success vs complexity with independence curve |
| 20 | §10 M-E11 |
| 21 | §10 |
| 22 | Floor at high complexity |
| 23 | n/a |
| 24 | P2 |

#### E14 — Adherence–quality–diversity–compute trade-off (P2)
| Field | Content |
|---|---|
| 1 | What does each CFG, steps or correction setting cost in latency, memory, CLIP similarity, preference score and diversity? |
| 2 | PS asks for quality and diversity; pilot CLIP/diversity effect is strong (VERIFIED) |
| 3 | PROPOSED: adherence gains of the targeted arm come at < 5% latency overhead |
| 4 | Extra attention recomputation is local |
| 5 | CFG, steps, arm |
| 6 | Scene, seed |
| 7 | Adherence, CLIP score, HPS v2 or ImageReward, diversity, seconds, peak VRAM |
| 8 | E11/E12 images |
| 9 | n/a |
| 10 | As E11/E12 |
| 11 | 4 |
| 12 | 0 new |
| 13 | Core cell |
| 14 | Pareto comparison |
| 15 | Scene-bootstrap CI per metric |
| 16 | Per-image timing/VRAM logged |
| 17 | `generate.py`, `intervene.py` log time and memory |
| 18 | `tradeoff.csv` |
| 19 | Fig. trade-off |
| 20 | §10 M-E14 |
| 21 | §10 |
| 22 | Shared GPU contention |
| 23 | n/a |
| 24 | P2 |

#### E15 — Object-identity robustness (P2)
| Field | Content |
|---|---|
| 1 | Does the H1 result survive splits that hold out object pairs, not only scenes? |
| 2 | Pilot: categories use different objects (§2.5) |
| 3 | PROPOSED: ΔAUC under leave-object-pair-out ≥ 0.5 × ΔAUC under scene split |
| 4 | Features could encode object-pair difficulty |
| 5 | Split type |
| 6 | Model, features |
| 7 | ΔAUC by split |
| 8 | DEV + TEST |
| 9 | n/a |
| 10 | As E05 |
| 11 | 4 |
| 12 | 0 |
| 13 | B1 incl. detector-prior difficulty |
| 14 | Scene split vs object-pair split |
| 15 | Bootstrap over object pairs |
| 16 | CPU |
| 17 | `splits.json` includes object-pair groups |
| 18 | `h1/identity_robustness.csv` |
| 19 | Table |
| 20 | §10 M-E05 |
| 21 | §10 |
| 22 | Too few object pairs |
| 23 | n/a |
| 24 | P2 |

#### E16 — Cross-model replication on SDXL (P2)
| Field | Content |
|---|---|
| 1 | Does the diagnostic (H1) and the benchmark baseline transfer to SDXL? |
| 2 | SDXL GenEval 0.55 vs SD 1.5 0.43 (FT); SDXL keeps UNet cross-attention |
| 3 | PROPOSED: ΔAUC sign agrees with SD 1.5 |
| 4 | Same attention mechanism, different capacity |
| 5 | Model |
| 6 | Scenes (100 canonical TEST), seeds |
| 7 | Baseline success; ΔAUC with frozen features |
| 8 | TEST subset |
| 9 | CAN |
| 10 | 100 |
| 11 | 4 |
| 12 | 400 images + 400 extractions (FULL), 240 (LIMITED) |
| 13 | SD 1.5 results |
| 14 | Sign and size agreement |
| 15 | Scene bootstrap CI of ΔAUC difference |
| 16 | SDXL fp16 must be measured (E03); offload possible; **UNVERIFIED that it fits 16 GB with the hook** |
| 17 | `generate.py` pipeline factory (`AutoPipeline`), attention processors for SDXL transformer blocks, tests |
| 18 | `outputs/sdxl/…` |
| 19 | Table |
| 20 | §10 M-E16 |
| 21 | §10 |
| 22 | Resolution 1024 changes map scale |
| 23 | n/a |
| 24 | P2 |

#### E19 — Error analysis and failure taxonomy (P2)
| Field | Content |
|---|---|
| 1 | What are the concrete failure modes in each category? |
| 2 | Pilot gallery shows colour leakage and dropped objects (VERIFIED visually) |
| 3 | (descriptive) |
| 4 | — |
| 5 | Failure type (omission, duplication, leakage, swap, wrong count, wrong side) |
| 6 | — |
| 7 | Failure-type proportions per category; ambiguous cases |
| 8 | Audit sample + stratified failures |
| 9 | n/a |
| 10 | ≈ 150 failure images |
| 11 | n/a |
| 12 | 0 |
| 13 | n/a |
| 14 | Proportions by category |
| 15 | Wilson CIs; inter-annotator agreement |
| 16 | Human time |
| 17 | `audit_sample.py` mode |
| 18 | `taxonomy.csv` |
| 19 | Failure gallery (selected by rule, not by appearance) |
| 20 | Descriptive |
| 21 | Used to interpret H1–H3 |
| 22 | Selection bias |
| 23 | n/a |
| 24 | P2 |

#### E20 — Seed sufficiency and variance decomposition (P2)
| Field | Content |
|---|---|
| 1 | How is variance split between scene, seed and noise, and is k = 4 enough? |
| 2 | Pilot ICC(binding_ok) = 0.52 (DERIVED) |
| 3 | PROPOSED: the between-scene share stays > 0.3 on V2 |
| 4 | — |
| 5 | k ∈ {2, 4, 6, 8} (subsampling) |
| 6 | — |
| 7 | ICC, design effect, effective n |
| 8 | PILOT + DEV |
| 9 | n/a |
| 10 | — |
| 11 | 6 (pilot), 4 (V2) |
| 12 | 0 |
| 13 | n/a |
| 14 | k-subsampling curves |
| 15 | Variance components by ANOVA/GLMM |
| 16 | CPU |
| 17 | analysis script |
| 18 | `variance_components.json` |
| 19 | Table |
| 20 | Any |
| 21 | If ICC is higher than expected, shift budget from seeds to scenes |
| 22 | — |
| 23 | n/a |
| 24 | P2 |

#### E22 — Synthetic ground-truth evaluator probe (P2)
| Field | Content |
|---|---|
| 1 | How accurate are J1/J2 on images whose ground truth is exact (rendered coloured shapes at known locations)? |
| 2 | Robustness paper uses simple shape + location scenes (FT); isolates the evaluator from generation |
| 3 | PROPOSED: J1 colour/spatial accuracy on synthetic scenes ≥ its audit accuracy on natural images |
| 4 | Removes generator uncertainty |
| 5 | Judge; object size; overlap |
| 6 | Scene layout and colours |
| 7 | Accuracy by type/overlap |
| 8 | ≈ 500 programmatically rendered scenes |
| 9 | Programmatic |
| 10 | 500 |
| 11 | n/a |
| 12 | 0 T2I images |
| 13 | n/a |
| 14 | J1 vs J2 vs truth |
| 15 | Wilson CI |
| 16 | CPU/GPU for judges |
| 17 | `src/synth_probe.py` |
| 18 | `synth_probe.json` |
| 19 | Table |
| 20 | Any |
| 21 | Separates "evaluator logic" errors from "natural-image" errors |
| 22 | Synthetic domain gap |
| 23 | n/a |
| 24 | P2 |

#### E13 — Intervention component ablations (P3)
| Field | Content |
|---|---|
| 1 | Which components matter: token selection, trigger, stage window, locality, bias vs loss? |
| 2 | §5.3 overlap concerns |
| 3 | PROPOSED: removing the trigger does not change the effect size by more than 0.03 |
| 4 | — |
| 5 | Token set {object-only, attribute-only, pair}; trigger {on, off}; window {early, middle, late, all}; locality {local, global} |
| 6 | β, q, CFG, steps |
| 7 | Binding-correct rate |
| 8 | 60 binding-bearing TEST scenes |
| 9 | CAN |
| 10 | 60 |
| 11 | 4 |
| 12 | `6 arms × 60 × 4 = 1,440` |
| 13 | Targeted full arm |
| 14 | Each ablation vs full |
| 15 | Paired scene bootstrap; FDR-BH; EXPLORATORY |
| 16 | As E12 |
| 17 | `intervene.py` flags |
| 18 | arm logs |
| 19 | Ablation matrix |
| 20 | §10 M-E12 |
| 21 | §10 |
| 22 | Underpowered at 60 scenes |
| 23 | n/a |
| 24 | P3 |

#### E17 — SD3 Medium generation + evaluation only (P3 / Defer)
| Field | Content |
|---|---|
| 1 | What are SD3 baseline adherence levels on the same benchmark? |
| 2 | SD3 Medium GenEval 0.74 (aggregator, ABS); MMDiT joint attention changes extraction |
| 3 | (none; descriptive) |
| 4 | Joint text/image attention |
| 5 | Model |
| 6 | Scenes, seeds |
| 7 | Success by category |
| 8 | 100 canonical TEST |
| 9 | CAN |
| 10 | 100 |
| 11 | 4 |
| 12 | 400 images, 0 extractions |
| 13 | SD 1.5/SDXL |
| 14 | Descriptive |
| 15 | Scene bootstrap |
| 16 | **UNVERIFIED** fit on a 16 GB card; docs warn < 24 GB is difficult (ABS) → offload or text-encoder drop (quality caveat) |
| 17 | `generate.py` pipeline factory; latent shape for 16 channels |
| 18 | `outputs/sd3/…` |
| 19 | Table |
| 20 | Any |
| 21 | Descriptive context only |
| 22 | T5 removal lowers quality (ABS) |
| 23 | n/a |
| 24 | P3 / Defer |

### 9.3 Revision 2: new and re-prioritised experiments
| ID | Title | Priority | New generation? | Hypothesis |
|---|---|---|---|---|
| **E23** | Constraint commitment curves by interventional re-noising | **P1** | yes (branch runs) | **HA** |
| **E25** | Initial-noise determinism and noise-only readout | **P1** (mandatory baseline) | no | HB baseline |
| E23b | Sensitivity of `t50` to the restart magnitude Δ | P2 | yes (DEV) | HA robustness |
| E24 | Outcome fragility under micro-perturbation | P2 | yes | supporting |
| E26 | Causal layer-group knockout with matched random control | P2 | yes | supporting |
| E27 | Guidance attribution by constraint tokens | P2 | no generation (UNet evaluations) | supporting |
| E28 | Negative-prompt timing for binding repair | P3 | yes | supporting |

**Changes to existing cards (Revision 2):**
| Card | Change |
|---|---|
| E00 | Also record cross-hardware **flip rate** of constraint outcomes (data for E24) |
| E01 / E18 | Audit and sensitivity analyses cover **conditional** outcomes (`Y_bind\|pres`, `Y_spat\|pres`) and branch/edit images; per-type accuracy reported conditionally |
| E03 | Also time one **branch run** (restart + re-denoise) and measure the cost of saving latents at the six grid steps |
| E04 | Core generation additionally **saves `x_t` at grid steps {6, 12, 18, 24, 30, 36}** for the 100 commitment scenes × 2 seeds (≈ 32 KB per latent in fp16) |
| E05 | Outcome redefined to `Y_bind\|pres`; adds **B_noise**; becomes the HB analysis pipeline; τ windows come from DEV `t50` |
| E08 | Demoted to P2 (supporting H2); unchanged otherwise |
| E12 | **Onset arms:** targeted@early / @mid / @late, random@early / @late, global s = 2@0 (6 arms; `140 × 6 × 4 = 3,360` images FULL; 4 arms `70 × 4 × 4 = 1,120` LIMITED); window length L = 12 |
| E21 | Gate now also reports the conditional-label AUC and the noise-only baseline on the pilot images (noise reproducible from seeds; no new generation) |

#### E23 — Constraint commitment curves by interventional re-noising (P1, HA)
| Field | Content |
|---|---|
| 1 | At which step is each constraint's outcome decided, per constraint type? |
| 2 | Pilot F1 (outcomes persist across CFG for the same seed); critical-window theory (ICML 2024, ABS); conflicting early-layout vs late-binding claims (ABS) |
| 3 | HA (§8.1), two-sided |
| 4 | Trajectory information about a constraint is progressively fixed; local re-noising of magnitude Δ at step `t` destroys it only if it has not yet been committed |
| 5 | Constraint type; grid step `t ∈ {6, 12, 18, 24, 30, 36}` |
| 6 | Scene, base seed, prompt, CFG 7.5, 40 steps, DDIM eta 0, Δ = 8, `M = 2`, branch-noise seeds, judge |
| 7 | `C_κ(t)` and `t50_κ` (§8.0) |
| 8 | DEV (40 scenes, rehearsal + `t50_DEV`) and TEST (≈ 100 canonical scenes); base images from E04 |
| 9 | CAN prompts only |
| 10 | DEV 40; TEST ≈ 100 (presence all; binding-bearing ≈ 70; spatial-bearing ≈ 35) |
| 11 | 2 base seeds per scene (of the 4 in E04); `M = 2` branch noises. Rationale: the estimand is within-scene, so more scenes beat more seeds (ICC of outcomes across prompts is high, §15.4) |
| 12 | FULL: `100 × 2 × 6 × 2 = 2,400` branch-runs ≈ **1,600 image-equivalents** (average remaining steps 26.7 of 40); DEV rehearsal `40 × 2 × 6 × 2 = 960` runs. LIMITED: `60 × 2 × 5 × 2 = 1,200` runs ≈ 890 equivalents (five grid steps). 0 attention extractions |
| 13 | **Negative controls:** (a) `Δ = 0` branch (ε = 0) must reproduce the base image exactly; (b) a **full-resample** branch (restart to the first-step noise level) must give `C ≈ 0`; (c) an **independent-seed** baseline gives `A0` |
| 14 | `C_κ(t)` curves by type; `Δt50` between binding\|pres and presence (primary); other pairs EXPLORATORY |
| 15 | Ratio estimator; 4000-draw scene bootstrap; Holm over {HA, HB, HC} |
| 16 | Cost = Σ remaining steps / 40 per branch; same VRAM as generation; latents stored as fp16; the **measured** `t_branch` comes from E03 |
| 17 | `generate.py`: `--save-latents-at 6,12,…`; new `src/branch.py` (restart operation, branch generator, deterministic re-denoise), `src/commitment.py` (agreement, ratio estimator, `t50`, bootstrap); tests (§17.1) |
| 18 | `outputs/e23/branches.jsonl` (scene, seed, t, Δ, branch seed, outcome per constraint, judge), `curves.csv`, `t50.json` |
| 19 | F16 (commitment curves), T3 (`t50` with CIs), F17 (`C` vs CFG, EXPLORATORY) |
| 20 | See §10 matrix M-E23 |
| 21 | See §10 |
| 22 | Δ is a free parameter; judge noise attenuates agreement; restart noise interacts with CFG; conditional outcomes shrink the usable sample; deterministic re-denoising after restart means branch variability comes only from the injected noise |
| 23 | CI of `Δt50` inside (−1.5, +1.5) steps |
| 24 | P1 |

#### E23b — Sensitivity of `t50` to the restart magnitude Δ (P2)
| Field | Content |
|---|---|
| 1 | Is a type ordering of `t50` stable when Δ changes? |
| 2 | Δ defines "local" noise |
| 3 | PROPOSED: ordering of types is the same for Δ ∈ {4, 8, 12} |
| 4 | Larger Δ erases more information at the same `t` |
| 5 | Δ ∈ {4, 12} (8 from E23) |
| 6 | As E23 |
| 7 | `t50(Δ)`, rank correlation of types across Δ |
| 8 | 20 DEV scenes |
| 9 | CAN |
| 10 | 20 |
| 11 | 2 base seeds, M = 2 |
| 12 | `20 × 2 × 6 × 2 × 2 = 960` runs (≈ 640 equivalents) |
| 13 | Δ = 8 |
| 14 | Orderings across Δ |
| 15 | Spearman across types with scene bootstrap; EXPLORATORY |
| 16 | As E23 |
| 17 | `branch.py` parameter |
| 18 | `e23b/t50_by_delta.csv` |
| 19 | Table |
| 20 | Stable / unstable ordering |
| 21 | If unstable, `t50` is reported only relative to Δ and HA is qualified |
| 22 | Few scenes |
| 23 | n/a |
| 24 | P2 |

#### E25 — Initial-noise determinism and noise-only readout (P1)
| Field | Content |
|---|---|
| 1 | How much of the final outcome is determined by the initial latent, and how much is readable from it? |
| 2 | F1 (corr 0.79 between CFG 7.5 and 15); InitNO / noise-refinement papers (ABS); mandatory baseline for HB |
| 3 | PROPOSED: a noise-only model predicts `Y_bind\|pres` above the prompt-only baseline (AUC gain ≥ 0.03) |
| 4 | The initial latent fixes coarse layout, which constrains later binding |
| 5 | Feature set (prompt-only, noise-only, both) |
| 6 | Scenes, splits |
| 7 | Variance components (noise vs prompt vs CFG); AUC of noise-only readout |
| 8 | Pilot (now) and DEV/TEST (later) |
| 9 | CAN |
| 10 | Pilot 12 prompts; later ≈ 70 DEV / ≈ 140 TEST binding-bearing |
| 11 | Existing seeds; 4 in V2 |
| 12 | **0 new images** — `x_T` is regenerated from the seed on the CPU (`initial_latents`); no storage needed |
| 13 | B1 prompt-only; shuffled-noise control |
| 14 | noise-only vs B1; seed main effect |
| 15 | Scene-bootstrap AUC differences; permutation test for a seed main effect (pilot seed-level success 0.19, 0.286, 0.268, 0.274, 0.31, 0.31; EXPLORATORY); ANOVA variance components |
| 16 | CPU |
| 17 | `features.py` noise features; `analyze_noise.py` |
| 18 | `outputs/e25/noise_readout.json` |
| 19 | F18 (variance decomposition), T4 (readout AUCs) |
| 20 | §10 M-E25 |
| 21 | §10 |
| 22 | Features capture object-pair difficulty only; leakage across scenes |
| 23 | Noise-only AUC gain CI upper < 0.02 |
| 24 | P1 |

#### E24 — Outcome fragility under micro-perturbation (P2)
| Field | Content |
|---|---|
| 1 | How often does a constraint outcome flip under tiny perturbations, and does fragility relate to failure and to commitment? |
| 2 | Same-noise reproducibility results exist for CIFAR-scale models (ABS); no T2I constraint-level study found (UNVERIFIED gap) |
| 3 | PROPOSED (exploratory): failures are less stable than successes |
| 4 | Samples near a decision boundary flip under tiny noise |
| 5 | Perturbation size `η ∈ {0.02, 0.05}` with variance-preserving perturbation `x_T' = sqrt(1 − η²) x_T + η δ`; precision fp16 vs fp32; hardware (E00) |
| 6 | Scene, seed, prompt, settings |
| 7 | Flip rate; margin `m = 1 − flip rate`; AUC of `m` for the base outcome |
| 8 | 100 TEST scenes (or DEV for tuning) |
| 9 | CAN |
| 10 | 100 |
| 11 | 2 base seeds × `R = 3` replicates |
| 12 | FULL `100 × 2 × 2 × 3 = 1,200` + fp32 control `100 × 2 = 200` = **1,400**; LIMITED `60 × 2 × 1 × 3 + 120 = 480` |
| 13 | `η = 0` (must reproduce exactly) |
| 14 | Margin of successes vs failures; relation to `t50` |
| 15 | Scene bootstrap; EXPLORATORY |
| 16 | fp32 SD 1.5 ≈ twice the fp16 weights (**UNVERIFIED fit with the hook**; check in E03) |
| 17 | `src/fragility.py` |
| 18 | `e24/flips.jsonl` |
| 19 | F19 |
| 20 | §10 M-E24 |
| 21 | §10 |
| 22 | Judge noise inflates flips |
| 23 | n/a |
| 24 | P2 |

#### E26 — Causal layer-group knockout with matched random control (P2)
| Field | Content |
|---|---|
| 1 | Which cross-attention layer groups are necessary for each constraint type? |
| 2 | Per-layer ablation for editing (arXiv 2403.03431; ABS); CoBind's resolution assumption (ABS); the ACL 2026 head-level paper (unread) |
| 3 | PROPOSED (exploratory): knocking out constraint tokens in 16×16/32×32 layers lowers `Y_bind\|pres` more than in 64×64 or 8×8 layers |
| 4 | Resolution tracks layout vs detail |
| 5 | Layer group (R64: 5 layers; R32: 5; R16: 5; R8 mid: 1) × token set (constraint tokens vs matched random non-content tokens) |
| 6 | Scene, seed, settings, number of knocked-out tokens |
| 7 | Change in `Y_pres`, `Y_count`, `Y_bind\|pres`, `Y_spat\|pres` relative to control |
| 8 | 40 DEV scenes (exploratory) |
| 9 | CAN |
| 10 | 40 |
| 11 | 4 |
| 12 | `40 × 4 × (4 groups × 2) = 1,280` images |
| 13 | Control (bias ≡ 0) and the random-token knockout in the same groups |
| 14 | Real − random per group and outcome |
| 15 | Paired scene bootstrap; FDR-BH; EXPLORATORY |
| 16 | As E12 |
| 17 | `intervene.py`: `--mode knockout --layer-group R16 --tokens {constraint, random}` (implemented as per-layer additive bias −B on key columns; B large and finite) |
| 18 | arm logs |
| 19 | F20 (matrix of effects) |
| 20 | §10 M-E26 |
| 21 | §10 |
| 22 | Random-token knockouts also degrade (generic effect) |
| 23 | n/a |
| 24 | P2 |

#### E27 — Guidance attribution by constraint tokens (P2)
| Field | Content |
|---|---|
| 1 | Which constraint tokens carry the guidance vector `ε_cond − ε_uncond`, and how does this change with CFG? |
| 2 | CFG raises similarity more than adherence in the pilot (VERIFIED, §2.3); token-removal attribution and guidance-by-token ideas exist (ABS) |
| 3 | PROPOSED (exploratory): guidance mass concentrates on object tokens more than attribute or relation tokens |
| 4 | Content tokens dominate the conditional signal |
| 5 | Token class (object, colour, count, relation, other); CFG levels along fixed trajectories |
| 6 | Trajectories fixed to the saved baselines |
| 7 | `c_w(t) = ‖g_t − g_t^{(−w)}‖ / ‖g_t‖` with `g^{(−w)}` computed after replacing word `w`'s token embeddings by the empty-prompt embeddings at the same positions |
| 8 | 60 scenes × 2 seeds, 10 steps |
| 9 | CAN |
| 10 | 60 |
| 11 | 2 |
| 12 | 0 generated images; `60 × 2 × 10 × 7 = 8,400` UNet evaluations ≈ **105** full-generation equivalents (80 evaluations per 40-step CFG generation) |
| 13 | Random non-content word ablation |
| 14 | Token-class contributions; scale dependence |
| 15 | Scene bootstrap; EXPLORATORY |
| 16 | GPU; same VRAM as generation |
| 17 | `src/guidance_attr.py` |
| 18 | `e27/attribution.csv` |
| 19 | F21 |
| 20 | §10 M-E27 |
| 21 | §10 |
| 22 | Norm-based importance may not equal causal importance |
| 23 | n/a |
| 24 | P2 |

#### E28 — Negative-prompt timing for binding repair (P3)
| Field | Content |
|---|---|
| 1 | Does a negative prompt naming the wrong binding help, and does the timing matter? |
| 2 | Negative prompts act with a delay and can show "reverse activation" (arXiv 2406.02965; ABS) |
| 3 | PROPOSED (exploratory): a late negative window helps less than an early one for binding |
| 4 | Delayed effect of the negative pathway |
| 5 | Window {all, early 0–11, late 12–39}; negative string {swapped-binding, random control string} |
| 6 | Scene, seed, settings |
| 7 | `Y_bind\|pres`, `Y_pres` (reverse-activation guard) |
| 8 | 140 TEST binding-bearing scenes (exploratory) |
| 9 | CAN |
| 10 | 140 |
| 11 | 4 |
| 12 | `140 × 5 arms × 4 = 2,800` images |
| 13 | No-negative control (core images); random-string negative |
| 14 | Arms vs control and vs random string |
| 15 | Paired scene bootstrap; FDR-BH; EXPLORATORY |
| 16 | As E12 |
| 17 | `generate.py`: `--negative-window` support |
| 18 | arm logs |
| 19 | Table |
| 20 | Helps / no effect / reverse activation |
| 21 | See §19.5 |
| 22 | The effect may be prompt-specific |
| 23 | n/a |
| 24 | P3 |

---

## 10. Observation and interpretation matrices

*Every row is a **POSSIBLE FUTURE OUTCOME** (nothing has been observed). "Phrase" = the sentence the paper may use **if** the pattern occurs; "Never claim" = the sentence it must not use.*

### M-E05 — Early attention diagnostics (H1)
| Pattern | Measured as | Supports / contradicts | Explanations | Distinguishing test | Allowed phrasing | Never claim |
|---|---|---|---|---|---|---|
| ΔAUC ≥ 0.05, CI excludes 0 on TEST, holds under E15 | `delta_auc.json`, tau-curve | Supports H1 | Early attention carries constraint-specific information | E12 (is it causal?), E16 (transfer) | "Under the tested setting, early attention features improved held-out prediction of colour-binding beyond prompt-structure and guidance covariates." | "attention causes binding"; "generalises to other models" |
| ΔAUC > 0 on scene split but ≈ 0 on object-pair split | E15 table | Contradicts H1's generality | Features encode object-pair difficulty | Identity-matched sub-analysis | "The signal did not generalise to unseen object pairs." | "attention explains binding" |
| ΔAUC ≈ 0 with narrow CI | CI upper < 0.03 | Rejects H1 | Prompt structure + CFG already carry the information | None needed | "Attention features added no practically meaningful predictive information." | "attention is uninformative" (only for this feature set) |
| ΔAUC point > 0 but CI includes 0 and is wide | CI width | Inconclusive | Underpowered | DEV-based power simulation; more scenes | "The study could not resolve a difference of 0.05." | "no effect" |
| Late τ beats early τ | tau-curve | Does not support *early* prediction | Binding is settled late (SD-as-feature-extractor, ABS) | E06 finer τ | "The signal strengthened later in denoising." | "failure is predictable early" |
| M_att beats B1 but not B3 (x0-detector) | B3 comparison | Mixed | A cheaper image-space signal contains the same information | Combined model | "Attention is no better than an early detector signal." | "attention is uniquely informative" |
| Shuffled-attention control matches M_att | E07 control | Contradicts H1 | Leakage via scene-level covariates | Audit feature pipeline | "Gain was not specific to the correct scene's attention." | any diagnostic claim |
| Labels noisy (E01 low accuracy on binding) | audit | Evaluator-confounded | Label noise attenuates AUC | Use J2/human labels on subset | "Results depend on judge quality for binding." | binding claim without audit |

### M-E06/E07 — Timing and feature families
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| Signal present by step ≈ 12 and flat after | ΔAUC(τ) | Early diagnostics | Binding fixed early | E12 with early-only window | "Information was available by step ~12 (of 40)." | "binding is decided at step 12" |
| Signal only in one resolution (e.g. 16×16) | family table | Localises mechanism | Layout-scale layers carry it | Layer-group ablation | "The signal was concentrated in the 16×16 maps." | "those layers cause binding" |
| Only entropy matters, overlap does not | family ablation | Changes the mechanistic story | Dispersion vs overlap | Independent test on counting | "Dispersion, not overlap, carried the signal." | "binding is an overlap problem" |
| No single family dominates | ablation | Distributed signal | Redundancy | Combined model | "The signal was distributed across features." | — |

### M-E08 — Wording sensitivity (H2)
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| Excess RMS ≥ 0.10 for binding/spatial, smaller for presence/count | `wsi.json` | Supports H2 | Text encoder sensitivity; attention reallocation | Embedding-distance analysis | "Wording changes shifted binding/spatial success by more than seed noise." | "model does not understand language" |
| Large shifts for all types | WSI | Broader than H2 | Generic prompt sensitivity | Per-type breakdown | "Sensitivity was not specific to binding." | — |
| Small shifts everywhere | CI upper < 0.05 | Rejects H2 | Robust to these rewrites | Harder rewrites | "No wording sensitivity beyond seed noise was detected for these rewrite types." | "robust to paraphrase in general" |
| Shifts only for one rewrite type (ORD) | per-variant | Partial | Order-specific bias | Mention-order analysis | "Mention-order changes drove the effect." | "paraphrase hurts" |
| Judge less accurate on rewrites | audit by variant | Evaluator-confounded | Judge brittle | Re-score with J2/humans | "Measured sensitivity partly reflects judge error." | sensitivity claim |

### M-E09 — Counterfactual edit fidelity
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| Target changes correctly, others preserved at the rate expected under independence | fidelity vs expectation | Edits behave as intended | — | — | "Edits changed the target without measurable collateral change." | "disentangled representation" |
| Target correct but unchanged constraints break | preservation rate | Entanglement | Shared text context | E12 localisation | "Single-property edits produced collateral changes." | "colour edit breaks relations" without mechanism |
| Target often unchanged | target-change rate | Model ignores the edit | Insensitivity | Larger edit magnitude | "The model often did not realise the edit." | — |
| Differences by edit type | per-type | Heterogeneous | Type difficulty | Stratified | "Fidelity depended strongly on edit type." | — |

### M-E11/E10 — CFG × steps × complexity
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| CFG raises CLIP but not adherence | CLIP vs adherence | Replicates pilot | Guidance boosts global similarity | Per-constraint breakdown | "CFG increased similarity more than constraint adherence." | "CFG does not help" |
| CFG helps 2-constraint scenes, not ≥ 4 | interaction | Complexity dependence | Floor/conjunction | E10 independence curve | "The CFG benefit depended on constraint count." | — |
| Steps 20 vs 40 no effect at any CFG | contrasts | Steps null | Converged sampler | 5–10 steps | "No step effect was detected between 20 and 40." | "steps never matter" |
| Non-monotone CFG for spatial | curve | Matches pilot hint (lowest cell at CFG 15/40, EXPLORATORY) | Oversaturation | Finer CFG grid | "Spatial success peaked at intermediate CFG." | causal claim |
| Success follows the independence model | observed vs Πp | Constraints fail independently | — | — | "Joint success matched an independence prediction." | — |
| Success below independence | deviation | Interaction between constraints | Competition | pairwise analysis | "Constraints interfered with each other." | — |

### M-E12 — Causal probe (H3)
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| targeted − random ≥ 0.08, CI > 0, guards ok | arm contrasts | Supports H3 | Attention overlap is a causal handle | E13 ablations, E16 transfer | "A targeted attention edit improved binding relative to a matched random edit." | "new state-of-the-art method" |
| targeted ≈ random | CI includes 0 | Rejects specificity | Gain (if any) is generic perturbation | Larger β | "Specificity to the colour–object pair was not demonstrated." | "attention is not causal" |
| Global helps presence, targeted helps binding | per-outcome | Dissociation | Different mechanisms | Mixed arm | "Presence and binding responded to different edits." | — |
| Targeted raises duplicates | guard metric | Trade-off | Over-attribution | Count check | "Binding gains came with more duplicate objects." | "improved adherence" without guard |
| No arm differs from control | all CIs include 0 | Inconclusive/rejected | Underpowered, wrong rule | MDE-based statement | "The study could resolve effects ≥ ~0.07 only." | "intervention does not work" |
| Gain disappears under J2/humans | E18 | Evaluator-confounded | Judge artefact | Calibration | "The apparent gain was judge-dependent." | any gain claim |
| Implementation failure (unit tests fail, zero edits fired) | `n_edits` | Invalid | Bug/threshold | Fix and rerun on DEV | "The correction was not triggered." | any result |

### M-E01/E18 — Evaluator reliability
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| J1 ≥ 0.9 on presence/count, < 0.8 on binding/spatial | audit table | Per-type usability | Crop colour & box rule coarse | E22 synthetic probe | "Binding and spatial results are reported with judge-error caveats." | "metrics are valid" |
| J1 and J2 disagree on binding systematically | confusion matrices | Judge-dependent | Different failure modes | Human tie-break | "Binding conclusions depend on the judge." | — |
| Primary estimates unchanged across judges | E18 | Robust | — | — | "Conclusions were stable across judges." | "judge-independent" |
| Sign flip under J2 | E18 | Evaluator-confounded | Judge bias | Audit-based calibration | "The sign of the estimate depended on the judge." | main conclusion |

### M-E14/E16 — Trade-offs and generalisation
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| Targeted arm cost < 5% latency | seconds | Cheap | — | — | "Overhead was small in the tested setting." | "negligible on all hardware" |
| Expensive method, small effect vs cheap CFG change | Pareto | Poor trade-off | — | — | "The correction was not cost-effective." | — |
| SDXL ΔAUC same sign, smaller | E16 | Partial transfer | More capable model fails less | — | "The direction replicated on SDXL." | "architecture-independent" |
| SDXL ΔAUC opposite | E16 | Not general | Different attention layout | — | "The signal was specific to SD 1.5." | — |
| SDXL fits only with offload and hook breaks identity | E03/E16 | Implementation limit | — | bit-identity test | "SDXL replication was not feasible." | — |

### 10.1 Revision 2 matrices (every row is a **POSSIBLE FUTURE OUTCOME**)

#### M-E23 — Commitment curves (HA)
| Pattern | Measured as | Supports / contradicts | Explanations | Distinguishing test | Allowed phrasing | Never claim |
|---|---|---|---|---|---|---|
| `Δt50 ≥ 3` (binding later than presence), CI > 0 | `t50.json` | Supports HA, direction "binding later" | Binding is resolved after objects exist | E23b (Δ), E26 (layers) | "Under the tested restart magnitude, binding commitment occurred later than presence." | "binding is decided at step X"; "in all models" |
| `Δt50 ≤ −3`, CI < 0 | `t50.json` | Supports HA, opposite direction | Colours fixed with early layout | E23b | "Colour assignment was committed earlier than object presence." | a mechanism |
| CI ⊂ (−1.5, +1.5) | `t50.json` | Bounded null | One shared commitment time | E23b | "No difference in commitment time greater than 1.5 steps was detected." | "decided at the same time in general" |
| Wide CI, point > 3 | CI width | Inconclusive | Too few conditional pairs | More scenes / M | "Could not resolve a 3-step difference." | "no difference" |
| `t50` strongly depends on Δ | E23b | Qualifies HA | `t50` is Δ-relative by construction | Report as curves, not single numbers | "Commitment was reported relative to the restart magnitude." | absolute claims |
| `C` never reaches 0.5 for some type | `t50 = ">36"` | Late/continuous commitment | Outcome decided in the last steps or judge noise | E18 judge sensitivity | "Commitment for this type was not reached by step 36." | "never decided" |
| `C` non-monotone | curves | Estimator unreliable | Judge noise, few conditional pairs | Re-score with J2; larger M | Report as unreliable | any ordering |
| Spatial `t50` earliest | curves | Compatible with early layout (ABS) | Layout is formed early | E09/E23 with CF-R | "Spatial commitment was earliest." | "word meaning irrelevant" |

#### M-E25 — Noise readout
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| Noise-only AUC gain ≥ 0.03 over B1 | `noise_readout.json` | Outcome partly written in the noise | Coarse layout in `x_T` | E23 at small `t` | "The initial latent carried outcome information beyond the prompt features." | "the seed alone decides" |
| No gain | CI | Outcome not linearly readable from `x_T` | Nonlinear or not readable | Nonlinear probe | "No linear readout from the initial latent was detected." | "not determined by noise" |
| A seed main effect across prompts | permutation test | Some seeds are bad generically | Seed-specific low-frequency content | More seeds | "A seed-level effect was present." | causal story |
| High outcome persistence (as F1) but no readout | both | Determined but not readable | Nonlinear chaotic mapping | E24 fragility | "Outcomes were stable across CFG but not linearly readable from the noise." | — |

#### M-E24 — Fragility
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| Failures have lower margin than successes | AUC of margin | Failures sit near boundaries | Failure is an unstable outcome | E23 | "Failures were less stable under micro-perturbation." | "failures are noise" |
| Successes less stable | AUC < 0.5 | Reverse | Success is a narrow region | E23 | "Successes were less stable." | — |
| Flip rates near zero everywhere | flips | Outcomes robust to tiny perturbations | Strong attractors | Larger η | "No flips at the tested perturbation sizes." | "outcomes are deterministic" |
| fp16 vs fp32 flips comparable to `η = 0.05` | flips | Numerical noise is as large as input noise | Chaotic sensitivity | Cross-hardware flips (E00) | "Numerical precision changed outcomes as often as input perturbations of size X." | hardware claims beyond tested |
| Hardware flips ≫ micro-perturbation flips | E00 | Hardware-sensitive pipeline | Kernel differences | — | "Outcomes differed across hardware." | "unreliable" |

#### M-E26 — Layer-group knockout
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| R16/R32 knockout lowers binding, others not | per-group effects vs random | Resolution-specific necessity | Mid resolutions carry binding | Head-level analysis | "Knockout in the 16×16 layers reduced binding given presence." | "binding lives in these layers" |
| All groups reduce all outcomes | effects | Generic degradation | Removing tokens hurts | Matched random control | "No group-specific effect was detected." | — |
| Random knockout also reduces binding | real − random ≈ 0 | Non-specific | Collateral damage | Smaller knockout | "Effect was not specific to constraint tokens." | specificity |
| Knockout raises presence | effects | Unexpected | Attention redistribution | Inspect maps | Report as unexpected | any claim |
| Implementation fault (bias not applied) | unit tests | Invalid | Bug | Fix on DEV | "Not applied." | any result |

#### M-E27 — Guidance attribution
| Pattern | Measured as | Supports / contradicts | Explanations | Test | Phrasing | Never claim |
|---|---|---|---|---|---|---|
| Object tokens dominate `c_w` | attribution table | Guidance emphasises objects | Content words dominate conditioning | Compare with E26 | "Guidance contribution was concentrated on object tokens." | "CFG ignores attributes" |
| Attribute/relation tokens contribute little | table | Matches pilot (CLIP up, adherence not) | Under-weighted constraints | E12 edit raising these tokens | "Attribute tokens contributed little to the guidance vector." | causal claim |
| Equal contributions | table | No concentration | — | — | "No token class dominated." | — |
| Contribution shares change with CFG | scale dependence | Scale reallocates guidance | Nonlinear trajectory | Fixed-trajectory caveat | "Shares varied with CFG along fixed trajectories." | "CFG reallocates attention" |

---

## 11. Benchmark V2: schema and generation plan

### 11.1 What it must fix (and what it must not claim)
| Pilot limitation (verified, §2.5) | Benchmark V2 feature |
|---|---|
| 4–6 prompts per category | 300 scene graphs (DEV 100, TEST 200) — size justified in §15.4 |
| Object identity confounded with category | Object inventory balanced across categories (§11.4) |
| One wording per scene | Canonical + paraphrase + order variants |
| No counterfactuals | Declared one-property edits with preservation lists |
| No complexity coverage; floor at binding/multi (median pooled success 0.06; 3 of 28 prompts at 0) | Complexity-graded scenes, calibrated on **DEV only** to avoid floor/ceiling |
| Limited validation | Blinded audit (E01) and synthetic probe (E22) |

*Not claimed:* novelty from size. T2I-CompBench++ has 8,000 prompts (ABS), GenEval 553 (FT), GenEval 2 800 prompts (ABS via EPIC), TIFA 4K (FT). What V2 adds, **if** the §5 gap checks hold, is the *combination* of scene-graph constraint records, controlled counterfactual/paraphrase variants, scene-level splits and per-step attention data.

### 11.2 Data schema
```json
{
  "scene_id": "sg_0123",
  "split": "TEST",
  "category": "binding",
  "template_family": "T_bind_2obj_table",
  "objects": [
    {"oid": "o1", "noun": "chair", "detector_query": "chair", "count": 1, "attributes": {"colour": "red"}},
    {"oid": "o2", "noun": "vase",  "detector_query": "vase",  "count": 1, "attributes": {"colour": "blue"}}
  ],
  "relations": [{"rel": "left_of", "subject": "o1", "object": "o2"}],
  "background": "on a wooden table",
  "complexity": {"n_objects": 2, "n_attributes": 2, "n_relations": 1, "n_count_constraints": 0, "n_constraints": 6},
  "object_pair_group": "chair|vase",
  "variants": [
    {"variant_id": "sg_0123_CAN", "type": "CAN",
     "prompt": "a red chair to the left of a blue vase on a wooden table",
     "constraints": ["presence:o1", "presence:o2", "binding:o1:colour=red", "binding:o2:colour=blue", "spatial:o1 left_of o2"]},
    {"variant_id": "sg_0123_PAR", "type": "PAR", "rewrite_rule": "synonym_phrase",
     "prompt": "a crimson chair standing left of a blue vase on a wooden table",
     "constraints": "same as CAN"},
    {"variant_id": "sg_0123_CFC", "type": "CF-C", "target_edit": {"field": "colour", "swap": ["o1", "o2"]},
     "prompt": "a blue chair to the left of a red vase on a wooden table",
     "constraints": ["...edited..."], "preserve": ["presence:o1", "presence:o2", "spatial:o1 left_of o2"],
     "review": {"reviewer": "R1", "status": "approved", "double_checked": false}}
  ]
}
```
`n_constraints` counts every presence, count, binding and spatial constraint once (the pilot's definition; `build_benchmark.py`), so complexity stays explicit and recomputable.

### 11.3 Template grammar and transformations
**Grammar (slots).** `SCENE := OBJ_PHRASE (REL OBJ_PHRASE)? BACKGROUND?`; `OBJ_PHRASE := DET (COUNT)? (COLOUR)? NOUN(S)`; `REL := "to the left of" | "to the right of" | "above" | "below"` (extend only after the spatial judge is validated, §12.5). Surface families vary article, conjunction and background phrase.

| Transformation | Type | Rule | Declared effect |
|---|---|---|---|
| Synonym paraphrase (PAR) | **meaning-preserving** | Replace noun/colour/relation words by a **pre-registered synonym lexicon** (e.g. "crimson" for "red"; relation paraphrases) | none (`constraints` unchanged) |
| Clause-order change (ORD-clause) | meaning-preserving | Swap the order of independent clauses | none |
| Mention-order change (ORD-mention) | meaning-preserving | Swap the order in which objects are introduced (relation expressed with its converse where needed) | none |
| Colour swap (CF-C) | **counterfactual** | Exchange the colours of two objects | colours change; presence/relations preserved |
| Relation reversal (CF-R) | counterfactual | Replace relation by its converse | relation flips; others preserved |
| Count change (CF-N) | counterfactual | n → n ± 1 within {1…4} | count changes; others preserved |
| Object deletion/insertion (CF-O) | counterfactual | Remove or add one object | presence set changes |
| Attribute insertion/removal (CF-A) | counterfactual | Add/remove a colour on one object | one binding constraint added/removed |

*Rule:* a pair is a **counterfactual** only if exactly one declared field differs (checked mechanically from the scene graph) **and** a reviewer approves; otherwise the pair is discarded. A **meaning-preserving** pair must have an identical constraint list; a reviewer approves each one without seeing images.

### 11.4 Balancing
- **Object inventory:** ≈ 40 nouns whose detector queries pass E22 (synthetic probe) and E03 (calibration), each appearing in every category's quota ≥ 3 times across splits (Latin-square assignment).
- **Colour inventory:** ≈ 8 colours that J1's CLIP classification separates in E22 (the pilot's 10-colour list includes brown, which collides with "wooden table"; decide on DEV).
- **Complexity strata:** constraint counts 2–5, ≥ 20 scenes per level across splits; objects and colours balanced within a level.
- **Count words** 1–4 only; relations restricted to left/right initially.

### 11.5 Ambiguity and duplicate detection
| Check | Method | Action |
|---|---|---|
| Ambiguous nouns (hypernyms/synonyms: cup/mug) | Maintain a conflict list | Drop one of the pair per scene |
| Detector-confusable pairs (e.g. "ball" vs "orange") | E22 pairwise confusion | Exclude the pair |
| Natural-colour conflict that confounds the judge (green grass) | Rule + review | Avoid or label |
| Relation ambiguity (occlusion, viewpoint) | Review of 100% of relation templates; synthetic-probe accuracy | Keep only relations with probe accuracy ≥ 0.85 |
| Duplicates | Normalised-string hash; scene-graph canonical isomorphism; text-embedding cosine > 0.95; n-gram Jaccard | Remove duplicates; keep one |

### 11.6 Splits and leakage policy
1. **Scene-level split:** every variant of a scene lives in one split.
2. **Object-pair groups are unique to a split** (supports E15). Pilot object pairs are used **only** in DEV.
3. **TEST sealed** (§8.4): no generation, scoring or inspection of TEST images before the freeze; TEST scene IDs and hashes recorded.
4. **No TEST-informed edits** to grammar, lexicon, judge thresholds or features.
5. **PILOT 28 prompts** may be pooled into DEV (labelled) and never into TEST.
6. **Audit sample** drawn after generation and stratified; auditors blind to arm/model/variant label where possible.

### 11.7 Human review requirements
Every PAR, ORD and CF variant is approved by ≥ 1 reviewer (text only); 20% double-reviewed; inter-reviewer agreement reported. Rejected variants are removed, not edited after the freeze.

### 11.8 Reproducible generation and documentation
`src/benchmark_v2.py` takes `--seed`, writes `scene_graphs.json`, `prompts_v2.json`, `splits.json`; the SHA-256 of generator, lexicon and outputs go into the protocol addendum. `DATASHEET.md` documents composition, construction, intended use, limitations (single language, small object inventory, synthetic-style prompts, left/right only), known judge weaknesses, and the leakage policy.

### 11.9 Size decision (the brief proposed ≈ 200 scene graphs)
**VERIFIED/DERIVED:** with ICC(binding) = 0.52 (DERIVED FROM PILOT), the design effect at k = 4 is 2.57, so raw 480 outcomes are worth ≈ 187 effective observations for 120 scenes. The single-AUC 95% half-width is ≈ 0.08 at ≈ 140 binding-bearing scenes. A **total of 200** scene graphs (≈ 100 TEST) would leave ≈ 70 binding-bearing TEST scenes and a half-width of ≈ 0.10, which cannot resolve ΔAUC = 0.05. **Recommendation: 300 (DEV 100, TEST 200)** for the FULL plan; the LIMITED plan (160 total) is explicitly labelled underpowered for H1 (§16.5).

---

## 12. Evaluation and human-validation system

### 12.1 Audit of the existing OWL-ViT + CLIP evaluator
Summarised from §2.5. Strengths: simple, deterministic, thresholds fixed a priori, separated from generation, raw detections stored so rescoring is cheap. Weaknesses: coarse spatial rule (centre x only); CLIP-B/32 colour on box crops; one detector; no "uncertain" outcome; single non-independent annotator and 40 images; thresholds swept on the same images that were labelled; label history includes a non-blind second pass.

### 12.2 Separating eight things
| Aspect | Measured by | Notes |
|---|---|---|
| 1. Generator correctness | Primary-judge constraint outcomes | The target quantity |
| 2. Evaluator correctness | Judge vs consensus human on the audit; synthetic probe | E01, E22 |
| 3. Human disagreement | κ between annotators; share of "ambiguous" | E01 |
| 4. Image quality | CLIP score (similarity, **not** adherence), HPS v2 / ImageReward (preference proxies) | Report separately |
| 5. Diversity | 1 − mean pairwise cosine of CLIP image embeddings across seeds (existing) | Per scene |
| 6. Robustness across wording | WSI (H2) | E08 |
| 7. Response to semantic change | Edit fidelity (E09) | E09 |
| 8. Compute cost | Seconds/image, peak VRAM, hook overhead | E03, E14 |

### 12.3 Per-category outcome definitions
**Presence.** `Y_pres = 1` iff each requested object has a distinct detection above threshold. Also record: missing (score below threshold), partially visible (box touches image border or area < a fixed fraction — thresholds fixed on DEV), duplicates (> requested count), confidence and an **uncertain** flag when the best score lies in a band around the threshold.
**Counting.** `Y_cnt = 1` iff detected instances after NMS equal `n`; report by requested n, over-count vs under-count, and instance matching via distinct boxes.
**Binding.** `Y_bind = 1` iff every requested colour is assigned to the intended object (own-object colour correct); also record: *leakage* (attribute appears on another object), *swap* (two colours exchanged), *all-present-but-wrong-binding* (the "both colours exist but on the wrong objects" case the PS highlights).
**Spatial.** Left/right from box geometry; additionally report the stricter T2I-CompBench-style rule (offset-vs-vertical and IoU conditions, per its text) so the effect of rule leniency is measured (§2.5 #1); flag overlapping/ambiguous boxes as uncertain.
**Multi-constraint.** Per-constraint outcomes, all-correct success, soft fraction (mean of constraints), and **uncertainty for the all-correct score** (Wilson CI per scene-level binomial; scene-bootstrap overall).
**Counterfactual pairs.** Target-change success, preservation of unchanged constraints, pair fidelity = both hold on the same seed pair.
**Quality/efficiency.** As §12.2 rows 4, 5, 8.

### 12.4 Human audit size and selection
- For a per-type accuracy of ≈ 0.9 and half-width 0.05: `n = 1.96² · 0.9 · 0.1 / 0.05² ≈ 138` constraint judgements per type; ≈ 550 for four types; with ≈ 3 constraints per image ≈ **190 images**. To allow stratification by variant/CFG/model and a second annotator, use **≥ 300 images** (≈ 900 judgements) with **100 double-annotated**.
- Selection: stratified random by (constraint type × variant × arm/model × predicted outcome from J1) drawn **after** generation, with oversampling of J1-uncertain items (reported separately so the headline accuracy is not inflated).
- Annotators see the image and one constraint question at a time, blind to arm, model, variant and J1/J2 verdict.
- Limitation to state: even 300 images give wide intervals for rare strata (e.g. spatial positives); results by type are reported with CIs and the "exploratory only" rule (§9.0) applies when accuracy < 0.85.

### 12.5 Judges
| Judge | Role | Notes |
|---|---|---|
| **J1** OWL-ViT + CLIP-B/32 | Existing; cheap | Improve spatial rule; add uncertain flag; keep thresholds frozen |
| **J2** Atomic-VQA (Soft-TIFA-style) | Alternative judge for every atom (object, count, colour-of-object, relation) | GenEval 2 reports AUROC 94.5% with Qwen3-VL-8B (FT); an 8B VLM in fp16 needs ≈ 16 GB of weights and **will not fit one 16 GB card with activations (UNVERIFIED arithmetic)** → use a smaller VLM or 4/8-bit quantisation; choose in E03 |
| **J3** Human audit | Ground truth proxy | E01 |
| **Synthetic probe** | Exact truth for rendered shapes | E22 |

**Primary-judge rule (frozen):** per constraint type, whichever of J1/J2 has the higher DEV-audit accuracy; if neither ≥ 0.85, the type is EXPLORATORY only. Conclusions are always re-reported under the other judge (E18).

### 12.6 Annotation protocol
1. Written guidelines with decision rules: partially visible object counts as present only if ≥ a stated fraction is visible; multi-coloured objects are "bound" to the dominant colour or flagged ambiguous; "left/right" is image-left/right; unclear → "ambiguous".
2. A 20-image training round discarded from analysis.
3. Double-annotation on 100 images; κ per type; disagreements resolved by a third reader; the **pre-resolution** disagreement rate is reported.
4. Ambiguous items are excluded from accuracy but their rate is reported.
5. Label provenance recorded (annotator ID, date, blind status). *The current project's labels began as an AI draft; any future label that is AI-assisted must be recorded as such.*

### 12.7 Revision 2: conditional outcomes, branch images and judge reliability
- **Conditional outcomes.** `Y_bind|pres` and `Y_spat|pres` (§8.0) are computed beside the existing fields for every image. The pilot field `binding_ok` is kept for reproduction of pilot results only; primary claims use conditional outcomes plus an unconditional companion that counts omissions as failures (reported side by side).
- **Audit strata.** The human audit (E01) is stratified by `objects_present`, so that colour-assignment accuracy given presence is estimated separately from presence recall (pilot validation: presence accuracy 0.93, n = 58; binding 0.89, n = 37 — EXPLORATORY, labels began as an AI draft).
- **Branch and edited images.** Images from E23 (branches), E12/E26/E28 (edits) are added as audit strata; if judge accuracy on them differs from accuracy on base images by more than 0.05, E18 re-scores them with J2 and the audit-calibrated labels before any conclusion is reported.
- **Agreement-based estimands** (`A`, `C`, `t50`) are attenuated by judge error; a flip-rate sensitivity analysis (inject random label flips at the audited error rate into synthetic outcomes) reports how much `C` shrinks, so that `t50` is never presented as noise-free.

---

## 13. Method design and feasibility (the interventional probe, not a claimed method)

### 13.1 Purpose
H3 needs an edit that (a) targets the quantity H1 uses as a diagnostic, (b) can be matched by a **same-magnitude random control**, and (c) is cheap to implement on SD 1.5. The edit below is a *probe*; its distinctness from prior art is not asserted (§5.3).

### 13.2 Specification
| Item | Definition |
|---|---|
| Inputs | Conditional-branch cross-attention probabilities `A_{ℓ,t} ∈ R^{h × N × 77}` at step `t`, layer `ℓ` (SD 1.5: `h = 8`; `N = side²` with side 8/16/32/64); scene graph with, per binding pair `(a, n)` (attribute word, own noun), the set `O` of other nouns |
| Token alignment | Existing `word_token_indices` (each word tokenised separately, positions verified by an assertion); attribute tokens `I_a`, noun tokens `I_n` |
| Maps | `m_w(ℓ,t) = mean_h mean_{i∈I_w} A_{ℓ,t}[h, :, i] ∈ R^N`; normalised `p_w = m_w / Σ m_w`; analysis at 16×16 layers (as A&E, G&B, D&B) with 32×32 as an alternative |
| Conflict score | `C_{ℓ,t} = max_{o∈O} corr(m_a, m_o) − corr(m_a, m_n)` (the negative of the pilot's `bind_margin`; continuity with H1) aggregated by mean over the 16×16 layers; `C > 0` means the attribute map resembles another noun more than its own |
| Trigger | Edit at step `t` iff `C_t > τ_c` **and** `t ∈ [t_0, t_1]` (DEV-tuned from {0–12, 0–20}) **and** the number of edits so far `< M` |
| Spatial masks | After 3×3 Gaussian smoothing of `p_n` and `p_o`: `S_n` = top-`q`% pixels of `p_n`, `S_o` = top-`q`% of `p_o` (`q ∈ {20, 30}`); skip if `IoU(S_n, S_o) ≥ 0.5` |
| Modification | For attribute tokens `i ∈ I_a`: `logit'_{pix,i} = logit_{pix,i} + β·1[pix ∈ S_n] − β·1[pix ∈ S_o]`; implemented as a per-pixel additive bias in the SDPA mask of the conditional batch element; the unconditional branch is untouched |
| Magnitude bound | `0 < β ≤ log 2` (tuned in {log 1.5, log 2}); at most `M` edits per image |
| Normalisation | Softmax re-normalises; no manual rescaling |
| Fallback | If alignment fails, noun-map mass < ε, or masks overlap, **no edit** and the reason is logged (`skipped_reason`) |
| Overhead | Cost of an extra recompute of conditional probabilities (already done by the recording hook) plus one masked attention pass for edited layers; **not measured here** — measure in E03/E14 |
| Failure modes | Duplicate objects; colour bleeding to background; mask drift as layout evolves; over-saturation; interaction with high CFG; edit never fires |

### 13.3 Pseudocode
```text
for step t in denoising loop:
    for layer l in L16:
        A = cond_attention_probs(l, t)                # [h, N, 77]
        m_a, m_n, m_o = word_maps(A, I_a, I_n, I_o)   # each [N]
    C_t = mean_l( max_o corr(m_a, m_o) - corr(m_a, m_n) )
    fire = (C_t > tau_c) and (t0 <= t <= t1) and (n_edits < M)
    if fire and masks_ok(p_n, p_o, q):
        B = zeros(1, 1, N, 77)
        B[:, :, S_n, I_a] += +beta ;  B[:, :, S_o, I_a] -= beta
        run layer with attn_mask = B for the cond branch
        n_edits += 1
    else:
        run layer unchanged (log skipped_reason)
```

### 13.4 Matched controls (what makes it a probe)
| Arm | What differs from the targeted arm |
|---|---|
| **control** | Same code path, bias ≡ 0 (bit-identical to baseline, as shown for `s = 1` in the pilot) |
| **random-pair** | Same `β`, same mask area `q`, same edit count, but a randomly chosen token pair/region |
| **shuffled-mask** | Targeted tokens, mask taken from a different scene/seed |
| **always-on** | Trigger removed (tests the trigger) |
| **global s = 2** | The pilot's fixed re-weighting (existing `intervene.py`) |

### 13.5 Comparison with the closest published work (operation, trigger, locality, timing, objective)
| Method | Operator | Trigger | Spatial selection | Timing | Objective | Difference from the probe |
|---|---|---|---|---|---|---|
| Gather-and-Bind (FT) | Latent gradient step (`z' = z − α∇L`) | Always-on | None (whole 16×16 maps) | Steps 0–25 of 50 | Entropy + JSD(attr, noun) | Operator, trigger, locality |
| Divide & Bind (FT) | Latent gradient | Always-on | None | `t_end = 25` | TV attendance + JSD binding | Operator, trigger, locality |
| SynGen (ABS) | Latent gradient | Always-on | None | first 25/50 | Syntax-based overlap | Operator, trigger, locality |
| Attend-and-Excite (FT) | Latent gradient | Thresholded refinement at steps {0, 10, 20} | None | Steps < 25 | Max-attention of subject tokens | Operator; objective is presence, not binding |
| Attention Modulation (ABS) | Masked cross-attention + phase weights + self-attn temperature | Phase schedule | Object-focused mask | Phase-wise | Alignment (details unread) | **Possibly overlapping: mask + schedule** |
| CoBind (ABS) | Stage-aware optimisation | Adaptive strength by constraint satisfaction | Graph-based | Early/mid/late | Entity completeness + binding | **Possibly overlapping: adaptive trigger** |
| UCC (FT) | Embedding fusion + temperature-sharpened attention | Proactive/fixed | none | Deeper layers | Attribute leakage | Different architecture (DiT) |
| P2P reweight (FT) | Multiply attention probs after softmax | None | Per token | `cross_replace_steps` | Edit strength | My global arm is a variant (pre-softmax) |
| AnchorSteer / EPIC (FT/ABS) | Latent rollback / image edit | VLM or predicate failure | Local edit | Mid-generation / post | Constraint verification | Different signal and operator |

**Verdict.** The probe's *operator* (bounded per-pixel logit bias) differs from the gradient methods, and its *trigger/locality* differ individually from each; but Attention Modulation and CoBind (abstract-only) may already combine locality and adaptivity. **Do not name it, do not claim it as new.** If a full-text read shows substantial overlap, keep the experiment as "a matched-control test of whether attention overlap is a causal handle", which does not depend on novelty.

### 13.6 Acceptance tests (before any DEV tuning)
1. With trigger threshold `τ_c = +∞` the output is **bit-identical** to the control arm.
2. The bias tensor is zero outside the designated attribute-token columns.
3. Targeted and random-pair arms have equal mask area (±1 pixel) and equal β.
4. Edits occur only within `[t_0, t_1]` and never exceed `M`.
5. Same seed → same output (determinism on the test hardware).
6. Every image logs `n_edits`, `fired_steps`, `skipped_reason`.
7. Token alignment assertion (existing) holds for all V2 prompts including paraphrases (synonym lexicon words must tokenise as expected).
8. The shuffled-mask arm uses a mask from a different scene (assert scene IDs differ).

### 13.7 Feasibility on SD 1.5 / Kaggle
Pure-PyTorch changes to `src/intervene.py` and `src/attention.py`; no new dependencies. The attention hook's bit-identity must be **re-verified on the Kaggle GPU** (E00/E03). For SDXL (E16) the processor must be adapted to its transformer-block attention; for SD3 (joint attention) the probe is **not** applicable without new extraction code (deferred).

### 13.8 Revision 2: use of the probe in HC
The edit of §13.2 is used by HC with a fixed window length `L = 12` and three onsets (early = 0, mid = `t50_DEV`, late = `ceil(t50_DEV) + 4`, capped at 28). Bias magnitude `β` and mask area are matched between targeted and random arms **and across onsets**; the total bias mass `Σ_steps Σ_layers ‖b‖₁` is logged per image and the arms are accepted only if mass differs by at most 5%. The trigger of §13.2 is evaluated on DEV only (the always-on variant is the confirmatory one for HC, because a trigger that fires later would confound onset). The edit remains a **probe**, not a claimed method.

---

## 14. Baselines, controls and ablations

### 14.1 Arm table (status = how faithfully each is implemented)
| Arm | Status | Notes |
|---|---|---|
| Unmodified SD 1.5 (control path) | **Faithful reproduction** of the pilot baseline | pixel-identical to hook-off by design (verified in pilot for 168/168) |
| Global re-weighting s = 2 / s = 4 | **Faithful reproduction** of the pilot arm | existing `intervene.py`; s = 4 only on DEV |
| Targeted conflict-gated edit | **New implementation** (probe) | §13 |
| Random-pair, shuffled-mask, always-on | **Controls** | isolate target, content and trigger |
| Attend-and-Excite | **Adapted implementation** if run: the official repo targets SD 1.4/2.1 with diffusers-era code (CODE-level); gradient updates every step make it heavier | Optional P3 |
| Divide & Bind | **Adapted**: repository states compatibility with diffusers 0.21.4 and AGPL-3.0 licence (search snippet); this project uses 0.41.0 | Optional P3; licence implications must be checked |
| Gather-and-Bind | Code link printed in the paper (`github.com/huan085128/Gather-and-Bind`); **the link returned HTTP 404 on 2026-10-14** → availability **UNVERIFIED** | Reimplementation from the paper's equations would be an approximation |
| SynGen | Code per the paper's footnote (not inspected) | Optional |
| CoBind, EPIC, AnchorSteer, UCC | No code found; UCC/Regional Prompting are FLUX-based | **Infeasible / out of scope** |

**Fairness rule:** published methods are compared only if run on the **same scenes, seeds, model, scheduler, resolution and judge**, with latency reported; otherwise they appear only in the related-work table. At most **one** published baseline is recommended (D&B or A&E), as P3.

### 14.2 Ablations tied to a scientific question
| Question | Ablation (E13) | Why it matters |
|---|---|---|
| Does token selection matter? | object-only vs attribute-only vs pair | Which side of the pair carries the effect |
| Does the trigger matter? | trigger on vs always-on | Whether "failure-triggered" adds value |
| Does stage matter? | window early / middle / late / all | Links to E06 timing |
| Does localisation matter? | local masks vs global tokens | Specificity of the spatial edit |
| Bias vs loss? | logit bias vs optional gradient arm (published baseline) | Operator comparison, only if the baseline arm is built |
| Paired loss vs independent sharpening? | targeted pair edit vs independent sharpening of each token | Whether the *pairing* adds anything |

No ablation is run without a corresponding H3 signal on DEV (stop rule in §21.2).

### 14.3 Revision 2 controls and ablations (for HA, HB, HC)
| Control | Purpose | Applies to | Failure criterion (stop or relabel) |
|---|---|---|---|
| **Δ = 0 branch** (ε set to 0) | Proves the branching code path reproduces the base image | E23 | Any pixel difference from the base image → fix before DEV |
| **Full-resample branch** (restart to the first-step noise level) | Establishes the floor `C ≈ 0` | E23 | `C > 0.15` at full resample → the outcome is mostly set by the prompt (not the trajectory) or the estimator is biased; report and re-check `A0` |
| **Independent-seed agreement `A0`** | Chance level without shared information | E23 | `A0 ≈ 1` for a constraint type → that type carries no information in `C`; reported, not hidden |
| **Distribution unit test of the restart operation** | Checks the forward-noising formula numerically | E23 | Moment mismatch above numerical tolerance |
| **Δ sensitivity (4, 8, 12)** | `t50` is Δ-relative by construction | E23b | Type ordering unstable → HA qualified |
| **Judge sensitivity** (J1 vs J2 vs audit-calibrated) on branch images | Judge noise attenuates agreement | E18 on E23 | Conclusion changes with the judge → HA reported as judge-dependent |
| **Noise-only baseline B_noise** | Separates "forecast from attention" from "forecast from the seed" | E05 / HB | Mandatory; HB is not interpretable without it |
| **Shuffled-attention control** | Detects leakage into features | E05 / E07 | Shuffled features reach the real AUC → withdraw |
| **Random-pair edit at the same onset, magnitude and mask area** | Isolates targeting from "any edit at that time" | E12 / HC | Random@early ≈ targeted@early → the edit is not specific |
| **Onset-matched strength** | Prevents confounding of onset with dose | E12 | Total bias mass differs between early and late arms by more than 5% → rescale |
| **Matched random-token knockout** | Isolates specificity of layer-group knockouts | E26 | Random effect ≈ real effect → no specificity |
| **η = 0 fragility replicate** | Verifies exact reproducibility | E24 | Any flip → numerical non-determinism reported as a finding on its own |
| **Control arm reuses core images** | Avoids double counting | E12, E26 | Control path not bit-identical on Kaggle hardware → regenerate control per arm |

Ablations tied to a question (E13) are unchanged and remain P3; **no ablation is run before the corresponding DEV signal exists** (§21.2).

---

## 15. Sample-size and statistical-analysis plan

### 15.1 Structure and unit of analysis
Images are nested: **scene graph → variant → seed**. Scene graphs (or prompt constructions) carry the independent diversity; seeds are repeated draws within a scene. All inference resamples **scenes**, never images. Paired comparisons pair by (scene, seed). *Caution:* the number of generated images (thousands) is **not** the sample size; the sample size is the number of scenes (≈ 140 binding-bearing for H1/H3 in TEST).

### 15.2 Estimands and models
| Question | Estimand | Primary analysis | Secondary / sensitivity |
|---|---|---|---|
| H1 | ΔAUC between nested prediction models on TEST | Scene-level bootstrap over frozen predictions | ΔBrier, Δlog-loss, calibration slope; leave-object-pair-out (E15) |
| H2 | Excess RMS wording shift per constraint type | Within-seed label-swap permutation + scene bootstrap CI | Mixed logistic model with scene random intercept and variant effect |
| H3 | Risk difference in binding-correct rate, targeted − random | Paired scene bootstrap | Scene-clustered GEE with arm effect; per-guard-metric contrasts |
| CFG × complexity (E11) | Interaction on the log-odds / risk-difference scale | Scene-clustered logistic GEE | Scene bootstrap of per-level risk differences |
| Evaluator accuracy (E01) | Per-type accuracy, κ | Wilson CI; bootstrap over images (clustered by scene) | — |
| Seed sufficiency (E20) | ICC, design effect | One-way ANOVA / GLMM variance components | k-subsampling curves |

Model assumptions to disclose: exchangeability of scenes within a category (not exactly true; object quotas mitigate); independence of scenes across the split (guaranteed by construction); a binary judge outcome that is itself noisy (attenuation; handled by E18); frozen predictors for TEST so no optimism from refitting.

### 15.3 Multiple testing
Holm over {H1, H2, H3} (confirmatory family). Per-category and per-type analyses, E06/E07/E13 ablations and every feature-importance statement are **EXPLORATORY** and use FDR-BH only for ordering, never for confirmation. Corrections are *not* applied across outcomes within E12's guard metrics, but the guards are pre-specified thresholds, not tests.

### 15.4 Power and sample-size calculations (**DERIVED FROM PILOT**; assumptions stated)
All numbers below are computed by read-only scripts from `outputs/main/results/per_image.csv` and `outputs/intervention/results/intervention_per_prompt.csv`. **Assumption:** new scenes are exchangeable with the 28 pilot prompts in variance structure — plausible but unverified; the pilot has only 28 prompts, 12 for binding.

**(a) Variance components.** CFG 7.5 vs 2: the observed variance of the prompt-level paired difference is 0.0317 with 12 images per prompt per level; the expected seed-noise part is 0.022, leaving a between-prompt effect variance of 0.0097 (SD ≈ 0.10). Per-image Bernoulli variance of a paired difference (ignoring seed pairing, hence conservative) is `2·p̄(1 − p̄)` averaging **0.282**.
**(b) ICC and design effect** (one-way ICC across prompts; cluster size 36): binding_ok 0.52, primary 0.27, presence_ok 0.23, spatial_ok 0.05, count_ok 0.04. Design effect `1 + (k−1)ρ`: binding 2.57 at k = 4, 3.62 at k = 6.
**(c) Minimum detectable effect** (80% power, two-sided; normal approximation; `MDE = (z_{α/2}+z_{β}) · sqrt((σ_b² + σ_w²/k)/N)`) for an overall success-rate contrast with σ_b² = 0.0097:
| N scenes | k = 2 | k = 4 | k = 6 | k = 4, α = 0.0167 |
|---|---|---|---|---|
| 28 (pilot) | 0.206 | 0.150 | 0.126 | 0.173 |
| 100 | 0.109 | 0.079 | 0.067 | 0.092 |
| 200 | 0.077 | 0.056 | 0.047 | 0.065 |
| 300 | 0.063 | 0.046 | 0.039 | 0.053 |

Scaling by `1/√N` gives ≈ **0.067** (α = 0.05) and ≈ **0.078** (α = 0.0167) for **N = 140** binding-bearing scenes at k = 4 — the basis for H3's smallest effect of interest (0.08). For a single category with N = 30 (e.g. TEST counting) the MDE at k = 4 is ≈ 0.145; hence **all per-category claims are exploratory**.
**(d) AUC precision** (Hanley–McNeil, AUC 0.75, base rate 0.30, effective n = n/DEFF with the binding ICC):
| N_test scenes | k | raw n | effective n | single-AUC 95% half-width |
|---|---|---|---|---|
| 60 | 4 | 240 | 93 | 0.116 |
| 120 | 4 | 480 | 187 | 0.082 |
| 120 | 6 | 720 | 199 | 0.079 |
| 160 | 4 | 640 | 249 | 0.071 |
| ≈ 140 | 4 | ≈ 560 | ≈ 218 | ≈ 0.077 (interpolated) |

Extra seeds buy little (k = 4 → 6 raises the effective n by ≈ 6% at N = 120); **scenes are the scarce resource**.
**(e) Nested-model ΔAUC.** The SE of a nested-model difference depends on the correlation between the two models' scores and is **not derivable from the pilot**. Rule: after DEV is generated, simulate the TEST design by repeatedly drawing scene sets of the TEST size from DEV, computing ΔAUC and its bootstrap CI width; **if the 95% CI half-width exceeds 0.04**, H1 is downgraded to exploratory and the paper reports the CI only (§8.1).

### 15.5 Reporting rules
Report effect sizes and 95% CIs for every contrast; never infer "no effect" from a non-significant test — use the CI against the smallest effect of interest; report number of scenes, variants, seeds, images, exclusions and failed generations; report per-type judge accuracy beside every adherence result; show both "all-correct" and "soft fraction" outcomes; label EXPLORATORY analyses explicitly; state the MDE next to every null.

### 15.6 Pre-specified decision rules (summary)
| Hypothesis | Support | Bounded null | Else |
|---|---|---|---|
| H1 | CI lower > 0 and ΔAUC ≥ 0.05 | CI upper < 0.03 | Inconclusive |
| H2 | CI lower > 0 and RMS ≥ 0.10 (binding or spatial) | CI upper < 0.05 (both) | Inconclusive |
| H3 | CI lower > 0, point ≥ 0.08, guards respected | CI upper < 0.04 | Inconclusive / "unresolvable at this N" |

### 15.7 Excluded and failed samples
Every failed generation (crash, NaN, black image), every judge "uncertain" outcome and every audit-ambiguous item is logged and **reported**, not dropped silently. Failed generations are retried once with the same seed; persistent failures are excluded with their scene ID and reason in `exclusions.csv`.

### 15.8 Revision 2 additions to the statistical plan
**Confirmatory family.** Holm over **{HA, HB, HC}** replaces {H1, H2, H3}. HA yields one statistic (`Δt50`); HB two (`ΔAUC_pre`, `ΔAUC_post`) that are combined into a single pre-specified outcome label (forecast / readout only / none / inconclusive) so that HB contributes one test to the family (the primary test is `ΔAUC_pre`; `ΔAUC_post` is conditional on it); HC two (`D_early`, `G`) combined analogously (primary: `D_early`; `G` decides the interpretation, not the family size). Everything else, including supporting H1/H2, E23b, E24–E28, is **EXPLORATORY** with FDR-BH for ordering only.

| Quantity | Estimator | Interval | Notes |
|---|---|---|---|
| Agreement `A_{κs}(t)` | Mean over `M = 2` branches × `k = 2` base seeds of outcome equality (conditional outcomes: only pairs with required objects present in both images) | — | Pairs with missing objects are counted and reported |
| Chance level `A0_{κs}` | Mean agreement over the 6 pairs of the `k = 4` independent-seed outcomes of the same canonical prompt | — | From core data; no extra generation |
| Commitment `C_κ(t)` | `Σ_s [A − A0] / Σ_s [1 − A0]` | 4000-draw scene bootstrap (resample scenes, recompute the ratio) | Ratio of sums is stable when single scenes have `A0 ≈ 1` |
| `t50_κ` | First crossing of 0.5, linear interpolation on the grid | Bootstrap percentile; `> 36` when never crossed | Crossing is a coarse functional; always show the full curve |
| `Δt50` | `t50(bind\|pres) − t50(presence)` | Scene bootstrap with the two estimators recomputed on the same resampled scenes | Pairing preserved across types |
| `ΔAUC_pre/post` | Frozen-model AUC difference on TEST | Scene bootstrap | DEV simulation gate on CI width (0.04) |
| `D_early`, `G` | Paired risk differences by (scene, seed) | Paired scene bootstrap | `G = D_early − D_late` |

**Power for HA (honest statement).** No commitment data exist, so no power calculation is possible from the pilot. The design therefore relies on a **DEV gate**: run E23 on 40 DEV scenes (960 branch-runs), bootstrap the DEV CI of `Δt50`; if its half-width exceeds **4 steps**, increase `M` (branches per base) or the number of TEST scenes before the freeze; if even the maximum affordable design projects a half-width above 4 steps, HA is downgraded to **EXPLORATORY** and the paper reports the curves with CIs only. The 3-step minimum effect is a **design judgement**, not a derived quantity.

**Why more scenes rather than more seeds.** Within-scene estimands (`A`, `C`) are limited by the number of scenes; the binding-outcome ICC across prompts in the pilot is 0.52 (DERIVED FROM PILOT), so extra seeds within a scene add little new information about the scene population (§15.4). E23 therefore uses `k = 2` base seeds × `M = 2` branches over ≈ 100 scenes instead of fewer scenes with many branches.

**Conditional-outcome bookkeeping.** For every conditional analysis, report the number of scenes and images that survive the "both objects present" filter; the pilot proportion was 0.45 for binding prompts (F2), so roughly half of the binding images are lost to the filter. This is why binding-bearing scene counts (≈ 70 of the commitment scenes) and the DEV gate matter. **Selection caveat:** conditioning on presence selects images by a post-treatment variable; comparisons across **arms** in HC are therefore reported both conditional on presence and unconditionally (`Y_bind` with omissions counted as failures), and a conclusion is stated only if both agree.

**Decision rules (summary, Revision 2).**
| Hypothesis | Support | Bounded null | Else |
|---|---|---|---|
| HA | CI of `Δt50` excludes 0 and `abs(Δt50) ≥ 3` | CI ⊂ (−1.5, +1.5) | Inconclusive |
| HB | Forecast: `ΔAUC_pre ≥ 0.05`, CI > 0. Readout only: pre CI upper < 0.03 and post ≥ 0.05, CI > 0 | None: both CI upper < 0.03 | Inconclusive |
| HC | `D_early` ≥ 0.08, CI > 0, **and** `G` ≥ 0.10, CI > 0, guards respected | `D_early` CI upper < 0.04 | Inconclusive; handle-but-not-time-gated if `G` CI within (−0.05, +0.05) |

---

## 16. Two-GPU compute and runtime budget

### 16.1 What is known and what is not
| Item | Status |
|---|---|
| Pilot generation on the RTX 3050 6 GB: ≈ 4.7 s/image at 20 steps and ≈ 9.1 s at 40 steps (unshared pilot); the intervention runs logged ≈ 9.1–9.3 s/image (attention-extraction time per image was not logged); SD 1.5 fp16 used ≈ 3.3 GB VRAM | **VERIFIED EXISTING RESULT** (`samples.jsonl` `seconds`, `nvidia-smi` during the run) |
| Kaggle hardware (GPU type, VRAM, quota, session length, internet, secrets) | **UNVERIFIED**; secondhand sources say T4 ×2 (16 GB each) or P100 (16 GB), ≈ 30 h/week GPU quota, session limit 9 or 12 h (sources disagree), internet toggle required, secrets via the Add-ons menu |
| Any Kaggle throughput | **Not measured** — measure in E03; no speed-up factor is assumed anywhere |
| SDXL fp16 + hook within 16 GB | **UNVERIFIED** (Hugging Face's own SDXL memory post reports far larger figures without offload) |
| SD3 Medium within 16 GB | **UNVERIFIED**; Diffusers docs warn it is hard below 24 GB; dropping T5 saves memory but may reduce quality (ABS) |
| A VLM judge (J2) in 16 GB | **UNVERIFIED** for 8B-class models; choose by measurement |

### 16.2 First action on Kaggle (before any experiment)
Run a hardware/quota probe that prints `nvidia-smi`, `torch.cuda.get_device_name()`, VRAM per device, CUDA version, library versions, free disk and the remaining GPU quota shown by the notebook UI; save it to `outputs/e03/hardware.json`. All later decisions cite this file.

### 16.3 Parallelism design
- **Two independent worker processes**, each pinned with `CUDA_VISIBLE_DEVICES`, each owning a disjoint **scene shard** (`hash(scene_id) mod 2`). SD 1.5 inference is small enough that model-level multi-GPU distribution would only add communication.
- **Shard by scene** (not by seed): a scene's variants and seeds stay in one file; no cross-worker dependencies; paired seeds stay identical across workers.
- **Different models run in separate sessions** (they cannot share a GPU with the 16 GB budget safely).
- Never run two workers on the same output file.

### 16.4 Safe resume, uniqueness and integrity
1. Per-worker metadata files: `samples_gpu0.jsonl`, `samples_gpu1.jsonl`; per-session output directories (`outputs/<exp>/<session>/…`).
2. Atomic writes: write PNG to a temp name then rename; append the metadata line **after** the PNG exists; record the PNG's SHA-256.
3. Resume rule: a sample is done iff its metadata line exists **and** the PNG exists **and** its hash matches.
4. `src/verify_outputs.py`: unique names; every PNG opens at the expected size; hashes match; seeds equal the manifest; zero duplicates across shards; worker shard assignments disjoint and complete.
5. `src/merge_samples.py` merges shards after verification and refuses to proceed on any duplicate name.
6. Every record carries: model ID and revision, scheduler and config, CFG, steps, seed, resolution, library versions, GPU name, `max_memory_allocated`, seconds, hook on/off, arm parameters.
7. Kaggle interruptions: the loop is idempotent, so a restarted session continues from the last verified sample; outputs are saved to the persistent output store between sessions (confirm the mechanism in the notebook).

### 16.5 Experiment matrices (formulas, then totals)
**Definitions.** `S` scene graphs; `V` average variants per scene; `k` seeds; `S_cfg` scenes in the CFG study; `S_b` binding-bearing TEST scenes; arms/ablation counts as in §9.

| Quantity | FULL | LIMITED |
|---|---|---|
| Independent scene graphs | 300 (DEV 100, TEST 200) | 160 (DEV 60, TEST 100) |
| Prompt variants (`S × V`) | 300 × 5 = **1,500** | 160 × 4 = **640** |
| Seeds `k` (DEV tuning k = 2) | 4 | 4 |
| Core images `S·V·k` (E04) | **6,000** | **2,560** |
| CFG × steps study `S_cfg·(C·T − 1)·k` | `150·(5·2 − 1)·4 =` **5,400** | `100·(3·2 − 1)·4 =` **2,000** |
| Intervention arms, TEST `S_b · 3 · k` | `140·3·4 =` **1,680** | `70·3·4 =` **840** |
| Intervention tuning, DEV `S_b,dev · configs · 2` | `70·8·2 =` **1,120** | `41·6·2 =` **492** |
| Ablations (P3) | `6·60·4 =` **1,440** | 0 |
| SDXL replication (P2) | `100·4 =` **400** | `60·4 =` **240** |
| **Total generated images** | **16,040** | **6,132** |
| Attention extractions | E04 6,000 + E11 40-step new cells `150·4·4 = 2,400` + SDXL 400 = **8,800** | `2,560 + 100·2·4 + 240 =` **3,600** |
| Evaluator calls, J1 | = images: 16,040 | 6,132 |
| Evaluator calls, J2 (`images × atoms`; atoms ≈ 6 is an **assumption**, measure in E03) | ≈ 96,240 | ≈ 36,792 |
| Human judgements (audit) | ≈ 900 + 300 duplicates | ≈ 600 + 200 duplicates |

*No double counting:* the control arm **reuses** the core images (hook-on is image-neutral and the control path is pixel-identical); the CFG study shares the core cell; attention maps are produced in the **same pass** as the images; paraphrase/order/counterfactual images are part of the core count.

### 16.6 Runtime — how the estimate will be produced (not guessed)
Run **E03** and measure on the actual Kaggle accelerator: `t_gen` (seconds/image with hook on, 40 steps, 512²), peak VRAM, 1-vs-2-worker scaling `s₂`, crash rate `f`, `t_J1`, `t_J2`, load overhead per session `t_load`. Then
`T_gen = N_img · t_gen · (1 + f) / (W · s₂) + sessions · t_load`, `T_eval = N_img · t_J1 + N_atoms · t_J2`, plus analysis and annotation time.
**Reference only (not a Kaggle estimate):** at the RTX 3050's measured 9.1 s/image, FULL = 16,040 × 9.1 s ≈ **40.5 GPU-hours**, LIMITED ≈ **15.5 GPU-hours**. Use these only to size the question "is FULL plausible within the weekly quota", and decide after E03: **choose FULL iff the measured total (including a 25% allowance for failures, restarts and re-runs) fits within the quota available over the project timeline; otherwise LIMITED.**

### 16.7 Minimum experiment set that preserves the central contribution
E00, E01, E02, E03, E21 (gate), E04 (DEV + TEST, CAN + PAR + one CF), E05, E18-lite (J1 vs audit-calibrated), E11 (reduced to 3 CFG levels). Under LIMITED, H1 has ≈ 70 binding-bearing TEST scenes (half-width ≈ 0.11) and is declared underpowered for ΔAUC = 0.05; the paper then reports it as a **calibrated pilot with CIs**, H3 is exploratory, and H2 retains CAN + PAR only.

### 16.8 What can reuse saved data and what must be re-run
| Reusable (no new generation) | Must be re-run for valid confirmation |
|---|---|
| E00 (re-analysis), E21 (gate), E20 (variance), E06/E07 on pilot maps (exploratory), E15 design checks | Everything confirmatory on **new TEST scenes** (E05, E08, E12) |
| Pilot images as extra DEV data (hardware-flagged) | Pilot images cannot be TEST |
| Existing code paths and tests | Hardware-sensitive checks (bit-identity, throughput) on Kaggle |

### 16.5b Revision 2 experiment matrices (formulas, then totals)
**Definitions.** Branch-run = restart at grid step `t` plus deterministic re-denoise to the end. **Image-equivalent** = branch-run × (remaining steps / 40); with grid `{6, 12, 18, 24, 30, 36}` and Δ = 8 the average remaining fraction is **26.7 / 40 = 0.667** (**DERIVED**: design parameter, not a measurement; LIMITED uses five grid steps with Δ = 8, remaining fraction ≈ 0.74). UNet-evaluation equivalent for E27 = evaluations / 80 (a 40-step CFG generation needs 80 UNet evaluations). Saving six fp16 latents per base image costs ≈ 32 KB per latent and no extra compute.

| Quantity | FULL | LIMITED |
|---|---|---|
| E04 core images (unchanged, §16.5) | 6,000 | 2,560 |
| E11 CFG × steps (unchanged) | 5,400 | 2,000 |
| **E12 onset arms, TEST** `S_b · arms · k` | `140·6·4 =` **3,360** | `70·4·4 =` **1,120** |
| E12 DEV tuning (unchanged) | 1,120 | 492 |
| **Subtotal generated images (P0–P1)** | **15,880** | **6,172** |
| **E23 DEV rehearsal** `40·2·6·2` branch-runs | 960 runs ≈ **640** equiv. | 960 runs ≈ **640** equiv. |
| **E23 TEST** `100·2·6·2` (LIMITED `60·2·5·2`) | 2,400 runs ≈ **1,600** equiv. | 1,200 runs ≈ **890** equiv. |
| **P1 total image-equivalents** | **18,120** | **7,702** |
| E24 fragility (P2; `100·2·2·3 + 100·2`; LIMITED `60·2·1·3 + 120`) | 1,400 | 480 |
| **P0–P2-lite total (with E24)** | **19,520** | **8,182** |
| E26 knockout (P2) `40·4·(4·2)` | 1,280 | 0 |
| E23b Δ sensitivity (P2) `20·2·6·2·2` runs ≈ equiv. | 960 runs ≈ 640 | 0 |
| E27 guidance attribution (P2) `60·2·10·7` UNet evaluations ≈ equiv. | 8,400 evals ≈ 105 | 8,400 evals ≈ 105 |
| E28 negative-prompt timing (P3) `140·5·4` | 2,800 | 0 |
| E13 ablations (P3, unchanged) | 1,440 | 0 |
| SDXL replication (P2, unchanged) | 400 | 240 |
| **Everything (FULL), image-equivalents** | **≈ 26,200** (19,520 + 1,280 + 640 + 105 + 2,800 + 1,440 + 400) | — |
| Attention extractions | Core 6,000 + CFG 2,400 + SDXL 400 as before; **E23 and E24 need none** | as before |
| Evaluator calls J1 | = generated images (branch images included) | same |

**Reference only (not a Kaggle estimate).** At the RTX 3050's measured 9.1 s per 40-step image: FULL P0–P2-lite 19,520 × 9.1 s ≈ **49.3 GPU-hours**; LIMITED 8,182 × 9.1 s ≈ **20.7 GPU-hours**. As in §16.6 these figures size the question only; the choice between FULL and LIMITED is made after E03 measures Kaggle throughput, including the measured `t_branch`. **Priority under a quota shortfall:** keep E23 and E12; shrink E11 (3 CFG levels) and E24 first; E26–E28 are dropped before the primary programme is touched.

*No double counting:* branch bases come from E04 core images (no regeneration); the control arm of E12 and the baseline `A0` come from the core data; E25 needs no new image; E27 generates no image.

### 16.7b Revision 2 minimum set
E00, E01, E02, E03, E21 (gate, now also the conditional-label and noise-only checks on the pilot), E04 (DEV + TEST, CAN + one CF, **with latents saved**), **E23 (DEV rehearsal + TEST on ≈ 60 scenes)**, **E25 (free)**, E05 with `B_noise`, E18-lite, E11 reduced to 3 CFG levels. Under LIMITED, HA keeps its definition with ≈ 60 TEST scenes and five grid steps; HB is declared underpowered and reported as a calibrated pilot with CIs; HC retains four arms (targeted/random × early/late) and is reported with its MDE.

---

## 17. File-by-file implementation plan

*Paths were inspected in this repository. "New" files are proposals; no file has been changed.*

| File | Change | Acceptance test |
|---|---|---|
| `configs/experiment.json` | **Keep** (pilot reproduction). Add `configs/experiment_v2.json` with model ID/revision, resolution, CFG list, steps list, k, split manifest path, shard count | Loading both configs yields the pilot grid and the V2 grid respectively; V2 config rejects unknown keys |
| `src/build_benchmark.py` | **Keep** (pilot). Add `src/benchmark_v2.py` (scene-graph schema, grammar, lexicon, transformations, validators, splitter) | Unit tests: every counterfactual differs from its canonical in **exactly one** declared field; every meaning-preserving variant has an identical constraint list; no scene appears in two splits; no object-pair group spans splits; duplicate detector finds seeded duplicates |
| `data/prompts.json` | **Keep** as PILOT; new `data/benchmark_v2/{scene_graphs,prompts_v2,splits}.json`, `DATASHEET.md` | JSON-schema validation; recorded SHA-256 in the protocol |
| `src/generate.py` (`load_pipeline`, `generate_image`, `main`) | Pipeline factory by model ID; `--shard i/n`; per-worker output files; atomic writes; PNG hash; log GPU name, library versions, `max_memory_allocated`; optional hook-on generation | Shards are disjoint and cover the manifest; a killed-and-restarted run yields no duplicates; two workers' merged file passes `verify_outputs.py`; hook-on and hook-off images identical for 10 sampled items on the target GPU |
| `src/attention.py` (`StoreCrossAttention`, `install`, `generate_with_attention`) | Store per-step maps at 16×16 and 32×32 and per-layer-group summaries; run during main generation; configurable word list; SDXL processor variant | Existing sum-to-1 and bit-identity tests pass on Kaggle; shapes as specified; SDXL variant gets its own bit-identity test or is marked unsupported |
| `src/intervene.py` (`ReweightCrossAttention`, `build_bias`, `main`) | Add `--mode {global,targeted,random_pair,shuffled_mask,always_on}`; conflict score; mask builder; per-pixel bias; logging of `n_edits`, `fired_steps`, `skipped_reason` | The eight tests of §13.6 |
| `src/evaluate.py` (`detect`, `score`, `THRESHOLDS`) | Judge interface (`J1`, `J2`); optional stricter spatial rule; `uncertain` flag; store which judge produced which field | J1 outputs on the 40 pilot validation images unchanged by default (regression test); synthetic-probe unit tests |
| `src/judge_vqa.py` (new) | Atomic-VQA judge with a VLM chosen by E03 | Deterministic outputs; VRAM within budget; agrees with J1 on trivially easy synthetic items |
| `src/validate_metrics.py` | Keep for pilot; new `src/audit_sample.py` (stratified draw, blinded export) and `src/validate_human.py` (accuracy, κ, Wilson CIs) | Sampler reproduces the stratum counts; blinded export contains no arm/model/variant names; κ matches a hand-computed example |
| `src/features.py`, `src/predict_failure.py` (new) | Feature extraction from `.npz`; nested logistic models; frozen predictor; bootstrap ΔAUC | Features invariant to a uniform rescaling of maps; train/test scene IDs disjoint (assertion); permuting labels gives ΔAUC ≈ 0 on a synthetic dataset |
| `src/analyze_wording.py`, `src/analyze_counterfactual.py` (new) | WSI with permutation null; edit fidelity | On synthetic data with a known shift the estimator recovers it; under a null simulation the CI covers 0 at ≈ 95% |
| `src/analyze.py` (`load_table`, `cluster_bootstrap`, `paired_contrast`) | Scene-level keys; GEE interactions; per-call RNG seeding (fix finding #7); warn that the "all" row is not a population parameter | Results identical across repeated runs with the same seed; scene-level clustering used in all CIs |
| `src/analyze_attention.py` (`image_features`) | Unify box grounding with the scoring assignment; generalise bind margin to n objects; per-layer storage | `box_ratio` uses the same box as `score()` (test); two-object results equal the pilot's on pilot data |
| `src/analyze_intervention.py` | Generalise to arms; add guard metrics | Pilot numbers reproduce exactly |
| `src/synth_probe.py` (new) | Render coloured-shape scenes with known truth | Truth labels match rendering; probe runs in minutes on CPU |
| `src/merge_samples.py`, `src/verify_outputs.py` (new) | Shard merge and integrity checks | A seeded duplicate or corrupted PNG is detected |
| `src/figures.py`, `src/report_numbers.py`, `report/build_report.py`, `report/make_slides.py` | Add the figures/numbers of §18; rebuild all generated documents after the next result change | `build_report.py` assertions pass; no stale numbers (diff of rendered text against saved results) |
| `report/protocol.md` | Append **Addendum B** (freeze) with hashes before any TEST generation | Hashes in the addendum match the files |
| `tests/` | Add tests for every new module above | `pytest -q` green on the laptop (CPU tests) and on Kaggle (GPU tests) |
| `README.md` | Replace by a build from `README_template.md`; fix e2e status; add Kaggle two-worker commands | Statements match `outputs/*.log` |

**Kaggle commands (proposed):** `CUDA_VISIBLE_DEVICES=0 python src/generate.py --config configs/experiment_v2.json --shard 0/2 --out outputs/e04/s1` and the same with `1` and `--shard 1/2` in a second process, followed by `python src/verify_outputs.py` and `python src/merge_samples.py`.

### 17.1 Revision 2 additions and changes
| File | Change | Acceptance test |
|---|---|---|
| `src/generate.py` | Add `--save-latents-at 6,12,18,24,30,36` (fp16 `.pt` per image, written atomically with the PNG hash) and `--branch-from <manifest>`; both flags default to off, so the pilot path is unchanged | Default-off run reproduces the pilot images bit-for-bit; saved latent at step `t` equals the recomputed latent from a fresh run (same seed) |
| `src/branch.py` (new) | `restart(x_t, t, delta, generator)`, `branch_generate(...)` with deterministic DDIM re-denoise | Δ = 0 returns the base image exactly; moment test: noised-back samples match the theoretical mean and variance on a synthetic Gaussian `x_t`; the branch seed stream is independent of the base seed |
| `src/commitment.py` (new) | Agreement `A`, chance level `A0`, ratio estimator `C`, `t50` (interpolation, `> 36` handling), scene bootstrap | Synthetic data with a known commitment step: estimator recovers it within one grid step; with independent outcomes `C ≈ 0`; with identical outcomes `C = 1` |
| `src/evaluate.py` | Conditional outcomes `Y_bind\|pres`, `Y_spat\|pres` in `score()` (new fields; existing fields unchanged) | Existing outputs unchanged on the 40 validation images; conditional field is `null` when an object is missing |
| `src/features.py` | Noise features of `x_T` (regenerated from the seed on CPU): channel moments, pooled 4×4 grid statistics, low-frequency spectrum | Features identical across runs; invariant to image content (they use only the seed) |
| `src/predict_failure.py` | `B_noise` model and the three-window evaluation (`pre`, `post`, and the exploratory single τ = 12) | Permuting labels gives ΔAUC ≈ 0 on a synthetic dataset; frozen predictions reproduce |
| `src/intervene.py` | `--window-start`, `--window-len`, `--onset-from-dev`, `--mode knockout --layer-group {R64,R32,R16,R8} --tokens {constraint,random}`; log total bias mass per arm | Control path identical to the baseline; bias mass equal across onset arms within 5%; knockout bias applies only inside the chosen layer group (unit test on a toy attention layer) |
| `src/fragility.py` (new) | Variance-preserving latent perturbation, replicates, flip rate | η = 0 reproduces exactly; flip rate on a synthetic classifier matches theory |
| `src/guidance_attr.py` (new) | Token-ablation guidance norm ratio along fixed trajectories | Ablating a token not in the prompt changes nothing; shares sum within expected range |
| `src/analyze_noise.py` (new) | Variance components and noise-only readout | Reproduces the pilot seed-level success rates 0.19–0.31 (EXPLORATORY) |
| `src/analyze_commitment.py` (new) | Curves, `t50`, `Δt50`, Δ sensitivity | Reproduces the synthetic test results; a placeholder `hypothesis_status.json` records which analyses are confirmatory |
| `tests/` | One test file per new module above | CPU tests pass on the laptop; GPU tests pass on Kaggle |
| `report/protocol.md` | Addendum B now freezes HA–HC, the grid, Δ, `M`, `L`, onsets and thresholds | Hashes recorded before TEST |

---

## 18. Figure and table design

*A figure exists only if it answers a stated question. All are generated from saved data by `src/figures.py`.*

| # | Item | Question | Raw data | Axes / columns | Uncertainty | Interpretation rule |
|---|---|---|---|---|---|---|
| F1 | Methodology diagram | What is the design and which parts are confirmatory vs exploratory? | design | boxes | — | Colour-code confirmatory vs exploratory |
| T1 | Dataset composition and balance | Is the benchmark balanced and independent? | `scene_graphs.json` | split × category × object-group × complexity | — | Show leakage policy result (zero shared object pairs) |
| F2 | Prompt examples: canonical vs paraphrase vs counterfactual | What do the variants look like? | `prompts_v2.json` | scene rows × variant columns, with declared edit | — | Select by a stated rule (first scene per category) |
| F3 | Adherence vs CFG and steps | How do CFG and steps affect each constraint? | E11 | x = CFG; y = success; panels by type; lines = steps | scene-bootstrap CI | Always show the CLIP/diversity companion panel |
| F4 | Success vs constraint complexity | Does success fall as constraints increase, relative to independence? | E10 | x = #constraints; y = success; independence curve | CI | A gap to the independence curve ≠ proof of interference without E10 checks |
| F5 | Attention-map trajectories | How do word maps evolve over denoising in success vs failure? | `.npz` | rows = steps (early/middle/late), columns = words | — | Examples chosen by a fixed rule; never claimed as evidence |
| F6 | Conflict score vs binding correctness | Is conflict higher in failures, by stage? | E05 features | x = stage; y = conflict; success vs failure | CI | Association only |
| F7 | Held-out diagnostic performance | Does attention add ΔAUC beyond baselines? | E05 | bars = AUC by model; ΔAUC with CI | scene bootstrap | Apply §15.6 decision rule; show MDE |
| F8 | ΔAUC vs step | When does the signal appear? | E06 | x = step; y = ΔAUC | CI | EXPLORATORY label |
| F9 | Intervention arms comparison | Does targeted beat random/global/control? | E12 | arms × outcome | paired CI | Report guards next to it |
| F10 | Token/stage ablation matrix | Which components matter? | E13 | rows = ablations; columns = outcomes | CI | EXPLORATORY; P3 only |
| F11 | Evaluator vs human agreement | How good is each judge per type? | E01 | judge × type accuracy/recall; confusion matrices | Wilson CI | Types < 0.85 are flagged EXPLORATORY |
| F12 | Robustness under wording changes | Is there wording sensitivity beyond seed noise? | E08 | type × excess RMS | CI | Show permutation null |
| F13 | Quality / diversity / compute trade-off | What does each setting cost? | E14 | x = latency, y = adherence, size = CLIP | CI | Pareto only |
| F14 | Failure-case gallery | What do failures look like? | E19 | selected by rule | — | Include successes and failures at equal rates |
| F15 | Effect-size forest plot | What are all primary and secondary effects with CIs? | all | estimate ± CI vs MDE band | scene bootstrap | Shade the "unresolvable" region below MDE |
| T2 | Statistical results table | Numbers for every primary test | all | estimate, CI, p (Holm), MDE | — | Same numbers appear in text via `report_numbers.py` |

### 18.1 Revision 2 figures and tables
| # | Item | Question | Raw data | Axes / columns | Uncertainty | Interpretation rule |
|---|---|---|---|---|---|---|
| F16 | **Commitment curves** | At which step is each constraint decided? | `e23/curves.csv` | x = restart step; y = `C_κ(t)`; one line per constraint type; dashed `C = 0.5` | Scene bootstrap band | Show every type including `t50 = "> 36"`; never single out a type post hoc |
| T3 | `t50` and `Δt50` table | Primary result for HA | `e23/t50.json` | type × (`t50`, CI); pairwise `Δt50` | Bootstrap | Apply the HA decision rule; show number of scenes after the presence filter |
| F17 | Commitment vs CFG (EXPLORATORY) | Does guidance scale shift commitment? | E23 at CFG 7.5 plus E11 cells | x = step; y = `C`; panels by CFG | Bootstrap | Only if branches exist for >1 CFG; otherwise omit |
| F18 | Variance decomposition of outcomes | How much is noise, prompt, CFG? | pilot, later E04 | stacked bars per constraint type | Bootstrap | Components overlap; report as shares, not causes |
| T4 | Noise-only and attention readout AUCs | Does information precede commitment? | E05 + E25 | model × window (`pre`, `post`) AUC and ΔAUC | Scene bootstrap | Apply HB label (forecast / readout / none) |
| F19 | Fragility margin | Are failures closer to a decision boundary? | `e24/flips.jsonl` | margin distribution for successes vs failures | Bootstrap | EXPLORATORY |
| F20 | Layer-group knockout matrix | Which layer groups are necessary for which constraint? | E26 | rows = group; columns = outcome; effect real − random | Paired bootstrap | EXPLORATORY; show the random-token control side by side |
| F21 | Guidance attribution by token class | Which tokens carry guidance? | E27 | token class × CFG; share of `‖g‖` | Bootstrap | EXPLORATORY; fixed trajectories only |
| F9′ | Onset arms | Is the edit effect time-gated? | E12 | x = onset (early/mid/late); y = targeted − random; separate panels for conditional and unconditional outcomes | Paired bootstrap | HC rule; show guards |
| F1′ | Methodology diagram, Revision 2 | What is confirmatory vs exploratory? | design | commitment → readout → handle | — | Colour-code confirmatory (HA, HB, HC) vs exploratory |

Figures F3–F8 and F12–F15 of §18 remain unchanged; F7/F8 are re-labelled as the HB analysis (pre/post windows).

---

## 19. Result-interpretation handbook

*Written before any confirmatory run. Every cell is conditional ("if … then …"). Nothing here has been observed.*

### 19.1 H1 — Early attention diagnostics
| Scenario | If this happens | What the paper may say | What it may not say | Follow-up that resolves the ambiguity |
|---|---|---|---|---|
| Best case | ΔAUC ≥ 0.05, CI excludes 0, stable under E15 and across CFG | "Under the tested setting, early attention features improved held-out prediction of colour-binding beyond prompt-structure and guidance covariates." | "Attention causes binding"; "predicts failures in general"; "works on other models" | E12 (causal probe), E16 (transfer), E18 (judge robustness) |
| Mixed | Gain on scene split, none on object-pair split | "The signal did not generalise to unseen object pairs." | "attention is informative" without qualification | Identity-matched analysis; more object pairs |
| Mixed | Gain over B1 but not over B3 (x0-detector) | "Attention was no better than an early detector signal." | "uniquely informative" | Combined model |
| Null / inconclusive | CI wide, includes 0 | "The study could not resolve a gain of 0.05 (CI …)." | "no effect" | More scenes; DEV-based power simulation |
| Bounded null | CI upper < 0.03 | "Attention features added no practically meaningful information beyond B1." | "attention maps are uninformative" | None needed; report as a negative result |
| Unexpected opposite | M_att worse than B1 on TEST | "Added features reduced held-out performance (overfitting)." | any positive claim | Stronger regularisation; check DEV/TEST shift |
| Implementation failure | Shuffled-attention control matches M_att; leakage found | Report the audit and withdraw the result | any diagnostic claim | Fix the pipeline; re-run on DEV; TEST is **already spent** → state it |
| Evaluator-confounded | Gain vanishes under J2/human labels | "Apparent diagnostic value was judge-dependent." | the primary claim | Audit-calibrated labels on the audited subset |

### 19.2 H2 — Wording sensitivity
| Scenario | If this happens | May say | May not say | Follow-up |
|---|---|---|---|---|
| Best case | Binding and/or spatial excess RMS ≥ 0.10; presence/count smaller | "Meaning-preserving rewrites changed scene-level binding/spatial success by more than seed noise." | "the model does not understand language" | Embedding-distance analysis of which rewrites matter |
| Mixed | Large for one rewrite type only | Name the type; "driven by mention-order changes" | Generalisation across rewrites | Targeted rewrite sets |
| Null | CI upper < 0.05 | "No wording sensitivity beyond seed noise was detected for these rewrite types." | "robust to paraphrase" | Harder rewrites |
| Inconclusive | Wide CI | "Could not resolve shifts below X" | "no effect" | More scenes |
| Opposite | Rewrites systematically *improve* success | Report; investigate lexicon artefacts | "rewrites help" | Check lexicon and judge on rewritten prompts |
| Implementation failure | Rewrite validity review fails for > 10% items | Drop affected variants and say so | any result using them | Re-review |
| Evaluator-confounded | Judge accuracy drops on rewrites (E01 by variant) | "Part of the measured sensitivity reflects judge error." | the sensitivity claim | J2/human re-scoring |

### 19.3 H3 — Targeted edit as a causal handle
| Scenario | If this happens | May say | May not say | Follow-up |
|---|---|---|---|---|
| Best case | targeted − random ≥ 0.08, CI > 0, guards respected | "A targeted attention edit improved binding relative to a matched random edit." | "new state of the art"; "superior to published methods" (no fair comparison) | E13 ablations; E16 transfer; optional published baseline |
| Mixed | targeted > control but ≈ random | "Gains were not specific to the colour–object pair." | "attention overlap is causal" | Larger β; different masks |
| Mixed | Global helps presence, targeted helps binding | "Presence and binding responded to different edits." | one-edit-fits-all | Combined arm |
| Null | all CIs include 0, MDE ≈ 0.07 | "Effects below ≈ 0.07 could not be resolved." | "the intervention does not work" | More scenes; stronger edits |
| Bounded null | CI upper < 0.04 | "No meaningful improvement from this edit." | "attention cannot be a handle" (one rule only) | Other edit rules |
| Opposite | targeted lowers binding or raises duplicates | "The edit degraded outcomes (e.g. more duplicate objects)." | any adherence gain | Guard-metric analysis |
| Implementation failure | `n_edits` ≈ 0, or acceptance tests fail | "The correction rarely triggered." | any effect claim | Fix and re-run on DEV |
| Evaluator-confounded | Gain only under J1 | "Gain was judge-dependent." | any gain | Calibration with audit |

### 19.4 Secondary results (all EXPLORATORY unless stated)
| Result | Legitimate wording | Forbidden wording |
|---|---|---|
| CFG raises CLIP/diversity effect but adherence unresolved (pilot-like) | "CFG changed global similarity more than constraint adherence in this benchmark." | "CFG does not improve adherence" |
| CFG × complexity interaction found | "The CFG benefit depended on the number of constraints (CI …)." | A mechanism |
| Evaluator accuracy per type | "Judge accuracy was X–Y by type (CI …)." | "metrics are valid" without qualification |
| SDXL replicates the sign | "The direction replicated on SDXL (n = 100 scenes)." | "architecture-independent" |

### 19.5 Revision 2 handbook — HA, HB, HC (written before any run; every cell is conditional)

#### HA — Commitment timing
| Scenario | If this happens | What the paper may say | What it may not say | Follow-up |
|---|---|---|---|---|
| Best case | `abs(Δt50) ≥ 3`, CI excludes 0, same sign under Δ ∈ {4, 12} and under J2 | "Under restart magnitude Δ, the binding outcome (given presence) was committed [later/earlier] than presence (Δt50 = …, CI …)." | "Binding is decided at step X"; "in all models" | E12 onset arms (HC); E16 replication |
| Mixed | Sign holds at Δ = 8 but not at 4 or 12 | "The ordering depended on the restart magnitude." | A single commitment step | Report full curves |
| Bounded null | CI ⊂ (−1.5, +1.5) | "No difference in commitment time larger than 1.5 steps was detected between presence and binding." | "They are decided at the same time in general" | HB/HC still interpretable with a shared `t50` |
| Inconclusive | Wide CI | "Could not resolve a difference of 3 steps (half-width …)." | "No effect" | More scenes or branches (DEV gate) |
| Opposite | Binding committed earlier than presence | Report as found; check judge and conditional-set selection | Mechanism claims | E26 layer knockouts; E18 |
| Estimator unreliable | Non-monotone curves, `C` negative, full-resample control `C > 0.15` | "The estimator was unreliable for this type." | Any ordering | Larger `M`; J2 re-scoring |
| Implementation failure | Δ = 0 control not identical; moment test fails | Withdraw and fix on DEV | Any result | DEV rehearsal again |
| Evaluator-confounded | Ordering flips with J2 or audit labels | "Commitment ordering was judge-dependent." | The ordering | Audit-calibrated labels |

#### HB — Forecast, readout or neither
| Scenario | If this happens | May say | May not say | Follow-up |
|---|---|---|---|---|
| Forecast | `ΔAUC_pre ≥ 0.05`, CI > 0 over `B_noise` | "Attention features available before the binding commitment point added information beyond prompt structure and the initial noise." | "Attention predicts failure in general"; causality | HC |
| Readout only | pre CI upper < 0.03 and post ≥ 0.05 | "Attention features separated outcomes only after the commitment point." | "Attention is uninformative" | E06 timing detail |
| None | both CI upper < 0.03 | "No linear attention feature added information over prompt structure and noise." | "Attention is irrelevant" | Nonlinear probes (exploratory) |
| Noise explains most | `B_noise` ≫ B1 and attention adds nothing | "The initial latent carried most of the available information." | "The seed determines the outcome" | E24, E25 |
| Inconclusive | Wide CIs | "Could not resolve …" with MDE | "No effect" | More scenes |
| Gate failure | DEV simulation CI half-width > 0.04 | HB is exploratory | A confirmatory claim | Report CIs only |
| Leakage | Shuffled-attention control matches | Withdraw | Any claim | Fix, re-run on DEV (TEST spent: state it) |
| Evaluator-confounded | Result disappears under J2 | "Judge-dependent." | The claim | Audit-calibrated labels |

#### HC — Time-gated edit
| Scenario | If this happens | May say | May not say | Follow-up |
|---|---|---|---|---|
| Time-gated handle | `D_early ≥ 0.08` (CI > 0) and `G ≥ 0.10` (CI > 0) | "A targeted edit improved binding given presence relative to a matched random edit when applied before, but not after, the estimated commitment step." | "Attention causes binding"; "new method"; "state of the art" | E13 ablations; E16 |
| Handle, not gated | `D_early` supported, `G` ≈ 0 | "The edit helped but its effect did not depend on onset, which is inconsistent with the commitment interpretation of HA." | HA's mechanism | Re-examine HA (Δ, judge) |
| Random edit does as well | `D_early` ≈ 0 | "Gains were not specific to the targeted tokens." | "Attention is not a handle" | Different edit rules |
| Late edit works | `D_late` > 0 | "Edits after the estimated commitment point still changed the outcome, so the commitment estimate is not a point of no return." | Contradiction of HA without more data | Check Δ and edit strength |
| No effect, low power | CIs include 0 with MDE ≈ 0.07–0.08 | "Effects below ≈ MDE could not be resolved." | "The edit does not work" | More scenes |
| Degradation | Duplicates or CLIP guard violated | "The edit degraded outcomes." | Any adherence gain | Guard-metric breakdown |
| Presence/binding mismatch | Conditional and unconditional contrasts disagree | Report both | One selected | E18 |
| Implementation failure | Bias-mass mismatch > 5% or `n_edits` ≈ 0 | Withdraw and fix | Any result | DEV |

#### Supporting results (all EXPLORATORY)
| Result | Legitimate wording | Forbidden wording |
|---|---|---|
| Failures are less stable than successes (E24) | "Failures were less stable under micro-perturbation." | "Failures are random" |
| Layer-group knockout differs from the random control (E26) | "Knockout in group G reduced binding given presence more than the matched random knockout." | "Binding is localised in G" |
| Guidance concentrated on object tokens (E27) | "Guidance contribution was concentrated on object tokens along the fixed trajectories." | "CFG ignores attributes" |
| Late negative prompt helps less than an early one (E28) | "Timing of the negative window changed the effect." | "Negative prompts fix binding" |
| Count regresses toward ≈ 3 (pilot F5) | "Mean detected count compressed toward ≈ 3 in the pilot (28 prompts)." | A general numerosity law |

---

## 20. Titles, contribution statements and the paper outline

### 20.1 Candidate titles (ranked by scientific defensibility)
| Rank | Title | Why this rank |
|---|---|---|
| 1 | *Do Early Cross-Attention Patterns Predict Compositional Failure? A Held-Out, Counterfactual Study in Stable Diffusion* | Question form; commits to no result; matches the design |
| 2 | *Where Does the Prompt Go? Held-Out Diagnostics and Counterfactual Robustness of Prompt Adherence in Diffusion Models* | Continuity with the project; accurate scope |
| 3 | *Compositional Adherence under Inference-Time Guidance: Complexity, CFG and Evaluator Reliability* | Fits the fallback paper (C3 + C5 + C6) |
| 4 | *Early Attention Diagnostics for Constraint-Level Failure in Text-to-Image Diffusion* | Stronger claim; valid only if H1 is supported |
| 5 | *The Cost of Following the Prompt: Adherence, Quality and Compute under Inference-Time Correction* | Valid only if E14 yields a clear trade-off |
| 6 | *Attention as Symptom or Handle? A Matched-Control Test of Attention Correction for Attribute Binding* | Implies a result in the title; use only after H3 |

### 20.2 Recommended title and contribution statements
**Recommended:** Title 1 (or 2).
- **Central question:** Does early cross-attention carry constraint-specific information about eventual failure beyond prompt structure and guidance, and is it a causal handle?
- **Primary contribution:** A held-out, scene-graph-level evaluation of early-attention diagnostics with matched-control interventional probe.
- **Supporting contributions:** Benchmark V2 with counterfactual/meaning-preserving variants and leakage-safe splits; validated multi-judge evaluation with a blinded audit; controlled CFG × steps × complexity measurements; negative-result record for global attention re-weighting.
- **Most direct competitors:** Diffusion Probe, Gather-and-Bind / Divide & Bind / SynGen, MetaLogic, Attention Modulation, CoBind.
- **Evidence required for the main claim:** §8.1 support criterion on TEST, replicated across CFG, with E15 and E18 not contradicting it.
- **What would invalidate or weaken it:** bounded null on H1; failure under object-pair split; evaluator-confounded labels; shuffled-attention control matching the real features.

**Version A — conservative (use if only mechanistic/benchmark evidence is established).**
> "We introduce a scene-graph benchmark with counterfactual and meaning-preserving prompt variants, evaluate how well early cross-attention features predict constraint-level compositional failure on held-out scenes relative to prompt-structure baselines, and measure how much of our conclusions depend on the choice of evaluator. We report [support / bounded-null / inconclusive] for the diagnostic hypothesis, with confidence intervals and the smallest effect our design could resolve."

**Version B — method-focused (use *only* if the targeted edit shows a replicated improvement on held-out scenes **and** a fair comparison with at least one published method exists).**
> "We further show that a bounded, conflict-gated attention-logit edit improves colour binding relative to matched random and global edits on held-out scenes, [and compare it with <published method> under identical conditions]."
Version B requires: H3 supported on TEST; E13 ablations; E16 replication; a fair published baseline; and a completed full-text novelty audit. **Absent any of these, Version A applies.**

### 20.3 Paper outline (claims allowed, evidence needed, figures/tables)
| § | Section | Claims allowed | Evidence needed | Figures/tables | Status of evidence |
|---|---|---|---|---|---|
| 1 | Abstract | One-sentence question, design, main result with CI, limitation | All below | — | FUTURE |
| 2 | Introduction | Motivation: images look plausible but violate constraints; early attention may reveal it | Cited literature | F1 | VERIFIED prior work |
| 3 | Research gap and contributions | Stated precisely vs Diffusion Probe, MetaLogic, G&B/D&B/SynGen | Completed full-text audit (§5) | prior-art table | **PENDING (UNVERIFIED gaps)** |
| 4 | Related work | Foundations, binding methods, evaluation drift, robustness | §4 | table | VERIFIED (FT/ABS marked) |
| 5 | Formal problem definition | Scene graph, constraints, outcomes, complexity | §11 | schema figure | PROPOSED |
| 6 | Hypotheses | H1–H3 with decision rules | §8 frozen addendum | — | PLANNED |
| 7 | Benchmark and dataset | Composition, grammar, balance, splits, leakage policy | E02 | T1, F2 | FUTURE |
| 8 | Metrics and evaluator validation | Judges, audit, accuracy by type | E01, E22, E18 | F11 | FUTURE (pilot validation is EXPLORATORY) |
| 9 | Model and experimental setup | SD 1.5 fp16, DDIM eta 0, 512², CFG/steps, seeds, Kaggle hardware | E03 hardware file | setup table | FUTURE |
| 10 | Proposed method | The probe, **not** a new algorithm claim | §13 | pseudocode | PROPOSED |
| 11 | Main experimental results | CFG × steps × complexity; adherence by type | E11, E10 | F3, F4, T2 | FUTURE (pilot: CFG 2→7.5 inconclusive; CLIP/diversity clear) |
| 12 | Attention and mechanistic analysis | Diagnostics, timing, features | E05, E06, E07, E15 | F5–F8 | FUTURE (pilot EXPLORATORY) |
| 13 | Counterfactual robustness | Wording sensitivity; edit fidelity | E08, E09 | F12 | FUTURE |
| 14 | Intervention comparisons and ablations | Probe vs controls; ablations | E12, E13 | F9, F10 | FUTURE (pilot: global arm null, VERIFIED) |
| 15 | Efficiency and generalisation | Trade-offs; SDXL replication | E14, E16 | F13 | FUTURE |
| 16 | Statistical uncertainty and error analysis | CIs, MDE, failure taxonomy | §15, E19 | F14, F15 | FUTURE |
| 17 | Limitations | Scene count, single base model, judge error, hardware dependence, SD 1.5 age | §21 | — | WRITE NOW |
| 18 | Conclusion | Only what the evidence in §19 allows | — | — | FUTURE |
| 19 | References | §23 | — | — | VERIFIED list |
| 20 | Reproducibility statement | Hashes, seeds, versions, commands, hardware file | §16.4, §17 | — | FUTURE |

*Rule:* an outline row marked FUTURE may not be written as a completed result.

### 20.1b Revision 2 candidate titles (ranked by scientific defensibility)
| Rank | Title | Why this rank |
|---|---|---|
| 1 | *When Is a Prompt Constraint Decided? Commitment Timing in Text-to-Image Diffusion* | Question form; commits to no result; matches HA |
| 2 | *Where Does the Prompt Go? Commitment, Readout and Time-Gated Edits for Compositional Constraints* | Continuity with the project title; covers HA–HC |
| 3 | *Forecast or Readout? What Attention and Initial Noise Reveal Before a Constraint Is Decided* | Fits if HB is the strongest result |
| 4 | *Compositional Adherence under Inference-Time Guidance: Complexity, CFG and Evaluator Reliability* | Fallback (C3 + C5 + C6) |
| 5 | *Attention as Symptom or Handle? A Time-Gated Matched-Control Test* | Result-implying; use only after HC |

Rev. 1 titles 1–6 (§20.1) are retained as fallbacks if HA is not feasible.

### 20.2b Revision 2 contribution statements
**Recommended:** Title 1 or 2.
- **Central question:** At which denoising step is each compositional constraint decided; what do attention, prompt structure and initial noise reveal before and after that step; can an attention edit work only before it?
- **Primary contribution:** A causal, constraint-type-specific measurement of commitment timing by interventional re-noising on controlled scene graphs, with a pre-specified test of whether attention-based information (HB) and attention edits (HC) are time-gated by it.
- **Supporting contributions:** Benchmark V2; conditional (presence-aware) outcome definitions and an evaluator audit; controlled CFG × steps × complexity measurements; noise-determinism and fragility analyses; layer-group knockouts and guidance attribution (exploratory); negative-result record for the global re-weighting.
- **Most direct competitors:** Critical-windows theory (ICML 2024), Restart sampling (operation only), negative-prompt timing (arXiv 2406.02965), Diffusion Probe, InitNO, Gather-and-Bind / Divide & Bind / SynGen, CoBind, Attention Modulation, MetaLogic.
- **Evidence required for the main claim:** HA support on sealed TEST under §8.1, stable across Δ ∈ {4, 12} and judge change.
- **What would weaken it:** a bounded null for HA; strong Δ-dependence; non-monotone curves; evaluator-confounded outcomes.

**Version A (conservative).**
> "We measure, by interventional re-noising of the sampling trajectory on scene-graph-generated prompts, the denoising step at which presence, count, colour-binding and spatial outcomes become fixed, report whether early attention features and the initial noise carry information about the outcome before that step, and test whether an attention edit changes the outcome only before it. We report [support / bounded-null / inconclusive] for each hypothesis, with confidence intervals and the smallest effect our design could resolve."

**Version B (use only if HC is supported, E13 and E16 confirm it, and a fair published baseline plus a completed full-text novelty audit exist).** Adds: "…and a time-gated, bounded attention-logit edit that improves colour binding relative to matched random edits." Otherwise Version A applies.

### 20.3b Revision 2 outline changes
| § | Section | Change |
|---|---|---|
| 6 | Hypotheses | HA–HC (H1/H2 supporting) |
| 10 | Method | Commitment operation and branch protocol replace "proposed method"; the targeted edit remains a probe |
| 11 | Main results | Adds F16/T3 (commitment) before F3/F4 |
| 12 | Mechanistic analysis | Adds readout vs forecast (T4), noise determinism (F18), fragility (F19) |
| 14 | Intervention | Onset arms (F9′); knockouts (F20); guidance attribution (F21) |

---

## 21. Risks, limitations, alternative outcomes and stop conditions

### 21.1 Risk register
| Risk | Likelihood (judgement) | Impact | Mitigation |
|---|---|---|---|
| H1 underpowered (nested ΔAUC SE unknown) | Medium–high | High | DEV simulation gate; FULL plan; report CI honestly |
| Floor effect (binding/multi at ≈ 6% pilot success) | High | High | Calibrate complexity on DEV; use soft fraction; stronger model replication |
| Evaluator error confounds labels | High | High | E01/E18/E22; per-type "exploratory" rule |
| Novelty overlap discovered (Diffusion Probe, CoBind, Attention Modulation) | Medium | High | Read full texts first; fall back to C3+C5+C6 (§21.3) |
| Kaggle quota/session/limits unknown | Medium | Medium | E03 measurement; LIMITED plan |
| Two-worker file collisions | Medium | Medium | Shard-per-worker files, atomic writes, verify/merge scripts |
| Cross-hardware non-identity (3050 vs Kaggle) | Medium | Medium | E00; do not pool pilot images into TEST |
| Attention hook not image-neutral on SDXL | Medium | Medium | Bit-identity test or mark unsupported |
| Benchmark leakage (grammar/lexicon tuned on TEST) | Low–medium | High | Sealed TEST; policy §11.6 |
| Annotation effort exceeds capacity | Medium | Medium | 190-image minimum; stratified sampling; document the limitation |
| AI-assisted labels or code undisclosed | — | High | Record provenance of labels and code in the paper and datasheet |
| Documentation drift (stale report/README) | High (observed) | Low–medium | Rebuild generated docs after every result change; `verify_docs` check |

### 21.2 Stop conditions (explicit)
1. **After E21:** if leave-prompt-out ΔAUC has CI upper < 0, do **not** pursue Direction A as the headline; switch to the fallback paper.
2. **After E03:** if the measured FULL cost does not fit the quota, adopt LIMITED and downgrade H1 to a calibrated pilot.
3. **After DEV simulation:** if the projected CI half-width for ΔAUC exceeds 0.04, H1 is exploratory; do not spend TEST on it as a confirmatory test.
4. **After E01 on DEV audit:** if no judge reaches 0.85 on binding, H1 and H3 become exploratory.
5. **After DEV tuning of H3:** if no arm beats control by ≥ 0.03 on DEV, skip TEST arms and E13.
6. **After E02 QA:** if reviewers reject > 20% of counterfactual/paraphrase variants, revise the grammar before generating anything.
7. **Never** open TEST labels to tune anything; a violation is recorded and the affected analyses are re-labelled EXPLORATORY.

### 21.3 If the proposed intervention fails — the paper still stands
Combining outcomes:
| H1 | H3 | Judge robust | Paper becomes |
|---|---|---|---|
| Supported | Supported | Yes | Diagnostic + causal-handle paper (Version A, possibly B) |
| Supported | Null | Yes | "Attention is a diagnostic but this edit is not a handle" — a precise mechanistic finding |
| Null (bounded) | — | Yes | Negative result: "attention adds nothing beyond prompt structure and CFG on held-out scenes", plus Benchmark V2, wording sensitivity and CFG × complexity |
| Inconclusive | — | Yes | Calibrated pilot with explicit MDEs; benchmark and evaluator audit are the contributions |
| Any | Any | **No** | Evaluator-reliability paper: "which conclusions survive a change of judge" |
In every row, Benchmark V2 + evaluator audit + controlled CFG × steps × complexity measurement (C3, C5, C6) remain valid contributions.

### 21.4 Limitations to state in the paper now
28-prompt pilot (12 for binding diagnostics); one base model and scheduler (SD 1.5, DDIM); detector-based judge with coarse spatial rule; labels began as an AI draft and are non-independent; no aesthetic study; attention analysis conditional-branch only and layer-averaged; hardware-dependent identity; results do not transfer to DiT models without new extraction code.

### 21.5 Final quality audit (Revision 1; see §21.8 for Revision 2)
| Question | Answer | Basis / revision |
|---|---|---|
| Does the contribution differ from the closest papers in a precisely identified way? | **Conditionally.** vs Diffusion Probe: target granularity (constraint-level vs overall quality), interpretable features, scene-held-out test, prompt-only baselines, interventional follow-up. vs MetaLogic: declared-edit counterfactuals with preservation scoring. **UNVERIFIED** until full texts are read | §5; Phase 0 action |
| Is the research question more specific than "change CFG and see what happens"? | Yes: H1–H3 with operational definitions and decision rules | §8 |
| Can the experiments answer it with the available independent scene graphs? | **Only under FULL** (≈ 140 binding-bearing TEST scenes). A total of 200 scenes (≈ 70 binding-bearing TEST) cannot resolve ΔAUC = 0.05 → **revised the target from ≈ 200 to 300 scenes** | §11.9, §15.4 |
| Are images, prompts and the independent unit distinguished? | Yes (scene graph is the unit; counts tabulated separately) | §15.1, §16.5 |
| Is the evaluator credible for every constraint type? | **No, not yet.** Spatial (n = 9, 2 positives) and binding are unverified; the plan gates each type behind an audit accuracy of 0.85 | §12, §9.0 |
| Are the primary hypotheses falsifiable? | Yes (bounded-null rules) | §15.6 |
| Is the compute plan feasible on the actual Kaggle accelerators? | **Unknown** until E03; two plans provided; no speed-up factors used | §16 |
| Does each experiment resolve a question? | Each card states its hypothesis or is explicitly a gate/measurement | §9 |
| Are null/opposite results interpretable? | Yes | §10, §19 |
| Does the plan avoid invented observations and unsupported novelty? | Yes: every number is verified, derived or a labelled design parameter; novelty verdicts are provisional | §1–§5 |
| Can the study still yield a valid contribution if the intervention fails? | Yes | §21.3 |

### 21.6 Revision 2 risks, stop conditions and fallbacks
| Risk | Likelihood (judgement) | Impact | Mitigation |
|---|---|---|---|
| Commitment curves are judge-noise dominated (low `A − A0` signal) | Medium | High | Conditional outcomes; J2 re-scoring; audit-calibrated labels; larger `M` after the DEV gate |
| `t50` depends strongly on Δ | Medium | Medium | Report curves; E23b; qualify HA |
| Conditional sample too small (only ≈ 45% of binding images have both objects) | High | Medium | Raise binding-bearing scene count; report scene counts; DEV gate |
| Restart noise leaves the data manifold for large Δ | Low–medium | Medium | Δ sensitivity; moment unit test; full-resample control |
| Novelty overlap with critical-window or noise-timing work discovered | Medium | High | Read full texts first (Phase 0); fall back to the Rev. 1 programme or to the measurement paper (§21.3) |
| ACL 2026 head-level paper overlaps E26 | Medium | Low | Read before running E26; E26 is exploratory |
| Time cost of E23 exceeds quota | Medium | Medium | LIMITED plan (five grid steps, 60 scenes); drop E24, E26–E28 first |
| Latent storage mismatch (saved fp16 latent differs from recomputed) | Low | High | Acceptance test in §17.1; recompute-vs-load check on 10 items |
| fp32 fragility control does not fit the hook | Medium | Low | Check in E03; otherwise run on CPU for a subset or omit |

**Additional stop conditions (Revision 2).**
1. **After the E23 DEV rehearsal:** if the Δ = 0 control fails, or the full-resample control gives `C > 0.15`, stop and fix before any TEST branch is generated.
2. **After the DEV gate:** if the projected CI half-width of `Δt50` exceeds 4 steps even at the maximum affordable design, HA becomes exploratory; the paper is then the Rev. 1 measurement paper plus commitment curves with CIs.
3. **After E21 (extended):** if the pilot conditional-label AUC of the attention margin is not above 0.5 (CI lower bound ≤ 0.5), HB is not pursued as a confirmatory test.
4. **After E25 on DEV:** if `B_noise` already reaches the attention model's AUC, the attention claim is dropped and the noise result is the finding.
5. **After DEV tuning of E12 onset arms:** if no arm beats the random arm by ≥ 0.03 on DEV, HC is not run on TEST; the paper reports HA and HB only.
6. TEST labels remain sealed (rule 7 of §21.2 applies to HA–HC).

**Fallback matrix (Revision 2).**
| HA | HB | HC | Judge robust | Paper becomes |
|---|---|---|---|---|
| Supported | Forecast / readout | Time-gated | Yes | Commitment + readout + handle paper (Version A, possibly B) |
| Supported | Any | Not supported | Yes | "Commitment timing differs by type, and this edit is not a handle" |
| Bounded null | Any | Any | Yes | "No type-specific commitment difference ≥ 1.5 steps" plus Benchmark V2 and the evaluator audit |
| Inconclusive | — | — | Yes | Calibrated measurement paper with explicit MDEs |
| Any | Any | Any | **No** | Evaluator-reliability paper |
In every row Benchmark V2, the evaluator audit and the controlled CFG × steps × complexity measurement remain valid contributions.

### 21.7 Revision 2 limitations to state in the paper now
28-prompt pilot (12 for binding diagnostics); the pilot label counts an undetected object as a binding failure (corrected to conditional outcomes in V2); detector-based judge; single base model and scheduler; commitment measured relative to a restart magnitude Δ; commitment curves characterise the sampler (DDIM, 40 steps) and are not claims about training-time learning; conditional outcomes select on presence; hardware-dependent identity.

### 21.8 Revision 2 final quality audit (answers inside the document)
| Question | Answer | Basis |
|---|---|---|
| Does the new headline differ from the closest papers in an identified way? | **Conditionally.** vs critical-windows theory: constraint-type-specific interventional measurement on SD with controlled scene graphs. vs Restart sampling: the operation is used as a measurement, not a sampler. vs negative-prompt timing: positive-constraint commitment, not negative-prompt delay. vs Diffusion Probe: constraint-level timing and a noise-only baseline. **UNVERIFIED** until full texts are read | §6A, §5 |
| Is the question more specific than "change CFG and see what happens"? | Yes: three hypotheses with operational definitions and decision rules | §8 |
| Is HA testable with the available compute? | Probably under FULL; E23 costs ≈ 1,600 image-equivalents (TEST) plus ≈ 640 for the DEV rehearsal, **a design estimate, not a measurement**; confirm in E03 | §16.5b |
| Is the power for HA known? | **No**; no commitment data exist. A DEV gate decides | §15.8 |
| Is the pilot evidence overstated anywhere? | Corrected: the Rev. 1 binding label includes omissions (§1.2, §2.7); all §2.7 numbers are EXPLORATORY | §2.7 |
| Can the study yield a valid contribution if every hypothesis fails? | Yes (§21.6 fallback matrix) | §21.6 |

---

## 22. Prioritised execution roadmap (phases and gates; no calendar times are asserted)

| Phase | Work | Experiments | Gate to leave the phase |
|---|---|---|---|
| **0 — Hygiene and audit completion** | `git init` and tag the current state; rebuild generated documents (`build_report.py`, `make_slides.py`); fix `README.md`; delete autosave/backup artefacts; **obtain and read full texts** of Diffusion Probe, MetaLogic, CoBind, Attention Modulation, SynGen, ToMe, StructureDiffusion, VQAScore/GenAI-Bench, DSG, T2I-CompBench++; update §5 | E00 (re-analysis part), **E21** | **G0:** novelty audit updated; E21 does not rule out Direction A |
| **1 — Build** | Implement `benchmark_v2.py`, shard/merge/verify scripts, attention storage upgrade, `intervene.py` modes, judge interface; write all unit tests | E02, E22 (design) | **G1:** all tests green on CPU; QA gates of E02 pass |
| **2 — Kaggle pilot** | Run hardware probe and E03 (throughput, VRAM, hook identity, 2-worker scaling, SDXL fit, J2 VLM fit); E00 cross-hardware check | E00 (regeneration), E03 | **G2:** choose FULL vs LIMITED; fix model list (SDXL yes/no) |
| **3 — DEV generation and tuning** | Generate DEV; audit sampling design; fit predictors on DEV; tune H3 arms; choose primary judges; run power simulation | E04 (DEV), E01 (DEV audit), E20, parts of E05–E07, E12 tuning | **G3:** Addendum B written and hashed; TEST still sealed; H1 confirmatory vs exploratory decided |
| **4 — TEST generation** | Generate TEST in two workers; verify/merge; run blinded audit | E04 (TEST), E11, E12, E16 | **G4:** `verify_outputs.py` clean; zero duplicates |
| **5 — Confirmatory analysis** | Run frozen analyses exactly once; then supporting analyses | E05, E08, E12, E18, then E06, E07, E09, E10, E14, E15, E19 | **G5:** results tables generated from files; no analysis outside the addendum labelled confirmatory |
| **6 — Write-up** | Fill the outline using only the §19 permitted wording; rebuild documents; reproducibility statement | — | Final checklist: every number traces to a saved file; limitations listed |

**Next ten actions (concrete):**
1. Initialise git and commit the current tree (so hashes and history exist).
2. Rebuild `report.md`, `defense.md`, `presentation.pptx` from current results; correct `README.md`.
3. Add the missing full-text PDFs to `Researcha-thon Project/` (Diffusion Probe 2602.23783, MetaLogic 2510.00796, CoBind 2607.16307, Attention Modulation 2404.13899, SynGen 2306.08877, ToMe 2411.07132, StructureDiffusion 2212.05032, VQAScore 2406.13743, DSG 2310.18235, T2I-CompBench++ 2307.06350).
4. Run E21 (feasibility gate) on the saved attention maps.
5. Draft the synonym lexicon and object/colour inventories; run E22 (synthetic probe) on CPU.
6. Write `benchmark_v2.py` with the validators and unit tests.
7. Write the annotation guidelines and the audit sampler.
8. Implement shard/merge/verify scripts with tests.
9. Prepare the Kaggle probe notebook and the E03 protocol.
10. Draft Addendum B (leaving numeric tuning fields blank until DEV).

### 22b. Revision 2 roadmap changes
| Phase | Change in Revision 2 | Gate |
|---|---|---|
| 0 | Add: read the ACL 2026 head-level paper and the critical-windows paper in full; extend E21 with conditional-label and noise-only checks on the pilot | **G0′:** no full-text conflict with HA; E21 extension not negative |
| 1 | Add `branch.py`, `commitment.py`, `fragility.py`, `guidance_attr.py`, `analyze_noise.py`, `analyze_commitment.py`, latent saving in `generate.py`, onset flags in `intervene.py`, and their tests | **G1′:** all new tests green on CPU, including Δ = 0 identity and the restart moment test |
| 2 | Add: measure `t_branch`, latent-save overhead and (if feasible) fp32 fit in E03 | **G2′:** FULL vs LIMITED decided with the E23 cost included |
| 3 | Add: **E23 DEV rehearsal**, E25 on DEV, `t50_DEV`, DEV gate for HA CI width; tune E12 onset arms; choose Δ | **G3′:** Addendum B contains the numbers `t50_DEV`, `τ_pre`, `τ_post`, onsets; TEST sealed |
| 4 | Add: TEST latents saved during E04; E23 TEST branches; E12 onset arms | **G4′:** `verify_outputs.py` clean for branch files too |
| 5 | Confirmatory order: HA → HB → HC (each once); then E23b, E24–E28 (exploratory) | **G5′:** no analysis outside the addendum labelled confirmatory |
| 6 | Write-up uses §19.5 wording | Final checklist as before |

**Next ten actions (Revision 2, replace the list in §22 where they differ).**
1. Initialise git and commit the current tree.
2. Rebuild generated documents; fix `README.md` (e2e line and stray first line).
3. Add the missing PDFs (list in §22 item 3), plus the critical-windows paper and the ACL 2026 head-level paper if obtainable.
4. Extend E21: conditional-label AUC and a noise-only baseline on the pilot (no generation).
5. Implement `branch.py` + the moment test and the Δ = 0 identity test (CPU/GPU small).
6. Write `commitment.py` with the synthetic-recovery tests.
7. Write `benchmark_v2.py` with validators and unit tests.
8. Add conditional outcomes to `evaluate.py`; confirm the 40-image validation regression is unchanged.
9. Prepare the Kaggle probe notebook with E03 additions (`t_branch`, latent saving).
10. Draft Addendum B (numeric fields blank until DEV).

---

## 23. Verified references and official code repositories

*Verification level: FT = full text read from the provided PDF; ABS = abstract/listing page or search snippet only; CODE = repository page summarised. Checked 2026-10-14 unless stated.*

### 23.1 Papers
| # | Reference | Level | Notes |
|---|---|---|---|
| 1 | Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239 | FT | venue printed in PDF |
| 2 | Nichol, Dhariwal. *Improved DDPM.* arXiv:2102.09672 | FT | venue not printed in PDF |
| 3 | Song, Meng, Ermon. *DDIM.* ICLR 2021. arXiv:2010.02502 | FT | |
| 4 | Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. arXiv:2112.10752 | FT | CVPR from CVF listing |
| 5 | Ho, Salimans. *Classifier-Free Diffusion Guidance.* arXiv:2207.12598 (short version: NeurIPS 2021 workshop) | FT | |
| 6 | Karras et al. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022. arXiv:2206.00364 | FT | |
| 7 | Vaswani et al. *Attention Is All You Need.* NIPS 2017. arXiv:1706.03762 | FT | |
| 8 | Radford et al. *CLIP.* ICML 2021 (PMLR 139). arXiv:2103.00020 | FT + listing | |
| 9 | Hertz et al. *Prompt-to-Prompt.* ICLR 2023. arXiv:2208.01626 | FT + CODE | |
| 10 | Chefer et al. *Attend-and-Excite.* ACM TOG 42(4), 2023. arXiv:2301.13826 | FT + CODE | |
| 11 | Tang et al. *What the DAAM.* ACL 2023. arXiv:2210.04885 | FT + CODE | |
| 12 | Li, Keuper, Zhang, Khoreva. *Divide & Bind.* BMVC 2023. arXiv:2307.10864 | FT | |
| 13 | Fu, Cheng. *Enhancing semantic mapping in text-to-image diffusion via Gather-and-Bind.* Computers & Graphics 125 (2024) 104118 | FT | received 21 Oct 2023, accepted 28 Oct 2024 |
| 14 | Rassin et al. *Linguistic Binding in Diffusion Models (SynGen).* NeurIPS 2023. arXiv:2306.08877 | ABS | |
| 15 | Hu et al. *Token Merging for Training-Free Semantic Binding (ToMe).* NeurIPS 2024. arXiv:2411.07132 | ABS | |
| 16 | Wu et al. *Towards Better T2I Alignment via Attention Modulation.* arXiv:2404.13899 | ABS | |
| 17 | Xie et al. *BoxDiff.* arXiv:2307.10816 | FT abstract | venue not printed in PDF |
| 18 | Bar-Tal et al. *MultiDiffusion.* arXiv:2302.08113 | FT abstract | venue not printed in PDF |
| 19 | Chen et al. *Training-free Regional Prompting for Diffusion Transformers.* arXiv:2411.02395 | FT | |
| 20 | Liu, Xu, Zhang. *Unified Compositional Controller.* Information Sciences 745 (2026) 123380 | **FT** | received 11 Dec 2025, accepted 16 Mar 2026; FLUX.1-dev |
| 21 | Wang et al. *Anchoring and Steering Diffusion (AnchorSteer).* arXiv:2607.26647 (ICLR 2026 per PDF header) | FT | |
| 22 | Chen et al. *CoBind.* arXiv:2607.16307 (Jul 2026) | ABS | abstract only |
| 23 | Mun, Cho, Ok. *EPIC.* arXiv:2605.11722 (May 2026) | ABS | abstract only |
| 24 | Cui et al. *Diffusion Probe.* arXiv:2602.23783 (CVPR 2026 per arXiv page) | ABS | **full text unread** |
| 25 | Shen et al. *MetaLogic.* arXiv:2510.00796 (ICFEM 2025) | ABS | **full text unread** |
| 26 | Vemishetty, Arora, Sharma. *Towards Evaluating Robustness of Prompt Adherence in Text to Image Models.* arXiv:2507.08039 | FT | |
| 27 | Witteveen, Andrews. *Investigating Prompt Engineering in Diffusion Models.* arXiv:2211.15462 | FT | |
| 28 | Ghosh et al. *GenEval.* NeurIPS 2023 D&B. arXiv:2310.11513 | FT | |
| 29 | Kamath et al. *GenEval 2.* arXiv:2512.16853 | FT | |
| 30 | Huang et al. *T2I-CompBench.* NeurIPS 2023 D&B. arXiv:2307.06350 (++ extension) | FT | ++ details from listing |
| 31 | Hu et al. *TIFA.* arXiv:2303.11897 | FT | venue not printed in PDF |
| 32 | Cho et al. *DALL-Eval.* ICCV 2023. arXiv:2202.04053 | FT | |
| 33 | Bakr et al. *HRS-Bench.* arXiv:2304.05390 | FT | venue not printed in PDF |
| 34 | Cho et al. *Davidsonian Scene Graph (DSG).* ICLR 2024. arXiv:2310.18235 | ABS | |
| 35 | Lin et al. *GenAI-Bench / VQAScore.* ECCV 2024. arXiv:2406.13743 | ABS | |
| 36 | Bansal et al. *Universal Guidance for Diffusion Models.* ICLR 2024 | FT | |
| 37 | Kynkäänniemi et al. *Applying Guidance in a Limited Interval…* NeurIPS 2024. arXiv:2404.07724 | ABS | |
| 38 | Wang et al. *Analysis of CFG Weight Schedulers.* TMLR 2024. arXiv:2404.13040 | ABS | |
| 39 | Kim, Esmaeili, Qiu. *Text Embedding is Not All You Need.* arXiv:2411.15236 | ABS | |
| 40 | Feng et al. *Training-Free Structured Diffusion Guidance.* ICLR 2023. arXiv:2212.05032 | ABS | |
| 41 | *Magnet.* NeurIPS 2024. arXiv:2409.19967 | ABS | |
| 42 | Xu et al. *ImageReward.* arXiv:2304.05977; Wu et al. *HPS v2.* arXiv:2306.09341 | ABS | |
| 43 | Saharia et al. *Imagen (DrawBench).* arXiv:2205.11487; Yu et al. *Parti (PartiPrompts).* arXiv:2206.10789 | ABS | |
| 44 | *Towards Understanding the Working Mechanism of T2I Diffusion.* arXiv:2405.15330; *SD as Task-Aware Feature Extractor.* arXiv:2507.07106; *R&B.* arXiv:2310.08872; *When Do Diffusion Models Learn Multiple Objects.* arXiv:2605.00273 | ABS | context only |

### 23.2 Official or reported code repositories
| Work | Repository | Verification |
|---|---|---|
| DAAM | `github.com/castorini/daam` | CODE (README and `daam/trace.py` summarised) |
| Attend-and-Excite | `github.com/yuval-alaluf/Attend-and-Excite` | CODE (README and `pipeline_attend_and_excite.py` summarised) |
| Prompt-to-Prompt | `github.com/google/prompt-to-prompt` | CODE (README and `ptp_utils.py` summarised) |
| GenEval | `github.com/djghosh13/geneval` | Search listing |
| GenEval 2 | `github.com/facebookresearch/GenEval2` | URL printed in the PDF; not opened |
| BoxDiff | `github.com/showlab/BoxDiff` | URL printed in the PDF; not opened |
| Divide & Bind | `github.com/SnowdenLee/Divide-and-Bind` (search snippet: Diffusers 0.21.4, AGPL-3.0); also reported under Bosch Research | Not opened; canonical source **UNVERIFIED** |
| SynGen | `github.com/RoyiRa/Syntax-Guided-Generation` | From the paper footnote per search; not opened |
| Gather-and-Bind | `github.com/huan085128/Gather-and-Bind` | URL printed in the paper; **HTTP 404 on 2026-10-14** |
| VQAScore / GenAI-Bench | `github.com/linzhiqiu/t2v_metrics` | Search listing |
| Regional Prompting (FLUX) | `github.com/antonioo-c/Regional-Prompting-FLUX` | Listed on the arXiv page |
| ToMe | `github.com/hutaihang/ToMe` | arXiv page says code "will be publicly available" |
| Universal Guidance | `github.com/arpitbansal297/Universal-Guided-Diffusion` | URL printed in the PDF |
| DALL-Eval, TIFA | `github.com/j-min/DallEval`, `tifa-benchmark.github.io` | URLs printed in the PDFs |
| Diffusion Probe, CoBind, EPIC, MetaLogic, UCC, AnchorSteer | none found | **UNVERIFIED** |

### 23.3 Provenance and limits of this document
- Repository files were read directly; numbers marked VERIFIED were checked against saved CSV/JSON/log files; numbers marked DERIVED were computed by read-only scripts (variance components, ICC, MDE, AUC half-widths, matrix totals) from those files.
- Literature: full texts were read for the 28 provided PDFs where noted; everything else is abstract/listing level and flagged.
- Not done: no model was run, no image generated, no source file edited; Kaggle was not accessed; the Diffusion Probe, MetaLogic, CoBind, Attention Modulation, SynGen and ToMe full texts were not available, so **§5's novelty verdicts are provisional**.
- This document does not replace the preregistration: the hypotheses become confirmatory only when Addendum B is frozen with hashes (§8.4).

### 23.1b Revision 2 additional references (discovery phase; all ABS unless stated)
| # | Reference | Level | Notes |
|---|---|---|---|
| 45 | Li, Chen. *Critical windows: non-asymptotic theory for feature emergence in diffusion models.* ICML 2024 | ABS | Theory with Gaussian-mixture bounds and preliminary Stable Diffusion experiments; closest to HA |
| 46 | Xu et al. *Restart Sampling for Improving Generative Processes.* NeurIPS 2023 | ABS | Add-noise-then-deterministic-denoise operation used here as a measurement |
| 47 | Meng et al. *SDEdit.* ICLR 2022 | ABS | Partial noising and re-denoising |
| 48 | *Understanding the Impact of Negative Prompts: When and How Do They Take Effect?* arXiv:2406.02965 | ABS | Delayed effect, "reverse activation"; closest to E28 |
| 49 | *Noise refinement for compositional generation* arXiv:2503.06506 | ABS | Initial-noise route; baseline for E25 |
| 50 | Guo et al. *InitNO.* CVPR 2024 | ABS | Valid/invalid noise via attention scores |
| 51 | Ma et al. *Inference-time scaling for diffusion models beyond scaling denoising steps.* CVPR 2025 | ABS | Noise search; context |
| 52 | Zhang et al., work on reproducibility of diffusion outputs from the same noise (venue **UNVERIFIED**: recalled as NeurIPS 2023; title not confirmed) | ABS | CIFAR-scale same-noise reproducibility; context for E24 |
| 53 | *Layer-wise ablation for editing with cross-attention.* arXiv:2403.03431 | ABS | Closest to E26 |
| 54 | ACL 2026 Findings paper on head-level token-removal attribution (submission 1265) | snippet only | **Full text not retrievable; read before E26/E27** |
| 55 | Gokhale et al. *VISOR.* arXiv:2212.10015 | ABS | First-mentioned-object bias; conditional metric |
| 56 | *Make It Count / CountGen.* arXiv:2406.10210 | ABS | Counting; context for pilot F5 |
| 57 | *Understanding and Mitigating Compositional Issues in T2I.* arXiv:2406.07844; *Follow the Flow* arXiv:2504.01137; *Attention Regulation* arXiv:2403.06381 | ABS | Context for mechanism claims |
| 58 | Oriyad et al. TMLR 2025 | ABS | Attention overlap and missing entities |

*Entries 45–58 were checked on abstract or search-result level only on 2026-10-14; arXiv IDs are as printed on those pages. Author lists, exact titles and venues of entries 45–52 and 58 are given as recalled and **must be re-checked against the source pages before citation**. None replaces a full-text read.*

### 23.3b Revision 2 provenance
- §2.7 numbers come from read-only analyses of `outputs/main/results/per_image.csv`, `detections.jsonl`, `attention_features.csv`, `samples.jsonl`; they are EXPLORATORY.
- The restart formula in §8.0 is stated from the forward process definition and is verified by the unit test planned in §17.1; **it has not been run**.
- Costs in §16.5b are formulas and a reference time at the pilot hardware speed, not Kaggle measurements.
- The discovery phase used a limited number of web searches; "no prior interventional constraint-level commitment study found" means **not found in that search**, not that none exists.
- No experiment was run, no image generated, no existing result overwritten in this revision.
