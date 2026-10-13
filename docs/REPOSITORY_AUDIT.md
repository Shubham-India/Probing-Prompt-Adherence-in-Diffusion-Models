# Repository audit

Audit date: 2026-10-14. Base commit: `9633d87` on `main` (clean worktree at audit start). Audit branch: `docs/master-implementation-plan`.
No source file, output or data file was modified. No image was generated. Evidence commands are listed in section 9.

Status vocabulary: **VERIFIED** (executed or read and confirmed in this audit), **PARTIAL**, **BROKEN**, **PLANNED** (only in the blueprint), **MISSING**, **UNVERIFIED** (could not be checked here).

## 1. Authoritative documents

| Item | Finding |
|---|---|
| Research blueprint | `GRAND_RESEARCH_BLUEPRINT.md` (tracked, ~261 KB, header says **Revision 2**). No file named `GRAND_RESEARCH_BLUEPRINT(1).md` exists in the repository; the Revision 2 text is in this file. Revision 1 text is kept inside it as superseded material (sections 1.6 and 8.6). |
| Pre-registration | `report/protocol.md` holds the pilot protocol and Addendum A (intervention). **Addendum B (HA, HB, HC) does not exist yet.** |
| Earlier planning | `report/audit.md` (pilot audit, written by the project owner), `data/data.md`, `outputs/outputs.md`, `outputs/intervention/intervention.md`, `report/demo_cheatsheet.md`. |
| Problem statement | `Research-a-thon2.0 PS.pdf` (tracked). |

## 2. Git, environment and capabilities

| Item | Finding |
|---|---|
| Git | Repository present, branch `main`, 3 commits, remote `origin` = `https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models.git`. `git ls-remote origin` succeeds with the owner's cached credentials, but **fails anonymously** (`git -c credential.helper= ls-remote`: "could not read Username", re-checked in the review), so the repository must be treated as **private**: a Kaggle clone needs a token. Push access is unverified (not attempted). |
| Tracked content | 734 objects counted by git; about 623 files under `outputs/` (including 504 attention `.npz` files and `results/*.csv`), the 40 validation images, 31 reference PDFs/zips under `Researcha-thon Project/`. Images under `outputs/*/images/` and `models/` are ignored by `.gitignore`. |
| Working machine | Windows 11, NVIDIA RTX 3050 Laptop GPU, 6144 MiB. `.venv` with torch 2.6.0+cu124 (CUDA available), diffusers 0.41.0, transformers 5.19.0 (matches `requirements.txt`). |
| GitHub CLI | `gh` not installed. |
| Kaggle CLI / credentials | `kaggle` not installed, no `KAGGLE_USERNAME`, no `~/.kaggle`. **Programmatic Kaggle execution is not available from this environment.** |
| Kaggle hardware | UNVERIFIED (never measured). |
| Model assets | `models/` (3.7 GB, git-ignored): SD 1.5, OWL-ViT base patch32, CLIP ViT-B/32. |

## 3. Component inventory

Dependencies common to all `src/` files: `torch`, `diffusers`, `transformers`, `numpy`, `pandas`, `pillow`; model snapshots in `models/`.

| Component | Purpose | Status | Inputs | Outputs | Tests | Next action |
|---|---|---|---|---|---|---|
| `configs/experiment.json` | Pilot grid: model id, 512 px, CFG {2, 7.5, 15}, steps {20, 40}, seeds 0-5; `revision: null` | VERIFIED (read) | - | - | none (no schema validation) | Add V2 config + schema validation (M1). Pin model revision |
| `data/prompts.json` | 28 pilot prompts with constraint lists | VERIFIED (697 lines) | - | - | indirect (build test) | Preserve as PILOT; never use as TEST |
| `src/build_benchmark.py` | Builds pilot prompts | VERIFIED (read size 98 lines) | - | `data/prompts.json` | `test_logic.py` partially | Keep; `src/benchmark_v2.py` is new |
| `src/generate.py` | Resumable grid generation (`load_pipeline`, `initial_latents` on CPU, `generate_image`, `done_names`) | VERIFIED (read in full) | config, prompts | `images/*.png`, `samples.jsonl` | none direct | Add shards, atomic writes, hashes, latent saving, GPU/library metadata (M4). Note: fp16 pipeline, DDIM, negative prompt "" |
| `src/attention.py` | Cross-attention recorder (`StoreCrossAttention`, `install`, `word_token_indices`, `generate_with_attention` via `callback_on_step_end`) | VERIFIED (read, GPU test recorded) | run dir | `.npz` maps `[steps,32,32]` per word | `test_attention_gpu.py` | Extend only after M8 identity checks |
| `src/intervene.py` | Global attention re-weighting (additive log-scale bias on constraint tokens), control scale 1 | VERIFIED (read) | prompts, scales | `outputs/intervention/` | covered by GPU test indirectly | Add window/onset, targeted, random-pair, knockout modes (M7) |
| `src/evaluate.py` | OWL-ViT detection + CLIP colour; `score()` returns one 0/1 per constraint | VERIFIED (read `score`) | images, constraints | `detections.jsonl`, `eval.jsonl` | `test_logic.py` | Add conditional fields (see finding F-3); keep existing fields unchanged |
| `src/analyze.py` | Cluster bootstrap, paired contrasts, GEE | VERIFIED (read size 210 lines) | `eval.jsonl` | `results/*.csv` | `test_logic.py` | Scene-level keys, per-call RNG seeding |
| `src/analyze_attention.py` | Attention features, AUC | VERIFIED | `.npz`, eval | `attention_*.csv` | none | Generalise (M6) |
| `src/analyze_intervention.py` | Intervention contrasts | VERIFIED | intervention eval | `intervention_*.csv/json` | none | Generalise arms (M7) |
| `src/validate_metrics.py`, `data/validation_labels.json`, `data/validation_subset.json` | Manual validation of the evaluator on 40 images, single label set | VERIFIED | labels | `outputs/validation_metrics.txt` | none | Independent second reviewer still outstanding |
| `src/figures.py`, `src/report_numbers.py`, `report/build_report.py`, `report/make_slides.py` | Figures, numbers, documents | VERIFIED (exist) / generated documents STALE | results | `report/*`, slides | assertions in build | Rebuild after label merge (M0 hygiene) |
| `tests/test_logic.py` | CPU tests (6) | **VERIFIED: 6 passed in 14.3 s, run during this audit** | - | - | - | Keep green |
| `tests/test_attention_gpu.py` | GPU test (1) | PARTIAL: recorded `7 passed in 22.87 s` in `outputs/test_results.txt` (earlier run); **not re-run in this audit** | GPU | - | - | Re-run on target hardware |
| `outputs/main/` | Pilot: 1008 images, detections, eval, 504 attention maps, figures, results | VERIFIED (counts: `samples.jsonl` 1008 lines, attention 504 files) | - | - | - | Frozen; never overwrite |
| `outputs/intervention/` | Intervention pilot: 504 images | VERIFIED (504 lines) | - | - | - | Frozen |
| `outputs/e2e*`, `smoke`, `demo`, `pilot_*` | Smoke and end-to-end artefacts | VERIFIED (exist) | - | - | - | Keep; mark as non-evidential |
| `report/protocol.md` | Pilot protocol + Addendum A | VERIFIED | - | - | - | Append Addendum B only at freeze (M10) |
| Benchmark V2 (`src/benchmark_v2.py`, `data/benchmark_v2/`) | Scene graphs, variants, splits | **MISSING** (PLANNED) | | | | M3 |
| Restart branching (`src/branch.py`), `src/commitment.py`, `src/fragility.py`, `src/guidance_attr.py`, `src/features.py`, `src/predict_failure.py`, `src/analyze_noise.py`, `src/analyze_commitment.py` | HA/HB support | **MISSING** (PLANNED) | | | | M1, M6, M8 |
| Shard/merge/verify tooling (`src/merge_samples.py`, `src/verify_outputs.py`) | Integrity | **MISSING** (PLANNED) | | | | M4 |
| Judge J2 (`src/judge_vqa.py`), audit tools (`src/audit_sample.py`, `src/validate_human.py`) | Evaluator validation | **MISSING** (PLANNED) | | | | M5 |
| Kaggle notebook / run scripts | Hardware probe, runs | **MISSING** | | | | M9 |

## 4. Verified pilot facts the plan relies on (read from saved files; re-derive before citing)

| Fact | Source | Status |
|---|---|---|
| 28 prompts x 3 CFG x 2 steps x 6 seeds = 1008 images | `outputs/main/samples.jsonl` (1008 lines) | VERIFIED |
| Intervention arm: 504 images; global re-weighting effect +0.02 [-0.04, +0.09], p 0.378 | key `intc_primary_all_rw2_vs_rw1` in `outputs/intervention/results/intervention_numbers.json` | VERIFIED (read in review) |
| Hook is image-neutral in the pilot (168/168 control images pixel-identical) | stated in `outputs/intervention/intervention.md` lines 83, 91 (owner-written); no raw comparison log found | Documented claim; **not re-run**; re-check in M8 |
| Spatial rule uses box-centre x only; unmatched object counts as failure | `src/evaluate.py` `score()` | VERIFIED (read) |

## 5. Findings that change the plan

| ID | Finding | Evidence | Consequence |
|---|---|---|---|
| F-1 | **`README.md` contains committed merge-conflict markers** (`<<<<<<< HEAD` on line 1, `=======` line 168, `>>>>>>>` line 171). | `git grep -n "^<<<<<<<"`; commit `9633d87` | Resolve in M0 before any other doc work. |
| F-2 | README line 158 says the end-to-end run is "Not completed", but `outputs/e2e.log` ends with `E2E-DONE` and contains 0 `Traceback` lines (checked in review). | `grep -c Traceback outputs/e2e.log` = 0; `tail` shows `E2E-DONE` | README row is wrong; correct it in M0. |
| F-3 | `score()` is **not conditional**: binding returns 0 when the object is missing; spatial returns 0 when either object is missing. No `objects_present` flag is stored. | `src/evaluate.py` lines 103-114 | Conditional outcomes need new fields; existing columns must stay unchanged. |
| F-4 | Generated documents are stale: `report/report.md`, `defense.md`, `speaker_notes.md`, `presentation.pptx` (2026-10-11) and `validation.md` (2026-10-10) predate `data/validation_labels.json` (2026-10-12). | `ls -l` timestamps | Rebuild with `build_report.py` / `make_slides.py` (M0). |
| F-5 | DDIM scheduler is created with `DDIMScheduler.from_config` of a PNDM config (`steps_offset=1`, default `timestep_spacing="leading"`, `set_alpha_to_one=False`). 40 steps give timesteps 976, 951, ..., 1. | executed in this audit (`docs/evidence/sched_check.py`, output in `docs/evidence/sched_check_output.txt`) | Restart indexing and clipping depend on this; see DECISION_REGISTER D-01, D-02. |
| F-6 | Diffusers' DDIM `step` computes `prev_timestep = timestep - num_train_timesteps // self.num_inference_steps` (`scheduling_ddim.py` line 445, read in review), so the stride depends on `num_inference_steps`, which a truncated custom `timesteps` list would change. | source read; behavioural test still to be run | Branch re-denoising needs a custom loop that calls `scheduler.step` on a slice of the 40-step schedule (D-07). |
| F-7 | Model revision is `null` in the config; the cached snapshot is `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. | `models/.../snapshots/` | Pin the revision in V2 configs. |
| F-8 | Blueprint counts are not mutually consistent (binding-bearing scenes vs category quotas, P-level of E24, rehearsal cost). | blueprint sections 9.0, 16.5b | See DECISION_REGISTER D-05. |
| F-9 | Large artefacts already tracked (504 `.npz`, PDFs, zips, `outputs/` tables); 741 tracked files existing on disk total **696 MB** (summed file sizes, review); git reports 630 MiB of loose objects. (An earlier figure of 91 MB in this file was wrong: it came from a truncated `du` batch.) | `git ls-files`, `git count-objects -vH` | Policy for Kaggle artefacts in RUNBOOK section 6; do not add bulk outputs to Git. |
| F-10 | Kaggle automation is not available (no CLI, no credentials). | section 2 | Path A is the baseline; Path B stays optional (RUNBOOK section 2). |

## 6. What can be re-analysed without generating images

Pilot re-analysis from `outputs/main` and `outputs/intervention`: conditional-label rates and attention AUC on both-present images (blueprint 2.7 F2, F7); noise-only baseline and variance components (seeds regenerate noise on CPU via `initial_latents`); variance/ICC/power recomputation; evaluator metrics on the 40 validation images. These are G0' tasks (M2).

## 7. What must be regenerated or cannot be stored in Git

| Item | Reason |
|---|---|
| All Benchmark V2 images (DEV/TEST), branch images, edit-arm images | New experiments |
| Saved latents at steps {6,...,36} | Not saved by the pilot; fp16 latents about 32 KB each |
| Images, latents, attention maps from Kaggle | Large; keep in Kaggle output/dataset storage, commit only manifests and compact results |
| SD 1.5, OWL-ViT, CLIP weights | Large; downloaded at runtime |

## 8. Gaps not checked

Push permission; Kaggle quota/GPU type; whether SDXL or a VLM judge fits 16 GB; GPU test re-run; independence of validation labels; literature full texts (papers listed as missing in blueprint section 22); correctness of every pilot number quoted in the blueprint.

## 9. Review corrections (2026-10-14, independent review)
Corrected: anonymous remote read (section 2), tracked size (F-9), e2e status (F-2), F-6 source evidence, intervention number and the origin of the 168/168 claim. All paths marked existing in section 3 were re-checked by script; paths marked PLANNED/MISSING indeed do not exist.

## 10. Evidence commands (all read-only)

```
git status; git branch -a; git log --oneline; git remote -v; git ls-remote origin
git ls-files; git grep -n "^<<<<<<<"; git count-objects -vH
.venv/Scripts/python.exe -m pytest tests/test_logic.py -q        -> 6 passed
.venv/Scripts/python.exe -c "import torch,diffusers,transformers; ..."  -> 2.6.0+cu124 True 0.41.0 5.19.0
nvidia-smi --query-gpu=name,memory.total
python docs/evidence/sched_check.py   (DDIM schedule, restart ratios, moment test; CPU; output saved in docs/evidence/sched_check_output.txt)
ls -l --time-style=long-iso report/... data/validation_labels.json ...
```
