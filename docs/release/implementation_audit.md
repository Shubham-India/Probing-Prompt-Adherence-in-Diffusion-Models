# Implementation audit (Revision 3, 2026-10-18)

Baseline before any change: branch `claude/admiring-cori-pbn9wk`, HEAD `daef6c4`, uncommitted work from the 2026-10-14/17 sessions preserved; 121 tests passing (`local_audit/snapshots/tests_baseline_20261018.txt`); snapshot `local_audit/snapshots/source_snapshot_20261018_pre_r3.zip` (307 files) and key-file hashes `docs/evidence/baseline_hashes_20261018.json`. Pilot verified on disk: 1,008 images, 504 attention maps, 504 intervention images.
Status words: UNIT-TESTED (CPU tests with synthetic data or the tiny random pipeline), SMOKE-TESTED (ran on a real GPU with real SD 1.5), IMPLEMENTED BUT NOT RUN, NOT RUN. Nothing below is a research result.

| File | Existing implementation | Required Revision 3 behaviour | Gap found | Modification | Test | Status |
|---|---|---|---|---|---|---|
| `src/benchmark_v2.py` | 300 scene graphs, 1,410 variants, validators | size attribute, `larger_than`, CF-S, PS factorial family | no size anywhere; local `--validate` failed on CRLF hashes | additive PS family (separate files, 24 pairs x 8 subsets + CF-C/S/R); LF-normalised hashing | `test_r3_benchmark_manifests.py` (leakage, one-field CF, tamper detection, size words) | UNIT-TESTED; both validators pass |
| `configs/benchmark_v2.json` | quotas, lexicon parameters | PS family parameters | - | `ps_family` block | via benchmark tests | done |
| `configs/experiment_r3.json` | (new) | full R3 design incl. pinned evaluator revisions | - | created | manifests tests | done |
| `configs/analysis_plan_r3.json` | (new) | decision rules, multiplicity | - | created | analysis tests | done |
| `src/manifests.py` | (new) | executable job lists, scene sharding, counts | position-based sharding split scenes | scene-group sharding over the whole programme | closed-form counts, disjoint cover, base/branch co-location | UNIT-TESTED |
| `src/generate.py` | pilot/R2a generator | unchanged for pilot; R3 uses `r3_run.py` | - | none | existing tests | unchanged |
| `src/r3_run.py` | (new) | core/branch/switch/CAG jobs, atomic writes, hashes, resume, UNet counting, TEST guard | `run_branches.py` re-denoised every branch at `cfg_scales[0]` | per-job CFG; records after files; resume from any worker's verified records | `test_r3_worker_fake.py` (fake pipeline: all kinds, resume, corruption recovery, TEST refusal) | UNIT-TESTED; real GPU via preflight only |
| `src/pipeline_factory.py` | (new) | real SD 1.5 or fake tiny pipeline for dry-runs | - | created; fake outputs flagged and refused downstream | worker tests | UNIT-TESTED |
| `src/branch.py`, `src/run_branches.py` | restart maths, loop | reused | - | none | existing tests | unchanged |
| `src/switch.py` | (new) | prompt-switch probe + chance-corrected estimator | - | created | `test_r3_measures.py` | UNIT-TESTED |
| `src/cag.py` | (new) | schedules, s*, per-step guidance loop, w = 1 skips uncond | - | created | `test_cag.py` (constant schedule = pipeline; w = 1 = pipeline at 1 with 40 calls) | UNIT-TESTED + SMOKE-TESTED (RTX 3050, real SD 1.5: identities hold) |
| `src/attn_record.py` | (new; `attention.py` kept for the pilot) | per-step maps per constraint word at 16 and 32 | pilot averaged all layers at 32 | grouped recorder reusing the pilot processor | worker test; preflight hook identity | UNIT-TESTED + SMOKE-TESTED (hook-on image identical) |
| `src/attn_stability.py` | (new) | signals, velocity, half-path t_stab, concordance, HF rule | blueprint "top-10% mass" ambiguous | `mass_q = 0.9` recorded for owner confirmation | `test_r3_measures.py` | UNIT-TESTED |
| `src/diversity.py` | (new) | DINOv2 pairwise seed distance; LPIPS optional | - | created | metric test; DINOv2 load timed in the preflight | UNIT-TESTED + SMOKE-TESTED (load) |
| `src/evaluate.py` | J1 + D-04 conditional fields | size (area ratio >= 1.5), count/size conditional fields, pinned revisions | evaluator revisions unpinned | additive; pilot default unchanged | `test_pilot_regression.py` (1,008 pilot images re-score identically), size-rule test | UNIT-TESTED |
| `src/r3_evaluate.py` | (new) | judge merged records incl. CF outcomes for switches; embeddings | - | created; `--fake-judge` for dry-runs | dry-run | IMPLEMENTED; real judge only in preflight timing |
| `src/r3_merge.py` | (new; `merge_samples.py` kept) | reject missing / duplicate / corrupt / mismatching / fake outputs | - | created | dry-run, worker tests | UNIT-TESTED |
| `src/r3_analyze.py` | (new; `analyze_commitment.py` kept) | HA (+5 types), HD (per-w chance level, shared resample), HE (IUT), HF, Holm, DEV constants and gates | - | created | `test_r3_analyze.py` (synthetic worlds with known answers) | UNIT-TESTED |
| `src/r3_report.py` | (new) | tables/figures from result files | - | created | dry-run | UNIT-TESTED (dry-run) |
| `src/protocol_r3.py` + `docs/ADDENDUM_C.md` | Revision 2 `freeze.py` | Addendum C freeze binding code, data, judge, constants, audit, TEST manifest hashes | - | created; dry-run protocols rejected for real runs | dry-run harness | UNIT-TESTED; **not frozen** |
| `src/stage_guard.py`, `configs/stage_gates.json` | Rev. 2 stages | R3 stages; TEST requires the freeze | - | extended | `test_stage_guard.py` | UNIT-TESTED |
| `kaggle/nbtools.py`, `kaggle/r3_preflight.py`, `kaggle/nb0*.py` | Rev. 2 Kaggle scripts | notebooks 00-04, multipart archives, preflight measurements | - | created | `scripts/dryrun_notebooks.py` (full chain on CPU); preflight on the RTX 3050 | dry-run + SMOKE-TESTED (preflight) |
| `scripts/*.py` | (new) | validation, verification, merge, analysis, archive wrappers; release builder | - | created | builder self-validation | done |
| `src/build_draft.py`, `paper/` | results-to-paper chain (Rev. 2) | must not show illustrative values as results | `main_latex.pdf` does (DL-02) | `paper/` kept data-driven; R3 sections pending | `test_build_draft.py` | done (R3 results PENDING) |

Not implemented: HPS v2 quality guard (DL-15), secondary VQA judge (DL-16), interval-guidance/CADS baselines (optional P2), Revision 2 HC attention-edit arms in the R3 package (P2), E24/E26-E28 (P2/P3).

## Hardening pass (2026-10-18, after the Kaggle PREFLIGHT import failure)
| Area | Defect found | Fix | Test |
|---|---|---|---|
| Dependencies | diffusers 0.41.0 imports `torchao.quantization.FqnToConfig` whenever any torchao is installed; Kaggle's older torchao breaks `StableDiffusionPipeline` | `kaggle/env_bootstrap.py` in all five notebooks (DL-27) | `tests/test_env_bootstrap.py` (offline stub torchao reproduces the error) |
| PREFLIGHT | `VERIFY = {"ok": True}` default; subprocess exit codes and `gpu_identity.json` never checked | `validate_preflight`, `validate_gpu_identity`; `preflight_ok` in `r3_preflight.py` | `tests/test_fail_closed.py` incl. end-to-end notebook 00 runs |
| FREEZE_CHECK | stdout stored, exit code ignored | `validate_freeze_check` | same |
| Workers 00/01/02/03 | worker and merge exit codes ignored; stale `merge_report.json` from a resumed archive could be read; LIMIT runs not excluded | `stage_verdict`, fresh merge folder, `archive_is_verified` | same |
| Notebook 04 | `verified=True` hard-coded; missing stages skipped; DEV ran without the R2a core | required stages, explicit failures, computed verified flag | static notebook tests + dry run |
| Merge | records lacking a required latent/attention file accepted; malformed records crashed | required files per manifest job; malformed -> corrupt | `test_merge_rejects_a_record_missing_a_required_file` |
| Archives | no self-check of parts; part checksums unused on extraction | `verify_parts`, checksum check in `extract_and_verify` | `test_tampered_or_missing_parts_are_detected` |
| Release | untracked files packaged; `dirty` not scoped to the package; no notebook/package consistency check | tracked files only, scoped `dirty`, consistency checks (DL-29) | builder validation |
| Dry run | subset padded with unresolvable jobs (2 worker errors, hidden by the old fail-open flow) | resolvable jobs only (DL-30) | `tests/test_manifest_dependencies.py` |
