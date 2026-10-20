# Kaggle master runbook

Status 2026-10-17. Governs how every expensive stage is prepared, started, verified and returned. **Nothing in this file authorises a run.** A stage may start only when (a) its row in section 3 says the prerequisites hold, (b) `python src/stage_guard.py --stage <s>` exits 0, and (c) the owner has written an explicit go-ahead. All stage gates in `configs/stage_gates.json` are currently `false`; only the owner sets them, in a committed change, after the evidence exists.
Older detail (RUN CARDS R0-R4, measured numbers) stays in `docs/EXPERIMENT_RUNBOOK.md`; where the two disagree this file governs and the disagreement is listed in section 9.

## 1. Dependency diagram
```
local preflight (S0) --> code bundle (S1a) -----------------------------+
                         core dataset from R2a (S1b) --------------------+--> Kaggle stages
R0 identity [done] -> R1 throughput [done] -> R2a DEV core [done, audited]
                                                 |
   human DEV audit (H1) + pilot second review (H2) --> evaluator error rates --> D-13 re-sizing (local, CPU)
   variant review (H3) --> R2b DEV variants
   synthetic probe (E22, threshold approval) --> lexicon review
                                                 v
        R3 DEV restart branches --> verify --> J1 evaluation --> DEV analysis HA/HB/HC + gates --> Addendum B values
                                                 v
                    freeze: freeze.py --write / --verify exit 0 --> owner signs Addendum B
                                                 v
   R4 TEST core (+latents) -> R5 TEST branches/edits -> frozen evaluation -> confirmatory analysis HA -> HB -> HC -> report
```
Rules: each Kaggle run is a separate notebook session started by the owner; no "Run All"; no stage starts from an unverified predecessor; failed jobs never overwrite good outputs (new run id, new folder).

## 2. Common procedure for every Kaggle stage
1. **Local preflight (S0)** on the machine that has the repository:
   `python -m pytest tests -q` (110+ passed), `git status`, `git rev-parse HEAD`. The bundle is built from a **commit** (`git archive`): uncommitted work never reaches Kaggle. The owner commits (Claude does not).
2. **Bundle (S1a):** `python kaggle/make_bundle.py --commit <SHA> --out kaggle_bundle`, upload the folder as a **private** Kaggle dataset (a new dataset or a new version). Known earlier mount: `/kaggle/input/datasets/shubhamindia/probing-bundle-daef6c4/kaggle_bundle/`. This path belongs to the old commit `daef6c4`; **after a new upload verify the mount** (`ls /kaggle/input/...`) and read its `COMMIT` file.
3. **Core dataset (S1b, needed from R3 on):** `python kaggle/make_core_dataset.py --core-run local_audit/R2a_checkpoint/r2a-20261014T130457Z-daef6c4/dev_core --out kaggle_core_dataset --run-id r2a-20261014T130457Z-daef6c4`, then upload `kaggle_core_dataset` as a private dataset (about 270 MB: 400 images 185 MB, 400 latent files 80 MB, jsonl). Both folders are git-ignored.
4. **Notebook settings:** Accelerator GPU (the observed machines were 2x Tesla T4, 15,360 MiB each); Internet **On** (the SD 1.5 weights at the pinned revision are downloaded from the model hub; whether Kaggle Secrets are needed for the hub is unverified: none was required in R0-R2a); Add Inputs: the code bundle and, for R3, the core dataset. Do not paste tokens into cells; use Kaggle Secrets if a credential is ever needed.
5. **Cell 1, setup** (copies the repo from the bundle at the exact commit, installs the pins, creates the run folder, writes `setup_report.json`):
```
RUN="<stage>-$(date -u +%Y%m%dT%H%M%SZ)-<sha7>"          # unique run id, e.g. r3-20261101T090000Z-ab12cd3
BUNDLE=/kaggle/input/datasets/<user>/<bundle-name>/kaggle_bundle
!python $BUNDLE/repo/kaggle/setup.py --source dataset --dataset-dir $BUNDLE --commit <FULL_SHA> --run-id $RUN
```
   It asserts that `COMMIT` equals the requested SHA. Outputs: `/kaggle/working/repo`, `/kaggle/working/outputs/$RUN/setup_report.json`.
6. **Cell 2, environment record:** `!python /kaggle/working/repo/kaggle/env_probe.py --out /kaggle/working/outputs/$RUN/env.json` (no generation unless `--time-one` is passed).
7. **Cell 3, guard** (prints the independently computed job count, disk and time range; the next cell must not run if this exits non-zero):
   `!python /kaggle/working/repo/src/stage_guard.py --stage <s> ... --bundle-dir $BUNDLE --expect-commit <FULL_SHA>`
8. **Cell 4, the stage** (section 4). Two GPUs: one process per GPU with `CUDA_VISIBLE_DEVICES=0|1` and `--shard 0/2|1/2`, started from the same cell and waited for; never two processes writing one file.
9. **Cell 5, finish:** `!python /kaggle/working/repo/kaggle/finish_run.py --stage <r2a|r2b|r3> ...` verifies, writes `checksums.sha256` and `run_report.json`, and only on success writes `COMPLETE` and the ZIP. If it exits 1, stop and return `run_report.json` and the logs (nothing else is needed to diagnose).
10. **Return:** download `<RUN>.zip` immediately (`/kaggle/working` is not guaranteed to persist after the session; also optionally "Save output as dataset"). Bring back: the ZIP (or at least `run_report.json`, `checksums.sha256`, `setup_report.json`, `env.json`, all `samples*.jsonl`, worker logs) and the image/latent folders when the next stage consumes them.

**Resume:** all generators skip verified samples. After an interrupted session start a new session, run cells 1-3 with the **same commit**, a **new** run id for the log files but the **same output folder name** restored from the dataset where the previous partial output was saved; then run cell 4 again and cell 5. A partial folder without `COMPLETE` is never an input to another stage.

## 3. Stage table
Statuses: DONE (evidence in section 8), READY (can be issued after the owner's go-ahead and green guard), BLOCKED (named blockers).

| Stage | Purpose | Script / cell | Required inputs | Expected outputs | Acceptance checks | Stop condition | Status |
|---|---|---|---|---|---|---|---|
| S0 | Local preflight | `pytest`, `git rev-parse HEAD` | repo | test log, SHA | all tests pass; worktree understood | any failure | DONE (110 + new tests) |
| S1a | Code bundle | `kaggle/make_bundle.py` | committed SHA | `kaggle_bundle/{repo,COMMIT}` | `COMMIT` equals SHA | uncommitted required files | READY after owner commits the new scripts |
| S1b | Core dataset | `kaggle/make_core_dataset.py` | R2a `dev_core` | `kaggle_core_dataset/` + manifest | counts 400/400; LF text | count mismatch | READY |
| R0 | Hardware and numerical identity | `setup.py`, `env_probe.py --time-one`, `tests/gpu_identity.py` | bundle | `env.json`, `gpu_identity.json` | all six identities 0 | any difference | DONE (`r0-20261014T145606Z-daef6c4`) |
| R1 | Throughput | `kaggle/run_throughput.py` | bundle | `throughput.json` | parses; returncode 0 | - | DONE (two sessions, unreconciled) |
| R2a | DEV canonical core + latents | `generate.py ... --split DEV --variants CAN --save-latents-at 6,12,18,24,30,36` | bundle | 400 images, 400 latent files | audit passes | verifier errors | DONE and audited |
| H1 | DEV evaluator audit (human) | `local_audit/audit_dev/annotator_A/label.html` | owner | `labels_A.csv` (+B) | 540 scored items answered | - | BLOCKED on owner labels |
| H2 | Pilot second review (human) | `local_audit/audit_pilot40/...` | other reviewer | `labels_pilot40.csv` | 125 items | - | BLOCKED |
| H3 | Variant and lexicon review (human) | `data/benchmark_v2/prompts_v2.json` | owner | review sheet; < 20% rejected | rejection rate | >= 20% rejected | BLOCKED |
| E22 | Synthetic detector probe | `synth_probe.py --export-judge-run`, `evaluate.py`, `synth_probe.py --score` | detector models | per-type accuracy | floor 0.95 (**proposal, needs approval**) | below floor | BLOCKED on threshold approval; GPU/model job |
| R2b | DEV non-canonical variants (no latents) | section 4.2 | bundle | **1,480** images (370 prompts x 4 seeds) | `finish_run --stage r2b` | verifier errors | BLOCKED on H3; gate `variant_review_complete` |
| R3 | DEV restart branches | section 4.3 | bundle + core dataset | 7,200 branch images | `finish_run --stage r3` | guard non-zero | BLOCKED (H1, D-13 re-sizing, owner authorisation) |
| R3e | J1 evaluation of branches | `evaluate.py --conditional` | R3 output | `detections.jsonl`, `eval.jsonl` | row count equals images | - | after R3 (local GPU about 0.3 s/image, 36 min for 7,200; or Kaggle) |
| A-DEV | DEV analysis HA/HB/HC, gates, Addendum B values | `analyze_commitment.py`, `analyze_hc.py`, `dev_gate_observed.py` | R2a + R3e eval | `outputs/dev_ha/*` | no TEST rows read | gate failure reported, not relaxed | after R3e |
| FRZ | Freeze | `freeze.py --write`, `--verify` | signed Addendum B | `docs/freeze/*` | `--verify` exit 0 | any problem | BLOCKED |
| R4 | TEST core with latents | card written **after** FRZ | bundle at the frozen commit | - | - | freeze not verified | NOT ISSUED |
| R5 | TEST branches and edits | card written after R4 | R4 | - | - | - | NOT ISSUED |
| A-TEST | Frozen confirmatory analysis HA -> HB -> HC | frozen scripts | R4, R5 evaluated | tables | each run once | deviations logged | NOT ISSUED |

## 4. Run cards (copy-ready; all BLOCKED until their gates are true)

### 4.1 Counting jobs independently
`python src/stage_guard.py --stage r3 --core-run <dev_core>` (dry run, any machine, CPU) enumerates the jobs from the real core records and configuration. Verified against the closed form on 2026-10-17: R3 = 100 scenes x 2 base seeds x 6 grid steps x 3 variants x 2 noises = **7,200 jobs = 5,730 full-generation equivalents** (D8 1,600; ER30 1,730; FULL 2,400); four base seeds would double both.

### 4.2 R2b (non-canonical DEV variants)
```
GPU=0: CUDA_VISIBLE_DEVICES=0 python repo/src/generate.py --config repo/configs/experiment_v2.json --prompts repo/data/benchmark_v2/prompts_v2.json --split DEV --variants PAR,ORD,CF-C,CF-A,CF-R,CF-N,CF-O --out /kaggle/working/outputs/$RUN/dev_variants --run-id $RUN --shard 0/2
GPU=1: same with CUDA_VISIBLE_DEVICES=1 and --shard 1/2
finish: python repo/kaggle/finish_run.py --stage r2b --run-dir /kaggle/working/outputs/$RUN/dev_variants --config ... --prompts ... --run-id $RUN --split DEV --variants PAR,ORD,CF-C,CF-A,CF-R,CF-N,CF-O --zip-out /kaggle/working/$RUN.zip
```
Expected: 1,480 images, about 0.68 GB, 1.3-1.6 h wall with two workers at 6.2-7.8 s per image (no allowance). Canonical prompts are excluded: they are already in R2a.

### 4.3 R3 (DEV restart branches)
Inputs: the code bundle and `kaggle_core_dataset` (mounted e.g. at `/kaggle/input/datasets/<user>/r2a-core/kaggle_core_dataset`, verify). Guard first:
```
!python repo/src/stage_guard.py --stage r3 --core-run $CORE/dev_core --bundle-dir $BUNDLE --expect-commit <FULL_SHA>
```
It verifies the 400 core images/latents (hashes), the pinned revision and the gates, then prints jobs/disk/time (expected: 7,200 jobs; 3.3 GB; 4.9-6.2 h wall, 6.3-7.9 h with a 25% allowance). Stage (one cell, two processes, waited):
```
python repo/src/run_branches.py --core-run $CORE/dev_core --config repo/configs/experiment_v2.json --prompts repo/data/benchmark_v2/prompts_v2.json --out /kaggle/working/outputs/$RUN/dev_branches --variants D8 ER30 FULL --M 2 --split DEV --run-id $RUN --shard 0/2   # CUDA_VISIBLE_DEVICES=0
python repo/src/run_branches.py ... --shard 1/2                                                                                                   # CUDA_VISIBLE_DEVICES=1
python repo/kaggle/finish_run.py --stage r3 --run-dir /kaggle/working/outputs/$RUN/dev_branches --config repo/configs/experiment_v2.json --prompts repo/data/benchmark_v2/prompts_v2.json --run-id $RUN --core-run $CORE/dev_core --branch-variants D8 ER30 FULL --M 2 --split DEV --zip-out /kaggle/working/$RUN.zip
```
Progress: each worker prints `[i/N]` every 20 branches (N = 3,600). Success: `finish_run` exits 0, prints `"ok": true`, `records == expected == 7200`, `COMPLETE` exists. The 7.9 h upper bound approaches typical session limits (unverified: 9-12 h from secondary sources); if the session ends, resume per section 2. Bring back the ZIP (about 3.3 GB) or at minimum `run_report.json`, `checksums.sha256` and `samples_shard*.jsonl`; the images are needed for local J1 evaluation unless evaluation runs on Kaggle.
Failure: duplicate or missing record -> do not delete anything; return `run_report.json`; the fix is a re-run of cell 4 with the same folder (it only generates missing branches).

### 4.4 E22 synthetic probe (after threshold approval; inference only, 200 small images)
```
python src/synth_probe.py --out local_audit/synth_probe --n 200 --seed 0 --export-judge-run
python src/evaluate.py --run local_audit/synth_probe --prompts local_audit/synth_probe/prompts_synth.json --conditional
python src/synth_probe.py --out local_audit/synth_probe --score eval.jsonl
```
Expected output: per-type accuracy and confusion. This is a GPU/model job (about a minute on the local RTX 3050); it needs the owner's approval of the 0.95 floor first and is not run in the session that wrote this file.

## 5. Evidence to bring back, by stage
| After | Bring back | Consumed by |
|---|---|---|
| H1/H2 | `labels_A.csv`, `labels_B.csv`, `labels_pilot40.csv` (unedited) | `audit_dev.py score` |
| R3 | ZIP or `run_report.json` + `checksums.sha256` + `samples_shard*.jsonl` + images | `verify_branches.py`, `evaluate.py`, `analyze_commitment.py` |
| any stage that fails | `run_report.json`, `setup_report.json`, `env.json`, worker logs | diagnosis |

## 6. Resource facts used (measured, not assumed)
Seconds per 40-step image on one T4 with both workers busy: 6.2 s (R1 session and real R2a, 200 images per worker in 1,243 and 1,245 s) to 7.6-7.8 s (probe in the R0 session). Latent saving adds under 2%. Peak VRAM 2.81 GB. PNG 0.462 MB, latent file 0.200 MB. Free disk reported 20.9 GB. Kaggle weekly quota and session limit are **not measured**.

## 7. Credentials and privacy
The code repository and all datasets stay private. The bundle contains `src, tests, configs, data (without the pilot validation images), kaggle, docs, requirements`. No token is stored in files, outputs or logs; `setup.py --source github` (token from a Kaggle Secret) is the only credential path and is not used when the dataset path works.

## 8. Evidence for completed stages
R0 `r0-20261014T145606Z-daef6c4` (`docs/R0_R1_AUDIT.md`); R1 (two throughput files); R2a `r2a-20261014T130457Z-daef6c4` (`docs/R2A_AUDIT_REPORT.md`, `src/audit_run.py`, 22/22 hard checks). Infrastructure only.

## 9. Discrepancies found and how they were resolved
| Discrepancy | Resolution |
|---|---|
| EXPERIMENT_RUNBOOK R2b row: "470 prompts x 4 seeds = 1,880 images" | would regenerate the 400 canonical R2a images; correct plan is 370 prompts x 4 = **1,480** (`stage_guard.plan_r2b`, test) |
| Two `throughput.json` files without run ids differ (6.2 vs 7.8 s) | both kept; the estimate uses the range; not reconciled |
| Bundle mount path refers to commit `daef6c4` | must be re-verified after each upload; `stage_guard --bundle-dir --expect-commit` checks the `COMMIT` file |
| R0 pass claimed for the pinned revision | revision is not stored in the R0 artifacts; R2a records do carry `revision`, which equals the pin |
| New scripts exist only in the working tree | they reach Kaggle only after the owner commits and rebuilds the bundle |
