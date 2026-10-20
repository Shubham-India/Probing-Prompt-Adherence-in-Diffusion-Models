# Kaggle runbook — Guide Before Commitment (Revision 3)

Every expensive step is a separate, deliberate notebook run. Each notebook: verifies its inputs (SHA-256 manifests), records the environment, asks the stage guard for permission and an independent job count, builds its shard manifest (hash must equal the guard's), generates resume-safely on all visible GPUs, verifies the outputs against the manifest, and packages them as multipart ZIPs with an internal manifest. A notebook that exits without error has **not** necessarily produced a valid experiment: only `verified: true` in `ARCHIVE_MANIFEST.json` (and `"ok": true` in `verify/merge_report.json`) counts.

**Facts used.** T4 throughput measured in R0/R1/R2a: 6.2-7.8 s per 40-step image per GPU, two concurrent workers without slowdown, 2.81 GB peak VRAM. Kaggle session length and weekly GPU quota are **not verified** (the official pages could not be read; secondary sources: 12 h per GPU session, about 30 GPU-hours per week). Read your quota in Kaggle -> Settings before planning. Use **one Kaggle account**; do not create or borrow accounts to obtain more GPU time.

## 0. Upload the inputs (once per code version)
1. Locally: commit, then `python scripts/build_release.py`. Check `manifests/source_manifest_summary.json`: `dirty` must be `false` for research runs.
2. Kaggle -> Datasets -> New Dataset -> upload `source_package.zip` -> name `gbc-source`, visibility **Private**. For a new code version use "New Version" on the same dataset.
3. Same for `benchmark_upload.zip` -> `gbc-benchmark` (private).
4. For DEV_COMMIT: the R2a core dataset (`python kaggle/make_core_dataset.py --core-run local_audit/R2a_checkpoint/r2a-20261014T130457Z-daef6c4/dev_core --out kaggle_core_dataset --run-id r2a-20261014T130457Z-daef6c4`) as private dataset `gbc-r2a-core` (about 270 MB).
Notebooks find their inputs by marker files, wherever Kaggle mounts them (`/kaggle/input/...`): `SOURCE_MANIFEST.json`, `BENCHMARK_MANIFEST.json`, `core_manifest.json`, `ARCHIVE_MANIFEST.json`, `dev_constants.json`. Attached ZIPs are extracted automatically.

## 1. Open a notebook
Kaggle -> Code -> New Notebook -> File -> Import Notebook -> choose e.g. `00_PREFLIGHT_DEV_FREEZE.ipynb`. Settings (right panel): **Accelerator: GPU T4 x2**, **Internet: On** (downloads SD 1.5 at the pinned revision and the judge models; no token needed; never paste tokens into cells). Add Input -> your datasets. Edit only the CONFIG cell. Run cells top to bottom (Run All is safe because the guard stops blocked stages, but run each STAGE as its own session). After replacing a notebook or the `gbc-source` version, start a **fresh session** (or Run > Restart) before running it.

**Dependency bootstrap (cell 2 of every notebook).** `kaggle/env_bootstrap.py` installs `requirements-kaggle.txt` (torch/torchvision are not touched; it fails if the install would replace Kaggle's torch), **uninstalls torchao** (diffusers 0.41.0 imports `torchao.quantization.FqnToConfig` whenever any torchao is present, so the older torchao on the image breaks `from diffusers import StableDiffusionPipeline`; this project never quantizes), then imports `DDIMScheduler`, `StableDiffusionPipeline` and the judge classes in a **fresh Python process**, and runs `pip check`. It writes `env_bootstrap.json` (copied into the run archive) and raises `BootstrapError` on any failure; nothing after it runs. If it says modules are already imported, restart the kernel. A `pip check` conflict that names a project dependency stops the run; other conflicts of the image are recorded (`pip_check_after.other`, with `preexisting`).

**Fail-closed rule.** Every notebook packages its outputs (so diagnostics survive) and then **raises** if anything required failed; the archive then carries `verified: false` and `verification_problems`. `verified: true` is written only when every check passed: for PREFLIGHT, both subprocesses exited 0 and `preflight.json` (`checks.all_identities_hold`, `checks.preflight_ok`, judge/DINOv2 timings, two-worker exit codes) and `gpu_identity.json` (`all_required_identities_hold`, non-empty results, every max-diff 0) are complete and exactly true; for FREEZE_CHECK, `protocol_r3.py --verify` exited 0 and printed `VERIFIED`; for worker stages, every GPU worker exited 0 and a fresh merge exited 0 with the whole manifest merged and nothing missing, unexpected, conflicting, corrupt, mismatched, fake or of the wrong revision; smoke runs (`LIMIT`) and dry runs are never verified. DEV/TEST stages and notebook 04 refuse a `dirty` source package.

## 2. Stage table
| # | Notebook / STAGE | Purpose | Inputs | Expected jobs (image-equivalents) | Acceptance | Stop condition | Output dataset (private) |
|---|---|---|---|---|---|---|---|
| 1 | 00 `PREFLIGHT` | dependency bootstrap, hardware, versions, timing, VRAM, hook/CAG identities, 2-worker throughput, judge + DINOv2 timing, pilot GPU identity | gbc-source, gbc-benchmark | ~20 images, discarded | notebook ends with `PREFLIGHT PASSED` and `verified=True`; `env_bootstrap.json: ok = true` | `BootstrapError`, `PREFLIGHT FAILED`, any identity false | `gbc-preflight` |
| 2 | 00 `DEV_CORE` | DEV variants + CF bases (latents) + DEV PS family | + gate `variant_review_complete` | 1,784 (1,784) | `verified: true` | guard blocked | `gbc-dev-core` |
| 3 | 00 `DEV_COMMIT` | DEV restarts D8/ER30/FULL from R2a; CFG 2/15 bases + rehearsal; DEV switches | + gbc-r2a-core, gbc-dev-core; gates `evaluator_audit_complete`, `d13_resolved_for_r3`, `r3_authorised` | 8,896 (6,930) | `verified: true` | guard blocked | `gbc-dev-commit` |
| 4 | 04 `PHASE="DEV"` | judge DEV + R2a, compute `dev_constants.json` and DEV gates | gbc-source, gbc-benchmark, gbc-r2a-core, gbc-dev-core, gbc-dev-commit | judging only | merge `ok`; `dev_constants.json` written | merge rejected | `gbc-dev-final` |
| 5 | 00 `DEV_CAG_GUARD` | CAG w_lo 1 vs 3 on DEV | + dev_constants.json (attach gbc-dev-final) | 400 (~230-400) | `verified: true` | guard blocked | `gbc-dev-cag` |
| 6 | **local** | owner reviews DEV gates, audit results, OWNER rows of DECISION_LOG; fills Addendum C section C9; `python src/protocol_r3.py --draft --constants dev_constants.json --audit audit_verdicts.json`, then `--sign "<name>"`, `--verify`; commit; rebuild + upload gbc-source | - | - | `--verify` prints VERIFIED | any problem | - |
| 7 | 00 `FREEZE_CHECK` | confirm the uploaded package carries the verified freeze | gbc-source | - | `VERIFIED` | NOT VERIFIED | - |
| 8 | 01 shards 0, 1, 2 | TEST core all variants + CFG x steps | gbc-source (frozen), gbc-benchmark | per shard ~3,050 (~2,550) | `verified: true` | guard blocked | `gbc-test-core-s0/1/2` |
| 9 | 03 shards 0, 1, 2 | CAG arms + PS family (core, branches, CF-S switches) | gbc-source (frozen), gbc-benchmark | per shard ~780 PS-family jobs + ~1,070 CAG jobs (cost depends on s*) | `verified: true` | guard blocked | `gbc-test-cagsize-s0/1/2` |
| 10 | 02 shards 0, 1, 2 | TEST restarts at CFG 2/7.5/15 + CF-C/CF-R switches | + gbc-test-core-s<i> (at least the same shard) | per shard ~3,520 (~2,330) | `verified: true` | guard blocked or upstream unverified | `gbc-test-commit-s0/1/2` |
| 11 | 04 `PHASE="TEST"` | full merge against frozen manifests, judge once, HA/HD/HE/HF + Holm, tables, export | gbc-source (frozen), gbc-benchmark, all 9 TEST datasets | judging only | merge `ok` for 3 stages; `confirmatory.json` | merge rejected / protocol not verified | `gbc-test-final` |
Exact per-shard counts and hashes: `manifests/experiment_manifest.json`. Independent check on any machine: `python src/stage_guard.py --stage TEST_COMMIT --shard 0/3` (prints jobs, image-equivalents, disk, hours).

**Time.** At 6.2-7.8 s per image-equivalent on 2 GPUs: DEV_COMMIT about 6-7.5 h; one TEST_CORE shard about 2.2-2.8 h; one TEST_COMMIT shard about 2.0-2.5 h. Sessions that may exceed the limit end safely: package a PARTIAL archive (cell 8 still runs; `verified: false`), save it as a dataset version, start a new session with it attached; the notebook resumes inside it.

## 3. Saving outputs (do this before the session ends)
`/kaggle/working` is not persistent. At the end of each run: notebook menu -> **Save Version** -> "Save & Run All (Commit)" is NOT needed; instead use the Output panel -> the folder `archive_<RUN_ID>` -> "Create Dataset" (private), or download the ZIP parts and `ARCHIVE_MANIFEST.json`. Downstream notebooks attach these datasets. Notebook 02 only needs the `*.latents.*.zip` and `*.meta.*.zip` parts of the matching 01 shard (plus `ARCHIVE_MANIFEST.json`).

## 4. Reassembling multipart archives
Each part is a normal ZIP; files are never split across parts. Put all parts and `ARCHIVE_MANIFEST.json` in one folder; notebooks do the rest. Locally: `python -c "import sys; sys.path.insert(0,'kaggle'); import nbtools; print(nbtools.extract_and_verify('ARCHIVE_DIR','DEST'))"` (prints the list of problems; empty = complete and verified).

## 5. What to bring back to the local repository
`preflight.json`, `gpu_identity.json`, every `ARCHIVE_MANIFEST.json`, every `verify/merge_report.json`, `dev_constants.json`, `confirmatory.json`, `eval_rows.jsonl`, `RESULTS_CATALOGUE.json` and the `report/` folder. Images/latents stay in Kaggle datasets (or a local folder excluded from Git).

## 6. Failure recovery
| Symptom | Action |
|---|---|
| `BLOCKED:` from the guard | read the listed gates/files; fix the prerequisite; never edit `stage_gates.json` inside Kaggle |
| worker errors (`errors_*.jsonl`) | re-run cell 6; failing jobs are retried; persistent failures are reported, not dropped |
| merge rejected | inspect `verify/merge_report.json` (missing / conflicts / corrupt / mismatch); never package a rejected shard as verified |
| session ended | attach the PARTIAL archive dataset and re-run the same notebook and shard |
| out of disk | lower `PART_GB` is not enough: delete `/kaggle/working/_inputs` after extraction or split the stage into more shards (requires a new manifest build; record it) |
| protocol not verified on TEST | the package was rebuilt or edited after signing: rebuild from the signed commit |
| `BootstrapError` in cell 2 | read `/kaggle/working/_bootstrap/env_bootstrap.json` (`problems`, `steps`); "already imported" -> restart the kernel; "replaced torch" -> a pin pulled a new torch, do not continue; import failure -> the last stderr line names the module |
| `PREFLIGHT FAILED` / `ValidationError` | the listed check failed; the archive is kept with `verified: false`; download `preflight.json`, `gpu_identity.json`, `preflight_process.log`, `gpu_identity_process.log` and investigate before any experiment |
| `source package is dirty` | commit locally, rebuild (`dirty: false`), upload a new `gbc-source` version |
