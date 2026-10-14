# Experiment runbook (Kaggle and local)

Status (2026-10-13): scripts for R0-R3 now exist (`kaggle/`, `src/generate.py`, `src/run_branches.py`, `tests/gpu_identity.py`) and are CPU-tested with fakes or tiny random models; **no Kaggle run has been made and no real SD 1.5 GPU test has been executed since the pilot.** Each procedure is validated at its milestone (see MASTER_IMPLEMENTATION_BLUEPRINT section 16) before it is used for an expensive run.
Verified facts about the environment are in `docs/REPOSITORY_AUDIT.md` section 2. Kaggle limits (GPU type, VRAM, quota, session length, internet and secrets behaviour) are **UNVERIFIED** until the hardware probe (E03) has run; every number below that depends on them is a placeholder.

## 1. Roles

| Actor | Does |
|---|---|
| Claude | Plans, writes and tests code, prepares notebook/commands, validates returned artefacts, analyses, updates `docs/SESSION_STATE.md` |
| Owner | Runs the Kaggle job, downloads outputs, places them in the agreed folder, pushes commits (Claude cannot authenticate) |

## 2. Execution paths

### Path A (default, manual and reliable)
1. Claude commits the code and states the **commit SHA** and the **notebook/script**.
2. Owner pushes the branch to GitHub (`git push origin <branch>`; Claude cannot).
3. Owner opens the Kaggle notebook, sets **Accelerator = GPU** (T4 x2 or whatever is offered; record the choice) and **Internet = On** (assumed necessary to clone the repository and download model weights; UNVERIFIED until R0).
4. Owner runs the setup cell, then the experiment cell exactly as given in the experiment's "RUN CARD" (section 8).
5. Owner downloads `/kaggle/working/<run_id>.zip` (or the Kaggle "Output" tab) to `D:\prompt_Project\kaggle_returns\<run_id>\` and tells Claude the run ID.
6. Claude validates (section 5) and analyses.

### Path B (optional automation) - NOT AVAILABLE NOW
Preconditions to evaluate before building it: `kaggle` CLI installed and authenticated (not present); a GitHub token usable for push (not verified); Kaggle notebook push/run via API (not tested). Test with a 1-minute dummy notebook that writes a manifest and returns it. Until that test passes end-to-end, Path A is the only path. Secrets are expected to live in Kaggle Add-ons -> Secrets (UNVERIFIED mechanism, from secondary sources) or the local credential store and are never written to logs or Git.

## 2b. Capability status table
| Capability | Status |
|---|---|
| Local git read/commit | VERIFIED |
| GitHub read with owner's cached credentials | VERIFIED (`ls-remote`) |
| Anonymous GitHub read | VERIFIED **not possible** (private) |
| GitHub push from the Claude cloud session | VERIFIED to branch `claude/admiring-cori-pbn9wk` only (push succeeded 2026-10-13); never to `main` |
| `gh` CLI, `kaggle` CLI, Kaggle credentials in the cloud session | VERIFIED **absent**; the cloud session also cannot reach huggingface.co or the PyTorch wheel index (proxy 403) |
| Kaggle GPU type/VRAM/quota/session length/internet/secrets behaviour | UNVERIFIED (planned R0 measures them) |
| Kaggle notebook can clone the repo and install pins | UNVERIFIED |
| Automatic return of outputs to GitHub | UNVERIFIED and not planned |

## 3. Standard Kaggle setup cell (to be committed as `kaggle/setup.py` in M9; shown as the plan)

```python
# Kaggle cell 1: environment record + checkout (internet must be On)
import subprocess, json, os
RUN_COMMIT = "<SHA given by Claude>"
subprocess.run(["git","clone","https://github.com/Shubham-India/Probing-Prompt-Adherence-in-Diffusion-Models.git","repo"],check=True)
subprocess.run(["git","-C","repo","checkout",RUN_COMMIT],check=True)
subprocess.run(["pip","install","-q","-r","repo/requirements-kaggle.txt"],check=True)   # file to be created in M9; pins to be decided after the probe
subprocess.run(["python","repo/kaggle/env_probe.py","--out","/kaggle/working/env.json"],check=True)  # PLANNED
```
`env_probe.py` records: `nvidia-smi`, device names, VRAM per device, CUDA/torch/diffusers/transformers versions, free disk, python version, commit SHA, timestamp. **No tokens are printed.**
Note: the repository is **not** anonymously readable (verified: `git -c credential.helper= ls-remote origin` fails). Cloning on Kaggle therefore needs a GitHub token with read access, kept as a Kaggle Secret and never printed; alternatively upload the commit as a private Kaggle dataset or make the repository public (owner's decision, DECISION_REGISTER D-11). Whether a Kaggle Secret can be read inside the notebook as assumed here is UNVERIFIED.

## 4. Run identity, outputs and integrity

| Rule | Detail |
|---|---|
| Run ID | `<experiment>-<UTC yyyymmddThhmmss>-<short sha>`, e.g. `e03-20261101T120000-ab12cd3` |
| Output tree | `/kaggle/working/outputs/<experiment>/<run_id>/{samples_gpu0.jsonl, samples_gpu1.jsonl, images/, latents/, attention/, config_snapshot.json, env.json, manifest.json, checksums.sha256, run.log}` |
| Config | `config_snapshot.json` = the resolved config; `manifest.json` stores its SHA-256 and the commit SHA |
| Per-sample record | experiment ID, run ID, scene/prompt ID, variant, seed, CFG, steps, resolution, model id **and revision**, scheduler config, intervention parameters, hook on/off, GPU name, `max_memory_allocated`, seconds, torch/diffusers versions, PNG SHA-256 |
| Atomic writes | write `name.png.tmp`, `os.replace` to `name.png`, then append the JSON line; a sample is **done** only if the line exists, the file exists and the hash matches |
| Workers | Two processes, `CUDA_VISIBLE_DEVICES=0|1`, shard by `hash(scene_id) % 2`, separate metadata files, never a shared output file |
| Resume | Re-running the same command skips verified samples, regenerates missing or corrupt ones |
| Verification | `src/verify_outputs.py` (PLANNED): unique names, file opens at the expected size, hash matches, seeds equal the manifest, shards disjoint and complete, no duplicates |
| Merge | `src/merge_samples.py` (PLANNED): refuses to merge if verification fails |

## 5. Validation of returned artefacts (Claude performs, in this order)

1. Run ID, commit SHA and config hash match the request.
2. `verify_outputs.py` clean: counts equal the manifest, zero duplicates, zero corrupt files, zero unexpected shards.
3. Environment equals the probed environment (GPU, library versions); differences are recorded.
4. Acceptance tests of the experiment (its RUN CARD) pass: for example identity checks (`D=0`, control arm vs baseline) on a sampled subset.
5. Failures/retries listed in `exclusions.csv` with reasons; no silent drops.
6. Only then analysis. A job that exits with status 0 is **not** by itself a valid experiment.

## 6. What goes where

| Artefact | Location | Reason |
|---|---|---|
| Code, configs, schemas, tests, docs | Git | small |
| Manifests, checksums, `env.json`, compact metrics (`results/*.csv/json`, `samples_*.jsonl`) | Git under `runs/<run_id>/` | small, needed for provenance |
| Images, latents, attention maps | Kaggle output / private Kaggle dataset; local copy in `kaggle_returns/` (git-ignored) | large; `.gitignore` must add `kaggle_returns/` and `runs/**/images` in M0 |
| Model weights | downloaded at runtime to the Kaggle working disk, never committed | large |
| TEST labels/results | local only until the Addendum B freeze commit exists | sealing |
Git LFS is not planned; revisit only if compact artefacts exceed GitHub limits.

## 7. Failure recovery

| Symptom | Action |
|---|---|
| Session ended / interrupted | Start a new session, run setup cell and the **same** experiment command with the same run ID; verification decides what is regenerated |
| Out of memory | Reduce batch to 1, drop the hook for that run only if identity was verified, record the change; do not change precision silently |
| Download of weights fails | Confirm Internet = On; retry; do not use a different revision |
| Duplicate/partial shard files | Run `verify_outputs.py`, quarantine the run folder, regenerate affected samples |
| Quota exhausted | Stop; switch to the LIMITED design only through a recorded decision (DECISION_REGISTER D-05) |
| Result looks wrong | Do not retune on TEST; compare with the DEV rehearsal; log a deviation in `SESSION_STATE.md` |

## 3b. Code transfer without making the repository public (Path A, recommended)
1. On your machine: `git fetch origin claude/admiring-cori-pbn9wk && git checkout claude/admiring-cori-pbn9wk` and note `SHA=$(git rev-parse HEAD)`.
2. `python kaggle/make_bundle.py --commit $SHA --out kaggle_bundle` creates `kaggle_bundle/{repo/, COMMIT}`.
3. Kaggle -> Datasets -> New Dataset -> upload the folder `kaggle_bundle`, visibility **Private**; name it e.g. `probing-bundle`.
4. In the notebook: Add Input -> that dataset; Settings: Accelerator GPU, **Internet On** (needed to download the SD 1.5 weights; if the weights are not downloadable, add them as a second private dataset and set `HF_HUB_OFFLINE=1`).
Alternative (Path A2): store a read-only fine-grained GitHub token as Kaggle Secret `GITHUB_READ_TOKEN` and use `--source github`; the token is never printed and is removed from `.git/config` by `kaggle/setup.py`. Whether Secrets work as assumed is UNVERIFIED.

## 8a. R0 smoke test on Kaggle and the permanent fix (2026-10-14)
**Reported by the owner (not re-measured here):** on 2x Tesla T4, bundle commit `89ab854`: model load 16.7 s, one 512x512 40-step image 5.39 s, peak VRAM 2.81 GB. The GPU identity test then passed for three cases with zero max differences on every required identity - but only with a **temporary diagnostic patch in the Kaggle working copy**, not with the committed code.
**Original failure:** `tests/gpu_identity.py` called `pipe.unet.set_default_attn_processor()` after `edit_probe.install_edit`, raising `ValueError: Cannot call set_default_attn_processor when attention processors are of type AttnProcessor2_0` (diffusers 0.41.0). Cause (read from the diffusers source and reproduced on a tiny real UNet in `tests/test_intervene_modes.py`): that method only proceeds if *every* processor is in diffusers' accepted list; after `install_edit` the UNet holds a mix that includes our `EditProcessor`, so the check fails, and the message prints the first processor (a self-attention `AttnProcessor2_0`), which hides the real cause. Even when it does not raise it would install the non-SDPA `AttnProcessor`, a different baseline from the pipeline default.
**Permanent fix (committed):** `install_edit` now stores the exact baseline processors on the state and `edit_probe.restore_processors(unet, state)` puts them back; `tests/gpu_identity.py` uses it and gained one extra required check, `restored_baseline_maxdiff` (an image generated after restoring must equal the baseline). The temporary diagnostic patch (explicit `AttnProcessor2_0` assignment) is **not** the committed implementation.
**Model revision pinned** in `configs/experiment.json` and `configs/experiment_v2.json`: `stable-diffusion-v1-5/stable-diffusion-v1-5` @ `f03de327dd89b501a01da37fc5240cf4fdba85a1`. The pilot images in `outputs/main` were generated with `revision: null` (the hub default at the time); the pin is not proof that the pilot used this revision.
**Status:** CPU checks of the fix pass (see SESSION_STATE). **R0 is NOT validated against the new commit** until `gpu_identity.py` is rerun on Kaggle from the new bundle and returns `all_required_identities_hold: true` including `restored_baseline_maxdiff`.

## 8. RUN CARDS

`<RUN>` = `<exp>-<UTC yyyymmddThhmmss>-<sha7>`. All paths are inside the checked-out repo (`/kaggle/working/repo`) unless absolute. Two GPUs: run each command twice with `CUDA_VISIBLE_DEVICES=0` / `1` and `--shard 0/2` / `--shard 1/2`. A card is **not** issued for expensive stages until the earlier card passed and its outputs were returned and validated.

| Run | Where | Commands (in order) | Success criteria (all must hold) | Returned files |
|---|---|---|---|---|
| **R0** hardware probe + identity | Kaggle, 1 GPU | `python /kaggle/input/probing-bundle/repo/kaggle/setup.py --source dataset --dataset-dir /kaggle/input/probing-bundle --commit <SHA> --run-id <RUN>` then `python repo/kaggle/env_probe.py --out /kaggle/working/outputs/<RUN>/env.json --time-one` then `python repo/tests/gpu_identity.py --out /kaggle/working/outputs/<RUN>/gpu_identity.json --n 3` | `env.json` has GPU name/VRAM/lib versions/`t_gen_one_s`/`peak_vram_gb`; `gpu_identity.json` has `all_required_identities_hold: true` (determinism, loop-from-step-0, reload-resume, D=0 branch, hook-off control all maxdiff 0). **A nonzero maxdiff is a finding to record and decide (D-07), not a pass.** | `env.json`, `gpu_identity.json`, `setup_report.json` |
| **R1** throughput | Kaggle, 1-2 GPUs | `python repo/kaggle/run_throughput.py --out /kaggle/working/outputs/<RUN>/throughput.json --n 6` | `throughput.json` parses; plain vs latent-saving per-image seconds; two-worker wall time if 2 GPUs. Decides FULL vs LIMITED (recorded decision) | `throughput.json` |
| **R2a** DEV core (CAN, with latents) | Kaggle | `python repo/src/generate.py --config repo/configs/experiment_v2.json --prompts repo/data/benchmark_v2/prompts_v2.json --split DEV --variants CAN --out /kaggle/working/outputs/<RUN>/dev_core --save-latents-at 6,12,18,24,30,36 --run-id <RUN> --shard i/2` (per GPU) | 100 scenes x 4 seeds = **400 images + 400 latent files**;  : first `python repo/src/merge_samples.py --run <dir>`, then `python repo/src/verify_outputs.py --run <dir> --config repo/configs/experiment_v2.json --prompts repo/data/benchmark_v2/prompts_v2.json --split DEV --variants CAN --latents` must exit 0 | `dev_core/` (images, latents, samples*.jsonl) |
| **R2b** DEV variants (no latents) | Kaggle | same as R2a without `--variants`/`--save-latents-at` | 470 prompts x 4 seeds = 1,880 images | `dev_all/` |
| **R3** DEV restart branches (E23/E23b rehearsal) | Kaggle | `python repo/src/run_branches.py --core-run <dev_core> --config repo/configs/experiment_v2.json --prompts repo/data/benchmark_v2/prompts_v2.json --out /kaggle/working/outputs/<RUN>/dev_branches --variants D8 ER30 FULL --M 2 --split DEV --run-id <RUN> [--shard i/2]` | images = scenes x 2 base seeds x 6 grid x 3 variants x 2 = **7,200 for 100 scenes**; `D0` identity (R0) already passed; FULL control `C <= 0.15` after evaluation | `dev_branches/` |
| Evaluate (any run) | Kaggle or local GPU/CPU | `python repo/src/evaluate.py --run <dir> --prompts repo/data/benchmark_v2/prompts_v2.json --conditional` | `eval.jsonl` with D-04 fields | `detections.jsonl`, `eval.jsonl` |
| Analyse (CPU, Claude) | local | `python src/analyze_commitment.py --core <dev_core eval> --branches <dev_branches eval> --out outputs/dev_ha` | no TEST rows; `ha.json` | - |
| **R4** TEST | Kaggle | **not issued** until `python src/freeze.py --verify` exits 0 and the owner approves Addendum B | zero duplicates, manifests match | - |

Image-count formulas (use `throughput.json` for time): R2a = DEV scenes x 4; R3 = scenes x 2 x 6 x variants x M; a branch at grid step t costs about (40 - target_idx)/40 of a full generation. No time or quota estimate is made here because none has been measured.
`verify_outputs.py` accepts the same `--split/--variants` filters as `generate.py`; use identical filters for both.

## 9. Local (RTX 3050 6 GB) checks that precede Kaggle

CPU unit tests; the GPU identity tests on 3-10 items (SD 1.5 fp16 fits in about 3.3 GB per the pilot); no bulk generation locally for V2.
