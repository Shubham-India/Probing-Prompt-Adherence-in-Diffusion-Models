# Experiment runbook (Kaggle and local)

Status: **procedure design.** None of the scripts named here with status PLANNED exist yet; no Kaggle run has been made. Each procedure is validated at its milestone (see MASTER_IMPLEMENTATION_BLUEPRINT section 16) before it is used for an expensive run.
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
3. Owner opens the Kaggle notebook, sets **Accelerator = GPU** (T4 x2 or whatever is offered; record the choice) and **Internet = On** (needed to clone the repository and to download model weights).
4. Owner runs the setup cell, then the experiment cell exactly as given in the experiment's "RUN CARD" (section 8).
5. Owner downloads `/kaggle/working/<run_id>.zip` (or the Kaggle "Output" tab) to `D:\prompt_Project\kaggle_returns\<run_id>\` and tells Claude the run ID.
6. Claude validates (section 5) and analyses.

### Path B (optional automation) - NOT AVAILABLE NOW
Preconditions to evaluate before building it: `kaggle` CLI installed and authenticated (not present); a GitHub token usable for push (not verified); Kaggle notebook push/run via API (not tested). Test with a 1-minute dummy notebook that writes a manifest and returns it. Until that test passes end-to-end, Path A is the only path. Secrets must live in Kaggle Add-ons -> Secrets or the local credential store and are never written to logs or Git.

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
Note: the repository appeared readable without credentials (`git ls-remote` worked); if it is made private, cloning requires a token stored as a Kaggle Secret.

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

## 8. RUN CARDS

Each card is completed when the milestone is implemented; fields marked `TBD` must be filled before the card is issued to the owner.

| Run | Milestone | Where | Purpose | Command | Success criteria | Output |
|---|---|---|---|---|---|---|
| R0 hardware probe (E03a) | M9 | Kaggle | Hardware, VRAM, quota, load time, 1 image timing, hook identity | `python kaggle/env_probe.py` + `python src/generate.py ... --limit 20` | `env.json` complete; peak VRAM recorded; hook-on image equals hook-off image for the 10 sample items | `env.json`, `manifest.json` |
| R1 throughput (E03b) | M9 | Kaggle, 1 and 2 workers | `t_gen`, `t_branch`, latent-save overhead, scaling `s2`, J2 fit | TBD | Timings with variance; crash rate | `throughput.json` |
| R2 DEV core (E04-DEV) | M10 | Kaggle | DEV images with latents and attention | TBD | `verify_outputs` clean | `outputs/e04/<run_id>` |
| R3 DEV commitment rehearsal (E23-DEV) | M10 | Kaggle | `t50_DEV`, controls | TBD | `D=0` identity; full-resample control `C` below the threshold in the blueprint (0.15) | `curves.csv` |
| R4 TEST core, branches, onset arms | M11 | Kaggle | Frozen protocol | TBD | zero duplicates; manifests match | per experiment |

## 9. Local (RTX 3050 6 GB) checks that precede Kaggle

CPU unit tests; the GPU identity tests on 3-10 items (SD 1.5 fp16 fits in about 3.3 GB per the pilot); no bulk generation locally for V2.
