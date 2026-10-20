# R2a local audit report

Audited: run `r2a-20261014T130457Z-daef6c4` (Kaggle, 2x Tesla T4, torch 2.11.0+cu128), restored to `local_audit/R2a_checkpoint/` (git-excluded).
Audit date: 2026-10-14. Repository: branch `claude/admiring-cori-pbn9wk`, HEAD `daef6c4`, clean worktree at audit start.
No existing result or report was overwritten. Evidence level: **VERIFIED** = executed in this audit; **REPORTED** = stated by the owner or an earlier session, no artifact available here; **ASSUMPTION** = inferred.

## 1. Verdict

**R2a is valid and usable as the base for restart-branch runs (R3).** All hard checks pass: 400 unique manifest records match the 400 jobs derived independently from the config and prompt files; 400 readable 512x512 RGB PNGs and 400 latent files with matching SHA-256; no orphans, duplicates, temp files or blank images; every latent file contains all six checkpoints in the layout `run_branches.py` reads; the latents are tied to their seeds and to the intended step (direct regeneration, section 4). One caveat on interpretation, not on validity: the first-pass evaluator shows that only 40% of base images of binding-bearing scenes have all objects (section 6), which makes the planned DEV commitment gate unlikely to pass with the current base-seed setting (decision D-13).

## 2. What the run is (derived from code and configuration, not assumed)

| Item | Derivation | Value |
|---|---|---|
| Jobs | `jobs.enumerate_jobs(DEV CAN prompts, cfg)`: 100 prompts x `cfg_scales` [7.5] x `steps` [40] x `seeds` [0,1,2,3] | **400** |
| DEV CAN composition | `prompts_v2.json`: binding 30, multi 25, counting 15, spatial 15, presence 15; literal binding-bearing 70 | matches the quotas in the blueprint (DEV 100 / 70 binding-bearing) |
| Latent file format | `generate.py` `torch.save({"steps_completed": [...], "latents_fp16": {t: tensor}})`; **one `.pt` holds six checkpoints** | 400 files x 6 = 2,400 checkpoints; file size 199,973 bytes = 6 x (1x4x64x64 fp16) + container |
| Checkpoint meaning | callback after step index `i` stores key `i+1`: latent after `t` completed steps (input of step index `t`) | grid {6,12,18,24,30,36} |
| Shard rule | round-robin by position in the ordered job list (`k % 2`) | 200 per shard |
| Record fields | name, prompt, cfg, steps, seed, size, hashes, run id, revision, env, scene id, variant, split, shard | uniform; revision `f03de327dd89b501a01da37fc5240cf4fdba85a1` = config pin |

## 3. Checks and results (command: `src/audit_run.py`, section 8)

| Check | Result |
|---|---|
| Merged `samples.jsonl` equals the union of the two shard files; names unique; shards disjoint | pass (400 / 400 / 200+200) |
| Coverage equals the expected 400 names; no extra | pass |
| Each record on the shard `k % 2` predicts | pass (0 wrong) |
| Metadata equals prompt file and config (prompt text, category, scene, variant, split, cfg, steps, size, model, negative prompt, scheduler) | pass (0 mismatches) |
| Single run id; revision equals pinned config revision | pass |
| Environment uniform | Tesla T4, python 3.13.15, torch 2.11.0+cu128 for all 400 records (diffusers version is not recorded in the per-sample env: gap, see section 7) |
| Images: decode, 512x512 RGB, SHA-256 equals record | pass (0 failures) |
| Duplicate image bytes; blank images (pixel std < 8); orphans; missing; temp files | none (pixel std range 20.1 to 99.3) |
| Latents: SHA-256 equals record; `torch.load(weights_only=True)` succeeds; keys/steps/dtype/shape/finite | pass (fp16, shape (1,4,64,64), finite, six distinct checkpoints) |
| External `checksums.sha256` (803 entries: 400 images, 400 latents, 3 jsonl) | pass for all binary files; the 3 `.jsonl` entries match **only after CRLF to LF** (see 5.1) |
| Repo verifier `verify_outputs.py ... --latents` | exit 0, `records=400 unique=400 expected=400 errors=0` (re-run in this audit) |
| `provenance/experiment_v2.json` and `prompts_v2.json` equal the repo files at `daef6c4` | identical |
| Worker logs | no error / NaN / OOM / traceback lines (grep on `gpu0.log`, `gpu1.log`) |
| Timing | median 6.21 s (shard 0), 6.26 s (shard 1) per image including latent saving; consistent with `throughput.json` (plain 6.16 s, with latents 6.25 s) |

## 4. Are the latents the right latents? (the part that is not checkable from hashes)

1. **Link to the seed.** The initial noise for each seed was regenerated on the CPU with the same call as `initial_latents`. Correlation of the saved step-6 latent with its own seed's noise: mean 0.9977, minimum 0.9852 (n=400); with another seed's noise: mean 0.001, max |r| 0.0095. At step 12: matched mean 0.9866 (min 0.948), mismatched max |r| 0.011. Latents are attached to the right seed in every file.
2. **Step meaning (direct regeneration, local RTX 3050, 2 samples, diagnostic only).** I regenerated `sg_0001_CAN seed0` and `sg_0050_CAN seed1` with the local SD 1.5 cache (revision `None`, snapshot `451f4fe...`, not the pinned revision; different GPU and torch) and compared local step-`k` latents with the Kaggle latent at key `t`. Correlation is highest at `k = t` in all 8 comparisons (t = 6, 18, 30, 36 x 2 samples); at t = 6 and 18 the match is near-exact (r >= 0.999992 vs 0.9995 to 0.99993 at k = t+-1; max |difference| 0.01 to 0.06 vs 0.07 to 0.16). At t = 30 and 36 the margin shrinks to hardware drift (r 0.9979 to 0.99997 vs 0.9960 to 0.9992; max |difference| is not minimal at k = t for t = 36 in sample 1). Conclusion: key `t` = "after t completed steps". Resuming the loop from the local step-12 latent reproduced the full image exactly (max pixel difference 0.0).
3. **A check that did not work.** A fit of the latent variance against the scheduled noise level preferred an offset of -1 step (rss 1.6e-5 vs 1.5e-4 for offset 0). Given point 2 this is an artefact of the fit's assumption (sampled trajectories do not have the forward-process marginal variance because predicted x0 is smoother than data). It is kept in `audit_run.py` as an **observation only**, with that explanation, so nobody later mistakes it for an off-by-one error.
4. Not verified: bit-identity of the Kaggle latents with a Kaggle regeneration (R0 reports this; see 5.3) and with the pinned revision's weights on the local machine.

## 5. Findings

### 5.1 Local restore changed line endings (not a run defect)
The restored `samples*.jsonl`, `checksums.sha256` and provenance JSON files contain CRLF (git `core.autocrlf=true` on this machine). Their SHA-256 differs from `checksums.sha256` until CRLF is converted to LF; with conversion all three match. Consequence: never compare text-file hashes of restored artefacts byte-for-byte; `audit_run.py` handles this explicitly and reports it as a soft observation. Binary files are unaffected.

### 5.2 Bug found and fixed in freeze tooling (found by the full test suite, not by R2a)
`tests/test_freeze.py::test_freeze_verify_roundtrip_and_tamper_detection` failed on this Windows machine (97 passed / 1 failed). Cause: `freeze.collect()` keyed the manifest with `str(Path)` (backslashes on Windows) and `file_hash` hashed raw bytes (CRLF checkout differs from LF clone). A manifest written locally would not verify on the Linux Kaggle machine, or on a clone with other line endings. Fix: POSIX keys and CRLF-normalised hashing for text suffixes (`src/freeze.py`, 3 changed lines); new test `test_manifest_keys_are_posix_and_hashes_ignore_crlf`. Full suite after the fix: **99 passed**. Nothing was frozen, so no manifest is affected.

### 5.3 R0 and R1 evidence
Update 2026-10-14: R0 `r0-20261014T145606Z-daef6c4` artifacts were received and verified (`docs/R0_R1_AUDIT.md`): all six required identities hold on the T4 for the 3 tested cases. R1 throughput: two measurements exist (6.16/6.25 s and 7.76/7.59 s per image); R2a itself ran at 6.2 s. Earlier text in this report that called R0 unverified is superseded.

### 5.4 First-pass evaluator results on R2a (local RTX 3050, `evaluate.py --conditional`, EXPLORATORY; judge not yet audited)
Output in `local_audit/R2a_checkpoint/eval_view/` (the run folder itself was not modified; images are reached through a junction).

| Quantity (70 binding-bearing DEV scenes, 280 images) | Value |
|---|---|
| All required objects detected | 0.404 |
| Binding given presence (eligible n = 113) | 0.487 |
| Joint (present and bound) | 0.196 |
| Images flagged `uncertain` (all 400) | 27 |
| Seed-agreement chance level A0 | presence 0.693; bind given presence 0.72 (31 scenes with at least 2 eligible seeds); joint 0.86 |
| Scenes with >= 2 eligible seeds for bind given presence | **31 of 70** |
| Eligible base images among seeds 0 and 1 (the configured branch bases) | **62 of 140** |

By category, share of images with every constraint satisfied: binding 0.075, multi 0.010, counting 0.267, spatial 0.183, presence 0.40 (floor effects in binding/multi).

## 6. Consequence for the next stage (R3) — needs your decision (D-13)
`configs/experiment_v2.json` uses `seeds_branch_bases = [0, 1]`. Only 62 of 140 base images can enter the binding-given-presence branch analysis (about 31 scenes), and `dev_gate_sim.py` predicted a Delta-t50 half-width of about 7.8 steps (gate: 4) for about 28 eligible scenes at eligibility 0.45 (assumption-driven, 12 replicates, as documented in SESSION_STATE). All four seeds already have saved latents, so using all four bases would raise eligible base images to 113, **at twice the branch cost**. This is a protocol choice (bases, M, gate wording) that I did not make. A data-anchored simulation with the real estimator (D-13 in the decision register) shows that the 4-step gate is not realistically reachable with a 10% evaluator error rate, even with four bases; the revised R3 cost is 6.3-7.9 h (2 bases) or 12.4-15.7 h (4 bases), see `docs/R0_R1_AUDIT.md`.

## 7. Gaps and limitations of this audit
- No bit-level comparison with Kaggle regenerations (needs the Kaggle GPU).
- Per-sample environment lacks diffusers/transformers versions; only torch, GPU and python are recorded. Recommend adding them (code change deferred: it changes record schema; no protocol decision involved but touches `generate.py`).
- Regeneration diagnostic used a different revision snapshot and GPU, 2 samples only.
- Evaluator output is unaudited; 27 of 400 images are `uncertain`.
- I did not open `outputs/main` results again; pilot numbers are not part of this audit.

## 8. Reproduction
```
.venv/Scripts/python.exe src/verify_outputs.py --run <run>/dev_core --config configs/experiment_v2.json --prompts data/benchmark_v2/prompts_v2.json --split DEV --variants CAN --latents
.venv/Scripts/python.exe src/audit_run.py --run <run>/dev_core --config configs/experiment_v2.json --prompts data/benchmark_v2/prompts_v2.json --split DEV --variants CAN --checksums <run>/checksums.sha256 --out local_audit/R2a_checkpoint/audit_run.json
.venv/Scripts/python.exe local_audit/R2a_checkpoint/diag/offset_diag.py      # local GPU, 2 samples (diagnostic)
.venv/Scripts/python.exe src/evaluate.py --run local_audit/R2a_checkpoint/eval_view --prompts data/benchmark_v2/prompts_v2.json --conditional
.venv/Scripts/python.exe -m pytest tests -q                                  # 103 passed (99 + 4 new verify_branches tests)
```
`audit_run.py` exits 0 only if every hard check passes (it did). Results of the first three are stored locally (git-excluded) as `audit_run.json` and `diag/offset_diag.json`.
