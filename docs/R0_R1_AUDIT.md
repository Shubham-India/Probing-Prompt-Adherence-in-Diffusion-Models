# R0 / R1 evidence audit (2026-10-14)

Source: `local_audit/R0_R1_reports.zip` (git-excluded; extracted to `local_audit/R0_R1/`). ZIP member timestamps are 14:56-15:02 UTC on 2026-10-14. This audit covers infrastructure only; it does not support or refute any scientific hypothesis.

## Verdict
**R0 (`r0-20261014T145606Z-daef6c4`) is verified for the scope it tested.** The reported pass was recomputed from the per-case numbers, not copied.

| Check | Evidence | Result |
|---|---|---|
| Run id | `setup_report.json.run_id` | `r0-20261014T145606Z-daef6c4` |
| Commit | `setup_report.json.commit_requested` = `commit_in_dataset` = `env.json.commit` = `daef6c4740884276fc068109bbb48207f7034edd`; `git rev-parse HEAD` in the repo | identical |
| Requirements | `setup_report.requirements_sha256` `0ba05b00f049df3e...` vs `requirements-kaggle.txt`: raw file hash differs (`a88ebf6a...`, CRLF working copy), LF-normalised hash equals the reported one | match after line-ending normalisation |
| Environment | `env.json`: 2x Tesla T4 (15,360 MiB), python 3.13.15, torch 2.11.0+cu128, diffusers 0.41.0, transformers 5.19.0, accelerate 1.15.0, scikit-learn 1.9.1, CUDA 12.8, 20.9 GB free disk | pins in `requirements-kaggle.txt` (diffusers, transformers, accelerate, scikit-learn) equal the installed versions; torch/numpy/pandas are unpinned by design |
| Model revision | `configs/experiment.json` at `daef6c4` pins `f03de327dd89b501a01da37fc5240cf4fdba85a1` and `tests/gpu_identity.py` loads that config by default | **not recorded in any of the four files**; the pin is inferred from the code at that commit, not observed in the artifacts |
| Identity checks (`gpu_identity.json`) | 3 cases (`presence_dog_bicycle`, `binding_chair_vase_v2`, `spatial_cup_book_lr`, seed 0, CFG 7.5, 40 steps, T4): `determinism`, `loop_from_step0`, `reload_resume`, `d0_branch`, `hook_off_control`, `restored_baseline` max differences all 0; `saved_latent_recompute_equal` true; `all_required_identities_hold` recomputed as true | pass |
| Not tested | `vs_pilot_image_maxdiff` is null (pilot images were not on Kaggle): cross-hardware identity with the laptop pilot is **unknown**; only 3 prompts, 1 seed, 1 setting; GPU 0 only | open |
| Single-image probe (`env.json`) | model load 22.0 s, one 40-step image 6.16 s, peak VRAM 2.81 GB | consistent with R2a |

Consequence: the restart branch loop, D=0 identity, latent reload and the restored attention processors behave exactly on this hardware for the tested cases. This validates machinery, not the persistence hypotheses.

## R1 throughput
`throughput.json` carries no run id or timestamp; its ZIP timestamp (15:02 UTC) places it in the R0 session. A different copy (`local_audit/R2a_checkpoint/provenance/throughput.json`) is from the earlier 10:10 session. The two differ.

| Measurement (T4, 40 steps, 512x512, per image) | 10:10 session | R0 session (this ZIP) | Real R2a run (400 images) |
|---|---|---|---|
| Plain | 6.16 s | 7.76 s | - |
| With latent saving | 6.25 s | 7.59 s | median 6.21 s (worker 0), 6.26 s (worker 1) |
| Two workers, wall for 6 images each | 53.9 s | 60.5 s (vs 60.3 s for one worker doing 6 incl. load) | 1,243 s and 1,245 s for 200 images per worker, run concurrently |

Reading: latent saving costs about 0-1.5%; two concurrent workers show no slowdown per worker (about 2x throughput); sessions differ by up to about 25%, so any estimate must carry a 6.2-7.8 s range. (Per-image times exclude the first image of each probe; the probe wall time includes model loading.)

## Revised R3 estimate (replaces the earlier 4.9 h figure)
Computed from `run_branches.enumerate_branch_jobs` and `plan_for` on the real R2a records (D8, ER30, FULL; M = 2; 100 DEV scenes):

| Design | Jobs | Image-equivalents | Wall, 2 workers, 6.2-7.8 s | With 25% allowance + 0.1 h loading |
|---|---|---|---|---|
| 2 base seeds (configured) | 7,200 | 5,730 | 4.9-6.2 h | 6.3-7.9 h |
| 4 base seeds | 14,400 | 11,460 | 9.9-12.5 h | 12.4-15.7 h |

A branch image costs the remaining fraction `(40 - target_idx) / 40` of a full generation plus decode; a fixed 0.5 s per-image overhead changes the totals by about 2%. Kaggle session length and weekly quota are not measured, so whether 4 bases fit is unknown. Branch evaluation (J1) is extra: about 0.3 s/image locally (400 images in about 2 min), roughly 36 min for 7,200.

## Remaining caveats
- Throughput variation between sessions is unexplained (shared hardware, clocks): treat estimates as ranges.
- The earlier R0 (`r0-20261014T100526Z`) and R1 (`r1-20261014T101024Z`) runs have no artifacts here except the older throughput copy.
