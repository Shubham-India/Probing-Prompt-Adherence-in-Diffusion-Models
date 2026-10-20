# Validation report (local, 2026-10-18, hardening pass)

Run on the owner's laptop (Windows 11, RTX 3050 6 GB, Python 3.13.5, torch 2.6.0+cu124, diffusers 0.41.0, transformers 5.19.0, accelerate 1.15.0). **No Kaggle job was run. No DEV or TEST experiment stage was run on a GPU. No TEST image was generated, scored or inspected. Nothing here is a research result.**

## 1. Kaggle import failure (`cannot import name 'FqnToConfig' from 'torchao.quantization'`)
| Check | Where | Result |
|---|---|---|
| Root cause in the installed code | `diffusers/quantizers/torchao/torchao_quantizer.py` line 54-55: `if is_torchao_available(): from torchao.quantization import FqnToConfig, quantize_`; `is_torchao_available()` tests presence only, although `dependency_versions_table.py` lists `torchao>=0.15.0`. transformers 5.19.0 checks a minimum torchao version itself | confirmed by reading the installed source |
| Reproduction | `--target` install of torchao 0.12.0 on `PYTHONPATH`; `from diffusers import DDIMScheduler, StableDiffusionPipeline` | **same error as on Kaggle** |
| Offline reproduction in the test suite | stub torchao 0.12.0 distribution without `FqnToConfig` | same error (`tests/test_env_bootstrap.py`) |
| Fix end to end | throwaway venv with torchao 0.12.0 -> `env_bootstrap.bootstrap()` -> torchao uninstalled -> fresh-process import | **OK**; `pip check` after: no blocking conflicts |
| Local `pip check` | project venv | "No broken requirements found." |
| Kaggle image (Python 3.13, torch 2.11.0+cu128, 2x T4) | - | **not verified**: needs a PREFLIGHT run (`env_bootstrap.json: ok = true`) |

## 2. Tests
`python -m pytest tests -q`: **247 passed**, 0 failed (2 numpy "mean of empty slice" warnings from synthetic edge cases). New in this pass: `test_env_bootstrap.py` (bootstrap failure paths, offline torchao reproduction), `test_fail_closed.py` (PREFLIGHT / GPU identity / FREEZE_CHECK / stage verdict / merge / archive validators, notebook source checks, and **end-to-end CPU runs of notebook 00**: an injected `all_identities_hold: false` with exit code 0 and a crashing preflight both stop the notebook and leave only `verified: false` archives; FREEZE_CHECK without a signed protocol fails; a passing fake PREFLIGHT is still never `verified`), `test_manifest_dependencies.py` (every base latent of the full manifests is produced upstream).

## 3. CPU dry run of the whole notebook chain (fake pipeline, fake judge)
`python scripts/dryrun_notebooks.py --sandbox local_audit/dryrun5`: 00 PREFLIGHT, 00 DEV_CORE, 00 DEV_COMMIT, 04 DEV, 00 DEV_CAG_GUARD, dry-run freeze, 01/02/03 shard 0, 04 TEST: all **OK**; every worker exited 0; all nine archives carry `verified: false, dry_run: true`; the analysis is labelled `DRY-RUN (fake pipeline/judge; not a result)`; a real-run verification refuses the dry-run protocol. The first dry run after the fail-closed change failed at DEV_COMMIT because the dry-run subset contained 2 switch jobs whose base latents the limited DEV_CORE run never produced; earlier dry runs had the same 2 errors but ignored the exit code. Fixed (DL-30); the full manifests have 0 unresolved bases.

## 4. GPU smoke test (earlier today, real SD 1.5 on the local RTX 3050; code paths unchanged by this pass)
CAG loop = pipeline at constant w = 7.5 and at w = 1 (40 UNet evaluations); attention recording leaves the image unchanged; 12.42 s per 40-step image; peak 2.86 GB.

## 5. Not validated locally
The bootstrap on the Kaggle image; the two-GPU worker path; T4 timings; judge / DINOv2 downloads; Kaggle session and quota limits; the human evaluator audit; the protocol freeze. See `KNOWN_LIMITATIONS.md`.
