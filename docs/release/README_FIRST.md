# README FIRST — Guide Before Commitment (Kaggle package)

**What this is.** Everything needed to run the Revision 3 programme ("When does a diffusion model decide what it draws, and what does classifier-free guidance change?") on Kaggle: code, small benchmark inputs, five notebooks and the scripts that verify, merge and analyse their outputs. **No experiment of the Revision 3 programme has been run yet.** Only the pilot (28 prompts) and the R0/R1/R2a infrastructure runs exist; see `EXPERIMENT_STATUS.md`.

## What is in the folder
| Item | Contents | Upload to Kaggle as |
|---|---|---|
| `source_package.zip` | `gbc_source/`: code (`src/`, `kaggle/`, `scripts/`, `tests/`), configs, benchmark files, protocol documents, `SOURCE_MANIFEST.json` (hashes, git commit, `dirty` flag) | private dataset `gbc-source` |
| `benchmark_upload.zip` | `gbc_benchmark/`: scene graphs, prompts (main + PS size family), DEV/TEST splits, datasheet, configs, `BENCHMARK_MANIFEST.json` | private dataset `gbc-benchmark` |
| `kaggle_notebooks/` | `00`-`04` as `.ipynb` (upload) and identical `.py` exports (read/diff) | Kaggle notebooks |
| `scripts/` | benchmark / protocol validation, output verification, shard merge, final analysis, report, archive builder, release builder, CPU dry-run | (local) |
| `manifests/` | dataset, scene-split, experiment (jobs and image-equivalents per stage and shard) and required-artifact manifests | (reference) |
| `checksums.sha256` | SHA-256 of every file in this folder | (verify after download) |

## Order
1. **Locally**: `LOCAL_SETUP.md` (tests, benchmark validation, dry-run).
2. **Before any GPU stage**: commit the code, rebuild this folder (`python scripts/build_release.py`) so `SOURCE_MANIFEST.json` shows `dirty: false`, and upload `gbc-source` + `gbc-benchmark`.
3. **Notebook 00** stage `PREFLIGHT` (hardware, timing, identities).
4. **Human work**: DEV evaluator audit labels (`docs/AUDIT_PROTOCOL_DEV.md`), variant review; the owner sets the stage gates in `configs/stage_gates.json`.
5. **Notebook 00** stages `DEV_CORE`, `DEV_COMMIT` -> **notebook 04** `PHASE="DEV"` -> `dev_constants.json` -> notebook 00 `DEV_CAG_GUARD` -> owner fixes the open decisions, drafts and **signs** the freeze locally -> rebuild + re-upload `gbc-source`.
6. **Notebooks 01 and 03** (shards 0, 1, 2; independent), then **02** (needs the 01 outputs) -> **notebook 04** `PHASE="TEST"`.
Full instructions: `KAGGLE_RUNBOOK.md`. Open owner decisions: `DECISION_LOG.md` (rows marked OWNER). Limits: `KNOWN_LIMITATIONS.md`.

## Do not
Do not run any TEST notebook before the freeze (they refuse anyway); do not upload images, latents or credentials to GitHub; do not use several Kaggle accounts to get more GPU time; do not cite `main_latex.pdf` numbers — they are illustrative placeholders.
