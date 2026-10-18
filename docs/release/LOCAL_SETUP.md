# Local setup (Windows / PowerShell)

Tested on Windows 11, Python 3.13, RTX 3050 6 GB, torch 2.6.0+cu124, diffusers 0.41.0, transformers 5.19.0 (2026-10-18).

## 1. Environment
```powershell
cd D:\prompt_Project
py -3.13 -m venv .venv                      # skip if .venv already exists
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
`requirements-local.txt` in the release is a copy of `requirements.txt` (exact pilot pins).

## 2. Tests (CPU; the GPU attention test also runs when CUDA is present)
```powershell
python -m pytest tests -q
```
Expected on 2026-10-18: all tests pass (count in `EXPERIMENT_STATUS.md`). Tests that need the local CLIP tokenizer cache (`models/`) skip themselves when it is absent.

## 3. Benchmark
```powershell
python src/benchmark_v2.py --validate        # main benchmark: regenerates identically, no leakage, hashes current
python src/benchmark_v2.py --validate-ps     # PS size family
python src/manifests.py --summary            # all jobs and image-equivalents per stage (no GPU)
```
Rebuild only when the design changes (never after the freeze): `python src/benchmark_v2.py --build`, `python src/benchmark_v2.py --build-ps`.

## 4. CPU dry-run of the whole notebook chain (fake pipeline, fake judge; about 20-40 min)
```powershell
python scripts/dryrun_notebooks.py           # sandbox: local_audit/dryrun (git-excluded)
```
It must end with `DRY RUN COMPLETE` and `real-run verification correctly refuses the dry-run protocol`.

## 5. Optional local GPU smoke test (real SD 1.5; downloads only small files if the model cache exists)
```powershell
python kaggle/r3_preflight.py --out local_audit/dbg/preflight_local.json --n 2
```
`all_identities_hold` must be `true`.

## 6. Build the Kaggle package
```powershell
git status                                   # commit first; the package records dirty=true otherwise
python scripts/build_release.py              # -> release/guide_before_commitment/ (self-validates)
```

## 7. After a Kaggle run
Download the archive folder, then:
```powershell
python scripts/verify_outputs.py --manifest <archive>\manifest.jsonl --inputs <extracted folder> --out <tmp>\verify
python src/build_draft.py ; python paper/check_paper.py
```
