# DEV next steps (2026-10-20)

## Verified state
DEV_CORE 1,784/1,784, DEV_COMMIT 8,896/8,896, DEV analysis archive (run 89e309e, 11,080 judged images, verified), DEV_CAG_GUARD 400/400 (made with the old constants, see below). Nothing on TEST has been generated or seen; the protocol is not frozen.

## Findings from the DEV outputs
| Item | Result | Status |
|---|---|---|
| A0 bug (DL-40) | chance agreement pooled other variants/CFG levels; fixed; constants recomputed locally from the verified eval rows | fixed, tested |
| HA DEV precision gate (half-width <= 4) | 7.94 after the fix (10.25 before) | FAILED -> HA is EXPLORATORY (predeclared rule) |
| FULL-resample control (max abs C <= 0.15) | 0.568 after the fix | FAILED; gate uncalibrated at DEV size; unexplained +0.16 presence offset (DL-41) |
| HD presence gates (w = 2, 15) | 45 and 59 scenes | passed |
| Evaluator audit (A only) | only presence meets 0.85 (DL-38/39) | binding, count, spatial, size EXPLORATORY under J1 |
| CAG quality guard | HPS v2 not implementable/verifiable today (DL-42) | PENDING, owner decision |
| DEV_CAG_GUARD images | s* changed 13 -> 12 for 73/100 prompts after the fix | stale; regenerate (400 jobs) |

Corrected constants: `docs/evidence/dev_analysis_20261020/dev_constants_A0fix.json`, upload copy `kaggle_outputs/gbc-dev-constants-v2/dev_constants.json`; originals kept in the same evidence folder and in `kaggle_outputs/dev_analysis/`.

## Order of work
1. Local: commit, `python scripts/build_release.py` (dirty=False, VALIDATION PASSED).
2. Kaggle: new `gbc-source` version; new private dataset `gbc-dev-constants-v2` (the folder above).
3. Owner: decide DL-42 (guard metric) before looking at any guard image.
4. Kaggle notebook 00, `STAGE = "DEV_CAG_GUARD"`, inputs gbc-source (new), gbc-benchmark, gbc-dev-constants-v2 (NOT gbc-dev-final) -> `gbc-dev-cag-v2`.
5. Guard metric on the new images -> w_lo decision (owner).
6. Owner decisions for the freeze: DL-07, DL-11, DL-17, DL-18; whether HA/HF enter TEST as exploratory; size/binding status under J1.
7. Freeze (`protocol_r3.py --draft/--sign/--verify`), new gbc-source, FREEZE_CHECK, then TEST (about 18-19 GPU-hours: 01 x3, 03 x3, 02 x3), then 04 TEST.
