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
| CAG quality guard | notebook 05 (DL-43): HPS v2 diagnostic + guard on existing images | PENDING until notebook 05 runs |
| DEV_CAG_GUARD images | regenerated with corrected constants (`gbc-dev-cag-v2`, manifest 80251995, verified) | done |

Corrected constants: `docs/evidence/dev_analysis_20261020/dev_constants_A0fix.json`, upload copy `kaggle_outputs/gbc-dev-constants-v2/dev_constants.json`; originals kept in the same evidence folder and in `kaggle_outputs/dev_analysis/`.

## Order of work
1. Done: A0 fix (99a0e5b), corrected constants (`gbc-dev-constants-v2`), DEV_CAG_GUARD regenerated (`gbc-dev-cag-v2`, verified).
2. Kaggle notebook 05 (`05_DEV_HPS_GUARD`): inputs gbc-source (new version), gbc-benchmark, gbc-r2a-core, gbc-dev-cag-v2; GPU T4, Internet ON -> `gbc-dev-hps` (hps_diagnostic.json, hps_scores.jsonl, hps_guard.json). Diagnostic failure -> report the error; nothing else changes.
3. Guard result -> w_lo by the predeclared rule (1 if pass, else 3; both fail -> owner).
4. Owner decisions for the freeze: DL-07, DL-11, DL-17, DL-18; confirm HA, HF and binding/count/spatial/size enter TEST as EXPLORATORY (failed DEV gates DL-40/41, audit DL-38); variant-review reviewer name (placeholder) fixed and checker passing; size B / pilot-40 still not labelled.
5. Freeze (`protocol_r3.py --draft --constants <corrected dev_constants with w_lo> --audit ...`, `--sign`, `--verify`), new gbc-source, FREEZE_CHECK.
6. TEST: 01 x3 and 03 x3, then 02 x3 (about 18-19 GPU-hours), then 04 PHASE=TEST.
