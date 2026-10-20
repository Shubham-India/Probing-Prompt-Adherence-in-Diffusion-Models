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
| CAG quality guard | HPS v2 (run 93dbad8, verified, DL-44): w_lo=1 -0.518 failed, w_lo=3 -0.259 passed | w_lo = 3 adopted |
| DEV_CAG_GUARD images | regenerated with corrected constants (`gbc-dev-cag-v2`, manifest 80251995, verified) | done |

Corrected constants: `docs/evidence/dev_analysis_20261020/dev_constants_A0fix.json`, upload copy `kaggle_outputs/gbc-dev-constants-v2/dev_constants.json`; originals kept in the same evidence folder and in `kaggle_outputs/dev_analysis/`.

## Order of work
1. Done: A0 fix, corrected constants, DEV_CAG_GUARD v2, HPS guard (w_lo = 3), audit verdicts, owner decisions DL-07/11/17/18, Addendum C, draft `protocol/protocol_freeze.json` (101 files, 0 problems).
2. Owner: `python src/protocol_r3.py --sign "<name>"`, then `--verify` (VERIFIED). Do not edit bound files (src, kaggle, scripts, configs, benchmark, ADDENDUM_C) between draft and sign.
3. Commit, `python scripts/build_release.py` (dirty=False), upload gbc-source, notebook 00 FREEZE_CHECK -> VERIFIED.
4. Before TEST: variant-review reviewer name (placeholder) -> `adopt_candidate_review.py --rename-reviewer`, checker exit 0, owner sets `variant_review_complete` (TEST gates also need evaluator_audit_complete, already set).
5. TEST: 01 x3 and 03 x3, then 02 x3 (about 18-19 GPU-hours), then 04 PHASE=TEST.
