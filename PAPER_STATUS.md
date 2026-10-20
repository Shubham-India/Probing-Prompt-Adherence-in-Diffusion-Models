# Paper status — Guide Before Commitment (R3)

Updated 2026-10-20. Manuscript: `paper/main.tex`. Overleaf package: `paper/paper_overleaf.zip` (60 files; rebuilt by `python paper/make_overleaf_zip.py`, not committed). **No PDF has been compiled**: there is no TeX engine on this machine. Static checks pass: 31 cited keys, 0 uncited; all macros defined; all references, labels and inputs resolve; braces, math and environments balanced in all 46 `.tex` files.

## Evidence used (hierarchy)
1. **Signed protocol.** `protocol/protocol_freeze.json` (source 40ade4d), VERIFIED after all edits. No protocol-bound file was changed (`src/`, `kaggle/`, `scripts/`, `configs/`, benchmark, `docs/ADDENDUM_C.md`).
2. **Partial TEST archive.** `kaggle_outputs/gbc-test-partial/r3-test-partial-20261020T134300Z-40ade4d.{meta,merged}.part01of01.zip`. The SHA-256 checksums match. This is the canonical source for every partial-TEST number.
   - The run id `r3-test-partial-20261020T133549Z-40ade4d` named in the request was **not found** anywhere on disk; only the 134300Z run exists.
3. **Local replication.** `local_audit/partial_run/work/final_r3-test-partial-20261020T095729Z-40ade4d`: the same frozen code, re-judged on an RTX 3050 with torch 2.6. All 16,974 de-duplicated judged rows are identical and all four rule outcomes are identical; bootstrap p-values differ by ≤ 0.002.
4. **DEV evidence.** `docs/evidence/` holds the DEV constants, audit verdicts and HPS guard results.

## Integrity checks on the partial archive
- **Coverage.** CORE shards 0–1 (6,172 jobs, 133 scenes), COMMIT 0–1 (7,080, 85), CAG_SIZE 1–2 (3,750, 222). Merged = manifest jobs for every present shard. Missing: CORE s2, COMMIT s2, CAG_SIZE s0. Nothing is imputed.
- **The 17,002 vs 14,592 discrepancy is not in the saved data.** `eval_rows.jsonl` has 17,002 rows. `dino_embeddings.npz` has 17,002 embeddings: unique ids, all finite, no zero vectors, 8,501 per GPU part. The local run also logged "judged rows: 17002 | embeddings: 17002". The 14,592 figure is not reproduced by any saved file; it may come from the 133549Z run, which isn't on disk.
- **De-duplication** removes 28 repeated generations, leaving 16,974 rows. The embeddings have 16,594 distinct vectors, because identical generations share an embedding.
- **Recorded warnings.**
  - `RuntimeWarning: Mean of empty slice` (r3_analyze.py:296): scenes with no eligible pair for a type; set to NaN by design.
  - `ConstantInputWarning` (attn_stability.py:142): degenerate HF bootstrap resamples, which are dropped.
  - Neither changes a point estimate. The size-type Spearman interval [−1, 1] is uninformative.

## Verified findings (all PARTIAL TEST, EXPLORATORY)
| Claim in paper | Value | Evidence file → field |
|---|---|---|
| HD presence Δt50 (w15 − w2) | −8.67 [−16.34, −4.72], Holm p = 0.046, rule "supported" | `partial_results.json` → `results.HD.estimands.presence`, `results.HD.p_holm` |
| t50 presence by w | 22.5 / 16.2 / 13.8 at w = 2 / 7.5 / 15 | `reports/partial/robustness.json` → `hd.by_error["0.0"].presence.t50_by_w` |
| Not a chance-correction artefact | mean A0: 0.729 at w=2 vs 0.697 at w=15; raw agreement at t=18: 0.79 vs 0.90 | `robustness.json` → `hd.A0_mean`, `hd.raw_agreement` |
| HD fragile to independent judge error | e = 0.05: −12.9 [−15.5, 0.3]; e = 0.10: interval reaches 0; e = 0.144 (audit): all curves left-censored | `robustness.json` → `hd.by_error` |
| HD heterogeneity | both shards negative; counting scenes +11.4 (10 scenes) | `robustness.json` → `hd_by_shard`, `hd_by_category` |
| DEV ordering does not replicate | binding ≤6 → 19.8, presence 9.3 → 16.2, count 18.9 → 15.3 (D8; ER30 same direction) | `dev_constants_final.json` → `ha_dev`; `partial_results.json` → `results.HA.variants` |
| HE not supported | A2 − A1 on F: −0.016, 90% CI [−0.035, 0.003] (margin −0.03); beats-reversed fails | `partial_results.json` → `results.HE` |
| CAG diversity is a dose effect | mean w 4.64 vs 7.5; A6 (constant at the same mean) +0.027; A2 − A6 −0.013 [−0.026, −0.001]; A2 − A4 −0.030 | `robustness.json` → `he_arms` |
| HF rejected | concordance 0.30 [0.20, 0.75]; within-type Spearman intervals all include 0 | `partial_results.json` → `results.HF` |
| CFG dose–response on adherence | presence 0.32 → 0.43; all-satisfied 0.09 → 0.17 (w 2 → 15; 102 common scenes) | `robustness.json` → `cfg_rates` |

Every number in the text is a generated macro:
- `tables/r3_numbers.tex` (DEV) comes from `paper/make_paper_assets.py`.
- `tables/partial_robust_numbers.tex` (partial TEST) comes from `partial/robustness.py`.
- The `tables/partial_*.tex` tables come from `partial/partial_report.py`.

## Unsupported or inconclusive
- **HA:** inconclusive and exploratory, because its DEV precision gate failed. The binding judge was not validated.
- **HE:** not supported. The registered contrast confounds timing with dose.
- **HF:** rejected.
- **All non-presence results** are exploratory. Only presence met the audit criterion (0.856), and there is a single annotator with no valid double annotation.

## Changes in this pass
- **New:** `partial/robustness.py` (exploratory analyses that reuse the frozen estimator functions) and `sections/results_partial_robustness.tex`.
- **New tables:** `partial_t50_by_w`, `partial_robust_hd`, `partial_dev_vs_test`, `partial_he_arms`, `partial_robust_numbers`.
- **New figures:** `partial_hd_dose.pdf`, `partial_cfg_rates.pdf`, `partial_he_arms.pdf`.
- **Methods:** added the judge-error stress-test equation (Eq. `eq:err-corr`).
- **Rewritten:** abstract, the introduction's findings paragraph and contribution (vi), discussion, conclusion, and the limitations (5 new items).
- **Bibliography:** author lists completed and checked against arXiv metadata for dynamiccfg2025, grimal2025step, ban2024negative, kamath2025geneval2, diffusionprobe2026 (CVPR 2026), guo2024initno (CVPR 2024) and minderer2022owlvit (ECCV 2022). The description of Grimal et al. was corrected to their abstract's actual claim (later steps favoured).
- **Fixes in `partial_report.py`:**
  - The HF table emitted truncated JSON with an unclosed brace, which would have broken compilation.
  - The HE interval is now shown to 3 decimals; at 2 it hid the −0.035 lower bound against the −0.03 margin.
  - Descriptive figure captions.
- **Source of the partial-TEST section:** now built from the canonical Kaggle archive rather than the local replication.

## Remaining
- **Confirmatory analysis is blocked.** It needs TEST_CORE s2, TEST_COMMIT s2 and TEST_CAG_SIZE s0; the folders `kaggle_outputs/gbc-test-core-s2` and `gbc-test-commit-s2` are empty. When all nine shards exist, the integration path is: notebook 04 with `PHASE=TEST`, then `python scripts/final_report.py --final <archive>`, which replaces `sections/results_confirmatory.tex`; then `python paper/make_overleaf_zip.py`.
- **Author names and affiliations** remain a placeholder in `main.tex`. Fill them in yourself; they were not invented.
- **15 bibliography entries are verified at abstract level only**, marked `x-verification={ABS}` in `references.bib`.
- **Single next action:** upload `paper/paper_overleaf.zip` to Overleaf (New Project → Upload Project; compiler pdfLaTeX; main file `main.tex`; BibTeX runs automatically), compile, and check float placement and table widths.

## Regenerate partial-TEST assets
```
python partial/partial_report.py --final kaggle_outputs/gbc-test-partial
python partial/robustness.py --final <extracted meta folder> --records <extracted merged/all_records.jsonl> [--compare <second eval_rows.jsonl>]
python paper/make_overleaf_zip.py
```
