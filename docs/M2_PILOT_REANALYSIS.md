# M2 pilot re-analysis (saved artifacts only, no generation)

Script: `python src/analyze_pilot_conditional.py --check` (CPU, ~1 min). Outputs: `outputs/m2_pilot_reanalysis/{per_image_conditional.csv, conditional_rates.csv, eligibility.csv, summary.json}`; console log `outputs/m2_pilot_reanalysis_stdout.txt`. `evaluate.py` was not modified (D-04 outcomes are a standalone function). **Status: EXPLORATORY.** Pilot prompts are DEV-only; nothing here is confirmatory (D-06).

## Reproduced (executed)
- All 1008 images re-score identically from `detections.jsonl` with the unchanged `evaluate.score()` (0 mismatches vs `eval.jsonl`); legacy binding / spatial / all_satisfied agree with `results/per_image.csv` (0 disagreements). The `--check` flag asserts this.

## Conditional and joint outcomes (new, D-04; scene = prompt, 95% CI by prompt-cluster bootstrap, 2000 draws, seeded)
Binding category (6 prompts x 12 images per CFG), full table in `conditional_rates.csv`:

| CFG | P(all objects present) | P(binding given present), n eligible | P(present and binding) |
|---|---|---|---|
| 2 | 0.403 [0.26, 0.54] | 0.172 [0.03, 0.41], 29 | 0.069 [0.01, 0.15] |
| 7.5 | 0.486 [0.31, 0.64] | 0.229 [0.08, 0.39], 35 | 0.111 [0.03, 0.21] |
| 15 | 0.472 [0.32, 0.63] | 0.235 [0.11, 0.39], 34 | 0.111 [0.06, 0.18] |

- Missing-object rate in binding images is 51-60%; across all 18 binding-bearing pilot prompts (binding 6, counting 6, multi 6 = D-05 definition: at least one `binding` constraint) 395/648 images (61%) are eligible. Multi-constraint prompts: only 33-42% eligible, 3 prompt/CFG cells have prompts with zero eligible images. Counting prompts are 100% eligible (their objects are always detected at the 0.10 presence threshold).
- Legacy `binding_ok` already scored 0 when an object was missing, so legacy binding equals the joint outcome here (0 differences on 648 images); the new information is the conditional rate and its denominator. Conditional binding is subject to selection (presence is itself a generation outcome); do not read it alone as improved adherence.

## Noise-only baseline (steps = 40 images, n = 504, base rate 0.236; AUC from out-of-fold predictions, folds grouped by prompt)
| model | AUC |
|---|---|
| design (category + CFG) | 0.719 |
| + seed identity (only noise information available in the pilot) | 0.721 (delta +0.002, 95% CI [-0.028, 0.027]) |
| + early/middle/late attention features | 0.786 (delta vs design +0.067, CI [0.028, 0.122]) |
| noise + attention vs noise alone | delta +0.050, CI [0.017, 0.097] |

- Within prompt x CFG x steps cells, a permutation test on a seed main effect (all 1008 images) gives p = 0.012 (seed 0 is 9 points worse than the mean; the other seeds are within +/-5 points). Evidence of a seed effect, but not predictive gain in prompt-grouped CV.
- Caveats: only 6 distinct noise vectors exist in the pilot, so "noise features" are limited to seed identity (an upper bound on any function of those 6 vectors); only 28 prompts; the saved `attention_classifier_auc.json` (0.713 / 0.740 / 0.750) used a different CV scheme and is not directly comparable. This is a preliminary look at the E21 gate and is not a substitute for the V2 design.

## Cannot be computed from pilot artifacts
- Restart persistence (no latents saved, no branches); noise features beyond the seed; binding-given-presence for scenes outside the 28 prompts; anything about CFG x steps x complexity at V2 scale.
