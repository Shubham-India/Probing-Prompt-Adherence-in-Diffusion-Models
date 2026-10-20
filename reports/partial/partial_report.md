# Guide Before Commitment R3 — partial TEST report (PARTIAL TEST, EXPLORATORY)

Source: `D:\prompt_Project\local_audit\partial_x\r3-test-partial-20261020T134300Z-40ade4d` (archive r3-test-partial-20261020T134300Z-40ade4d; judged with torch 2.11.0+cu128 on 2 GPU(s))

## Coverage
| Stage | present | missing | jobs merged | scenes |
|---|---|---|---|---|
| TEST_CORE | [0, 1] | [2] | 6172/6172 | 133 |
| TEST_COMMIT | [0, 1] | [2] | 7080/7080 | 85 |
| TEST_CAG_SIZE | [1, 2] | [0] | 3750/3750 | 222 |

## Hypotheses (exploratory on partial data)
| H | estimand | estimate | CI | p | Holm p | rule outcome |
|---|---|---|---|---|---|---|
| HA | $\Delta t_{50}$ binding$-$presence (steps) | 3.59 | [-10.43, 13.06] | 0.5652 | 1.0000 | EXPLORATORY (DEV precision gate failed (half-width 7.94 > 4); binding judge below the audit criterion); rule outcome: inconclusive |
| HD | $\Delta t_{50}$ presence, $w$15$-$2 (steps) | -8.67 | [-16.34, -4.72] | 0.0116 | 0.0464 | supported |
| HE | A2$-$A1 on $F$ (90\% CI; margin $-0.03$) | -0.02 | [-0.04, 0.00] | 0.4825 | 1.0000 | not supported (non_inferior, beats_reversed failed) |
| HF | type-order concordance | 0.30 | [0.20, 0.75] | 0.8042 | 1.0000 | rejected (ordering concordance <= 0.5) |

## Limitations
- Missing TEST shards (not imputed): TEST_CORE shard 2; TEST_COMMIT shard 2; TEST_CAG_SIZE shard 0; estimates cover only the scene groups of the present shards.
- This is not the registered confirmatory analysis: that requires all nine shards and notebook 04; Holm-adjusted p-values are shown for completeness only.
- HA is exploratory by its failed DEV precision gate; the full-resample control failed on DEV (Section on DEV results).
- Only the presence judge met the audit criterion (single annotator, no valid double annotation); count, binding, spatial and size outcomes are exploratory.
- HE uses only scenes with both A1 and CAG images (67 scenes).

Figures: ['partial_curves_D8.pdf', 'partial_curves_ER30.pdf', 'partial_he.pdf', 'partial_s2_presence.pdf']
LaTeX: paper/sections/results_partial.tex, paper/tables/partial_*.tex
