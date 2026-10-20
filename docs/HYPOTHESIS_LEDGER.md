# Hypothesis ledger

> **Revision 3 (2026-10-18) supersedes the primary family below.** Confirmatory family (Holm over four): **HA** restart persistence differs between presence, count, size, spatial and colour binding (primary estimand Delta t50 = t50(bind|pres) - t50(presence); count's position EXPLORATORY); **HD** higher CFG moves t50 earlier (presence and binding|pres, chance level per CFG); **HE** Commitment-Aligned Guidance is non-inferior on F, raises seed diversity and beats the reversed schedule (intersection-union); **HF** attention stabilisation tracks t50 (type concordance >= 0.8 and within-type Spearman > 0 for >= 2 types). HB and HC are supporting (P2). Exact rules: `docs/ADDENDUM_C.md`, `configs/analysis_plan_r3.json`; decisions: `docs/R3_DECISION_LOG.md`. **No data for any of them.** The Revision 2 entries below are kept for traceability.


Updated 2026-10-17. Definitions, thresholds and decision rules come from `GRAND_RESEARCH_BLUEPRINT.md` (Revision 2) sections 8.0-8.5 and the **draft, unsigned** `docs/ADDENDUM_B.md`; this ledger records state, not new definitions. Working defaults D-02..D-06, D-12 are approved as working defaults only (2026-10-14); nothing is frozen; TEST is sealed. Evidence labels follow `docs/RESULTS_PROVENANCE.md`. **No confirmatory evidence exists for any hypothesis.**
Rule: HA, HB and HC are never edited after results are seen. A change is an amendment: dated, reasoned, and every affected analysis is relabelled EXPLORATORY.

## 1. Primary (Holm family: HA, HB, HC)
### HA - restart-persistence timing differs between presence and binding given presence
| Field | Content |
|---|---|
| Motivation / prior work | Conflicting reports on when layout and binding are formed (R&B, CoBind assumptions, abstract level); critical-window theory (abstract level); pilot seed persistence (outcome correlation 0.79 between CFG 7.5 and 15 for the same prompt and seed, exploratory). Prior-art gap is **provisional**: narrow search, full texts of the closest papers not read |
| Variables | IV: constraint type (presence vs binding given presence), grid step `t` in {6,...,36}, restart magnitude D=8 steps (sensitivity: equal-ratio 0.30, D in {4,12}); DV: persistence `C_k(t)`, `t50_k` |
| Unit / estimand | scene graph; `Delta t50 = t50(bind given pres) - t50(presence)`, two-sided |
| Method | agreement `A` minus chance `A0` from independent seeds, ratio estimator, 4000-draw scene bootstrap, Holm over {HA, HB, HC}; per-type judge from the audit |
| Controls | D=0 identity (passed on tested cases), full-resample control (C <= 0.15 gate), independent-seed agreement, equal-ratio and D sensitivity, judge sensitivity |
| Support / reject | support: CI excludes 0 and abs(Delta t50) >= 3 steps; bounded null: CI within (-1.5, 1.5); else inconclusive; **precision gate: half-width <= 4 steps on DEV or HA is exploratory** |
| Evidence now | none. Sizing simulation (planning only, 2026-10-14): gate unlikely at 70 DEV binding scenes with realistic judge error; to be re-run with measured error |
| Threats | judge noise, noise ratio varies with t (absolute t50 not a model property), selection by presence conditioning, small effective scenes (31 of 70 DEV scenes have >= 2 eligible seeds under J1) |
| Needs | evaluator audit; R3 + J1 evaluation on DEV; Addendum B values; freeze; TEST core and branches |
| Prereg state | working defaults approved; not frozen. Next decision: after the audit, whether R3 is run as an exploratory rehearsal (D-13) |

### HB - readout versus forecast
| Field | Content |
|---|---|
| Motivation | pilot: early attention features add held-out AUC over design+seed (+0.050, CI +0.017 to +0.097, 28 prompts, exploratory); noise features add none; early prediction of quality from attention exists (abstract level) |
| Variables / unit | features: prompt structure (B1), initial-noise statistics (B_noise), attention-conflict features at steps up to `tau_pre` or `tau_post` (from DEV `t50`); DV: binding given presence; unit: scene |
| Method | frozen L2-logistic models, grouped CV by scene on DEV, one evaluation on TEST; scene bootstrap of Delta AUC |
| Support / reject | forecast: Delta AUC_pre >= 0.05, CI > 0; readout only: pre CI upper < 0.03 and post >= 0.05; none: both upper < 0.03; DEV simulation gate: projected CI half-width <= 0.04 else exploratory |
| Evidence now | pilot only (motivation); design: single-AUC half-width 0.083 (140 scenes) / 0.094 (110) |
| Threats | label noise, shuffled-attention leakage control, object-identity confounds, windows undefined when `t50_DEV` extreme (D-12 rules) |
| Needs | V2 attention maps (not yet generated), DEV timing reference, audit |

### HC - time-gated targeted edit
| Field | Content |
|---|---|
| Motivation | pilot global re-weighting gave +0.02 (CI -0.04 to +0.09), i.e. no resolved effect; consistency test of HA's timing and of attention as a handle |
| Variables / unit | arms: targeted vs random-pair at early/mid/late onset (window 12 steps), global s=2; DV: binding given presence and joint; unit: scene, paired by scene and seed |
| Support / reject | `D_early >= 0.08` (CI > 0) and `G = D_early - D_late >= 0.10` (CI > 0), guards (CLIP drop <= 0.5, duplicate rate +0.03); reject if D_early CI upper < 0.04 or G inside (-0.05, 0.05) while D_early supported |
| Evidence now | none (infrastructure: control path identical to baseline on tested cases) |
| Threats | mask error, strength confounded with onset (bias-mass equality enforced within 5%), judge dependence, undefined onsets (D-12) |
| Needs | DEV onsets, E12 arms (3,360 images FULL), freeze |

## 2. Supporting hypotheses and questions (outside the Holm family; exploratory unless stated)
| ID | Question | Status | Notes |
|---|---|---|---|
| S1 (RB H1) | Early attention features add held-out information about binding | subsumed by HB; pilot EXPLORATORY | outcome corrected to binding given presence |
| S2 (RB H2) | Meaning-preserving rewrites shift scene-level binding/spatial success beyond seed noise (within-seed permutation, excess RMS >= 0.10) | PLANNED; variants not reviewed | MetaLogic/HRS-Bench overlap (abstract level) |
| S3 | CFG x steps x constraint complexity | pilot EXPLORATORY (no CFG effect resolved, MDE about 0.15) | required by the problem statement; E11 planned |
| S4 | Evaluator reliability per type; sensitivity of conclusions to the judge | audit package ready; labels pending | criterion accuracy >= 0.85 per type |
| S5 | Noise-only baseline: how much of an outcome is readable from the initial noise | pilot EXPLORATORY (no gain over design, only six noise vectors) | mandatory baseline for HB |

## 3. Open hypothesis backlog (all POST HOC until separately specified, dated and approved)
Template: `BL-nn | question | origin (which exploratory observation) | variables | unit | method | controls | support/reject | data needed | compute | novelty uncertainty | readiness | decision`.
| ID | Question | Origin | Value | Novelty uncertainty | Feasibility / compute | Readiness |
|---|---|---|---|---|---|---|
| BL-01 | Do failures mostly come from missing objects rather than wrong colours? | pilot: only 61% of binding-bearing images have all required objects | high (changes what is measured) | low (known omissions literature) | free (existing data + audit) | ready after audit |
| BL-02 | First-mentioned object keeps its colour more often than the second (62% vs 38% in the pilot) | pilot F3, n = 98 | medium | high overlap (mention-order / causal-mask work, abstract level) | exploratory inside E08/E09 | needs variants reviewed |
| BL-03 | Requested counts compress toward about three | pilot F5 | medium | high overlap (counting work) | descriptive | ready |
| BL-04 | Is a seed-0 deficit real or an artefact of six noise vectors? | pilot permutation p = 0.012 | low | low | free with V2 seeds | needs V2 data |
| BL-05 | Which tokens carry the CFG guidance vector? (guidance attribution) | CFG raises CLIP score more than adherence | medium | medium | 8,400 UNet evaluations | after P1 stages |
| BL-06 | Layer-group knockouts with matched random control | blueprint E26 | medium | medium (head-level work unread) | 1,280 images | after P1 |
| BL-07 | Outcome fragility under micro-perturbation | blueprint E24 | medium | medium | 1,400 images | after P1 |
| BL-08 | Negative-prompt timing for binding repair | blueprint E28 | low-medium | high overlap | 2,800 images | P3 |
| BL-09 | Second architecture (e.g. SDXL) | generalisation | medium | n/a | hook port and memory unverified | deferred |
None of BL-01..09 may be run to "rescue" a failed primary hypothesis; each needs its own specification first.

## 4. Amendment log
2026-10-14 owner approval of working defaults and D-13 deferral (see `docs/DECISION_REGISTER.md`). 2026-10-17 ledger created. No hypothesis text changed.
