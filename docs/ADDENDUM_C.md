# Addendum C — Revision 3 protocol ("Guide Before Commitment")  — DRAFT, NOT FROZEN

Becomes binding only when `python src/protocol_r3.py --sign "<owner>"` succeeds and `--verify` exits 0. Until then nothing here is a preregistration and no TEST job may run (workers and the confirmatory analysis refuse). Source: `GRAND_RESEARCH_BLUEPRINT.md` sections 0 and 24 (Revision 3, authoritative), with Revision 2 working defaults D-02..D-06, D-12 (approved by the owner 2026-10-14 as working defaults). Machine-readable twin: `configs/experiment_r3.json`, `configs/analysis_plan_r3.json`; conflicts are resolved in `docs/R3_DECISION_LOG.md`.

## C1 Splits, scenes, prompts, seeds
- Main benchmark V2: 300 scene graphs (DEV 100, TEST 200), 1,410 prompts (design maximum of 1,500 = five per scene is not reached because some variant types are inapplicable; DL-04). Split and pair-group manifests: `data/benchmark_v2/splits.json`, `scene_graphs.json` (hashes in the freeze).
- PS flagship family (relative size): 24 object pairs (DEV 8, TEST 16) x 8 subsets of {colour, size, spatial} = 192 canonical prompts + 72 counterfactuals (CF-C, CF-S, CF-R on the full subset) in separate files (`ps_family.json`, `prompts_ps.json`, `splits_ps.json`). Size exists only in this family (DL-05).
- Seeds: core generation 0-3 for every prompt; restart/switch bases 0 and 1 (D-13: four bases not approved); CFG x steps study 0-3.
- Commitment scenes: DEV all 100 canonical scenes; TEST the first scenes per category in scene_id order (binding 30, multi 25, counting 15, spatial 15, presence 15). CFG x steps study: first 150 TEST scenes per category (45/37/23/22/23); it contains the commitment scenes, so its 40-step w = 2 and 15 images are the HD bases.

## C2 Generation settings
Stable Diffusion 1.5 fp16, revision `f03de327dd89b501a01da37fc5240cf4fdba85a1`, DDIM eta = 0, 40 steps, 512x512, empty negative prompt, CFG 7.5 core. Latents saved after {6, 12, 18, 24, 30, 36} completed steps (one file per image). Attention: constraint words, conditional branch, heads averaged, layers averaged within the 16x16 and 32x32 groups, every step. Evaluator models: OWL-ViT `cbc355fb...`, CLIP `3d74acf9...`; diversity DINOv2-small `ed25f3a3...`.

## C3 Primary outcomes (per image)
Presence (all required objects detected), count (all count constraints exact), binding given presence, spatial given presence, size given presence (larger object's box area >= 1.5x the other's), and the joint versions. Conditional outcomes are undefined (not 0) when an object is missing. Overall satisfaction F = mean over the families present of presence, count, binding|pres, spatial|pres, size|pres (conditional families only when defined).

## C4 Commitment estimation
Re-noising probe: restart from the latent after t steps to index max(t - 8, 0) with fresh branch-specific noise (sqrt(r) x_t + sqrt(1 - r) eps, r = abar(j)/abar(t)), M = 2 branches, deterministic DDIM to the end at the base's CFG. Equal-ratio sensitivity ER30 (ratio about 0.30 at every t). Chance level A0 from the independent seeds of the canonical prompt at the same CFG. C(t) = sum_s[A - A0] / sum_s[1 - A0]; t50 = interpolated first crossing of 0.5; "> 36" if never reached. Prompt-switch probe (E31): CF trajectory to step t, canonical prompt afterwards; C_switch(t) = sum_s[W(t) - W0]/sum_s[W1 - W0]. Wording: "restart persistence" / "operational commitment step"; never "the model decided".

## C5 Confirmatory hypotheses (Holm over four, alpha 0.05)
| ID | Estimand | Support | Bounded null | Other |
|---|---|---|---|---|
| HA | Delta t50 = t50(bind given pres) - t50(presence), w = 7.5, D8 | CI excludes 0 and abs >= 3 steps; joint companion and ER30 agree in sign | CI within (-1.5, 1.5) | else inconclusive. Ordering of five types reported; the position of count is EXPLORATORY (DL-09) |
| HD | t50(w=15) - t50(w=2) for presence and for binding given presence; A0 per w; one shared scene resample | at least one estimand: CI excludes 0, abs >= 3, negative; p_HD = min(1, 2 min p) | both CIs within (-1.5, 1.5) | positive with CI excluding 0 = contradicted |
| HE | A2 - A1 on F (90% CI lower > -0.03), A2 - A1 diversity (95% CI > 0), A2 - A4 on F (95% CI > 0) | all three (intersection-union; p = max of the three) | - | otherwise "not supported" naming the failed condition |
| HF | concordance of type orderings of t_stab and t50 (95% CI); per-type Spearman of scene t_stab vs scene commitment AUC | concordance >= 0.8 and Spearman CI > 0 for >= 2 types | concordance <= 0.5 or all within-type CIs include 0 with upper < 0.15 | else inconclusive |
A hypothesis is reported SUPPORTED only if its rule passes and its Holm-adjusted p < 0.05. Bootstrap: 10,000 scene-level draws with fixed keys.

## C6 CAG
s*(prompt) = ceil(max over the prompt's constraint types of t50_DEV at w = 7.5) + 2, capped at 40; unresolved t50 for a present type gives s* = 40 (CAG = constant CFG; reported). Arms: A1 constant 7.5 (core images), A2 CAG, A3 global s* = ceil(median t50_DEV) + 2, A4 reversed (w_lo before s*, w_hi' after with A2's mean guidance; undefined when s* = 40), A6 constant at A2's mean. w_lo = 1 (unconditional pass skipped; UNet evaluations counted) unless the DEV quality guard fails, in which case w_lo = 3 is adopted BEFORE TEST and logged. Guard metric: HPS v2 drop <= 0.5 — **the scorer is not implemented; the owner must fix the guard metric before the freeze (DL-15).**

## C7 Judges
J1 thresholds: presence 0.10, count 0.15, distinct IoU 0.5, NMS 0.3, size ratio 1.5, uncertain band 0.03 (`src/evaluate.py`; snapshot in the freeze). A constraint type is confirmatory only if its DEV-audit weighted accuracy >= 0.85 (`docs/AUDIT_PROTOCOL_DEV.md`); size needs >= 140 audited judgements (blueprint 24.6), otherwise EXPLORATORY. J2 (VQA) is not implemented and plays no confirmatory role.

## C8 Exclusions and missing data
Jobs whose output is missing or corrupt after one retry are excluded and listed; no outcome-dependent exclusion. Conditional outcomes are never imputed; eligibility counts accompany every estimate.

## C9 DEV-derived constants (filled from `dev_constants.json` at the freeze; currently UNSET)
t50_DEV per kind at w = 7.5; s*_global; chosen w_lo; DEV gate results (HA half-width <= 4 steps; full-resample control |C| <= 0.15; >= 40 DEV scenes with a both-present base at w = 2 and 15; CAG quality guard).

## C10 Hashes
Code, configs, benchmark and split files, this document, the judge definition and the three TEST stage manifests are hashed into `protocol/protocol_freeze.json` (LF-normalised SHA-256).
