# Experimental protocol (written before the main experiment)

## Research question
How do classifier-free guidance (CFG) scale and the number of denoising steps affect different kinds of
compositional prompt adherence in Stable Diffusion 1.5, and can the effects be related to cross-attention behaviour?

## Hypotheses (predictions fixed before the main run)
- **H1 (CFG, non-monotonic by constraint type).** Raising CFG from a low value (~2) to the default (~7.5)
  increases the success rate of every constraint type (presence, binding, counting, spatial). Raising it further
  (~15) gives little extra gain for presence, and does not improve binding, counting or multi-constraint prompts.
  Concretely: paired difference (CFG 7.5 − CFG 2) > 0 in all categories, while the (CFG 15 − CFG 7.5) difference in
  binding, counting and multi is <= 0.05 in absolute accuracy and smaller than the first difference.
- **H2 (steps).** Going from 20 to 40 steps with a fixed deterministic scheduler matters much less than CFG
  (|Cohen's d| < 0.2 per category), and there is no CFG x steps interaction.
- **H3 (mechanism, exploratory).** (a) Higher CFG makes the attention maps of object tokens more spatially
  concentrated early in denoising (higher top-10% mass fraction, lower normalised entropy), because the
  guidance-amplified update commits latent structure to prompt-conditioned layouts earlier. (b) In binding
  prompts, failures show more overlap between the colour token's map and the *other* object's map than successes.
  (c) Attention strength alone does not predict adherence: a model using only attention features will have
  AUC well below an oracle (we expect AUC < 0.75).

## What would contradict
- H1: no category gains from 2 -> 7.5, or binding/counting keep gaining from 7.5 -> 15.
- H2: steps effect comparable to CFG effect, or a clear interaction.
- H3: object-token concentration does not change with CFG; success/failure show no attention difference
  (AUC ~ 0.5) or attention separates them nearly perfectly (would suggest attention is a strong explanation).

## Alternative explanations
- The evaluator (OWL-ViT + CLIP) is biased by saturation/contrast, which CFG changes, so measured gains may be
  detector artefacts. Mitigation: manual check on a labelled subset.
- Attention in the conditional branch evolves with the latent, so correlations with CFG are confounded by the
  different trajectory; attention maps are not causal evidence.
- Per-prompt difficulty dominates; therefore analyses are paired by prompt and seed and clustered by prompt.

## Variables
- **Independent:** CFG scale (3 levels), steps (2 levels). Chosen after a pilot to confirm they are technically valid.
- **Controlled:** checkpoint (stable-diffusion-v1-5 fp16, revision logged), scheduler (DDIM, eta=0), 512x512,
  empty negative prompt, same seed list for every cell, same prompt text, fp16 on one GPU.
- **Unit of analysis:** prompt x seed; seeds are repeated across all cells so comparisons are paired.

## Outcomes and analysis plan
- Per-constraint success (0/1) and per-image overall success (all constraints satisfied), by category.
- Per-image mean of constraint successes (soft score) for effect sizes.
- Paired comparisons between CFG levels (and step levels) per category: difference in mean success with
  cluster-bootstrap 95% CI (resampling prompts, then seeds within prompt), paired Cohen's d_z on prompt-level means,
  and a Wilcoxon signed-rank test on prompt-level paired means. Holm correction across CFG pairwise comparisons.
- CFG x steps interaction: logistic GEE (prompt-clustered) on overall success, if counts allow.
- Secondary: CLIP image-text similarity (not an adherence measure).
- Attention: per-token concentration and overlap summaries at early/middle/late steps; association with
  adherence via logistic regression / AUC; all labelled exploratory.
- Sample count and every excluded/failed sample are reported.

## Revisions log
(Any change made after seeing data is listed here with date and reason.)

- 2026-10-09 (after pilot, before main analysis): validation on 40 hand-labelled pilot images (see `report/validation.md`).
  Detection thresholds were swept on that subset; accuracy was insensitive, so the pre-set values (presence 0.10,
  count 0.15) were kept. CLIP colour classification rejects yellow-green apples as "yellow" (3 of 4 binding
  errors), so for counting prompts the *primary* outcome is the count constraint only; the colour constraint is
  reported as a secondary outcome with this caveat. Spatial validation has only 2 positive examples among 10,
  so spatial-metric reliability is poorly established and flagged as a limitation.
- Pilot (seeds 0-1) was a subset of the main grid and was kept; parameter values (CFG 2/7.5/15; steps 20/40)
  were not changed because all images were technically valid (no NaN/black images).

## Addendum A (2026-10-10, written BEFORE running the intervention): attention re-weighting
**Motivation.** The main experiment showed frequent neglected objects and colour leakage at CFG 7.5, and attention maps of failed
binding examples where the second object/colour tokens have no region of their own. Prompt-to-Prompt (Hertz et al. 2023) shows that
scaling the cross-attention of chosen tokens controls how strongly a word is reflected in the image; Attend-and-Excite (Chefer et al. 2023)
strengthens neglected subject tokens through gradient-based latent updates. We test the cheapest gradient-free variant.
(Checked and rejected as a rationale: whether the second-mentioned object is dropped more often. Presence rates by mention order flip between
categories and are confounded by object identity, so the intervention is symmetric across tokens.)

**Intervention.** In every text cross-attention layer and every denoising step, the conditional branch's pre-softmax scores of the *content tokens*
(subword tokens of every constraint object word, plural included, and every colour word in the prompt) are increased by `log(s)`, i.e. their unnormalised
attention weights are multiplied by `s` and the softmax is re-normalised. The unconditional branch is untouched. Implemented as an additive attention bias
to the fused attention kernel (`src/intervene.py`).

**Design.** All 28 prompts x seeds 0-5 at CFG 7.5, 40 steps (matched to the baseline cell). Conditions: `rw1` = control through the *same code path* with s = 1 (checks
that the code path itself does not change outcomes), `rw2` (primary), `rw4` (dose-response, secondary). The original baseline images (plain pipeline) are also compared with `rw1`.
**Primary comparison:** rw2 vs rw1, overall primary success, paired by prompt and seed, cluster bootstrap CI, Wilcoxon on prompt-level differences.
**Predictions.** Supported: overall success rises by >= 0.05 (CI excluding 0) and per-category gains in binding/multi, with CLIP score not falling by more than 0.5 and diversity not collapsing.
Contradicted: change <= 0, or gain accompanied by large CLIP/quality loss. Holm correction over the two intervention strengths.
**Failure modes to look for:** over-saturated colours, duplicated objects, higher count errors (reported by category).

- 2026-10-12 (validation labels finalised, after the main analysis): the second-pass overrides (images 2, 9, 29, 35) and the project owner's revisions (images 26, 32, 38) were merged into one final label set in `data/validation_labels.json`; `src/validate_metrics.py` now reports against that single set. Evaluator thresholds are unchanged. The primary outcomes and all main-experiment numbers do not depend on these labels.
