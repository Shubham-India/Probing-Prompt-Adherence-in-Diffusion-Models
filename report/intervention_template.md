@@ABSTRACT
scaling the attention weight of object and colour tokens by s = 2 (control: the same code path with s = 1, whose images were pixel-identical to the plain pipeline in {int_ctrl_pixel_identical} cases) changed overall success at CFG 7.5 / 40 steps by {intc_primary_all_rw2_vs_rw1} (95% CI {intc_primary_all_rw2_vs_rw1_ci}; Holm p = {intc_primary_all_rw2_vs_rw1_holm}), below the pre-registered +0.05 threshold, so H4 is not supported; an exploratory presence-only outcome improved by {intc_presence_ok_all_rw2_vs_rw1} ({intc_presence_ok_all_rw2_vs_rw1_ci}) with no detectable CLIP-score change ({intc_clip_score_all_rw2_vs_rw1}).

@@SECTION
**Design (pre-registered in `report/protocol.md`, Addendum A).** In every text cross-attention layer and denoising step the conditional branch's pre-softmax scores of the content tokens (constraint objects and colour words) get `+log(s)`, i.e. their attention weights are multiplied by s and the softmax re-normalises
(`src/intervene.py`, an additive bias to the fused attention kernel; no gradients, no retraining). All {n_prompts} prompts x {n_seeds} seeds at CFG 7.5, 40 steps: control s = 1 (same code path), s = 2 (primary), s = 4 (dose-response); {int_n_images} images. The plain-pipeline baseline images from the main experiment were also compared with the control:
the control images were pixel-identical to the baseline images in {int_ctrl_pixel_identical} cases and gave the same primary outcome for a fraction {int_ctrl_identical_outcomes} of the prompt x seed pairs, so the code path itself changes nothing. Contrast = intervention minus control, paired by prompt and seed; Holm correction over the two strengths; other outcomes are secondary and not corrected across outcomes.

Primary success (mean [95% cluster-bootstrap CI]):

| category | prompts | control s=1 | s=2 | s=4 |
|---|---|---|---|---|
| presence | {nprompts_presence} | {int_presence_rw1} {int_presence_rw1_ci} | {int_presence_rw2} {int_presence_rw2_ci} | {int_presence_rw4} {int_presence_rw4_ci} |
| binding | {nprompts_binding} | {int_binding_rw1} {int_binding_rw1_ci} | {int_binding_rw2} {int_binding_rw2_ci} | {int_binding_rw4} {int_binding_rw4_ci} |
| counting | {nprompts_counting} | {int_counting_rw1} {int_counting_rw1_ci} | {int_counting_rw2} {int_counting_rw2_ci} | {int_counting_rw4} {int_counting_rw4_ci} |
| spatial | {nprompts_spatial} | {int_spatial_rw1} {int_spatial_rw1_ci} | {int_spatial_rw2} {int_spatial_rw2_ci} | {int_spatial_rw4} {int_spatial_rw4_ci} |
| multi | {nprompts_multi} | {int_multi_rw1} {int_multi_rw1_ci} | {int_multi_rw2} {int_multi_rw2_ci} | {int_multi_rw4} {int_multi_rw4_ci} |
| all | {nprompts_all} | {int_all_rw1} {int_all_rw1_ci} | {int_all_rw2} {int_all_rw2_ci} | {int_all_rw4} {int_all_rw4_ci} |

Contrasts versus the control (all prompts): primary success s = 2: {intc_primary_all_rw2_vs_rw1} {intc_primary_all_rw2_vs_rw1_ci} (Wilcoxon p = {intc_primary_all_rw2_vs_rw1_p}, Holm {intc_primary_all_rw2_vs_rw1_holm}, d_z = {intc_primary_all_rw2_vs_rw1_dz}); s = 4: {intc_primary_all_rw4_vs_rw1} {intc_primary_all_rw4_vs_rw1_ci} (Holm {intc_primary_all_rw4_vs_rw1_holm}).
Per category, no s = 2 contrast has Holm p < 0.05 (binding {intc_primary_binding_rw2_vs_rw1} {intc_primary_binding_rw2_vs_rw1_ci}, multi {intc_primary_multi_rw2_vs_rw1} {intc_primary_multi_rw2_vs_rw1_ci}, presence {intc_primary_presence_rw2_vs_rw1} {intc_primary_presence_rw2_vs_rw1_ci}).
At prompt level, s = 2 raised success for {int_prompts_improved_rw2} prompts and lowered it for {int_prompts_worse_rw2} (s = 4: {int_prompts_improved_rw4} and {int_prompts_worse_rw4}).
Secondary constraint-level outcomes (s = 2 vs control, all prompts): presence {intc_presence_ok_all_rw2_vs_rw1} {intc_presence_ok_all_rw2_vs_rw1_ci} (Wilcoxon p = {intc_presence_ok_all_rw2_vs_rw1_p}, Holm over strengths {intc_presence_ok_all_rw2_vs_rw1_holm}), colour binding {intc_binding_ok_all_rw2_vs_rw1} {intc_binding_ok_all_rw2_vs_rw1_ci}, count {intc_count_ok_all_rw2_vs_rw1} {intc_count_ok_all_rw2_vs_rw1_ci}, spatial {intc_spatial_ok_all_rw2_vs_rw1} {intc_spatial_ok_all_rw2_vs_rw1_ci}.
The presence result is the only one that excludes zero; it is one of six outcomes examined, is not corrected across outcomes, and is therefore an exploratory lead, not a confirmed effect.
Image quality: CLIP score s = 2 vs control {intc_clip_score_all_rw2_vs_rw1} {intc_clip_score_all_rw2_vs_rw1_ci}, s = 4 {intc_clip_score_all_rw4_vs_rw1} {intc_clip_score_all_rw4_vs_rw1_ci}; seed diversity {int_div_rw1} (control), {int_div_rw2} (s = 2), {int_div_rw4} (s = 4).
Qualitatively (`intervention.png`, fixed gallery rule: the alphabetically first prompt of the presence, binding, counting and multi categories, seed 0), re-weighting changes the rendering of the objects that are already present but did not create the missing second object in the chair/vase examples.
Figure: `outputs/main/figures/intervention.png`; tables: `outputs/intervention/results/`.

@@VERDICT
**not supported on the pre-registered primary outcome.** Overall success changed by {intc_primary_all_rw2_vs_rw1} {intc_primary_all_rw2_vs_rw1_ci}, below the +0.05 threshold with a CI that includes 0, and neither binding nor multi-constraint prompts improved. The only signal is an exploratory gain in object presence ({intc_presence_ok_all_rw2_vs_rw1} {intc_presence_ok_all_rw2_vs_rw1_ci}) with no CLIP-score cost; it needs confirmation on new prompts.

@@CONCLUSION
In Stable Diffusion 1.5, going from a low CFG to the default is associated with higher image-text similarity, lower diversity, more localised object attention and a small, statistically inconclusive increase in prompt adherence; guidance beyond the default and more steps showed no benefit in this benchmark.
Compositional failures, especially colour leakage and dropped objects, persist at every setting, and the colour-object attention overlap is a promising but still non-causal indicator of binding failure. A simple gradient-free attention re-weighting of object and colour tokens did not improve overall adherence (H4 not supported), though it raised object presence in an exploratory secondary outcome without lowering CLIP score.
A larger benchmark, more CFG levels and an intervention aimed directly at the colour/object overlap are needed before any of these associations can be turned into conclusions.

@@SLIDE_HEADLINE
Attention re-weighting (2x object/colour tokens): overall success {intc_primary_all_rw2_vs_rw1} {intc_primary_all_rw2_vs_rw1_ci}, so H4 is not supported

@@SLIDE_LINE1
Primary: s = 2 vs control {intc_primary_all_rw2_vs_rw1} {intc_primary_all_rw2_vs_rw1_ci}, Holm p = {intc_primary_all_rw2_vs_rw1_holm}; s = 4 {intc_primary_all_rw4_vs_rw1} {intc_primary_all_rw4_vs_rw1_ci}. Pre-registered threshold was +0.05 with CI above 0.

@@SLIDE_LINE2
Exploratory: object presence {intc_presence_ok_all_rw2_vs_rw1} {intc_presence_ok_all_rw2_vs_rw1_ci}; binding {intc_binding_ok_all_rw2_vs_rw1} {intc_binding_ok_all_rw2_vs_rw1_ci}; CLIP {intc_clip_score_all_rw2_vs_rw1}. Matched prompts, seeds, CFG 7.5, 40 steps; control reproduces baseline.

@@SLIDE_TAKEAWAY
Re-weighting object/colour attention did not raise overall success ({intc_primary_all_rw2_vs_rw1} {intc_primary_all_rw2_vs_rw1_ci}); only object presence moved (exploratory)

@@NOTES
Hypothesis four was written down before any intervention image existed: multiply the attention weight of the object and colour words by 2 in the conditional branch, a gradient-free relative of Prompt-to-Prompt re-weighting. The control uses exactly the same code with a factor of 1 and reproduces the baseline images, so any difference comes from the factor and not from the code path. The pre-registered prediction was a gain of at least 0.05 in overall success with an interval above zero. We got {intc_primary_all_rw2_vs_rw1}, interval {intc_primary_all_rw2_vs_rw1_ci}, so the hypothesis is not supported, and a factor of 4 does not change that. Binding and multi-constraint prompts did not improve. One secondary outcome, whether all requested objects are present, rose by {intc_presence_ok_all_rw2_vs_rw1}, interval {intc_presence_ok_all_rw2_vs_rw1_ci}; that is one of six outcomes I looked at without correction, so I present it as a lead for follow-up and not as a finding. CLIP score did not change.

@@NOTES_TAKEAWAY
The intervention did not help overall success, so the hypothesis about re-weighting is not supported; only object presence moved, as an exploratory lead.
