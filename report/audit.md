# Finalization and Verification Audit

**Dates:** 10–11 October 2026

The report's numerical results are kept in the generated `report/report.md` and the files under `outputs/*/results/`. This audit records the checks made before finalizing the report and slides, the intervention experiment, and the work that remains open.

## 1. Checks completed

### Report numbers and claims

The report, defense notes, README, and slides now take their numbers from saved experiment results rather than numbers typed in by hand. `src/report_numbers.py`, `report/build_report.py`, and `report/make_slides.py` are used to collect the results and generate the documents. The report was regenerated, replacing the earlier hand-written version.

The report build also checks the direction of the main claims against the data before writing the final text. These checks cover statements such as whether every category is higher at CFG 7.5 than at CFG 2, whether any per-category Holm-adjusted p-value is below 0.05, and whether presence is the only intervention result whose confidence interval excludes zero. All of these assertions passed. If the saved results contradict one of these statements, the build is set to fail rather than produce a report with unsupported wording.

### References and unsupported statements

All eight citations were checked against arXiv or publisher/venue pages on 10 October 2026. Several reference details were corrected:

- Ho and Salimans is cited as an arXiv paper from 2022; it was first presented at the NeurIPS 2021 workshop.
- Chefer et al. is listed in *ACM Transactions on Graphics*, volume 42, issue 4.
- Full author lists were added.
- The Stable Diffusion 1.5 repository is described as an unaffiliated mirror of a deprecated repository, rather than being presented as an official repository.

The report was also reread against the saved results. An unverified claim about image posterisation was removed. The cost statement was corrected to reflect summed generation time and the fact that GPU use was partly shared. The qualitative comment about spatial relations was rewritten to describe what the figure actually shows.

### Attention extraction and evaluation

The attention extraction code was reviewed and checked with `tests/test_attention_gpu.py`. Across all 16 cross-attention layers, the maps summed to 1 over the 77 tokens. Generating images with the attention hooks also produced pixel-identical results to plain generation in the tested setup, so the hooks did not change the generated pixels in that test.

The evaluation code was reviewed, and no logic error was found. The threshold note was clarified: thresholds had been fixed in advance, and the validation sweep showed that the results were insensitive to those thresholds.

### Analysis-code bug fix

A bug was found in `constraint_type_table`: it called the bootstrap routine twice, meaning the lower and upper confidence-interval bounds could come from different random draws. This was fixed so the bounds are taken from the same bootstrap results. The report now also notes that the Wilcoxon p-values are approximate when ties are present.

### Validation labels

All 40 validation images in `data/validation_labels.json` were reviewed again at a larger size. During this second pass by Claude, two constraints were marked ambiguous:

- **Image #2:** three round objects were visible, but only one was red, making the count-of-three-balls judgement ambiguous.
- **Image #9:** the chair had a blue back and cushion but a red frame, so its colour was ambiguous.

Two yellow-green apples, **#29 and #35**, were relabelled as not yellow. This second pass was **not blind**: the evaluator's disagreements had already been seen. The project owner then revised labels for **#26, #32, and #38**. These overrides were merged into the single final `labels` set.

### Slides

The slide deck was exported through PowerPoint using COM and inspected. The original deck had small figures, centred titles, and little visual structure, so it was rebuilt with a clearer layout, auto-fitted text, and figures chosen for individual slides. Three further rounds of fixes addressed overlapping boxes in the method diagram, truncated labels, an unreadable contact sheet, and file paths that wrapped awkwardly.

## 2. Intervention experiment

The intervention experiment was registered in advance in Addendum A of `report/protocol.md`, before any intervention images were generated. It used **504 images**: three conditions (`s=1` control, `s=2`, and `s=4`), with 28 prompts and 6 seeds per condition. CFG was fixed at 7.5 and generation used 40 steps. The images were evaluated with the unchanged evaluator, then analysed with the same paired-statistics approach in `src/analyze_intervention.py`.

The pre-registered hypothesis **H4 was not supported**; the results are discussed in section 13 of the report. The control path reproduced the baseline images pixel for pixel.

There was one operational issue during generation. A stopped run was resumed twice, and for a short time two generation processes wrote to the same folder. Duplicate metadata lines were removed, keeping the first occurrence. Every image was checked to make sure it opened at 512 × 512, and the final set contained exactly 504 unique images: 168 for each condition. Generation is deterministic, so the duplicate writes produced identical images.

## 3. Work still to do

The remaining limitations are described in section 15 of the report and in the README status table. The following work has not been completed:

- An independent human review of the validation labels; this is still an action for the project owner.
- A separate confirmation set to check the exploratory result for the presence category.
- More prompts in each category and a wider range of CFG values.
- Tests with other models and schedulers.
- Additional intervention variants, including colour-only, adaptive, and gradient-based approaches.

These limitations should be kept in mind when interpreting the results. In particular, the exploratory presence result still needs confirmation, and the validation labels have not yet had an independent human check.
