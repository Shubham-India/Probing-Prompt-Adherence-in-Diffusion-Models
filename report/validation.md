# Evaluator validation

**Subset.** 40 images drawn at random (seed 123, 8 per category) from the pilot images (seeds 0-1). The list is in
`data/validation_subset.json`; the labels are in `data/validation_labels.json`.

**Annotator.** The labels were written by Claude from the images (contact sheets), per constraint, before looking at any
automatic score. Ambiguous constraints are `null` and excluded (9 constraints). This is a single, non-independent annotator,
so the project owner should spot-check the labels (open `data/validation_subset.json`, view the images in `outputs/main/images`).

**Result** (`python src/validate_metrics.py --run outputs/main`, thresholds fixed in advance: presence 0.10, count 0.15):

| constraint type | n | accuracy | precision | recall | notes |
|---|---|---|---|---|---|
| presence | 59 | 0.93 | 1.00 | 0.91 | misses are low-confidence OWL-ViT detections (partially visible horse/bench/dog) |
| binding (colour) | 36 | 0.81 | 0.93 | 0.68 | CLIP calls yellow-green apples "green"/"orange" (3 of 7 errors) |
| count | 11 | 0.91 | 0.83 | 1.00 | only 11 labelled examples |
| spatial | 10 | 0.90 | 1.00 | 0.50 | only 2 positive examples: poorly validated |

A sweep of the presence threshold (0.05-0.15) and count threshold (0.10-0.35) changed accuracy by at most 3 constraints, so the pre-set
values were kept. Not evaluated: size words (small/large), because box area is not a reliable size cue.
Known biases: OWL-ViT detects stylised or partial objects less reliably; CLIP colour on a crop is affected by background and lighting;
a failure to detect counts as a failure of the generation (false negatives deflate every success rate, most for binding).
