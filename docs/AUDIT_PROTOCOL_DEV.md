# DEV evaluator audit protocol (E01) — predeclared before any label exists

Status: **DRAFT FOR OWNER SIGN-OFF**. It must be agreed, and committed or hash-recorded, before the first label is made. Nothing here freezes the research protocol or opens TEST. Claude has not seen, and must not be shown, any human label until the owner returns the files described in section 6.
Scope: judge J1 (OWL-ViT + CLIP-B/32) on the DEV core images of run `r2a-20261014T130457Z-daef6c4` (100 canonical DEV prompts x 4 seeds). J2 does not exist, so no judge comparison and no "primary judge selection" is possible now. Branch/edited images are not covered; they need their own audit stratum later (RB 12.7).

## 1. Population and sample (generated, seeded, reproducible)
| Item | Value |
|---|---|
| Unit | one (image, constraint) judgement |
| Population | 1,352 items in 400 images: presence 632, binding 500, count 108, spatial 112 (from `local_audit/R2a_checkpoint/eval_view/eval.jsonl`, sha256 in `key/manifest.json`) |
| Strata | type x J1 verdict x J1-uncertain flag (x J1 "all objects present" for binding and spatial); equal allocation over non-empty strata, capped by stratum size; seed 1 |
| Scored sample | **540 items**: presence 160, binding 160, count 108 (census), spatial 112 (census; only 18 J1-positive) |
| Practice round | 20 extra items shown first, never analysed (drawn from items not in the sample, so only presence/binding) |
| Double annotation | 100 items (25 per type) answered also by a second person |
| Images | 335 distinct opaque-named images (annotator A), 110 (annotator B) |
| Reproduction | `python src/audit_dev.py sample --eval <eval.jsonl> --images <images> --prompts data/benchmark_v2/prompts_v2.json --out local_audit/audit_dev --seed 1` |

Because strata are sampled unequally, all accuracy figures are **re-weighted to the 1,352-item population** (weight N_h / n_h) with a stratified bootstrap (2,000 draws, seed 0); unweighted Wilson intervals are shown beside them. Limitation: the bootstrap resamples items, not images or scenes, so intervals are slightly optimistic.

## 2. What the annotator sees (blinding)
Only an opaque image, a neutral question and answer buttons. No prediction, target colour/count/side, prompt, scene id, seed, CFG, model or arm appears in the page (`label.html` contains none; checked by a test and by search). Questions never state the requested value:
| Type | Question | Answers |
|---|---|---|
| presence | Is there at least one X? | yes / no / ambiguous |
| count | How many Xs are in the image? | 0-9, 10 or more, cannot tell |
| binding | What colour is the X? (several: colour of most) | 12 colours incl. "other colour", "no X visible" (key `n`), "mixed / ambiguous" (key `m`) |
| spatial | Which one is further to the LEFT? | the A / the B (random order) / one or both not visible / ambiguous |
The truth for each constraint is derived afterwards from the neutral answer (`audit_dev.truth`): presence yes; count equals the requested n; colour equals the requested colour; chosen object equals the requested left object. A missing object makes binding/spatial/count truth 0 (the legacy judge counts omissions as failures, so this is the like-for-like comparison); presence of the object is recorded separately.

## 3. Labelling rules (fixed before labelling; unclear means "ambiguous", never guess)
1. Work alone, without any AI tool, without discussing items, without looking at model outputs. Labels are recorded as human and not AI-assisted.
2. Present = a recognisable real instance of the class, even if partly cut off or occluded, as long as you would confidently name it. Toys, statues, drawings or pictures of the object inside the image: choose "ambiguous".
3. Colour = the dominant colour of the object's main surface as it would look in neutral light (ignore shadows and highlights). Patterned or multi-coloured with no dominant colour: "mixed / ambiguous". Colour not in the list (cyan, gold, silver, beige): "other colour". Wooden objects: brown.
4. Count = distinct whole objects you can separate; partly visible ones count if you can tell what they are; reflections and pictures do not; dense piles you cannot count: "cannot tell".
5. Left/right = image left, comparing horizontal centres; with several instances, use the most prominent one of each class.
6. The first 20 items are practice and are discarded. The rules above may be clarified only by adding a dated amendment line to this file **before** continuing, never after seeing any prediction.
7. Sessions: any length; progress is saved in the browser (do not clear site data); finish with "Download labels CSV".

## 4. Second annotator and second review
- **DEV double annotation:** annotator B must be a different person. If none is available, B may be the same person repeating the B package after at least 48 hours; this is reported as intra-rater agreement and is **not** inter-annotator kappa. Items on which A and B disagree are excluded from accuracy and listed; a third reader may resolve them if one exists.
- **Pilot validation labels (40 images, 125 constraints):** package `local_audit/audit_pilot40/annotator_A/label.html`. The reviewer must not be the person who edited `data/validation_labels.json`, must not have seen those labels or the J1 outputs, and answers the same neutral questions. Score with `python src/audit_dev.py score-pilot --key local_audit/audit_pilot40/key/audit_key.csv --b <labels.csv> --existing data/validation_labels.json --subset data/validation_subset.json`. Reported: raw agreement, Cohen's kappa, per-type agreement, each disagreement, and the reviewer's agreement with J1. Existing labels are **not** changed automatically; any change is made by the owner with a dated note.

## 5. Predeclared analysis (no judge is selected here)
Computed by `python src/audit_dev.py score --key local_audit/audit_dev/key/audit_key.csv --a labels_A.csv [--b labels_B.csv] --out report.json`, separately for presence, count, binding, spatial:
1. n items labelled, n used, n ambiguous/disagreement (excluded, reported).
2. Population-weighted accuracy of J1 against the human truth, stratified-bootstrap 95% CI, unweighted accuracy with Wilson 95% CI.
3. Sensitivity and specificity (weighted, with CIs) -> false-negative rate = 1 - sensitivity, false-positive rate = 1 - specificity.
4. Binding only: accuracy among items where J1 says every object is present and the human sees the object; and the number of items where J1 says "all present" but the human sees no such object.
5. Double annotation: raw agreement and kappa on the 100 items.
**Criterion (RB 9.0/12.5, unchanged):** a constraint type is eligible for confirmatory use only if its weighted accuracy point estimate is at least **0.85**; otherwise it stays EXPLORATORY. No other threshold, CI-based rule or metric is added after seeing data. If the owner wants a stricter rule (for example CI lower bound), it must be written here first.
**Object-level detectability (for the lexicon review):** from the presence items, report J1 vs human agreement per object noun with n; nouns with fewer than 10 items are listed as "insufficient". Nouns below 0.85 are flagged for review, not removed.
**Use of the result in D-13:** the false-negative/false-positive rates of the presence and binding types are passed to `src/dev_gate_observed.py --err-presence FN FP --err-bind FN FP`. Error rates of the joint outcome are a proxy for those of the conditional outcome; the simulation states this.

## 6. What the owner does and returns
1. Open `local_audit/audit_dev/annotator_A/label.html` in a browser (double-click). Do **not** open anything in `local_audit/audit_dev/key/`.
2. Enter annotator ID (initials), label 20 practice + 540 scored items (about 560 clicks; shortcuts: number keys, `n` = no object visible, `m` = ambiguous, `b` = back).
3. Click "Download labels CSV" (also automatic at the end). Save it unedited as `local_audit/audit_dev/labels/labels_A.csv`.
4. Annotator B (different person) does the same with `local_audit/audit_dev/annotator_B/label.html` (120 items incl. practice) -> `labels_B.csv`.
5. The pilot-40 reviewer does `local_audit/audit_pilot40/annotator_A/label.html` (125 items) -> `labels_pilot40.csv`.
6. **Size judge (Revision 3, added 2026-10-19; same rules as above):** `local_audit/audit_size_dev/annotator_A/label.html` (176 items: census of every DEV size judgement in the verified DEV_CORE run; question "which one looks LARGER") -> `local_audit/audit_size_dev/labels/labels_size_A.csv`; the re-annotation subset `annotator_B/label.html` (40 items; a different person, or the same person after >= 48 h, reported as intra-rater) -> `labels_size_B.csv`. Built by `python scripts/build_size_audit.py --dev-core <verified DEV_CORE folder>`. Criterion unchanged: size is confirmatory only with >= 140 usable judgements and weighted accuracy >= 0.85 (Addendum C); otherwise size results are EXPLORATORY. Score: `python src/audit_dev.py score --key local_audit/audit_size_dev/key/audit_key.csv --a labels_size_A.csv --b labels_size_B.csv --out local_audit/audit_size_dev/size_audit_report.json`.
7. Tell Claude when the files are in place. Claude then runs the scoring commands, nothing else.

## 7. Synthetic detector probe (E22) — specification only (not run)
| Item | Specification |
|---|---|
| Purpose | Ceiling/sanity check of the J1 code path and thresholds on scenes with exact truth. It does **not** validate J1 on natural images and does not certify the benchmark nouns (flat vector shapes stand in for objects). |
| Generator | `python src/synth_probe.py --out local_audit/synth_probe --n 200 --seed 0 --export-judge-run`: 200 scenes of 2-3 coloured shapes (cube=square, sphere=circle, cone=triangle), 8 colours, optional counts of 2-3, constraints (presence, binding, count, spatial) with a seeded fraction deliberately wrong; truth computed from the scene description, not from pixels |
| Judge run | `python src/evaluate.py --run local_audit/synth_probe --prompts local_audit/synth_probe/prompts_synth.json --conditional` (needs the detector models; inference on 200 small images; **a GPU/CPU job: not run in this session**) |
| Scoring | `python src/synth_probe.py --out local_audit/synth_probe --score eval.jsonl` -> per-type accuracy and confusion |
| Reading | Accuracy below 0.95 on any type signals a code or threshold defect to investigate before anything is concluded; at or above 0.95 means only that the pipeline works on trivial scenes |
| Gate status | The 0.95 floor is a proposal needing owner approval; it is not part of the lexicon or primary-judge criteria |

## 8. After the labels (Claude's next steps, in order)
1. `score` for DEV; `score-pilot` for the pilot reviewer; report per-type tables and the measured error rates.
2. Re-run the D-13 sizing with the measured rates: `python src/dev_gate_observed.py --eval <eval.jsonl> --prompts data/benchmark_v2/prompts_v2.json --bases 2 4 --M 2 --delta 0 6 12 --err-presence FN FP --err-bind FN FP --reps 40 --n-boot 500`. Scenes are the bootstrap unit; seeds are never counted as scenes; the four-step gate is not altered.
3. Reassess whether R3 is worth running as an exploratory rehearsal, and update the decision register and session state. No protocol freeze.
