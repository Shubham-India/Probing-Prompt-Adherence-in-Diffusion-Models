# Variant review (H3): owner decisions (2026-10-18)

> **Status: answered by the owner on 2026-10-18 (Q1 = A, Q2 = A) and applied.** Q1: benchmark amendment DL-35 (`golden -> lemon`, `on the left/right side of -> on the left/right of`, CF-O `n_objects`); Q2: `src/dedup.py` + `configs/analysis_plan_r3.json` `duplicates` (DL-33). The text below is the decision sheet as presented (pre-amendment numbers); the pre-amendment copy is in `pre_amendment_20261018/`.

Everything else in the review is decided by rules that follow from the specification (`candidate_review.csv`, `candidate_summary.json`). Two choices are research-design choices and need your approval. Evidence is text-encoder proxies computed on CPU (no images); image-level confirmation is possible only after generation.

**Policy used (from the repository, not new):** PAR is meaning-preserving and keeps the constraints (blueprint §1806, "pre-registered synonym lexicon"). The judge scores a colour as the argmax over 10 basic colours (`src/evaluate.py` COLORS) and a relation from box positions. So a colour synonym must name a shade **inside** the canonical basic-colour category and add no object or material that changes a constraint; a relation phrase must keep the viewer's left/right. A narrower shade is acceptable (the judge is categorical), a different category is not.

## Q1. Lexicon amendment for two PAR synonyms (affects 77 FAIL candidates)
| Option | What changes | Consequence |
|---|---|---|
| **A (recommended)** | `yellow: golden -> lemon`; relation synonym `on the left/right side of -> on the left/right of`; in the same amendment, `n_objects` of CF-O rows becomes the prompt's own object count (DL-34) | 120 PAR prompt texts change (40 DEV, 80 TEST); ids, constraints, splits, scenes and all other prompts unchanged (simulated: 0 validation errors). New `prompts_v2.json`/`scene_graphs.json`/`lexicon.json` hashes; new `gbc-benchmark` upload; the 120 changed rows are re-reviewed (carry-over keeps every other decision). Projected rejection after amendment: 0% |
| B | Keep both synonyms | 74 PAR rows and 3 lexicon entries stay FAIL (4.4% < 20%), but a FAIL item may not be generated, so B still needs an amendment of some kind; PAR-yellow would measure the synonym (judge reads "golden" as brown), not the paraphrase |
| C | Use other words (e.g. `mustard`, `on the left of` vs `to the left side of`) | as A, with the alternative words |
Evidence:
- `golden`: nearest basic colour in the judge's CLIP is **brown** in 33/47 PAR prompts and yellow in only 10/47 (SD's encoder: yellow 47/47, smallest margin of all synonyms). Dictionary sense: the colour of gold (metallic), not basic yellow.
- `lemon`: yellow nearest in 47/47 (SD) and 45/47 (judge); the colour reading beats an extra-fruit reading for 40/40 nouns, like `tangerine`. `mustard` scores 47/47 in both, but is a darker brownish-yellow, the same failure direction as golden. `bright yellow` / `lemon-yellow` score highest but keep the canonical word, unlike the other seven synonyms.
- `on the left/right side of`: for 30 of 82 prompts the reference object has its own front and sides (car, chair, cow, bench, guitar, boat, dog, bus, cat), so "on the left side of a car" admits the car's own left and contact; the judge scores image-left. In SD's encoder the current phrase keeps the canonical direction in 76/82 prompts, `on the left/right of` in 82/82.
- The other six synonyms (crimson, azure, violet, tangerine, emerald, ivory) and `ebony` stay (proposed PASS; ivory/emerald/ebony with medium confidence because of material readings that do not change the colour).
- Files: `synonym_evidence.json`, `synonym_candidates.json`.

## Q2. Duplicate prompt texts (73 rows; benchmark unchanged under the recommendation)
Facts: identical text plus identical CFG, steps and seed plus the same noise draw gives **byte-identical images**. 128 setting groups, 156 redundant generations (0.6% of 25,083 image-equivalents); 32 groups span DEV and TEST (44 TEST generations identical to DEV ones; all CF-O single-object prompts such as "a shoe on a wooden table"). No confirmatory analysis (HA, HD, HE, HF) uses CF-O or CF-A images. One canonical pair, `sg_0091_CAN` (main) = `psD05_P_CAN` (PS), is pooled twice in the DEV HA constants; its TEST counterpart pair (`sg_0249_ORD` = `psT12_R_CAN`) does not enter HA twice.
| Option | What changes | Consequence |
|---|---|---|
| **A (recommended)** | Keep all items; pre-register one rule before the freeze: *every pooled estimate counts each distinct generation (normalised prompt text, CFG, steps, seed, model revision) once, keeping the first item id; TEST items whose text also occurs in DEV are excluded from TEST variant-level (S2) estimates; `psD05_P` is excluded from the DEV HA pool* | No benchmark or manifest change; the rule is added to the analysis plan and code before the freeze; 0.6% compute spent on duplicates |
| B | Remove the duplicate CF-O/CF-A items (keep the first per text and split; drop TEST items duplicating DEV) | Benchmark amendment: fewer items (counts, hashes, manifests change) |
| C | Keep everything, no rule | Duplicated observations get double weight in variant summaries; identical DEV/TEST images; not recommended |

## How to answer
Reply with e.g. `Q1=A, Q2=A` (or name the alternative). With A/A: the amendment is applied, the sheets are rebuilt with `--carry-over`, the 120 changed rows are re-proposed, and you then adopt the candidates:
`python scripts/adopt_candidate_review.py --reviewer "<your name>" --date <YYYY-MM-DD> --answers Q1=A,Q2=A`, followed by `python scripts/check_variant_review.py`.
Not decided here: the n_objects metadata (follows Q1, DL-34) and detector validity of nouns (evaluator audit / E22, not H3).
