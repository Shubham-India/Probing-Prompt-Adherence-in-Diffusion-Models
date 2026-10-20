# Owner review packet: variant and lexicon review (H3)

Candidate decisions are proposals derived from rules, evidence and the approved owner answers (Q1 = A, Q2 = A, 2026-10-18). **None of them is your decision until you adopt it.** Check each group below (the examples are representative; open `candidate_review.csv` filtered on `rules` for all rows), then adopt the groups you agree with (command at the end). Rows you disagree with: put their item ids in `--exclude` and decide them in the sheets yourself.

**What the confidence means.** `high` = every automatic check passes (text vs constraints, variant definition, grammar, duplicates, confusable pairs; checks verified to fire on broken prompts). The checks cannot judge naturalness, odd wording or a misleading reading: read at least the examples of every group and a random sample of the large R-OK group before adopting it.

Prompt rows: 1674 (proposed PASS 1674); lexicon rows: 73 (proposed PASS 73); projected rejection rate 0.0% (criterion < 20%). Rows changed by the 2026-10-18 amendment: 120 PAR prompts (dispositions of the 77 former FAIL candidates: `amendment_dispositions.csv`).

| # | Sheet | Rules | Decision | Confidence | Depends on | Rows |
|---|---|---|---|---|---|---|
| G01 | lexicon_review | R-LEX-NOUN | PASS | high | - | 40 |
| G02 | lexicon_review | R-LEX-CONFLICT | PASS | high | - | 12 |
| G03 | lexicon_review | R-LEX-COLOUR | PASS | high | - | 8 |
| G04 | lexicon_review | R-LEX-REL | PASS | high | - | 2 |
| G05 | lexicon_review | R-REL-SYN | PASS | medium | Q1 | 2 |
| G06 | lexicon_review | R-SYN-crimson | PASS | high | Q1 | 1 |
| G07 | lexicon_review | R-SYN-azure | PASS | high | Q1 | 1 |
| G08 | lexicon_review | R-SYN-lemon | PASS | medium | Q1 | 1 |
| G09 | lexicon_review | R-SYN-emerald | PASS | medium | Q1 | 1 |
| G10 | lexicon_review | R-SYN-tangerine | PASS | high | Q1 | 1 |
| G11 | lexicon_review | R-SYN-violet | PASS | high | Q1 | 1 |
| G12 | lexicon_review | R-SYN-ivory | PASS | medium | Q1 | 1 |
| G13 | lexicon_review | R-SYN-ebony | PASS | medium | Q1 | 1 |
| G14 | lexicon_review | R-LEX-NSYN | PASS | high | - | 1 |
| G15 | variant_review | R-OK | PASS | high | - | 1346 |
| G16 | variant_review | R-DUP | PASS | medium | Q2 | 53 |
| G17 | variant_review | R-OK; R-REL-SYN | PASS | medium | Q1 | 45 |
| G18 | variant_review | R-SYN-ebony; R-SYN-lemon | PASS | medium | Q1 | 32 |
| G19 | variant_review | R-SYN-azure; R-SYN-ivory | PASS | medium | Q1 | 22 |
| G20 | variant_review | R-SYN-violet; R-SYN-crimson | PASS | high | Q1 | 21 |
| G21 | variant_review | R-SYN-emerald; R-SYN-tangerine | PASS | medium | Q1 | 21 |
| G22 | variant_review | R-DUP-CROSS | PASS | medium | Q2 | 20 |
| G23 | variant_review | R-SYN-tangerine; R-SYN-crimson | PASS | high | Q1 | 11 |
| G24 | variant_review | R-SYN-ivory; R-SYN-violet | PASS | medium | Q1 | 11 |
| G25 | variant_review | R-SYN-azure; R-SYN-emerald | PASS | medium | Q1 | 10 |
| G26 | variant_review | R-SYN-ebony; R-SYN-lemon; R-REL-SYN | PASS | medium | Q1 | 9 |
| G27 | variant_review | R-SYN-violet; R-SYN-crimson; R-REL-SYN | PASS | medium | Q1 | 7 |
| G28 | variant_review | R-SYN-ebony | PASS | medium | Q1 | 6 |
| G29 | variant_review | R-SYN-lemon | PASS | medium | Q1 | 6 |
| G30 | variant_review | R-SYN-tangerine | PASS | high | Q1 | 6 |
| G31 | variant_review | R-SYN-ivory | PASS | medium | Q1 | 6 |
| G32 | variant_review | R-SYN-azure | PASS | high | Q1 | 6 |
| G33 | variant_review | R-SYN-emerald; R-SYN-tangerine; R-REL-SYN | PASS | medium | Q1 | 6 |
| G34 | variant_review | R-OK; R-SYN-crimson | PASS | high | Q1 | 5 |
| G35 | variant_review | R-SYN-violet | PASS | high | Q1 | 5 |
| G36 | variant_review | R-SYN-emerald | PASS | medium | Q1 | 5 |
| G37 | variant_review | R-SYN-azure; R-SYN-ivory; R-REL-SYN | PASS | medium | Q1 | 5 |
| G38 | variant_review | R-SYN-azure; R-SYN-emerald; R-REL-SYN | PASS | medium | Q1 | 4 |
| G39 | variant_review | R-SYN-ivory; R-SYN-violet; R-REL-SYN | PASS | medium | Q1 | 3 |
| G40 | variant_review | R-SYN-tangerine; R-SYN-crimson; R-REL-SYN | PASS | medium | Q1 | 3 |

## G01: R-LEX-NOUN (lexicon_review, 40 rows, proposed PASS, confidence high)

**Reason (first row):** plural 'dogs' is correct English; detector validity of the noun is assessed by the evaluator audit / E22, not by H3

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-NOUN-apple |  | noun | apple -> apples |  |
| LEX-NOUN-dog |  | noun | dog -> dogs |  |
| LEX-NOUN-cat |  | noun | cat -> cats |  |
| LEX-NOUN-horse |  | noun | horse -> horses |  |

## G02: R-LEX-CONFLICT (lexicon_review, 12 rows, proposed PASS, confidence high)

**Reason (first row):** confusable pair kept out of every scene (co-occurs in 0 prompts)

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-CONFLICT-apple-ball |  | conflict_pair | apple -> ball |  |
| LEX-CONFLICT-box-cube |  | conflict_pair | box -> cube |  |
| LEX-CONFLICT-ball-sphere |  | conflict_pair | ball -> sphere |  |
| LEX-CONFLICT-bus-car |  | conflict_pair | bus -> car |  |

## G03: R-LEX-COLOUR (lexicon_review, 8 rows, proposed PASS, confidence high)

**Reason (first row):** basic colour term; also a judge colour (src/evaluate.py COLORS)

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-COLOUR-black |  | colour | black -> black |  |
| LEX-COLOUR-red |  | colour | red -> red |  |
| LEX-COLOUR-blue |  | colour | blue -> blue |  |
| LEX-COLOUR-yellow |  | colour | yellow -> yellow |  |

## G04: R-LEX-REL (lexicon_review, 2 rows, proposed PASS, confidence high)

**Reason (first row):** 'to the left of' is the canonical relation phrase; the judge scores image left/right; direction checked in every prompt

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-REL-left_of |  | relation | left_of -> to the left of |  |
| LEX-REL-right_of |  | relation | right_of -> to the right of |  |

## G05: R-REL-SYN (lexicon_review, 2 rows, proposed PASS, confidence medium)

**Reason (first row):** 'on the left of' replaces 'side of' (amendment 2026-10-18, Q1=A): viewer frame; canonical direction kept in 82/82 PAR prompts (SD text encoder)

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-REL-left_of_synonym (amended) |  | relation | left_of_synonym -> on the left of |  |
| LEX-REL-right_of_synonym (amended) |  | relation | right_of_synonym -> on the right of |  |

## G06: R-SYN-crimson (lexicon_review, 1 rows, proposed PASS, confidence high)

**Reason (first row):** deep red: inside red; blueprint's own PAR example; nearest basic colour red in all occurrences in both text encoders

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-CSYN-red |  | colour_synonym | red -> crimson | darker red; usually acceptable |

## G07: R-SYN-azure (lexicon_review, 1 rows, proposed PASS, confidence high)

**Reason (first row):** bright/sky blue: inside blue; nearest basic colour blue in 47/47 (SD) and 46/47 (judge CLIP) amended PAR prompts

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-CSYN-blue |  | colour_synonym | blue -> azure | light/sky blue, narrower than blue |

## G08: R-SYN-lemon (lexicon_review, 1 rows, proposed PASS, confidence medium)

**Reason (first row):** lemon (focal) yellow: inside yellow; nearest yellow in 47/47 (SD) and 45/47 (judge CLIP) PAR prompts; colour reading beats an extra-fruit reading for 40/40 nouns (proposed replacement for 'golden', OWNER_DECISIONS Q1)

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-CSYN-yellow (amended) |  | colour_synonym | yellow -> lemon | also a fruit noun (replaces 'golden', amendment 2026-10-18) |

## G09: R-SYN-emerald (lexicon_review, 1 rows, proposed PASS, confidence medium)

**Reason (first row):** bright green: inside green; nearest green in all occurrences in both encoders; gemstone reading about as strong as the colour reading in the judge's CLIP (material only, colour unchanged)

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-CSYN-green |  | colour_synonym | green -> emerald | gemstone; narrower hue than green |

## G10: R-SYN-tangerine (lexicon_review, 1 rows, proposed PASS, confidence high)

**Reason (first row):** reddish orange: inside orange; nearest orange in all occurrences in both encoders; colour reading stronger than the fruit reading

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-CSYN-orange |  | colour_synonym | orange -> tangerine | also a fruit noun |

## G11: R-SYN-violet (lexicon_review, 1 rows, proposed PASS, confidence high)

**Reason (first row):** bluish purple: inside purple; nearest purple in all occurrences in both encoders; colour reading stronger than the flower reading

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-CSYN-purple |  | colour_synonym | purple -> violet | also a flower noun |

## G12: R-SYN-ivory (lexicon_review, 1 rows, proposed PASS, confidence medium)

**Reason (first row):** off-white: inside white; nearest white in SD for all and in the judge's CLIP for 44/47 occurrences; strong 'made of ivory' reading (material only, still white)

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-CSYN-white |  | colour_synonym | white -> ivory | material connotation; off-white |

## G13: R-SYN-ebony (lexicon_review, 1 rows, proposed PASS, confidence medium)

**Reason (first row):** very dark/black: inside black; nearest black in SD for all, in the judge's CLIP for 42/47 (5 brown); colour reading stronger than the wood reading

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-CSYN-black |  | colour_synonym | black -> ebony | material (wood) connotation |

## G14: R-LEX-NSYN (lexicon_review, 1 rows, proposed PASS, confidence high)

**Reason (first row):** 'bike' is the common synonym of 'bicycle' and the same detector object

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| LEX-NSYN-bicycle |  | noun_synonym | bicycle -> bike |  |

## G15: R-OK (variant_review, 1346 rows, proposed PASS, confidence high)

**Reason (first row):** text expresses exactly the listed constraints and nothing else; the variant makes only its defined change; grammar checks pass; unusual colour-noun pairs are intended for binding (DATASHEET)

**Evidence:** build_variant_review text/change/grammar/duplicate/conflict checks: no flag

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| psD00_CR_CAN | DEV | CAN | a yellow hat to the left of a white vase on a wooden table |  |
| sg_0001_CFA | DEV | CF-A | an orange book and a cup |  |
| psD00_CSR_CFC | DEV | CF-C | a small white hat to the left of a large yellow vase on a wooden table |  |
| sg_0032_CFN | DEV | CF-N | three black kites and a yellow teapot on a concrete floor |  |
| sg_0071_CFO | DEV | CF-O | a basket |  |
| psD00_CSR_CFR | DEV | CF-R | a small yellow hat to the right of a large white vase on a wooden table |  |

## G16: R-DUP (variant_review, 53 rows, proposed PASS, confidence medium)

**Reason (first row):** prompt is valid; its text equals sg_0062_CFA (DEV); identical generation settings with 1 of them -> byte-identical images; no confirmatory analysis (HA/HD/HE/HF) uses CF-O/CF-A images; handled by the Q2 de-duplication rule for variant-level summaries

**Evidence:** 1 same-text item(s); shared settings: 1; splits ['DEV']

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| psD05_P_CAN | DEV | CAN | an apple and a pillow on a wooden table | DUPLICATE: same text as sg_0091_CAN (DEV) |
| sg_0057_CFA | DEV | CF-A | two laptops on a wooden table | DUPLICATE: same text as sg_0062_CFA (DEV) |
| sg_0072_CFO | DEV | CF-O | a ball on a wooden table | DUPLICATE: same text as sg_0099_CFO (DEV), sg_0100_CFO (DEV) |
| psT12_R_CAN | TEST | CAN | a book to the right of a hat on a wooden table | DUPLICATE: same text as sg_0249_ORD (TEST) |
| sg_0213_CFA | TEST | CF-A | two horses on a concrete floor | DUPLICATE: same text as sg_0233_CFA (TEST) |
| sg_0241_CFO | TEST | CF-O | a bowl on a wooden table | DUPLICATE: same text as sg_0297_CFO (TEST) |

## G17: R-OK; R-REL-SYN (variant_review, 45 rows, proposed PASS, confidence medium)

**Reason (first row):** text expresses exactly the listed constraints and nothing else; the variant makes only its defined change; grammar checks pass \| relation synonym 'on the right of' (amendment 2026-10-18, Q1=A): viewer-frame phrase without 'side of'; keeps the canonical direction in 82/82 PAR prompts in SD's text encoder (synonym_candidates.json); direction checked against the constraint

**Evidence:** build_variant_review text/change/grammar/duplicate/conflict checks: no flag

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0071_PAR (amended) | DEV | PAR | a basket on the right of a bus |  |
| sg_0241_PAR (amended) | TEST | PAR | a bowl on the right of a bench on a wooden table |  |
| sg_0072_PAR | DEV | PAR | a ball on the left of an umbrella on a wooden table |  |
| sg_0242_PAR | TEST | PAR | a hat on the right of a basket |  |

## G18: R-SYN-ebony; R-SYN-lemon (variant_review, 32 rows, proposed PASS, confidence medium)

**Reason (first row):** 'ebony': very dark/black: inside black; nearest black in SD for all, in the judge's CLIP for 42/47 (5 brown); colour reading stronger than the wood reading \| 'lemon': lemon (focal) yellow: inside yellow; nearest yellow in 47/47 (SD) and 45/47 (judge CLIP) PAR prompts; colour reading beats an extra-fruit reading for 40/40 nouns (proposed replacement for 'golden', OWNER_DECISIONS Q1)

**Evidence:** ebony: SD nearest {'black': 47}, judge nearest {'black': 42, 'brown': 5} \| lemon: SD nearest {'yellow': 47}, judge nearest {'yellow': 45, 'white': 1, 'black': 1}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0004_PAR (amended) | DEV | PAR | an ebony umbrella and a lemon pillow on a wooden table | SYNONYM-CHECK: ['ebony', 'lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0101_PAR (amended) | TEST | PAR | an ebony bottle and a lemon chair | SYNONYM-CHECK: ['ebony', 'lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0008_PAR (amended) | DEV | PAR | an ebony book and a lemon sphere on a concrete floor | SYNONYM-CHECK: ['ebony', 'lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0012_PAR (amended) | DEV | PAR | an ebony cow and a lemon basket | SYNONYM-CHECK: ['ebony', 'lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G19: R-SYN-azure; R-SYN-ivory (variant_review, 22 rows, proposed PASS, confidence medium)

**Reason (first row):** 'azure': bright/sky blue: inside blue; nearest basic colour blue in 47/47 (SD) and 46/47 (judge CLIP) amended PAR prompts \| 'ivory': off-white: inside white; nearest white in SD for all and in the judge's CLIP for 44/47 occurrences; strong 'made of ivory' reading (material only, still white)

**Evidence:** azure: SD nearest {'blue': 47}, judge nearest {'blue': 46, 'green': 1} \| ivory: SD nearest {'white': 47}, judge nearest {'white': 44, 'blue': 2, 'brown': 1}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0103_PAR | TEST | PAR | an azure guitar and an ivory hat | SYNONYM-CHECK: ['azure', 'ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0107_PAR | TEST | PAR | an azure dog and an ivory bus | SYNONYM-CHECK: ['azure', 'ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0111_PAR | TEST | PAR | an azure guitar and an ivory backpack on a concrete floor | SYNONYM-CHECK: ['azure', 'ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0115_PAR | TEST | PAR | an azure boat and an ivory bowl | SYNONYM-CHECK: ['azure', 'ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G20: R-SYN-violet; R-SYN-crimson (variant_review, 21 rows, proposed PASS, confidence high)

**Reason (first row):** 'violet': bluish purple: inside purple; nearest purple in all occurrences in both encoders; colour reading stronger than the flower reading \| 'crimson': deep red: inside red; blueprint's own PAR example; nearest basic colour red in all occurrences in both text encoders

**Evidence:** violet: SD nearest {'purple': 47}, judge nearest {'purple': 47} \| crimson: SD nearest {'red': 47}, judge nearest {'red': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0102_PAR | TEST | PAR | a violet cow and a crimson boat on a concrete floor | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0106_PAR | TEST | PAR | a violet guitar and a crimson shoe on a wooden table | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0110_PAR | TEST | PAR | a violet bowl and a crimson hat on a concrete floor | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0114_PAR | TEST | PAR | a violet bottle and a crimson hat | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G21: R-SYN-emerald; R-SYN-tangerine (variant_review, 21 rows, proposed PASS, confidence medium)

**Reason (first row):** 'emerald': bright green: inside green; nearest green in all occurrences in both encoders; gemstone reading about as strong as the colour reading in the judge's CLIP (material only, colour unchanged) \| 'tangerine': reddish orange: inside orange; nearest orange in all occurrences in both encoders; colour reading stronger than the fruit reading

**Evidence:** emerald: SD nearest {'green': 46}, judge nearest {'green': 46} \| tangerine: SD nearest {'orange': 47}, judge nearest {'orange': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0104_PAR | TEST | PAR | an emerald car and a tangerine shoe | SYNONYM-CHECK: ['emerald', 'tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0108_PAR | TEST | PAR | an emerald candle and a tangerine shoe | SYNONYM-CHECK: ['emerald', 'tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0112_PAR | TEST | PAR | an emerald bench and a tangerine pillow on a concrete floor | SYNONYM-CHECK: ['emerald', 'tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0116_PAR | TEST | PAR | an emerald dog and a tangerine ball | SYNONYM-CHECK: ['emerald', 'tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G22: R-DUP-CROSS (variant_review, 20 rows, proposed PASS, confidence medium)

**Reason (first row):** prompt is valid; its text equals sg_0269_CFO (TEST); identical generation settings with 1 of them -> byte-identical images; no confirmatory analysis (HA/HD/HE/HF) uses CF-O/CF-A images; handled by the Q2 de-duplication rule for variant-level summaries

**Evidence:** 1 same-text item(s); shared settings: 1; splits ['TEST']

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0079_CFO | DEV | CF-O | a shoe on a wooden table | DUPLICATE-CROSS-SPLIT: same text as sg_0269_CFO (TEST) |
| sg_0252_CFO | TEST | CF-O | a shoe | DUPLICATE-CROSS-SPLIT: same text as sg_0086_CFO (DEV), sg_0294_CFO (TEST) |
| sg_0081_CFO | DEV | CF-O | an apple on a wooden table | DUPLICATE-CROSS-SPLIT: same text as sg_0091_CFO (DEV), sg_0262_CFO (TEST) |
| sg_0084_CFO | DEV | CF-O | a book on a concrete floor | DUPLICATE-CROSS-SPLIT: same text as sg_0260_CFO (TEST) |

## G23: R-SYN-tangerine; R-SYN-crimson (variant_review, 11 rows, proposed PASS, confidence high)

**Reason (first row):** 'tangerine': reddish orange: inside orange; nearest orange in all occurrences in both encoders; colour reading stronger than the fruit reading \| 'crimson': deep red: inside red; blueprint's own PAR example; nearest basic colour red in all occurrences in both text encoders

**Evidence:** tangerine: SD nearest {'orange': 47}, judge nearest {'orange': 47} \| crimson: SD nearest {'red': 47}, judge nearest {'red': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0001_PAR | DEV | PAR | a tangerine book and a crimson cup | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0005_PAR | DEV | PAR | a tangerine teapot and a crimson pillow on a concrete floor | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0009_PAR | DEV | PAR | a tangerine bottle and a crimson umbrella on a concrete floor | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0013_PAR | DEV | PAR | a tangerine pillow and a crimson hat on a wooden table | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G24: R-SYN-ivory; R-SYN-violet (variant_review, 11 rows, proposed PASS, confidence medium)

**Reason (first row):** 'ivory': off-white: inside white; nearest white in SD for all and in the judge's CLIP for 44/47 occurrences; strong 'made of ivory' reading (material only, still white) \| 'violet': bluish purple: inside purple; nearest purple in all occurrences in both encoders; colour reading stronger than the flower reading

**Evidence:** ivory: SD nearest {'white': 47}, judge nearest {'white': 44, 'blue': 2, 'brown': 1} \| violet: SD nearest {'purple': 47}, judge nearest {'purple': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0002_PAR | DEV | PAR | an ivory cat and a violet umbrella on a wooden table | SYNONYM-CHECK: ['ivory', 'violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0006_PAR | DEV | PAR | an ivory pillow and a violet cup on a concrete floor | SYNONYM-CHECK: ['ivory', 'violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0010_PAR | DEV | PAR | an ivory pillow and a violet candle on a concrete floor | SYNONYM-CHECK: ['ivory', 'violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0014_PAR | DEV | PAR | an ivory basket and a violet candle on a wooden table | SYNONYM-CHECK: ['ivory', 'violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G25: R-SYN-azure; R-SYN-emerald (variant_review, 10 rows, proposed PASS, confidence medium)

**Reason (first row):** 'azure': bright/sky blue: inside blue; nearest basic colour blue in 47/47 (SD) and 46/47 (judge CLIP) amended PAR prompts \| 'emerald': bright green: inside green; nearest green in all occurrences in both encoders; gemstone reading about as strong as the colour reading in the judge's CLIP (material only, colour unchanged)

**Evidence:** azure: SD nearest {'blue': 47}, judge nearest {'blue': 46, 'green': 1} \| emerald: SD nearest {'green': 46}, judge nearest {'green': 46}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0003_PAR | DEV | PAR | an azure chair and an emerald cup | SYNONYM-CHECK: ['azure', 'emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0007_PAR | DEV | PAR | an azure backpack and an emerald chair | SYNONYM-CHECK: ['azure', 'emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0011_PAR | DEV | PAR | an azure boat and an emerald hat on a concrete floor | SYNONYM-CHECK: ['azure', 'emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0015_PAR | DEV | PAR | an azure kite and an emerald hat on a wooden table | SYNONYM-CHECK: ['azure', 'emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G26: R-SYN-ebony; R-SYN-lemon; R-REL-SYN (variant_review, 9 rows, proposed PASS, confidence medium)

**Reason (first row):** 'ebony': very dark/black: inside black; nearest black in SD for all, in the judge's CLIP for 42/47 (5 brown); colour reading stronger than the wood reading \| 'lemon': lemon (focal) yellow: inside yellow; nearest yellow in 47/47 (SD) and 45/47 (judge CLIP) PAR prompts; colour reading beats an extra-fruit reading for 40/40 nouns (proposed replacement for 'golden', OWNER_DECISIONS Q1) \| relation synonym 'on the right of' (amendment 2026-10-18, Q1=A): viewer-frame phrase without 'side of'; keeps the canonical direction in 82/82 PAR prompts in SD's text encoder (synonym_candidates.json); direction checked against the constraint

**Evidence:** ebony: SD nearest {'black': 47}, judge nearest {'black': 42, 'brown': 5} \| lemon: SD nearest {'yellow': 47}, judge nearest {'yellow': 45, 'white': 1, 'black': 1}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0036_PAR (amended) | DEV | PAR | an ebony sphere on the right of a lemon basket | SYNONYM-CHECK: ['ebony', 'lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0177_PAR (amended) | TEST | PAR | an ebony dog on the left of a lemon cow on a wooden table | SYNONYM-CHECK: ['ebony', 'lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0040_PAR (amended) | DEV | PAR | an ebony backpack on the right of a lemon candle | SYNONYM-CHECK: ['ebony', 'lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0052_PAR (amended) | DEV | PAR | an ebony apple on the left of a lemon candle on a concrete floor | SYNONYM-CHECK: ['ebony', 'lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G27: R-SYN-violet; R-SYN-crimson; R-REL-SYN (variant_review, 7 rows, proposed PASS, confidence medium)

**Reason (first row):** 'violet': bluish purple: inside purple; nearest purple in all occurrences in both encoders; colour reading stronger than the flower reading \| 'crimson': deep red: inside red; blueprint's own PAR example; nearest basic colour red in all occurrences in both text encoders \| relation synonym 'on the right of' (amendment 2026-10-18, Q1=A): viewer-frame phrase without 'side of'; keeps the canonical direction in 82/82 PAR prompts in SD's text encoder (synonym_candidates.json); direction checked against the constraint

**Evidence:** violet: SD nearest {'purple': 47}, judge nearest {'purple': 47} \| crimson: SD nearest {'red': 47}, judge nearest {'red': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0166_PAR (amended) | TEST | PAR | a violet cup on the left of a crimson boat | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0162_PAR | TEST | PAR | a violet boat on the right of a crimson ball on a concrete floor | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0174_PAR | TEST | PAR | a violet basket on the left of a crimson ball on a wooden table | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0178_PAR | TEST | PAR | a violet cow on the right of a crimson candle | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G28: R-SYN-ebony (variant_review, 6 rows, proposed PASS, confidence medium)

**Reason (first row):** 'ebony': very dark/black: inside black; nearest black in SD for all, in the judge's CLIP for 42/47 (5 brown); colour reading stronger than the wood reading

**Evidence:** ebony: SD nearest {'black': 47}, judge nearest {'black': 42, 'brown': 5}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0056_PAR | DEV | PAR | three ebony vases on a wooden table | SYNONYM-CHECK: ['ebony'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0215_PAR | TEST | PAR | two ebony sheep on a wooden table | SYNONYM-CHECK: ['ebony'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0064_PAR | DEV | PAR | two ebony suitcases on a concrete floor | SYNONYM-CHECK: ['ebony'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0223_PAR | TEST | PAR | four ebony horses | SYNONYM-CHECK: ['ebony'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G29: R-SYN-lemon (variant_review, 6 rows, proposed PASS, confidence medium)

**Reason (first row):** 'lemon': lemon (focal) yellow: inside yellow; nearest yellow in 47/47 (SD) and 45/47 (judge CLIP) PAR prompts; colour reading beats an extra-fruit reading for 40/40 nouns (proposed replacement for 'golden', OWNER_DECISIONS Q1)

**Evidence:** lemon: SD nearest {'yellow': 47}, judge nearest {'yellow': 45, 'white': 1, 'black': 1}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0057_PAR (amended) | DEV | PAR | two lemon laptops on a wooden table | SYNONYM-CHECK: ['lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0216_PAR (amended) | TEST | PAR | two lemon forks on a wooden table | SYNONYM-CHECK: ['lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0065_PAR (amended) | DEV | PAR | three lemon cubes | SYNONYM-CHECK: ['lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0224_PAR (amended) | TEST | PAR | four lemon bananas | SYNONYM-CHECK: ['lemon'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G30: R-SYN-tangerine (variant_review, 6 rows, proposed PASS, confidence high)

**Reason (first row):** 'tangerine': reddish orange: inside orange; nearest orange in all occurrences in both encoders; colour reading stronger than the fruit reading

**Evidence:** tangerine: SD nearest {'orange': 47}, judge nearest {'orange': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0058_PAR | DEV | PAR | two tangerine boxes on a concrete floor | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0214_PAR | TEST | PAR | three tangerine bananas on a concrete floor | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0066_PAR | DEV | PAR | three tangerine vases on a concrete floor | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0222_PAR | TEST | PAR | three tangerine bikes on a concrete floor | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G31: R-SYN-ivory (variant_review, 6 rows, proposed PASS, confidence medium)

**Reason (first row):** 'ivory': off-white: inside white; nearest white in SD for all and in the judge's CLIP for 44/47 occurrences; strong 'made of ivory' reading (material only, still white)

**Evidence:** ivory: SD nearest {'white': 47}, judge nearest {'white': 44, 'blue': 2, 'brown': 1}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0060_PAR | DEV | PAR | four ivory cubes on a concrete floor | SYNONYM-CHECK: ['ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0212_PAR | TEST | PAR | two ivory bikes on a concrete floor | SYNONYM-CHECK: ['ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0068_PAR | DEV | PAR | two ivory boxes on a concrete floor | SYNONYM-CHECK: ['ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0220_PAR | TEST | PAR | three ivory clocks on a wooden table | SYNONYM-CHECK: ['ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G32: R-SYN-azure (variant_review, 6 rows, proposed PASS, confidence high)

**Reason (first row):** 'azure': bright/sky blue: inside blue; nearest basic colour blue in 47/47 (SD) and 46/47 (judge CLIP) amended PAR prompts

**Evidence:** azure: SD nearest {'blue': 47}, judge nearest {'blue': 46, 'green': 1}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0062_PAR | DEV | PAR | two azure laptops on a wooden table | SYNONYM-CHECK: ['azure'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0211_PAR | TEST | PAR | four azure mirrors on a concrete floor | SYNONYM-CHECK: ['azure'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0070_PAR | DEV | PAR | four azure cubes on a concrete floor | SYNONYM-CHECK: ['azure'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0219_PAR | TEST | PAR | three azure skateboards | SYNONYM-CHECK: ['azure'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G33: R-SYN-emerald; R-SYN-tangerine; R-REL-SYN (variant_review, 6 rows, proposed PASS, confidence medium)

**Reason (first row):** 'emerald': bright green: inside green; nearest green in all occurrences in both encoders; gemstone reading about as strong as the colour reading in the judge's CLIP (material only, colour unchanged) \| 'tangerine': reddish orange: inside orange; nearest orange in all occurrences in both encoders; colour reading stronger than the fruit reading \| relation synonym 'on the left of' (amendment 2026-10-18, Q1=A): viewer-frame phrase without 'side of'; keeps the canonical direction in 82/82 PAR prompts in SD's text encoder (synonym_candidates.json); direction checked against the constraint

**Evidence:** emerald: SD nearest {'green': 46}, judge nearest {'green': 46} \| tangerine: SD nearest {'orange': 47}, judge nearest {'orange': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0172_PAR (amended) | TEST | PAR | an emerald chair on the right of a tangerine guitar | SYNONYM-CHECK: ['emerald', 'tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0168_PAR | TEST | PAR | an emerald boat on the left of a tangerine apple | SYNONYM-CHECK: ['emerald', 'tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0188_PAR (amended) | TEST | PAR | an emerald car on the right of a tangerine boat on a concrete floor | SYNONYM-CHECK: ['emerald', 'tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0192_PAR (amended) | TEST | PAR | an emerald basket on the left of a tangerine chair on a concrete floor | SYNONYM-CHECK: ['emerald', 'tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G34: R-OK; R-SYN-crimson (variant_review, 5 rows, proposed PASS, confidence high)

**Reason (first row):** text expresses exactly the listed constraints and nothing else; the variant makes only its defined change; grammar checks pass \| 'crimson': deep red: inside red; blueprint's own PAR example; nearest basic colour red in all occurrences in both text encoders

**Evidence:** build_variant_review text/change/grammar/duplicate/conflict checks: no flag \| crimson: SD nearest {'red': 47}, judge nearest {'red': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0059_PAR | DEV | PAR | four crimson suitcases |  |
| sg_0218_PAR | TEST | PAR | four crimson birds on a concrete floor |  |
| sg_0067_PAR | DEV | PAR | two crimson laptops |  |
| sg_0226_PAR | TEST | PAR | two crimson forks on a concrete floor |  |

## G35: R-SYN-violet (variant_review, 5 rows, proposed PASS, confidence high)

**Reason (first row):** 'violet': bluish purple: inside purple; nearest purple in all occurrences in both encoders; colour reading stronger than the flower reading

**Evidence:** violet: SD nearest {'purple': 47}, judge nearest {'purple': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0061_PAR | DEV | PAR | four violet vases | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0217_PAR | TEST | PAR | four violet lamps | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0069_PAR | DEV | PAR | two violet suitcases on a wooden table | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0225_PAR | TEST | PAR | four violet sheep on a concrete floor | SYNONYM-CHECK: ['violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G36: R-SYN-emerald (variant_review, 5 rows, proposed PASS, confidence medium)

**Reason (first row):** 'emerald': bright green: inside green; nearest green in all occurrences in both encoders; gemstone reading about as strong as the colour reading in the judge's CLIP (material only, colour unchanged)

**Evidence:** emerald: SD nearest {'green': 46}, judge nearest {'green': 46}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0063_PAR | DEV | PAR | two emerald boxes | SYNONYM-CHECK: ['emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0213_PAR | TEST | PAR | two emerald horses on a concrete floor | SYNONYM-CHECK: ['emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0221_PAR | TEST | PAR | two emerald mirrors | SYNONYM-CHECK: ['emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0229_PAR | TEST | PAR | four emerald skateboards on a wooden table | SYNONYM-CHECK: ['emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G37: R-SYN-azure; R-SYN-ivory; R-REL-SYN (variant_review, 5 rows, proposed PASS, confidence medium)

**Reason (first row):** 'azure': bright/sky blue: inside blue; nearest basic colour blue in 47/47 (SD) and 46/47 (judge CLIP) amended PAR prompts \| 'ivory': off-white: inside white; nearest white in SD for all and in the judge's CLIP for 44/47 occurrences; strong 'made of ivory' reading (material only, still white) \| relation synonym 'on the left of' (amendment 2026-10-18, Q1=A): viewer-frame phrase without 'side of'; keeps the canonical direction in 82/82 PAR prompts in SD's text encoder (synonym_candidates.json); direction checked against the constraint

**Evidence:** azure: SD nearest {'blue': 47}, judge nearest {'blue': 46, 'green': 1} \| ivory: SD nearest {'white': 47}, judge nearest {'white': 44, 'blue': 2, 'brown': 1}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0179_PAR (amended) | TEST | PAR | an azure cup on the right of an ivory dog on a wooden table | SYNONYM-CHECK: ['azure', 'ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0171_PAR | TEST | PAR | an azure hat on the left of an ivory umbrella | SYNONYM-CHECK: ['azure', 'ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0195_PAR | TEST | PAR | an azure bowl on the right of an ivory bottle on a wooden table | SYNONYM-CHECK: ['azure', 'ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0199_PAR | TEST | PAR | an azure bowl on the left of an ivory sphere on a wooden table | SYNONYM-CHECK: ['azure', 'ivory'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G38: R-SYN-azure; R-SYN-emerald; R-REL-SYN (variant_review, 4 rows, proposed PASS, confidence medium)

**Reason (first row):** 'azure': bright/sky blue: inside blue; nearest basic colour blue in 47/47 (SD) and 46/47 (judge CLIP) amended PAR prompts \| 'emerald': bright green: inside green; nearest green in all occurrences in both encoders; gemstone reading about as strong as the colour reading in the judge's CLIP (material only, colour unchanged) \| relation synonym 'on the right of' (amendment 2026-10-18, Q1=A): viewer-frame phrase without 'side of'; keeps the canonical direction in 82/82 PAR prompts in SD's text encoder (synonym_candidates.json); direction checked against the constraint

**Evidence:** azure: SD nearest {'blue': 47}, judge nearest {'blue': 46, 'green': 1} \| emerald: SD nearest {'green': 46}, judge nearest {'green': 46}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0031_PAR (amended) | DEV | PAR | an azure bottle on the right of an emerald car on a concrete floor | SYNONYM-CHECK: ['azure', 'emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0047_PAR | DEV | PAR | an azure bowl on the right of an emerald basket on a concrete floor | SYNONYM-CHECK: ['azure', 'emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0035_PAR (amended) | DEV | PAR | an azure chair on the right of an emerald car on a wooden table | SYNONYM-CHECK: ['azure', 'emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0039_PAR (amended) | DEV | PAR | an azure apple on the right of an emerald cow on a concrete floor | SYNONYM-CHECK: ['azure', 'emerald'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G39: R-SYN-ivory; R-SYN-violet; R-REL-SYN (variant_review, 3 rows, proposed PASS, confidence medium)

**Reason (first row):** 'ivory': off-white: inside white; nearest white in SD for all and in the judge's CLIP for 44/47 occurrences; strong 'made of ivory' reading (material only, still white) \| 'violet': bluish purple: inside purple; nearest purple in all occurrences in both encoders; colour reading stronger than the flower reading \| relation synonym 'on the left of' (amendment 2026-10-18, Q1=A): viewer-frame phrase without 'side of'; keeps the canonical direction in 82/82 PAR prompts in SD's text encoder (synonym_candidates.json); direction checked against the constraint

**Evidence:** ivory: SD nearest {'white': 47}, judge nearest {'white': 44, 'blue': 2, 'brown': 1} \| violet: SD nearest {'purple': 47}, judge nearest {'purple': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0046_PAR (amended) | DEV | PAR | an ivory umbrella on the right of a violet bench | SYNONYM-CHECK: ['ivory', 'violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0038_PAR | DEV | PAR | an ivory cat on the left of a violet shoe on a concrete floor | SYNONYM-CHECK: ['ivory', 'violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0042_PAR | DEV | PAR | an ivory umbrella on the left of a violet sphere on a concrete floor | SYNONYM-CHECK: ['ivory', 'violet'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## G40: R-SYN-tangerine; R-SYN-crimson; R-REL-SYN (variant_review, 3 rows, proposed PASS, confidence medium)

**Reason (first row):** 'tangerine': reddish orange: inside orange; nearest orange in all occurrences in both encoders; colour reading stronger than the fruit reading \| 'crimson': deep red: inside red; blueprint's own PAR example; nearest basic colour red in all occurrences in both text encoders \| relation synonym 'on the left of' (amendment 2026-10-18, Q1=A): viewer-frame phrase without 'side of'; keeps the canonical direction in 82/82 PAR prompts in SD's text encoder (synonym_candidates.json); direction checked against the constraint

**Evidence:** tangerine: SD nearest {'orange': 47}, judge nearest {'orange': 47} \| crimson: SD nearest {'red': 47}, judge nearest {'red': 47}

| Item | Split | Variant | Text | Flags |
|---|---|---|---|---|
| sg_0049_PAR (amended) | DEV | PAR | a tangerine kite on the right of a crimson bench | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0041_PAR | DEV | PAR | a tangerine chair on the left of a crimson ball on a concrete floor | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |
| sg_0053_PAR | DEV | PAR | a tangerine cow on the left of a crimson book on a concrete floor | SYNONYM-CHECK: ['tangerine'] may read as a material/noun or a narrower hue (see lexicon_review.csv) |

## Adopt

After checking the groups (and excluding any rows you want to decide yourself):

`python scripts/adopt_candidate_review.py --reviewer "<your name>" --date YYYY-MM-DD --answers Q1=A,Q2=A [--exclude ITEM_ID ...]`

`python scripts/check_variant_review.py`  (must exit 0 with `criteria_met: true`)
