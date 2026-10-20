# Open owner decisions (consolidated 2026-10-19)

Sources: `docs/R3_DECISION_LOG.md` (rows marked **OWNER**), `configs/stage_gates.json` + `src/stage_guard.py` (`REQUIRED_GATES`), `docs/ADDENDUM_C.md`. Ordered by the **latest stage that needs the decision**. Nothing here is decided; recommendations are mine.

## DEV_CORE (the next Kaggle stage)
DEV_CORE needs only the gate `variant_review_complete` (stage_guard) and no DEV constants, attention maps or audit. Nothing below blocks it.
| Item | What you do | Evidence |
|---|---|---|
| Variant review (H3) | check `docs/evidence/variant_review/OWNER_REVIEW_PACKET.md`, adopt, run the checker, then set the gate yourself | 1,674 + 73 rows, all proposed PASS after the approved amendment (DL-35) |

## Before DEV_COMMIT (largest DEV spend: 8,896 jobs, about 6,930 image-equivalents)
| ID | Decision | Recommended | Evidence | Consequence |
|---|---|---|---|---|
| Gate `evaluator_audit_complete` | finish the human DEV evaluator audit (`docs/AUDIT_PROTOCOL_DEV.md`) and accept the per-type judge error | complete it before DEV_COMMIT; DEV_CORE can run meanwhile | gate required by DEV_COMMIT, DEV_CAG_GUARD and all TEST stages | without it no restart data is interpretable (D-13) |
| Gates `d13_resolved_for_r3`, `r3_authorised` | resolve D-13 (owner decision 2026-10-14: R3 deferred until the evaluator audit) and authorise R3 | set both after the audit, together with DL-08 and DL-24 | stage_guard REQUIRED_GATES["DEV_COMMIT"] | DEV_COMMIT refuses until both are true |
| DL-08 | ER30 equal-ratio restart on TEST; FULL-resample control on DEV only | **approve** | the approved HA rule (D-02) requires ER30 sign agreement; FULL is the validity gate (C <= 0.15) | DEV_COMMIT already contains ER30 (2,400) and FULL (2,400) jobs; TEST +~1,730 image-equivalents |
| DL-24 | compute: about 27,200-28,700 image-equivalents (manifests) vs 22,196 (blueprint), about 47-62 h on 2 T4 | **approve as computed** (no silent reduction) | `python src/manifests.py --summary`; DL-04/07/08 explain the difference | sets the Kaggle quota plan; DEV_COMMIT is the first large commitment |

## Before the DEV analysis decision on CAG (after DEV_CAG_GUARD, latest at the freeze)
| ID | Decision | Recommended | Evidence | Consequence |
|---|---|---|---|---|
| DL-15 | CAG quality-guard metric (w_lo = 1 vs 3) | **HPS v2** as specified, if its licence and checkpoint verify on Kaggle (download in DEV_CAG_GUARD's environment); otherwise choose a documented substitute **before** looking at DEV_CAG_GUARD outputs | Addendum C: "HPS v2 drop <= 0.5"; scorer not implemented (`r3_analyze.dev_constants` gate PENDING) | without it w_lo cannot be chosen; DEV_CAG_GUARD images can still be generated |

## Before the protocol freeze (all TEST stages)
| ID | Decision | Recommended | Evidence | Consequence |
|---|---|---|---|---|
| DL-07 | E30 CAG arms on all 200 TEST canonical scenes (vs 140) | **200** | F is defined over every constraint family; 140 would drop presence/spatial/count scenes | 3,200 CAG jobs instead of 2,240 |
| DL-11 | supported = decision rule AND Holm-adjusted p < 0.05 (HF p = P(concordance <= 0.5); HE p = max of its three one-sided p) | **approve** | conservative, stated before data | fixes how the CI rules and Holm combine |
| DL-17 | HF attention-area threshold `area_mass_q = 0.9` | **confirm 0.9** | blueprint "top-10% mass"; HF has no DEV tuning (no DEV attention maps) | changing it after TEST would be post hoc |
| DL-18 | reversed arm A4 when s* is late (w_hi' up to about 124 at s* = 38) | **keep as specified, no cap**; pre-register reporting of max guidance and an exploratory sensitivity without the late-s* prompts | a cap breaks the equal-mean-guidance definition of A4, which is what HE's timing test needs | very high guidance images may degrade; that is a property of the reversed schedule |
| Addendum C §C9 | DEV-derived constants (t50, s*, gates) | fill from `dev_constants.json`, then `protocol_r3.py --draft/--sign/--verify` | Addendum C §C9 is UNSET | TEST refuses until the freeze verifies |

## Not required on the Revision 3 path
`synthetic_probe_threshold_approved` (Revision 2 stage `e22`) and `addendum_b_signed` (Revision 2 stage `r4`) gate no Revision 3 stage. Recommended: leave them false and record them as superseded by Addendum C.
