# Addendum B (DRAFT - NOT FROZEN, NOT SIGNED)

Status: draft text for `report/protocol.md`. It becomes binding only when the freeze procedure in `src/freeze.py` has been run, the model revision is pinned, the DEV-derived numbers below are filled in, and the owner has approved the open decisions. Until then every analysis is EXPLORATORY.

Source of truth for hypotheses and definitions: `GRAND_RESEARCH_BLUEPRINT.md` (Revision 2) sections 8.0-8.4 (HA, HB, HC verbatim by reference). Constants: `configs/analysis_plan.json` (tests assert equality with code defaults).

> Owner decision 2026-10-14: the working defaults D-02, D-03, D-04, D-05 (literal primary, competitive sensitivity), D-06 and D-12 below are approved **as working defaults only**. Addendum B remains a DRAFT: unsigned, DEV-derived values unset, nothing frozen, TEST not accessed. D-13: R3 deferred until the evaluator audit (`docs/AUDIT_PROTOCOL_DEV.md`) is complete; four base seeds not approved to meet the gate; the 4-step gate is not to be changed to make a result pass.

## Working decisions applied (owner instruction, 2026-10-13)
- **D-02** Restart: fixed jump D = 8 steps retained as the starting design; every branch logs requested/effective jump, scheduler timesteps and noise ratio; the clipped boundary `t = 6` (effective jump 6, ratio 0.21) is analysed separately; an equal-ratio sensitivity (ER30) and D in {4, 12} are mandatory; a persistence curve is not interpreted as semantic evidence when the trend can arise from changing perturbation severity. Support for HA additionally requires the same sign of Delta-t50 under the equal-ratio restart.
- **D-03** Wording: "restart persistence" / "restart-persistence curve"; no claim that the model "decided" a constraint.
- **D-04** Outcomes reported separately: entity presence; binding given all entities present; joint (present and bound); spatial given presence; joint spatial. Eligibility denominators and missing-object rates always shown; conditional outcomes are not read alone as improved adherence.
- **D-05** Binding-bearing = canonical constraints contain at least one explicit binding; counts derived from the manifest (`data/benchmark_v2/design_counts.json`): TEST 140 (literal) / 110 (competitive), DEV 70 / 55. Primary population: literal; competitive reported as sensitivity (owner may change this before the freeze).
- **D-06** TEST sealing: no TEST generation, scoring or inspection before `python src/freeze.py --verify` passes; hashes of code, configs, split manifest, metric definitions, analysis plan and thresholds are in the freeze manifest; deviations go to `docs/freeze/deviations.md` and relabel the affected analysis EXPLORATORY.
- **D-12** Undefined DEV-derived quantities follow `commitment.windows_from_t50_dev`.

## Values to be filled from DEV only (currently UNSET)
`t50_DEV(bind|pres)`, `tau_pre`, `tau_post`, HC onsets (early 0, mid, late), primary judge per constraint type, DEV half-width of Delta-t50 (gate: <= 4 steps) and of Delta-AUC (gate: <= 0.04), full-resample control C (gate: <= 0.15).

## Open items that block the freeze
Model revision pin (needs network access to the model hub); human review of PAR/ORD/CF variants (M3 gate: < 20% rejected); lexicon freeze after E22/E03; judge audit (E01); Kaggle hardware probe; owner approval of the primary population and of this text.
