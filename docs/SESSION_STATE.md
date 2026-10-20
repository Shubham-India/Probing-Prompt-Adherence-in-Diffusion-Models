# Session state (update this file first when a session ends)

## 2026-10-18 session: Revision 3 ("Guide Before Commitment") implementation and Kaggle package
- Read Revision 3 (sections 0, 24) and `main_latex.pdf` (illustrative values flagged, DL-02; pilot values verified). Snapshot + baseline hashes before any change; baseline 121 tests.
- Implemented (all uncommitted): PS size family (additive files), relative-size judge, Revision 3 config/analysis plan, executable manifests with scene-group sharding, worker (`src/r3_run.py`), CAG, prompt switch, attention groups and stabilisation, DINOv2 diversity, merge/judge/analysis/report, Addendum C + protocol freeze, guard stages, Kaggle notebooks 00-04, release builder, CPU dry-run harness.
- Real-GPU smoke test on the RTX 3050 (`local_audit/dbg/preflight_rtx3050.json`): hook-on image identical, CAG constant = pipeline, CAG w = 1 = pipeline at 1 with 40 UNet evaluations, 2.86 GB peak VRAM.
- Release: `release/guide_before_commitment/` (built from a DIRTY tree: commit before any research run). Nothing run on Kaggle; nothing frozen; no human labels yet.
- Next owner actions: `docs/release/README_FIRST.md`; OWNER rows of `docs/R3_DECISION_LOG.md`; DEV audit labels.


| Field | Value |
|---|---|
| Date of last update | 2026-10-17 (local session: execution system, results-to-paper chain, LaTeX draft; no Kaggle, no GPU job, no TEST; detailed state in `docs/PROJECT_STATUS.md`) |
| Branch | `claude/admiring-cori-pbn9wk` (pushed to origin; fast-forwarded from `main` at `6854b10`; the named `docs/master-implementation-plan` branch does not exist on the remote - the planning commit `6854b10` is on `main`) |
| Latest commit | see `git log -1` (the commit that contains this file is the resume point) |
| Authoritative files | `GRAND_RESEARCH_BLUEPRINT.md` (Revision 2), `docs/MASTER_IMPLEMENTATION_BLUEPRINT.md`, `docs/DECISION_REGISTER.md` (status block at top), `docs/KAGGLE_MASTER_RUNBOOK.md` (governs execution; `docs/EXPERIMENT_RUNBOOK.md` keeps older detail), `docs/ADDENDUM_B.md` (DRAFT), `docs/PROJECT_STATUS.md`, `docs/HYPOTHESIS_LEDGER.md`, `docs/DATASETS_AND_TRAINING.md`, `docs/RESULTS_PROVENANCE.md` (generated), `docs/AUDIT_PROTOCOL_DEV.md`, `paper/` |
| Kaggle runs | **R0** `r0-20261014T145606Z-daef6c4` (commit `daef6c4740884276fc068109bbb48207f7034edd` = repo HEAD): artifacts `env.json`, `gpu_identity.json`, `setup_report.json`, `throughput.json` verified locally (`docs/R0_R1_AUDIT.md`): all six required GPU identities hold (max diff 0, 3 cases, T4). The earlier R0 `r0-20261014T100526Z` and R1 `r1-20261014T101024Z` exist only as the older `throughput.json` copy under `local_audit/R2a_checkpoint/provenance/`. **R2a** `r2a-20261014T130457Z-daef6c4`: audited, **valid** (`docs/R2A_AUDIT_REPORT.md`). Infrastructure checks only: no scientific hypothesis is supported by any of this. |
| Frozen? | **No.** `python src/freeze.py --verify` correctly fails (no manifest, revision unpinned). No TEST scene was generated, scored or inspected. |

## Milestones (all evidence in `outputs/acceptance/*.txt`)
| MS | Status | What was actually run |
|---|---|---|
| M0 hygiene | DONE | README conflict markers removed, e2e row corrected (log ends `E2E-DONE`; not re-run), `.gitignore` for `kaggle_returns/`; `build_report.py` rerun (assertions pass; now reads pilot library versions from `samples.jsonl`/`requirements.txt`; **it overwrites README.md from a template - that overwrite was reverted on purpose**); slides rebuilt |
| M1 restart math + config | DONE (CPU) | `branch.py`, `config_schema.py`, `configs/experiment_v2.json` (draft; revision `UNPINNED-DRAFT`); 13 tests incl. diffusers timestep/alpha-bar equality, D=0 identity, clipping at t=6, synthetic moment test |
| M2 pilot re-analysis | DONE (exploratory) | `analyze_pilot_conditional.py --check`: 1008/1008 re-scores identical; conditional/joint rates, eligibility, seed effect (permutation p=0.012), nested AUC; report `docs/M2_PILOT_REANALYSIS.md` |
| M3 Benchmark V2 | DRAFT DONE | `benchmark_v2.py`, 300 scenes (DEV 100/TEST 200), 1,410 variants, validators, derived binding-bearing counts (TEST 140 literal / 110 competitive; DEV 70/55), datasheet, precision recompute. **Human review of variants and lexicon freeze NOT done** |
| M4 generate/verify/merge | DONE (CPU, fake pipeline) | shards, atomic writes, hashes, latent saving, `--split/--variants`, TEST guard; byte-identity vs pilot images on GPU NOT checked |
| M5 evaluator/audit | DONE (CPU) | additive conditional fields (old columns byte-identical), audit sampler (blinded), human-validation stats, synthetic probe, J2 interface (no VLM chosen) |
| M6 persistence stats | DONE (synthetic) | `commitment.py`, `analyze_commitment.py`, `features.py`, `predict_failure.py`, `analyze_noise.py`; synthetic recovery tests pass |
| M7 intervention modes | DONE (CPU toy layer) | `edit_probe.py` + `intervene.py --mode`; control identity, equal area/beta, bias-mass equality across onsets, knockout locality, caps, determinism |
| M8 GPU identity | DONE for the tested scope | Kaggle T4 `gpu_identity.json` verified (3 pilot prompts, seed 0, CFG 7.5, 40 steps; all required differences 0); local RTX 3050 diagnostic agrees. Not tested: other prompts/seeds, V2 prompts, TEST |
| M9 Kaggle probe | DONE (R0 and throughput verified; two throughput sessions differ by about 25%, see D-13) | `kaggle/{env_probe,setup,make_bundle,run_throughput}.py`, `requirements-kaggle.txt`; syntax-checked, `env_probe` ran on this CPU box; **no Kaggle run** |
| M10 DEV | PARTIAL | R2a (DEV CAN core with latents) generated and audited. Not done: R2b variants, R3 restart branches, DEV audit (E01), power gates, Addendum B values. `freeze.py` portability bug fixed (CRLF/POSIX keys) |
| M11-M13 | NOT STARTED | need TEST freeze / GPU |
| M14 report | PARTIAL | pilot documents regenerated; V2 report needs real results |

## Tests actually run (this container, CPU, torch 2.14.1 / diffusers 0.41.0)
Local Windows machine, `.venv` torch 2.6.0+cu124, RTX 3050: `python -m pytest tests -q` gave 97 passed / 1 failed before the fix and **99 passed** after fixing (103 after adding the branch-run verifier tests) `src/freeze.py` (the failing test was the freeze round-trip; cause: backslash manifest keys and CRLF hashing). The GPU attention test ran and passed. The older cloud-container remark (CLIP tokenizer download failure) does not apply locally.

## Key findings to remember (all EXPLORATORY unless stated)
- Pilot: only 61% of binding-bearing images (395/648) have every required object; binding category 40-49% by CFG. Legacy `binding_ok` already equals the joint outcome. Seed 0 is ~9 points worse than other seeds (p=0.012) but seed identity adds no AUC (+0.002) and noise features add none (-0.007); early attention features add +0.05 AUC over noise+design (CI 0.017-0.097; different CV scheme than `outputs/main/results/attention_classifier_auc.json`).
- Design: spatial-bearing TEST scenes are 54, not ~70 (RB assumption). Single-AUC half-width 0.083 (140 scenes) / 0.094 (110), DEV 0.118 / 0.133.
- `dev_gate_sim.py` (assumption-driven, 12 replicates): with ~28 binding-bearing DEV scenes and eligibility 0.45 the Delta-t50 half-width exceeds the 4-step gate (about 7.8); about 55 scenes at eligibility 0.6 or 110 scenes pass. Plan the DEV commitment run on all binding-bearing DEV scenes, not 40. Percentile bootstrap showed excess false positives at small N (3/12 at 110 scenes, eligibility 0.6, delta 0) - needs more replicates before relying on it.
- The `|C|<0.1` acceptance criterion for independent outcomes holds on average (MC mean 0.0017) but single runs can exceed 0.1 (SD ~0.04 per grid step, shared A0 noise); tested as mean<0.025 and max<0.15.

## Open blockers
1. ~~R0 artifacts~~ received and verified 2026-10-14.
2. ~~Model revision pin~~ DONE 2026-10-14 (`f03de327dd89b501a01da37fc5240cf4fdba85a1`, both configs); the freeze itself is still not done.
3. Owner approvals: D-02..D-06/D-12 wording in `docs/ADDENDUM_B.md`; primary population (literal 140 vs competitive 110).
4. Human review of PAR/ORD/CF variants (M3 gate) and lexicon freeze after E22/E03 (detector probe never run through OWL-ViT/CLIP).
5. Independent second reviewer of the 40 validation labels (still outstanding from the pilot).
6. Not implemented: HB attention-window analysis on V2 data (`features.py`/`predict_failure.py` provide the machinery; window features need V2 attention maps), `fragility.py`, `guidance_attr.py` (exploratory P2), per-call RNG seeding in the old `analyze.py` (deferred: it would change the published pilot CIs), V2 figures/report.

7. **D-13 (new):** base seeds / branch design for the HA DEV gate: only 62 of 140 base images (seeds 0,1) of binding-bearing DEV scenes have all objects, so the planned DEV gate is unlikely to pass; choose bases, M and gate wording before R3 (R2A report 6).

## 2026-10-17 session summary
Added and tested (121 tests pass): `src/stage_guard.py` + `configs/stage_gates.json` (all gates false; owner-controlled), `kaggle/finish_run.py`, `kaggle/make_core_dataset.py`, `src/build_draft.py`, `paper/` LaTeX draft with `paper/check_paper.py` (static checks pass; **not compiled: no LaTeX toolchain**), and the docs listed above. Found: the older runbook's R2b count would have regenerated canonical images (1,480 is correct). **The new scripts are uncommitted; they reach Kaggle only after the owner commits and rebuilds the bundle.** Still no human labels; R3 stays blocked.

## Owner decisions on record (2026-10-14)
- D-02, D-03, D-04, D-05 (literal primary, competitive sensitivity), D-06, D-12: approved **as working defaults only** (not the final protocol, not the freeze, no TEST access).
- D-13: R3 deferred until the evaluator audit is complete; no 4 base seeds to meet the CI gate; the 4-step gate unchanged.

## Evaluator audit status
Packages are ready (local, git-excluded): `local_audit/audit_dev/` (annotator A: 20 practice + 540 scored items; annotator B: 100 double items + practice) and `local_audit/audit_pilot40/` (125 items for the second review of the 40 pilot labels). Protocol: `docs/AUDIT_PROTOCOL_DEV.md` (predeclared criterion: weighted accuracy >= 0.85 per type; no judge is selected). Tool: `src/audit_dev.py` (9 tests). **No labels exist yet.** The synthetic probe is specified and has an export/score adapter but was not run (GPU/detector job).

## Requirements still open before the freeze (none is satisfied yet)
| Item | What is needed | Evidence of completion |
|---|---|---|
| DEV evaluator audit (E01) | Blinded human audit of J1 on DEV images (`src/audit_sample.py` export, owner labels, `src/validate_human.py`); per-type accuracy; types under 0.85 stay exploratory; primary judge chosen on DEV only; measured flip rate feeds D-13 | filled primary-judge values in Addendum B |
| Second reviewer of the 40 pilot validation labels | independent person | recorded agreement (kappa) |
| Variant human review (M3 gate) | owner review of PAR/ORD/CF variants and lexicon; < 20% rejected; E22 synthetic probe through OWL-ViT/CLIP never run; lexicon frozen only afterwards | review sheet + frozen `lexicon.json` |
| Addendum B approval | owner confirms D-02..D-06, D-12, D-13 wording and the primary population (literal 140 vs competitive 110); DEV-derived numbers filled (t50_DEV, tau_pre/post, onsets, half-width gates, full-resample control) | signed-off `docs/ADDENDUM_B.md` |
| DEV stages | R3 branches (+ evaluation); R2b only after variant review | verified outputs |
| Freeze | `python src/freeze.py --write`, copy the hash into the config, `--verify` exits 0, commit; TEST allowed only afterwards | `docs/freeze/*`, verify exit 0 |

## Next exact task
Owner: label the DEV audit package (`local_audit/audit_dev/annotator_A/label.html`, 20 practice + 540 items) and return `labels_A.csv`; arrange annotator B (different person) for `annotator_B` and a reviewer for `audit_pilot40`. Then Claude scores the audit, re-sizes D-13 with measured error rates and reassesses R3 as an exploratory rehearsal. No Kaggle job and no GPU job until then.
