# Session state (update this file first when a session ends)

| Field | Value |
|---|---|
| Date of last update | 2026-10-13 (autonomous milestone session, Claude cloud container, CPU only, no GPU) |
| Branch | `claude/admiring-cori-pbn9wk` (pushed to origin; fast-forwarded from `main` at `6854b10`; the named `docs/master-implementation-plan` branch does not exist on the remote - the planning commit `6854b10` is on `main`) |
| Latest commit | see `git log -1` (the commit that contains this file is the resume point) |
| Authoritative files | `GRAND_RESEARCH_BLUEPRINT.md` (Revision 2), `docs/MASTER_IMPLEMENTATION_BLUEPRINT.md`, `docs/DECISION_REGISTER.md` (status block at top), `docs/EXPERIMENT_RUNBOOK.md` (RUN CARDS), `docs/ADDENDUM_B.md` (DRAFT) |
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
| M8 GPU identity | CODE ONLY | loop/resume/D=0 identity verified on a TINY RANDOM CPU pipeline; **real SD 1.5 fp16 identity not run** (`tests/gpu_identity.py`) |
| M9 Kaggle probe | PREPARED | `kaggle/{env_probe,setup,make_bundle,run_throughput}.py`, `requirements-kaggle.txt`; syntax-checked, `env_probe` ran on this CPU box; **no Kaggle run** |
| M10 DEV | PREPARED | `run_branches.py`, `freeze.py`, `analysis_plan.json`, `ADDENDUM_B.md` (draft), `dev_gate_sim.py`, `analyze_hc.py`; nothing generated |
| M11-M13 | NOT STARTED | need TEST freeze / GPU |
| M14 report | PARTIAL | pilot documents regenerated; V2 report needs real results |

## Tests actually run (this container, CPU, torch 2.14.1 / diffusers 0.41.0)
`python -m pytest tests -q`: see last line of `outputs/acceptance/FINAL_tests.txt`. One test, `tests/test_logic.py::test_word_token_alignment`, FAILS HERE ONLY because it downloads the CLIP tokenizer from huggingface.co and the proxy returns 403 (passes on the owner's machine per `outputs/test_results.txt`, earlier run). `tests/test_attention_gpu.py` is skipped (no CUDA). Nothing about SD 1.5 was run on a GPU in this session.

## Key findings to remember (all EXPLORATORY unless stated)
- Pilot: only 61% of binding-bearing images (395/648) have every required object; binding category 40-49% by CFG. Legacy `binding_ok` already equals the joint outcome. Seed 0 is ~9 points worse than other seeds (p=0.012) but seed identity adds no AUC (+0.002) and noise features add none (-0.007); early attention features add +0.05 AUC over noise+design (CI 0.017-0.097; different CV scheme than `outputs/main/results/attention_classifier_auc.json`).
- Design: spatial-bearing TEST scenes are 54, not ~70 (RB assumption). Single-AUC half-width 0.083 (140 scenes) / 0.094 (110), DEV 0.118 / 0.133.
- `dev_gate_sim.py` (assumption-driven, 12 replicates): with ~28 binding-bearing DEV scenes and eligibility 0.45 the Delta-t50 half-width exceeds the 4-step gate (about 7.8); about 55 scenes at eligibility 0.6 or 110 scenes pass. Plan the DEV commitment run on all binding-bearing DEV scenes, not 40. Percentile bootstrap showed excess false positives at small N (3/12 at 110 scenes, eligibility 0.6, delta 0) - needs more replicates before relying on it.
- The `|C|<0.1` acceptance criterion for independent outcomes holds on average (MC mean 0.0017) but single runs can exceed 0.1 (SD ~0.04 per grid step, shared A0 noise); tested as mean<0.025 and max<0.15.

## Open blockers
1. Real-GPU validation of every GPU-dependent piece (R0: `gpu_identity.py`, `env_probe.py`) - owner must run.
2. Model revision pin (Hugging Face unreachable from the cloud session) and `UNPINNED-DRAFT` in the config.
3. Owner approvals: D-02..D-06/D-12 wording in `docs/ADDENDUM_B.md`; primary population (literal 140 vs competitive 110).
4. Human review of PAR/ORD/CF variants (M3 gate) and lexicon freeze after E22/E03 (detector probe never run through OWL-ViT/CLIP).
5. Independent second reviewer of the 40 validation labels (still outstanding from the pilot).
6. Not implemented: HB attention-window analysis on V2 data (`features.py`/`predict_failure.py` provide the machinery; window features need V2 attention maps), `fragility.py`, `guidance_attr.py` (exploratory P2), per-call RNG seeding in the old `analyze.py` (deferred: it would change the published pilot CIs), V2 figures/report.

## Next exact task
Owner: run **R0** (RUNBOOK section 8) on Kaggle or the local RTX 3050: build the bundle (`python kaggle/make_bundle.py --commit <SHA>`), upload as a private Kaggle dataset, run the three R0 commands, and return `env.json`, `gpu_identity.json`, `setup_report.json`. (Local alternative for the identity part only: `python tests/gpu_identity.py --out outputs/acceptance/gpu_identity.json` on the RTX 3050.)
Claude after the return: validate against RUNBOOK section 5; if identities hold, issue R1 then R2a; otherwise record and decide per D-07.
