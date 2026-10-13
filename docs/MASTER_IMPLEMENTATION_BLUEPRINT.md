# Master implementation blueprint

Scope: turns `GRAND_RESEARCH_BLUEPRINT.md` (Revision 2, called **RB** below; section numbers refer to it) into a dependency-ordered implementation plan. It does not repeat RB. Written 2026-10-14 on branch `docs/master-implementation-plan` from `main` at `9633d87`. Nothing in this document is implemented yet; no experiment was run for it.
Companion files: `REPOSITORY_AUDIT.md` (facts), `DECISION_REGISTER.md` (open decisions D-xx), `EXPERIMENT_RUNBOOK.md` (Kaggle procedure), `SESSION_STATE.md` (progress).

## 1. Executive overview and verified status

| Item | Status |
|---|---|
| Pilot (SD 1.5, 28 prompts, 1008 + 504 images, CPU tests 6 passed) | Implemented, results saved, frozen |
| Revision 2 programme (HA, HB, HC, Benchmark V2, branching, conditional evaluation, shard tooling, Kaggle workflow) | **CPU-side implemented 2026-10-13 (M0-M8 code, M9/M10 prep, HC/HA analysis drivers); no GPU, Kaggle or TEST step has been run.** See `docs/SESSION_STATE.md` for exact status |
| Git | Repo present with remote; README has committed merge-conflict markers (F-1) |
| Kaggle / GitHub | No Kaggle CLI or credentials; GitHub repo is private (anonymous read fails), so a Kaggle clone needs a token; push unverified; hardware unmeasured |
| Blocking decisions | D-02 (restart magnitude), D-03 (terminology), D-04 (conditional outcome), D-05 C-1 (binding-bearing counts), D-06 (freeze procedure) |
| Single best next action | Owner approvals (D-05 C-1, D-04 first); in parallel Claude does M0 hygiene and M1 (CPU only) |

## 2. Research goals and hypothesis map

| Hypothesis | Question (RB) | Primary statistic | Experiments | Main code |
|---|---|---|---|---|
| HA (RB 8.1) | Does restart-persistence timing differ between presence and binding given presence? | `Delta t50` | E23 (+E23b, E24) | `branch.py`, `commitment.py`, `analyze_commitment.py` |
| HB (RB 8.2) | Do attention features add information beyond prompt and noise features, before vs after `t50`? | `Delta AUC_pre/post` over `B_noise` | E05, E25, E21 gate | `features.py`, `predict_failure.py`, `analyze_noise.py` |
| HC (RB 8.3) | Is a targeted attention edit effective only before `t50`, relative to a random edit? | `D_early`, `G` | E12 (+E26) | `intervene.py` modes |
| Supporting | H1/H2 (RB 8.5), CFG x steps x complexity (PS requirement) | various | E08, E11, E01, E18 | `analyze*.py` |
| Wording rule | Use "restart-persistence" until validated (D-03); no novelty claims (RB 5.6) | | | |

## 3. Repository audit summary
See `REPOSITORY_AUDIT.md`. Key items: working pilot modules (generate, attention, intervene, evaluate, analyze*), six CPU tests green, stale generated documents, conditional fields missing in `score()`, README conflict, no Kaggle automation.

## 4. Existing versus planned inventory

| Group | Existing (preserve) | Modify | Create |
|---|---|---|---|
| Generation | `generate.py`, `attention.py` | `generate.py` (shards, atomic writes, latents, metadata) | `branch.py`, `verify_outputs.py`, `merge_samples.py` |
| Evaluation | `evaluate.py`, `validate_metrics.py` | `evaluate.py` (new conditional fields, judge interface) | `judge_vqa.py`, `audit_sample.py`, `validate_human.py`, `synth_probe.py` |
| Benchmark | `build_benchmark.py`, `prompts.json` | - | `benchmark_v2.py`, `data/benchmark_v2/*`, `configs/experiment_v2.json`, DATASHEET |
| Analysis | `analyze*.py`, `figures.py`, `report_numbers.py` | `analyze.py`, `analyze_attention.py`, `analyze_intervention.py` | `commitment.py`, `analyze_commitment.py`, `features.py`, `predict_failure.py`, `analyze_noise.py`, `fragility.py`, `guidance_attr.py` |
| Intervention | `intervene.py` | modes: window/onset, targeted, random pair, knockout | - |
| Process | `report/protocol.md`, tests | Addendum B at freeze | `kaggle/` scripts, `requirements-kaggle.txt`, `runs/` |

## 5. Decision register
`DECISION_REGISTER.md` is authoritative. A milestone may start only if the decisions marked "blocking" for it are RESOLVED (column "Needs" in section 16).

## 6. Architecture and dependency graph

```
configs/experiment_v2.json --> benchmark_v2.py --> data/benchmark_v2/{scene_graphs,prompts_v2,splits}
                                      |
generate.py (shards, atomic, latents) -+--> images, latents, samples_gpuN.jsonl --> verify_outputs --> merge_samples
attention.py (hook, same pass) ---------+--> attention maps
                                      |
evaluate.py (J1 + conditional fields) --+--> eval rows <-- judge_vqa.py (J2) <-- audit_sample / validate_human (J3)
                                      |
branch.py (restart math, custom loop) --+--> branch images --> evaluate --> commitment.py --> analyze_commitment (HA)
features.py (prompt, noise, attention) --> predict_failure.py --> HB
intervene.py (onset arms) --> evaluate --> analyze_intervention (HC)
analyze.py (bootstrap, paired contrasts) <-- all analyses --> figures.py / report_numbers.py --> report
```
Phase order: **audit -> decisions -> CPU modules and tests -> pilot re-analysis (no GPU) -> local GPU identity checks -> Kaggle hardware pilot -> DEV generation and rehearsal -> freeze -> TEST -> confirmatory analysis -> report.**

## 7. File-by-file plan

Columns: **Ex** = exists; **Act** = Preserve/Modify/Create; **Resp** = single responsibility; **Deps** = upstream; **I/O** = interface; **Tests** = unit/integration; **Accept** = acceptance; **MS** = milestone; **Where** = validation location; **Next** = what depends on it.

| File | Ex | Act | Resp | Deps | I/O (key schema/config) | Tests | Accept | MS | Where | Next |
|---|---|---|---|---|---|---|---|---|---|---|
| `configs/experiment_v2.json` + `src/config_schema.py` | no | Create | Validated V2 config (model id + **revision**, CFG, steps, k, seeds, grid, `D`, `M`, shard count, split manifest path) | - | JSON; unknown keys rejected | Schema tests (valid, missing key, unknown key) | pilot config still loads | M1 | CPU | all generation |
| `src/benchmark_v2.py` | no | Create | Scene graph schema, grammar, lexicon, variants (CAN/PAR/ORD/CF-*), validators, splitter | config | `scene_graphs.json`, `prompts_v2.json`, `splits.json` | exactly-one-field counterfactuals; no scene or object-pair across splits; duplicate detector | RB 17 acceptance list | M3 | CPU | generation, audit |
| `data/benchmark_v2/DATASHEET.md` | no | Create | Provenance, composition, limits | benchmark_v2 | md | doc check | hashes recorded | M3 | CPU | freeze |
| `src/generate.py` | yes | Modify | Image generation with shards, atomic writes, hashes, metadata, optional latent saving | config, benchmark | `--shard i/n`, `--save-latents-at`, `--run-id` | default flags reproduce pilot bytes on the same GPU; killed-and-restarted run has no duplicates | RB 17 list | M4 (+M8 on GPU) | CPU for logic, local GPU, Kaggle | everything generated |
| `src/branch.py` | no | Create | Restart math (pure), custom denoise loop from a saved latent | generate (pipeline factory), scheduler | `restart(x_t, t, D, generator)`, `branch_generate(...)` | CPU: moment test, index mapping, `D=0` returns input. GPU: loop from step 0 equals pipeline image (D-07) | D-01, D-07 resolved | M1 (math), M8 (GPU) | CPU, local GPU | commitment |
| `src/commitment.py` | no | Create | `A`, `A0`, `C`, `t50`, scene bootstrap | outcomes | tables in/out | Synthetic: known commitment recovered within one grid step; independent outcomes `C~0`; identical `C=1`; stable-wrong case returns no information | D-02/D-03 text implemented as written | M6 | CPU | HA |
| `src/evaluate.py` | yes | Modify | Add `objects_present`, `bind_given_pres`, `spatial_given_pres`, `uncertain`; judge interface | detections | new columns appended | Regression: old columns unchanged on the 40 validation images and on `outputs/main/eval.jsonl` | byte-identical old columns | M5 | CPU | all adherence analyses |
| `src/judge_vqa.py` | no | Create | J2 atomic-question judge | VLM chosen in E03 | deterministic outputs | agrees with J1 on trivial synthetic cases | fits VRAM | M9-M10 | Kaggle | E18 |
| `src/audit_sample.py`, `src/validate_human.py` | no | Create | Stratified blinded draw; accuracy, kappa, Wilson CI | eval rows | CSV export without arm/variant names | sampler reproduces strata; kappa vs hand example | RB 17 | M5 | CPU | J3 |
| `src/synth_probe.py` | no | Create | Coloured-shape scenes with known truth (E22) | - | images + truth | truth matches render | CPU minutes | M5 | CPU | judge gates |
| `src/features.py`, `src/predict_failure.py`, `src/analyze_noise.py` | no | Create | Prompt, noise (from seed on CPU) and attention features; nested models; noise readout | attention, eval | frozen predictions | invariance to map rescale; train/test scene disjoint; label permutation gives ~0 | RB 17 | M6 | CPU | HB |
| `src/intervene.py` | yes | Modify | `--mode {global,targeted,random_pair,shuffled_mask,always_on,knockout}`, window/onset, bias-mass log | attention, tokens | arm logs | control equals baseline; mass equal across onsets within 5%; knockout only in chosen layer group | RB 13.6 | M7 | CPU (toy layer) + local GPU | HC |
| `src/fragility.py`, `src/guidance_attr.py` | no | Create | E24, E27 | generate | | `eta=0` reproduces; ablating an absent token changes nothing | exploratory | M13 | Kaggle | P2 |
| `src/analyze.py`, `analyze_attention.py`, `analyze_intervention.py`, `analyze_commitment.py` | partly | Modify/Create | Scene-level keys, per-call RNG seeds, arm generalisation | | | pilot numbers reproduce exactly | | M2, M6, M7 | CPU | report |
| `src/verify_outputs.py`, `src/merge_samples.py` | no | Create | Integrity | samples | | seeded duplicate and corrupt PNG detected | | M4 | CPU | all runs |
| `src/figures.py`, `src/report_numbers.py`, `report/build_report.py`, `report/make_slides.py` | yes | Modify | Figures/tables from saved data; no hand-typed numbers | results | | assertions pass | | M14 | CPU | final report |
| `kaggle/{setup.py, env_probe.py}`, `requirements-kaggle.txt` | no | Create | Environment record, pinned installs | | `env.json` | run on Kaggle | | M9 | Kaggle | all Kaggle runs |
| `tests/` | yes | Extend | One file per new module; tests for old behaviour stay | | | | CPU suite green | each MS | CPU/Kaggle | gate |
| `report/protocol.md` | yes | Append | Addendum B at freeze | all DEV | hashes | hash check | | M10 | CPU | TEST |
| `README.md` | yes | Fix, later rebuild | conflict markers removed first | | | | | M0, M14 | - | - |

Rule: no new module may duplicate an existing one; additions to `evaluate.py`, `generate.py`, `intervene.py` keep their old behaviour under default flags.

## 8. Experiment catalogue to code and outputs

| Experiment (RB 9) | Code | Output |
|---|---|---|
| E00 reproduce pilot | existing analyses | re-run tables equal saved |
| E21 gate (extended) | `analyze_attention.py` + `analyze_noise.py` on pilot | `outputs/e21/*.json` |
| E01, E18, E22 | `evaluate.py`, `audit_*`, `synth_probe.py` | accuracy/kappa tables |
| E02 | `benchmark_v2.py` | `data/benchmark_v2/` |
| E03 | `kaggle/env_probe.py` + generate | `hardware.json`, `throughput.json` |
| E04 | `generate.py` + `attention.py` | images, latents, maps |
| E05, E25 | `features.py`, `predict_failure.py`, `analyze_noise.py` | HB tables |
| E23, E23b | `branch.py`, `commitment.py` | `branches.jsonl`, `curves.csv`, `t50.json` |
| E12, E26 | `intervene.py` | arm logs, contrasts |
| E11, E10, E08, E09, E14, E15, E19, E20 | `analyze*.py` | tables |
| E24, E27, E28 | `fragility.py`, `guidance_attr.py`, generate flag | exploratory |
| E16 SDXL, E17 SD3 | deferred | needs hook port |

## 9. Tests and acceptance per milestone (summary; details in section 16)
CPU-first: no GPU run is requested while CPU tests, schema validation and synthetic recovery tests fail. A test counts only after it has been executed and its output shown; writing a test is not a pass.

## 10. Kaggle environment and hardware protocol
Procedure in `EXPERIMENT_RUNBOOK.md` sections 3-4. Measured, not assumed: GPU type, VRAM, quota, session length, `t_gen`, `t_branch`, latent-save cost, two-worker scaling, hook identity, J2 fit. FULL vs LIMITED (RB 16) is chosen from `throughput.json`.

## 11. GitHub-Kaggle artefact workflow
Path A (manual) is the baseline; Path B only after a minimal end-to-end dummy run succeeds (RUNBOOK 2). Bulk outputs stay out of Git (RUNBOOK 6).

## 12. Run, validate, resume, recover
RUNBOOK sections 4, 5, 7. Idempotent resume; verification before analysis.

## 13. DEV/TEST separation and pre-registration
1. Pilot prompts: DEV-only. 2. DEV: tuning, `t50_DEV`, windows, onsets, judge choice, power simulation. 3. Freeze (G3'): Addendum B with hypotheses verbatim, definitions (D-01..D-04 outcomes), grid, `D`, `M`, `L`, thresholds, split manifest, analysis scripts, DEV-derived numbers, **SHA-256 hashes**, commit SHA. 4. TEST generated only after the freeze commit exists; scripts refuse TEST paths without the Addendum hash (D-06). 5. Confirmatory order HA -> HB -> HC, each analysed once; deviations appended with reason and relabelled EXPLORATORY.

## 14. Statistics, evaluator validation, integrity
Scene is the unit; scene-level bootstrap (per-call seeded RNG); Holm over HA, HB, HC; conditional and joint outcomes reported together (D-04); MDE next to every null; judge sensitivity (E18) before any headline claim; types with audit accuracy under 0.85 stay exploratory; failed samples logged; no inspection of TEST before the freeze.

## 15. Claude session protocol
Start each session by reading `docs/SESSION_STATE.md` only (not the whole RB). Read RB sections on demand by number. Report per milestone: what changed, what was tested, results, files and commit, blocker, next owner action. Update `SESSION_STATE.md` before the context ends.

## 16. Milestones and gates

| MS | Content | Needs (decisions) | Test / gate before leaving | Where |
|---|---|---|---|---|
| M0 | Hygiene: remove README conflict markers; verify e2e log and correct README row; `.gitignore` for `kaggle_returns/`; rebuild stale generated docs; pin model revision | - | `git grep "^<<<<<<<"` empty; `build_report.py` assertions pass; CPU tests green | local |
| M1 | `config_schema.py`, `experiment_v2.json`, `branch.py` math, tests for D-01 | D-01 | moment test, index map, `D=0` identity (CPU) | local CPU |
| M2 | Pilot re-analysis: conditional rates, eligibility, noise-only baseline, variance components (E21 extended). Conditional outcomes are computed by a **standalone function in the analysis script** from saved `detections.jsonl`; `evaluate.py` is not touched before M5 | D-04 (definition) | outputs reproducible; pilot numbers unchanged; **G0'** | CPU |
| M3 | `benchmark_v2.py`, splits, datasheet, QA; **draft lexicon only** (frozen after M5 and M9, before M10; D-05 C-8) | D-05 C-1 | validators pass; reviewer rejects under 20% of variants | CPU + human review |
| M4 | `generate.py` upgrade, verify/merge | - | seeded duplicate/corruption detected; default flags reproduce pilot bytes locally on 3 items | CPU + local GPU |
| M5 | Conditional `evaluate.py`, audit tools, J2 interface | D-04, D-10 | old columns unchanged (regression); audit export blinded | CPU |
| M6 | `commitment.py`, features, predict, stats | D-02, D-03, D-09, D-12 | synthetic recovery tests; **G1'** (all CPU tests green) | CPU |
| M7 | `intervene.py` modes | - | control identity; bias-mass equality; knockout locality | CPU + local GPU |
| M8 | Local GPU identity: custom loop equals pipeline, saved-latent reload equals recompute, `D=0` image identical | D-07, D-08 | all identities pass on the 3050 | local GPU |
| M9 | Kaggle probe and throughput (E03) | - | `hardware.json`, `throughput.json`, FULL/LIMITED decision; **G2'** | Kaggle |
| M10 | DEV generation, E01 DEV audit, E23 rehearsal, E25, power simulation, tuning; write Addendum B | D-02..D-06, D-09 | DEV gate on `Delta t50` half-width; hashes; **G3'** | Kaggle + CPU |
| M11 | TEST generation | M10 approved | `verify_outputs` clean; **G4'** | Kaggle |
| M12 | HA -> HB -> HC analysis | frozen Addendum | each analysed once; tables from files; **G5'** | CPU |
| M13 | Exploratory P2/P3 (E24, E26-E28, E16) | | labelled EXPLORATORY | Kaggle |
| M14 | Report, figures, README rebuild, reproduction check | | every number traces to a saved file | CPU |

## 16b. Executable acceptance checks
A milestone is complete only when every listed command has been run and its output saved under `outputs/acceptance/<MS>.txt`. Test file names are planned (none exists yet unless marked). "Pass" means exit code 0 **and** the stated criterion.

| MS | Command | Pass criterion |
|---|---|---|
| M0 | `git grep -n "^<<<<<<<" -- README.md`; `python report/build_report.py`; `python -m pytest tests/test_logic.py -q` | no match; build assertions pass; 6 passed (the pytest part exists today) |
| M1 | `python -m pytest tests/test_config.py tests/test_branch_math.py -q` | schema accepts pilot + V2 configs, rejects unknown/missing keys; `restart` with `D=0` returns the input; ratios equal `docs/evidence/sched_check_output.txt` (0.2099, 0.1843, 0.2795, 0.3994, 0.5381, 0.6834) within 1e-3; moment-test variance within 3% of `1 - abar` on 200,000 samples |
| M2 | `python src/analyze_pilot_conditional.py --check` | old-field tables reproduce byte-for-byte; prints P(pres), P(bind given pres), joint, eligibility by CFG |
| M3 | `python -m pytest tests/test_benchmark_v2.py -q`; `python src/benchmark_v2.py --validate` | every counterfactual differs in exactly one declared field; zero scenes/object pairs across splits; seeded duplicates detected; counts equal the quotas in the config (after D-05 C-1) |
| M4 | `python -m pytest tests/test_shards.py -q`; local `python src/generate.py ... --limit 3` twice | disjoint complete shards; seeded duplicate and corrupt PNG detected; second run generates 0; default-flag images byte-identical to the pilot images of the same names on the local GPU (otherwise the difference is recorded) |
| M5 | `python -m pytest tests/test_evaluate_conditional.py tests/test_audit.py -q`; `python src/validate_metrics.py` | old columns identical on `outputs/main/eval.jsonl`; metrics equal `outputs/validation_metrics.txt`; blinded export has no arm names |
| M6 | `python -m pytest tests/test_commitment.py tests/test_predict.py -q` | synthetic known commitment recovered within 6 steps; independent outcomes give `abs(C) < 0.1`; identical give `C = 1`; stable-wrong scene has weight 0; permuted labels give `abs(ΔAUC) < 0.03` |
| M7 | `python -m pytest tests/test_intervene_modes.py -q`; local `python tests/gpu_identity.py --mode control` | control equals baseline; bias mass equal across onsets within 5%; knockout bias only in the chosen layer group |
| M8 | `python tests/gpu_identity.py --loop --reload --d0` | custom loop image equals pipeline image; reloaded latent equals recomputed; `D=0` branch image equals base (pixel difference 0) |
| M9 | Kaggle R0 and R1 (RUNBOOK 8) | `env.json`, `hardware.json`, `throughput.json` exist and parse; hook-on equals hook-off on 10 items; FULL/LIMITED decision written |
| M10 | Kaggle R2, R3; `python src/verify_outputs.py --run <id>`; DEV gate script; `python src/verify_freeze.py` (planned) | zero duplicates/corrupt files; `D=0` identity; full-resample control `C <= 0.15`; DEV half-width of `Δt50` reported; Addendum B hashes verified |
| M11 | Kaggle R4 + `verify_outputs.py` | counts equal manifest; zero duplicates |
| M12 | `python src/analyze_commitment.py --frozen`, then HB and HC scripts | each run once; tables carry the Addendum hash; no TEST path read before the freeze check |
| M13 | exploratory scripts | outputs labelled EXPLORATORY |
| M14 | `python report/build_report.py`; clean-clone reproduction of one figure | assertions pass; numbers equal saved files |

## 17. Limitations, cost risks, fallback
Kaggle quota/limits unknown; E23 cost is a design estimate (RB 16.5b: P1 about 18,120 image-equivalents FULL, 7,702 LIMITED, reference only); HA power unknown; conditional sample about half of binding images; judge accuracy on branch images unknown; novelty gap unverified; single base model. Fallbacks: LIMITED design; drop E24, E26-E28 first; measurement paper (RB 21.6).

## 18. Final reproducibility checklist
- [ ] `git grep "^<<<<<<<"` empty; clean `main` tag for the frozen protocol
- [ ] Config snapshots, model revision pins, library versions recorded for every run
- [ ] Manifests and checksums verified; zero duplicates
- [ ] Addendum B hashes match files
- [ ] DEV/TEST separation audit logged
- [ ] CPU tests and GPU tests executed, outputs saved (`outputs/test_results.txt` regenerated)
- [ ] All report numbers generated from saved results
- [ ] README instructions run from a clean clone
- [ ] Limitations and exploratory labels present; AI-assistance disclosure per event policy
