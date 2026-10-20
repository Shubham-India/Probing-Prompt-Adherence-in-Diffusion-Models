# Project status

Written 2026-10-17 from the working tree (branch `claude/admiring-cori-pbn9wk`, HEAD `daef6c4`, **uncommitted changes present and preserved**). Status words: VERIFIED COMPLETE, PARTIAL, PLANNED, BLOCKED, FAILED, NOT VERIFIED. "Verified" means a test or an artifact check was executed; a mock or toy pipeline never counts as proof of real-model GPU correctness.

## 1. Test evidence
`python -m pytest tests -q` (Windows, `.venv`, torch 2.6.0+cu124, RTX 3050): **121 passed, 1 warning** (2026-10-17). The GPU attention test is included and ran. Not covered by any test: real Stable Diffusion 1.5 behaviour on Kaggle (covered only by the R0 artifact), the evaluator on natural images (covered only by the unaudited first pass), and LaTeX compilation (no LaTeX toolchain on this machine; `paper/check_paper.py` performs static checks only).

## 2. Component inventory and gap matrix
| Component | Path | Status | Evidence | Remaining work | Safe to change now? |
|---|---|---|---|---|---|
| Configs, schema, pinned revision | `configs/experiment*.json`, `src/config_schema.py` | VERIFIED COMPLETE (SD 1.5 pinned) | tests; R2a records carry the revision | pin and record OWL-ViT/CLIP revisions (DATASETS 5) | yes, protocol-neutral |
| Benchmark V2 | `src/benchmark_v2.py`, `data/benchmark_v2/` | PARTIAL | validators and counts; DATASHEET | human variant review, lexicon freeze | no (review first) |
| Generation, sharding, latents, merge, verify | `generate.py`, `jobs.py`, `merge_samples.py`, `verify_outputs.py` | VERIFIED COMPLETE for R2a scope | R2a audit 22/22 checks; CPU tests | byte-identity with pilot images on the same GPU not checked | yes |
| Run integrity (independent) | `audit_run.py`, `verify_branches.py`, `kaggle/finish_run.py` | VERIFIED COMPLETE (CPU, fakes + real R2a) | tests; real R2a audit | first use on real branch output | yes |
| Stage guard and gates | `stage_guard.py`, `configs/stage_gates.json` | VERIFIED COMPLETE | tests; dry run against R2a prints 7,200 jobs | gates stay false until the owner sets them | yes |
| Restart math and branching loop | `branch.py`, `run_branches.py` | PARTIAL | CPU math tests; R0 GPU identity (D=0, reload, loop) on 3 cases | real branch generation never run (R3 blocked) | no change needed |
| Attention hook and interventions | `attention.py`, `edit_probe.py`, `intervene.py` | PARTIAL | pilot runs; toy-layer tests; control identity on tested cases | targeted/random/onset arms on real SD never run | no |
| Evaluator J1 with conditional fields | `evaluate.py` | PARTIAL | old columns unchanged (regression); first pass on 400 R2a images | **human audit pending**; OWL-ViT/CLIP revisions unpinned | only additive |
| Evaluator audit tooling | `audit_dev.py`, `audit_sample.py`, `validate_human.py`, `synth_probe.py` | VERIFIED COMPLETE (tooling); audit itself BLOCKED | tests; packages built | owner labels; E22 threshold approval and run | yes |
| Persistence statistics | `commitment.py`, `analyze_commitment.py` | PARTIAL | synthetic recovery tests only | no real branch data | no |
| HB features/models | `features.py`, `predict_failure.py`, `analyze_noise.py` | PARTIAL | synthetic + pilot | V2 attention maps absent | no |
| HC analysis | `analyze_hc.py` | PARTIAL | synthetic | no data | no |
| Sizing simulations | `dev_gate_sim.py`, `dev_gate_observed.py` | PARTIAL (planning aid) | tests | rerun with measured error rates | yes |
| Freeze tooling | `freeze.py` | VERIFIED COMPLETE (portability bug fixed) | 6 tests | nothing frozen | yes |
| Results-to-paper chain | `build_draft.py`, `paper/` | PARTIAL | tests; 18 ledger rows; static checks pass | PDF not compiled; HA/HB/HC sections pending by design | yes |
| Kaggle scripts | `kaggle/*.py` | VERIFIED COMPLETE for R0-R2a use; new scripts CPU-tested only | R0/R1/R2a artifacts | new scripts need a commit and a new bundle | yes |
| Pilot documents | `report/*` | PARTIAL (stale vs final labels) | timestamps | rebuild when convenient | yes |
| Prior-art review | blueprint section 4-5 | PARTIAL | abstract level for the closest timing/noise papers | read full texts (list in blueprint section 22) | owner |

## 3. Scientific gates
| Gate | State |
|---|---|
| DEV evaluator audit (labels, per-type error) | pending owner labels |
| Pilot-40 second review | pending a reviewer |
| Synthetic probe + 0.95 floor approval | pending approval (and a GPU/model run) |
| Variant and lexicon review | pending |
| D-13 | decided 2026-10-14: R3 deferred, no 4 base seeds to chase the gate, gate unchanged; re-sizing after labels |
| Working defaults D-02..D-06, D-12 | approved as working defaults only |
| Addendum B | draft, unsigned, DEV values unset |
| Freeze | not done (`freeze.py --verify` fails as expected) |
| TEST | sealed; nothing generated, scored or inspected |

## 4. Findings so far (all EXPLORATORY or infrastructure; see `docs/RESULTS_PROVENANCE.md`)
Pilot: no CFG effect resolved with 4-6 prompts per category; global re-weighting +0.02 (CI -0.04 to +0.09); 61% of binding-bearing pilot images have all required objects (41% for binding+multi); early attention adds +0.050 AUC (CI +0.017 to +0.097) over design+seed in prompt-grouped CV; seed 0 lower (p = 0.012). R2a first pass (unaudited judge): all objects detected in 40% of binding-bearing DEV images, binding given presence 0.49 (n = 113), joint 0.20. Infrastructure: R0 identities hold on 3 cases, R2a integrity 22/22. **HA, HB, HC: no conclusions.**

## 5. What was done in the 2026-10-17 session
Added `stage_guard.py` + gates file, `kaggle/finish_run.py`, `kaggle/make_core_dataset.py`, `build_draft.py` with tests (121 pass), the LaTeX draft under `paper/`, and `docs/{KAGGLE_MASTER_RUNBOOK,DATASETS_AND_TRAINING,HYPOTHESIS_LEDGER,PROJECT_STATUS,RESULTS_PROVENANCE}.md`. Found and recorded: the runbook's R2b count (1,880) would have regenerated the 400 canonical images (correct: 1,480).

## 6. Human actions needed
1. Label the DEV audit (`local_audit/audit_dev/annotator_A/label.html`, 560 items) -> `local_audit/audit_dev/labels/labels_A.csv`.
2. Find annotator B (a different person) for `annotator_B` -> `labels_B.csv`; a reviewer for `audit_pilot40` -> `labels_pilot40.csv`.
3. Review PAR/ORD/CF variants and the lexicon (`data/benchmark_v2/prompts_v2.json`).
4. Approve or amend the synthetic-probe floor (0.95) in `docs/AUDIT_PROTOCOL_DEV.md` section 7.
5. When ready for any Kaggle stage: commit the new scripts, rebuild the bundle, set the relevant gate in `configs/stage_gates.json` in a commit.
6. Install a LaTeX toolchain and run `latexmk -pdf paper/main.tex` (or `python paper/check_paper.py --compile`).
7. Complete the AI-assistance disclosure placeholder in the paper according to the event policy.

## 7. Rebuild the draft (no generation)
```
python src/build_draft.py          # results.json, ledger, tables, number macros, figures, docs/RESULTS_PROVENANCE.md
python paper/check_paper.py        # static checks (add --compile when a LaTeX toolchain exists)
```

## NEXT ACTION FOR OWNER
Open `local_audit/audit_dev/annotator_A/label.html` in a browser, label the items following `docs/AUDIT_PROTOCOL_DEV.md` section 3, and save the downloaded CSV unedited as `local_audit/audit_dev/labels/labels_A.csv`. Then tell the assistant; it will score the audit, re-run the D-13 sizing with the measured error rates, and update the paper tables automatically.
