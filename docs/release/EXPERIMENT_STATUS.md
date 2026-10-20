# Experiment status (2026-10-18)

| Experiment | What | Depends on | Implementation | Checks done | Data | Outstanding |
|---|---|---|---|---|---|---|
| Pilot | 28 prompts x 3 CFG x 2 steps x 6 seeds, attention, global re-weighting | - | done | regression: 1,008 images re-score identically | PILOT RESULT (exploratory) | second review of 40 labels |
| R0 | T4 identity checks (Rev. 2 code) | - | done | 6 identities, 3 cases | infrastructure | - |
| R1 | throughput | R0 | done | two sessions (6.2 vs 7.8 s) | infrastructure | reconcile in PREFLIGHT |
| R2a | DEV canonical core + latents (400) | R0 | done | independent audit 22/22 | DEV artifacts (first-pass judge unaudited) | - |
| PREFLIGHT (00) | R3 hardware/identity/timing | gbc-source | implemented | smoke-tested on RTX 3050 (all identities hold); dry-run | NOT RUN on Kaggle | run |
| DEV audit (human) | J1 accuracy per type | R2a | package ready | blinded UI tested | no labels | owner labelling, annotator B, pilot reviewer |
| DEV_CORE (00) | DEV variants, CF bases, DEV PS family | variant review | implemented | dry-run | NOT RUN | gate |
| DEV_COMMIT (00) | E23-DEV, E29-DEV, E31-DEV | audit, D-13, R3 authorisation | implemented | dry-run | NOT RUN | gates |
| DEV analysis (04 DEV) | t50_DEV, DEV gates | DEV_COMMIT | implemented | unit tests, dry-run | NOT RUN | - |
| DEV_CAG_GUARD (00) | w_lo 1 vs 3 | dev constants | implemented | dry-run | NOT RUN | guard metric (DL-15) |
| Freeze | Addendum C + protocol_freeze.json | all DEV + owner decisions | implemented | dry-run freeze/sign/verify; dry-run protocol refused for real runs | NOT FROZEN | OWNER rows of DECISION_LOG |
| TEST_CORE (01) | E04-TEST, E11 | freeze | implemented | dry-run | NOT RUN | - |
| TEST_COMMIT (02) | E23, E29, E31 CF-C/R | 01 latents, freeze | implemented | dry-run | NOT RUN | - |
| TEST_CAG_SIZE (03) | E30, E32, E31 CF-S | freeze | implemented | dry-run | NOT RUN | - |
| Confirmatory (04 TEST) | HA, HD, HE, HF + Holm | all TEST | implemented | unit tests on synthetic worlds, dry-run | NOT RUN | - |
| Paper | `paper/` from results files | results | implemented (R3 sections pending) | static checks | - | LaTeX toolchain to compile |

Test suite and dry-run results of this session are recorded in `validation_report.md` in this folder.
