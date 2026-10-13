# Decision register

Status values: **OPEN** (needs a decision), **PROPOSED** (resolution recommended, awaiting approval), **RESOLVED**.
"Approval" = human approval required before implementation. Research blueprint = `GRAND_RESEARCH_BLUEPRINT.md` (Revision 2); section numbers refer to it.
Nothing below is yet approved. Entries marked *verified* rely on `docs/evidence/sched_check_output.txt` (executed on CPU during the audit) or on files read in the repository.

## D-01 Restart indexing and direction (task B1)

| Aspect | Content |
|---|---|
| Conflict / gap | Blueprint 8.0 writes `x_{t-D}' = sqrt(abar_{t-D}/abar_t) x_t + sqrt(1 - abar_{t-D}/abar_t) eps` with `t` as a **denoising-step count** (grid {6,...,36}). `abar` is defined on the 1000 **training timesteps**. Read literally, `abar_6` would be about 1, which is wrong. |
| Verified | SD 1.5 DDIM with 40 steps uses timesteps 976, 951, ..., 1 (stride 25, offset 1). After `t` completed steps the latent sits at noise level `ts[t]` (`ts[6]=826`, `ts[12]=676`, ..., `ts[36]=76`); after 40 steps the level is `final_alpha_cumprod = abar_0 = 0.99915`. Noising back by `D` steps means moving to `ts[t-D]` (a **larger** noise level, smaller `abar`), so `abar_{t-D}/abar_t < 1`. The formula's direction is therefore correct once `abar` is indexed through the schedule. |
| Moment test (executed) | With `x0 ~ N(1, 0.25)` samples, forward-noised to level `t` then restarted, the residual `x' - sqrt(abar_target) x0` had variance 0.9800 (expected 0.9819) at `t=12` and 0.6189 (expected 0.6165) at `t=30`, mean about -0.001: consistent within sampling error. This is a Gaussian-marginal check, not proof that a real SD latent follows that marginal. |
| Proposed resolution | Define `lvl(t) = abar[ts[t]]` for `t<40`; `abar_final` for `t=40`. Restart target index `j = max(t - D, 0)`. Implement and test in `src/branch.py` as pure functions with a documented derivation in the module docstring. |
| Alternatives | (a) Define the grid in training-timestep units; (b) define `D` as a fixed `abar` ratio (see D-02). |
| Resolving check | CPU unit test (moment test on a synthetic Gaussian) and a GPU test that `D=0` returns the identical image. |
| Approval | No for the mathematics; yes for any change to the grid or `D` (D-02). |

## D-02 Restart magnitude and the boundary at `t=6` (task B2)

| Aspect | Content |
|---|---|
| Verified | `D=8` is clipped at the first grid point: requested index -2 becomes 0 (`ts=976`), an **effective jump of 6 steps**, ratio `abar_976/abar_826 = 0.21`. Moreover a fixed step jump is **not a fixed noise magnitude**: the ratio `abar_{t-D}/abar_t` is 0.21, 0.18, 0.28, 0.40, 0.54, 0.68 for t = 6, 12, 18, 24, 30, 36 (from `sched_check_output.txt`). Later restarts therefore inject relatively **less** noise. Also, `ts[0]=976` is not pure noise (`abar=0.0061`); the "full-noise level" control in the blueprint is only approximately a full resample. |
| Consequence | Persistence `C(t)` will tend to rise with `t` by construction. The absolute value of `t50` is not interpretable as a property of the model. Differences between constraint types at the same `t` remain comparable because every type uses the identical operation. |
| Proposed resolution | (1) Keep `D=8` and the grid for the pre-registered primary analysis, but **log effective jump and ratio at each grid point** and flag `t=6` as a clipped point. (2) Add a mandatory sensitivity analysis (E23b) using an **equal-ratio** parameterisation (choose `j` so that `abar_j/abar_t` is about 0.30 at every `t`) and `D` in {4,12}. (3) Make HA's support statement conditional on sign agreement under the equal-ratio variant. (4) For the full-resample control, restart to the pipeline's own starting noise (fresh `N(0,I)`), not to level `ts[0]`. |
| Alternatives | Shift the grid to {8,14,20,26,32,38} (no clipping, changes `t50` resolution); report `t=6` as clipped without sensitivity. |
| Resolving check | Compute `C(t)` on a synthetic outcome model where the true commitment is known and the injected-noise ratio varies like the real schedule; check that `Delta t50` between types is recovered while absolute `t50` is biased. |
| Approval | **Yes** (changes the pre-registered design). |

## D-03 What "commitment" means (task B3)

| Aspect | Content |
|---|---|
| Concern | `C_k(t)` measures **persistence of a judge-scored outcome under bounded local re-noising**, relative to the agreement of independent seeds. It does not show that the model "decided" a constraint internally. |
| Assumptions | Judge error is random and symmetric; branch noise draws are independent of the base; the outcome distribution given the prompt is comparable across scenes; `A0` estimated from 4 seeds (6 pairs) is unbiased. |
| Edge cases | `A0` near 1 gives a near-zero denominator weight (carried by the ratio of sums); **stable wrong outcomes** (always failing) have `A=A0=1`, contribute nothing, and say nothing about commitment; negative `C` is possible with noise and must not be clipped silently; non-monotone `C(t)` marks an unreliable estimate; `t50` may not be reached ("> 36"); judge error attenuates `A` and `A0` differently; branch images may be out of distribution for the detector. |
| Proposed terminology | "restart-persistence" / "outcome lock-in under re-noising". Use "commitment" only after validation (synthetic recovery, Delta/judge sensitivity, E18) and then phrased as "operational commitment". Keep the blueprint's symbol names but define them in the Addendum with this wording. |
| Resolving check | Synthetic commitment-recovery test; Delta sensitivity; judge sensitivity. |
| Approval | **Yes** (terminology in the pre-registration). |

## D-04 Conditional outcomes and selection effects (task B4)

| Aspect | Content |
|---|---|
| Verified | Existing `score()` gives 0 for binding and spatial when an object is missing (F-3). In the pilot only about 45% of binding images have both objects (blueprint 2.7 F2; quoted, not re-derived here). |
| Selection problem | Eligibility for `Y_bind|pres` is itself an outcome of generation. Comparing arms or branch times on the eligible subset conditions on a post-treatment variable. For HA, pairs are eligible only if both the base and the branch image contain the objects, so eligibility depends on `t` and on the branch noise. |
| Proposed resolution | Report **three quantities** everywhere: `P(pres)`, `P(bind|pres)` with its n, and the joint `P(bind and pres)` (omission counted as failure). Primary HA outcome: agreement on `Y_bind|pres` with eligibility counts reported per grid point; **required companion**: agreement of the joint outcome. A conclusion is stated only if both agree in sign. HC: report conditional and unconditional contrasts (already in blueprint 15.8). Add `objects_present` and `bind_given_pres` / `spatial_given_pres` as **new** fields next to the old ones. |
| Alternatives | Principal-stratification bounds (more complex); restricting to scenes where presence is near-certain (loses coverage). |
| Resolving check | On the pilot: compute the joint and conditional rates and the eligibility fraction by CFG (no generation). |
| Approval | **Yes** (primary outcome definition). |

## D-05 Count reconciliation (task B5)

| ID | Observation (blueprint locations) | Problem | Proposed reconciliation |
|---|---|---|---|
| C-1 | 9.0: TEST binding 60 + multi 50 = 110, DEV 30 + 25 = 55; text says "about 140" / "about 70" binding-bearing (8.1, 8.6.1, 11.9, 15.4, 16.5) | 110 < 140 and 55 < 70 unless count/spatial/presence scenes also carry colour attributes. Power numbers (half-width 0.08, MDE 0.07) were computed for 140. Also affected: E12 DEV tuning count `70*8*2 = 1,120` (LIMITED 41) and every E05/E12/E28 count that uses 140. | Decide the definition of "binding-bearing" (any scene with at least one colour-object constraint). Then either change the quotas to guarantee 140/70 or recompute the power with the real number. **Open; blocks the Addendum.** |
| C-2 | E23: about 100 TEST canonical scenes, binding about 70 | 70 is 0.7 of 100, consistent with 140/200 but not with C-1 (110/200 = 0.55 gives about 55). After the presence filter (about 45% eligible per image) the effective binding sample is smaller still (a rough estimate: 25-30 scenes with usable pairs). | Recompute after C-1; enlarge the commitment subset if the DEV gate requires. |
| C-3 | 16.5b: E24 is P2 but is included in the "P0-P2-lite" total (19,520 / 8,182); P1 total 18,120 / 7,702 | A reader may take 19,520 as the P1 cost. | Present both totals; budget decisions use the P1 total first. |
| C-4 | E23 image-equivalents use 0.667 (six grid steps) but 0.74 for LIMITED (five steps) | Formula assumes the remaining fraction; an unclipped `t=6` branch actually has a longer path | Replace by measured `t_branch` after M9. |
| C-5 | E23b is P2 yet is needed to qualify HA | If it is dropped, HA is unqualified (D-02) | Promote the equal-ratio sensitivity to P1 (it reuses E23 code; DEV-scale cost). |
| C-6 | Pilot k=6 seeds vs V2 k=4 vs branch bases k=2 | Three seed counts in one design | Fix: V2 core k=4; branches use the first 2 of the 4; DEV tuning k=2. Write it into `configs/experiment_v2.json`. |
| C-8 | RB 11.4 requires nouns to pass the synthetic detector probe (E22) and calibration (E03) before the lexicon is final, but the plan builds the benchmark (M3) before E22 (M5) and E03 (M9) | Dependency inversion | M3 delivers schema, grammar, validators and a *draft* lexicon; the lexicon is frozen only after M5 and M9, before DEV generation in M10 (MASTER 16). |
| C-7 | Edit arms: E12 uses 6 arms FULL (3,360 images) vs 4 arms LIMITED | LIMITED drops mid and global arms; HC is still defined by early/late | Accept; record that LIMITED cannot report dose-response. |

Approval: **yes** for C-1 (changes the sample-size basis).

## D-12 Undefined cases of DEV-derived quantities (found in review)
| Quantity (RB) | Undefined when | Proposed rule |
|---|---|---|
| `tau_pre` = last grid step <= 0.5 * `t50_DEV` (8.2) | `t50_DEV` < 12 (no grid step >= 6 qualifies) or `t50_DEV` = "> 36" | If undefined, HB `ΔAUC_pre` uses `tau = 6` and is labelled EXPLORATORY; record the rule in the Addendum |
| `tau_post` = first grid step >= `t50_DEV` + 8 | `t50_DEV` > 28 | HB cannot test readout-after-commitment; mark `post` not estimable |
| HC mid onset = `t50_DEV`, late onset = `ceil(t50_DEV) + 4` capped at 28, window L = 12 (8.3) | `t50_DEV` > 28 or "> 36": a mid window would end after step 40 | Cap every onset so that onset + 12 <= 40; if `t50_DEV` > 28 the early/late contrast loses its meaning, so HC reduces to `D_early` and `G` is not estimated |
| `t50` interpolation | `C` never reaches 0.5 or is non-monotone | Report ">36"; use the first crossing; flag non-monotone curves (D-03) |
Approval: yes (Addendum B text).

## D-06 DEV/TEST separation and leakage (task B6)

| Aspect | Content |
|---|---|
| Risk points | (1) TEST core images must carry saved latents, so the E04 generation flags must be frozen before TEST is generated. (2) HB windows (`tau_pre`, `tau_post`) and HC onsets come from `t50_DEV`; the analysis must read frozen numbers from the Addendum, not recompute on TEST. (3) The branch `A0` uses TEST core images; no TEST label may be inspected before the freeze. (4) The pilot's 28 prompts may enter DEV only. (5) Scene-level splitting must hold object-pair groups together (blueprint 11.6). (6) Judge selection (J1 vs J2) is on DEV audit only. |
| Proposed safeguards | `splits/manifest.json` with SHA-256; a `test_sealed.flag` check in every script that reads TEST paths (the script refuses unless the Addendum hash is present); `SESSION_STATE.md` records the freeze commit. |
| Approval | Yes (freeze procedure). |

## D-07 Branch re-denoising implementation

| Aspect | Content |
|---|---|
| Question | Use the diffusers pipeline with a truncated `timesteps` list, or a custom loop? |
| Evidence | Diffusers' DDIM computes the previous timestep from `num_inference_steps` (F-6). A truncated list would change the stride. Not yet tested here. |
| Proposed resolution | Custom loop: `set_timesteps(40)`, then for `i` in `range(j, 40)` call the UNet with CFG and `scheduler.step`, **using the same code path as the pipeline** (including `init_noise_sigma`, fp16 cast, guidance formula, negative prompt ""). Acceptance: the loop started at `i=0` from the stored initial latent reproduces the pipeline image bit-for-bit on the target GPU; if not, record the difference and decide. |
| Approval | No. |

## D-08 Latent saving format

fp16 latents are what the pipeline holds, so saving them as fp16 is lossless for the pipeline; CPU-generated branch noise must be created in float32 then cast to fp16. Proposed: one `.pt` per base image holding the six latents, keyed by step, plus a hash in `samples_gpuN.jsonl`. Check: reload equals recompute (10 items). Approval: no.

## D-09 Statistical items still open

| Item | Question | Resolution path |
|---|---|---|
| HA power | Unknown (no commitment data) | DEV gate: half-width of `Delta t50` > 4 steps means increase M or scenes, else exploratory |
| HB nested-AUC variance | Unknown | DEV simulation of the TEST design; half-width > 0.04 means HB exploratory |
| `t50` as a functional | Coarse (grid 6, interpolation) | Always show full curves; the primary decision uses the bootstrap of `Delta t50` |
| Multiplicity | Holm over HA, HB, HC with composite outcomes | Fix in Addendum B; the primary statistic per hypothesis is named |
| Bootstrap | Scene-level, 4000 draws | Per-call seeded RNG (existing finding in blueprint 2.5) |
| Approval | Yes for the final Addendum. |

## D-10 Evaluator

J1 (OWL-ViT + CLIP) is the only judge that exists. Pilot validation accuracies (n small, labels began as an AI draft): presence 0.93 (n 58), binding 0.89 (n 37), count 1.00 (n 10), spatial 0.89 (n 9, 2 positives). J2 (VLM) choice and fit in 16 GB are unverified. Branch and edited images may shift detector behaviour. Resolution: E01/E18 on DEV; types below 0.85 audit accuracy stay exploratory. Approval: yes (primary judge rule in the Addendum).

## D-11 Repository and process

| Item | Decision needed |
|---|---|
| README conflict markers (F-1) | Fix in M0 (no scientific content change) |
| Kaggle artefact storage | Proposed: Kaggle output / private Kaggle dataset for bulk; Git for manifests and compact results only (RUNBOOK 6) |
| Pushing the branch | Not done; needs the owner's GitHub authentication. The repository is private (anonymous read fails), so Kaggle needs a read token, a private dataset upload, or a public repo: owner decision |
| AI-assistance disclosure | Check the event's policy; record provenance of labels and code (blueprint risk register) |
| Novelty | The commitment-curve gap is **not established**; do not use "first" or "novel" in any document before the full-text checks listed in blueprint section 22 |

## Summary of items blocking implementation

Blocking: D-02, D-03, D-04, D-05 (C-1), D-06, D-12 (approval). Non-blocking for the CPU modules in M1: D-01, D-07, D-08.
