# Known limitations (2026-10-18)

## Not verified / not run
- **No Revision 3 experiment has been run.** HA, HD, HE, HF have no data; `main_latex.pdf` shows illustrative placeholder values (DL-02) and must not be used as results.
- **Kaggle**: session length, weekly quota, current accelerator options and the multiple-account rule could not be read from Kaggle's own pages (JavaScript-rendered); figures in the runbook are measured throughput only. The T4 throughput range (6.2-7.8 s/image) comes from two unreconciled sessions.
- **GPU identity on Kaggle for Revision 3 code**: CAG loop identities, hook-on/off identity and w = 1 evaluation counts were smoke-tested on the local RTX 3050 with real SD 1.5 (2 prompts), not yet on a T4; notebook 00 `PREFLIGHT` checks them on Kaggle.
- **Evaluator**: the DEV human audit has no labels yet; types below 0.85 accuracy become exploratory; relative size needs >= 140 audited judgements. The secondary VQA judge is not implemented.
- **CAG quality guard**: HPS v2 is not implemented; the owner must choose the guard metric before the freeze (DL-15).
- **DINOv2 / LPIPS**: DINOv2 loading and timing measured locally; LPIPS (secondary) needs the `lpips` package, not installed.
- **CPU dry-run** uses a tiny random pipeline and fake detections: it validates control flow, manifests, verification, archives and analysis plumbing, not model behaviour.

## Design limitations
- One model (SD 1.5) and one sampler (DDIM, 40 steps); no claim for SDXL, SD3 or transformer-based models (hooks not verified there).
- Restart persistence is relative to the re-noising magnitude; the noise ratio varies with the step (0.21 at t = 6 to 0.68 at t = 36), so absolute t50 is not a model property; ER30 sensitivity is required for HA support.
- Conditional outcomes select on presence; joint outcomes are always reported beside them.
- Size exists only in the PS family (8 DEV / 16 TEST pairs), so size conclusions rest on few scene groups.
- The reversed CAG arm can require extreme guidance when s* is late (DL-18).
- Grid resolution is 6 steps (t50 is interpolated).
- The benchmark has 1,410 prompts, not 1,500 (DL-04); the lexicon and variants still await human review.
- Compute: about 27,200-28,700 image-equivalents from the manifests vs 22,196 in the blueprint (DL-24).

## Literature
The closest works (Prompt-Mixing, ProSpect, Dynamic CFG, Step Selection, CADS, Self-Guidance, critical windows, Diffusion Probe, CoBind) were checked at abstract/HTML level only; full-text reading is still pending, so novelty is stated only as "to our knowledge" and is not established.

## Kaggle environment (2026-10-18)
- The torchao fix (DL-27) was reproduced and verified locally (Windows, Python 3.13.5, torch 2.6.0+cu124) and in a throwaway venv with torchao 0.12.0; it has **not yet run on the Kaggle image** (Python 3.13, torch 2.11.0+cu128, 2x T4). The first Kaggle evidence is `env_bootstrap.json` from a PREFLIGHT run.
- `pip check` on the Kaggle image may list conflicts among preinstalled packages; only conflicts that name a project dependency stop the run. Whether the pinned set produces such a conflict on Kaggle is unknown until PREFLIGHT.
- The two-GPU worker path, T4 timings, the judge/DINOv2 downloads with Internet on, and session/quota limits are verified only by a real PREFLIGHT run.
