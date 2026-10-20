"""Restart (partial re-noising) mathematics for the restart-persistence experiments (HA).

Scope of this module (milestone M1): pure, scheduler-indexed functions only. The custom
denoising loop that continues from a restarted latent lives in a later milestone (M8) and
is validated on a GPU; nothing here has been validated on real SD 1.5 latents.

Indexing (verified against diffusers DDIMScheduler, see docs/evidence/sched_check_output.txt)
  * The sampler runs `n` steps over timesteps ts[0] > ts[1] > ... > ts[n-1]  (SD 1.5, n=40: 976, 951, ..., 1).
  * After `t` completed steps the latent sits at noise level  abar_level(t) = abar[ts[t]]  for t < n,
    and at abar_final (= abar[0] when set_alpha_to_one=False) for t = n.
  * A restart of step-jump D from step t targets index j = max(t - D, 0): a LARGER noise level
    (smaller abar). With r = abar_level(j) / abar_level(t) < 1 the forward-marginal-consistent
    transition is
        x' = sqrt(r) * x_t + sqrt(1 - r) * eps ,   eps ~ N(0, I)
    which has the marginal of a noised sample at level j if x_t has the marginal at level t.
    (Gaussian-marginal property only; it is a premise, not a verified fact for real SD latents.)
  * Requested jump D may be clipped at the start of the schedule (j would be < 0); the effective jump
    t - j and the ratio r are always reported, and clipped points are flagged.
  * A fixed step jump is NOT a fixed noise ratio (r varies with t); `equal_ratio_plan` gives the
    sensitivity parameterisation (D-02).
"""
from __future__ import annotations

import numpy as np

# Stable Diffusion 1.5 noise schedule (scaled_linear), DDIM "leading" spacing with steps_offset = 1.
SD15 = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, steps_offset=1, set_alpha_to_one=False)


def alphas_cumprod(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, **_):
    betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


def ddim_timesteps(n, num_train_timesteps=1000, steps_offset=1, **_):
    """Same rule as diffusers DDIMScheduler(timestep_spacing='leading'): (arange(n) * stride)[::-1] + offset."""
    stride = num_train_timesteps // n
    return (np.arange(n) * stride)[::-1].astype(np.int64) + steps_offset


class Schedule:
    def __init__(self, n=40, **cfg):
        c = {**SD15, **cfg}
        self.n = n
        self.ac = alphas_cumprod(**c)
        self.ts = ddim_timesteps(n, **c)
        self.final = float(self.ac[0]) if not c["set_alpha_to_one"] else 1.0

    def abar_level(self, t):
        """Noise level (alpha-bar) of the latent after t completed steps, 0 <= t <= n."""
        if not 0 <= t <= self.n:
            raise ValueError(f"step {t} outside [0, {self.n}]")
        return self.final if t == self.n else float(self.ac[self.ts[t]])


def _plan(sch, t, j, requested):
    r = sch.abar_level(j) / sch.abar_level(t)
    return dict(t=t, requested_jump=requested, target_idx=j, effective_jump=t - j, clipped=(t - requested) < 0,
                ts_t=None if t == sch.n else int(sch.ts[t]), ts_target=int(sch.ts[j]),
                abar_t=sch.abar_level(t), abar_target=sch.abar_level(j), ratio=r)


def restart_plan(sch, t, D):
    """Fixed step-jump design. D=0 gives j=t, ratio exactly 1."""
    if D < 0:
        raise ValueError("D must be >= 0")
    return _plan(sch, t, max(t - D, 0), D)


def equal_ratio_plan(sch, t, target_ratio=0.30):
    """Sensitivity design: choose j <= t whose ratio abar(j)/abar(t) is closest to `target_ratio` (D-02).
    The achieved ratio is returned; at small t it may differ from the target because the schedule is discrete/bounded."""
    best = min(range(0, t + 1), key=lambda j: abs(sch.abar_level(j) / sch.abar_level(t) - target_ratio))
    return _plan(sch, t, best, t - best)


def restart(x_t, ratio, noise):
    """x' = sqrt(r) x_t + sqrt(1 - r) noise. `noise` must be N(0, I) of the same shape (generate it on CPU, float32,
    from a seeded generator, then cast; D-08). ratio == 1 returns x_t unchanged."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if ratio == 1.0:
        return x_t
    return ratio**0.5 * x_t + (1.0 - ratio) ** 0.5 * noise


# ----- denoising loop (torch / diffusers) -------------------------------------------------------------------------------
# Same code path as StableDiffusionPipeline.__call__ for the DDIM, eta = 0, classifier-free-guidance case (D-07):
#   scale_model_input -> unet on [uncond, cond] -> guidance -> scheduler.step -> vae decode -> image_processor.postprocess.
# Validated here only against a tiny random UNet/VAE on CPU (tests/test_branch_loop.py); identity with the real SD 1.5 fp16 pipeline
# on a GPU is checked by tests/gpu_identity.py and is NOT established by this repository snapshot.
def encode_prompt_pair(pipe, prompt, device, negative_prompt=""):
    """(negative_embeds, positive_embeds) exactly as the pipeline computes them."""
    pos, neg = pipe.encode_prompt(prompt, device, 1, True, negative_prompt=negative_prompt)
    return neg, pos


def denoise_from(pipe, embeds, latents, start, steps=40, cfg=7.5, decode=True):
    """Continue denoising `latents` (the model input of step index `start`) through the last step. start=0 with the initial noise
    reproduces the pipeline. `embeds` = (negative, positive). Returns a PIL image (decode=True) or the final latents."""
    import torch
    neg, pos = embeds
    device = latents.device
    pipe.scheduler.set_timesteps(steps, device=device)
    ts = pipe.scheduler.timesteps
    prompt_embeds = torch.cat([neg, pos])
    with torch.no_grad():
        for i in range(start, steps):
            inp = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), ts[i])
            noise_pred = pipe.unet(inp, ts[i], encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            n_u, n_c = noise_pred.chunk(2)
            noise_pred = n_u + cfg * (n_c - n_u)
            latents = pipe.scheduler.step(noise_pred, ts[i], latents, eta=0.0, return_dict=False)[0]
        if not decode:
            return latents
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])[0]


def branch_noise(shape, seed, dtype, device):
    """Branch noise N(0, I): drawn in float32 on the CPU from a seeded generator, then cast (D-08)."""
    import torch
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


def branch_generate(pipe, embeds, x_saved, t_completed, plan, noise_seed, steps=40, cfg=7.5, decode=True):
    """Restart branch: re-noise the saved latent (after `t_completed` steps) by `plan` (branch.restart_plan / equal_ratio_plan),
    then denoise from step index plan['target_idx']. plan['ratio'] == 1 (D=0) returns the unmodified latent (identity)."""
    assert plan["t"] == t_completed, "plan does not belong to this latent"
    if plan["ratio"] == 1.0:
        x = x_saved
    else:
        eps = branch_noise(x_saved.shape, noise_seed, x_saved.dtype, x_saved.device)
        x = restart(x_saved.float(), plan["ratio"], eps.float()).to(x_saved.dtype)
    return denoise_from(pipe, embeds, x, plan["target_idx"], steps, cfg, decode)
