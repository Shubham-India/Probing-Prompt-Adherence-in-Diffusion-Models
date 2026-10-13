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
