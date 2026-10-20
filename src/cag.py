"""Commitment-Aligned Guidance (CAG, HE; GRAND_RESEARCH_BLUEPRINT.md section 24.5).
Schedule (all parameters fixed before TEST; nothing tuned on adherence):
    s*(prompt) = ceil( max over the prompt's constraint types of t50_DEV(type, w_hi) ) + margin      (margin m = 2)
    w(step)    = w_hi for step < s*, w_lo afterwards                                              (w_hi = 7.5; w_lo = 1 primary, 3 secondary)
    unresolved t50 (">36") for any present type  ->  s* = 40, i.e. CAG equals constant CFG for that prompt (reported, never hidden).
Arms: A1 constant w_hi (= the core images); A2 CAG; A3 one global s* (ceil(median of all t50_DEV) + margin); A4 reversed (w_lo before s*,
w_hi' after, with the same mean guidance as A2; undefined when s* = 40); A6 constant at A2's mean guidance.
The sampler is the same DDIM loop as branch.denoise_from. A step with w == 1 evaluates ONLY the conditional branch (as diffusers does when
guidance_scale <= 1), so the number of UNet evaluations is counted per image, not inferred. Pure schedule functions are torch-free."""
from __future__ import annotations

import math

KIND_OF_TYPE = {"presence": "presence", "count": "count", "binding": "bind_given_pres", "spatial": "spatial_given_pres", "size": "size_given_pres"}


def _resolved(v):
    return v is not None and v != ">36" and not (isinstance(v, float) and not math.isfinite(v))


def s_star(constraint_types, t50_dev, margin=2, steps=40):
    """Returns (s*, note). t50_dev maps outcome kind -> DEV t50 at w_hi (a number, or '>36' / None when unresolved)."""
    vals = []
    for ty in sorted(set(constraint_types)):
        v = t50_dev.get(KIND_OF_TYPE[ty])
        if not _resolved(v):
            return steps, f"unresolved commitment for '{ty}': s* = {steps} (CAG equals constant CFG for this prompt)"
        vals.append(float(v))
    return min(steps, math.ceil(max(vals)) + margin), "ok"


def s_star_global(t50_dev, margin=2, steps=40):
    vals = sorted(float(v) for v in t50_dev.values() if _resolved(v))
    if not vals:
        return steps
    mid = len(vals) // 2
    med = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
    return min(steps, math.ceil(med) + margin)


def schedules(s, w_hi=7.5, w_lo=1.0, steps=40):
    """Per-step guidance for A2 (CAG), A4 (reversed, equal mean) and A6 (constant at A2's mean). A4 is None when s >= steps."""
    if not 0 <= s <= steps:
        raise ValueError("s* outside [0, steps]")
    a2 = [w_hi if i < s else w_lo for i in range(steps)]
    mean = sum(a2) / steps
    out = {"A2": a2, "A6": [mean] * steps, "A4": None}
    if s < steps:
        w_hi_rev = w_lo + s * (w_hi - w_lo) / (steps - s)
        out["A4"] = [w_lo if i < s else w_hi_rev for i in range(steps)]
    return out, mean


def unet_calls(schedule):
    return sum(1 if w == 1 else 2 for w in schedule)


def sample_with_schedule(pipe, embeds, latents, schedule, start=0, decode=True):
    """DDIM eta=0 from step index `start` with a per-step guidance list. Returns (image or latents, unet_calls).
    Identity: a constant schedule [w]*n equals branch.denoise_from(..., cfg=w) and the diffusers pipeline (tests/test_cag_loop.py)."""
    import torch
    neg, pos = embeds
    steps = len(schedule)
    device = latents.device
    pipe.scheduler.set_timesteps(steps, device=device)
    ts = pipe.scheduler.timesteps
    calls = 0
    with torch.no_grad():
        for i in range(start, steps):
            w = float(schedule[i])
            if w == 1.0:
                inp = pipe.scheduler.scale_model_input(latents, ts[i])
                noise_pred = pipe.unet(inp, ts[i], encoder_hidden_states=pos, return_dict=False)[0]
                calls += 1
            else:
                inp = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), ts[i])
                noise_pred = pipe.unet(inp, ts[i], encoder_hidden_states=torch.cat([neg, pos]), return_dict=False)[0]
                n_u, n_c = noise_pred.chunk(2)
                noise_pred = n_u + w * (n_c - n_u)
                calls += 2
            latents = pipe.scheduler.step(noise_pred, ts[i], latents, eta=0.0, return_dict=False)[0]
        if not decode:
            return latents, calls
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])[0], calls
