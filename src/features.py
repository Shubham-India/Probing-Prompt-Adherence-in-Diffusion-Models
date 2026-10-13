"""Feature builders for HB (RB 8.2): prompt-structure features and initial-noise features. CPU only, no model weights.
Noise features are reproduced from the seed exactly as generate.initial_latents does (CPU generator, randn (1,4,size/8,size/8)).
Attention-conflict features already exist in analyze_attention.py / attention.py and are joined by name, not re-implemented here."""
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=None)
def initial_noise(seed, size=512):
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((1, 4, size // 8, size // 8), generator=g)[0].numpy().astype(np.float64)


@lru_cache(maxsize=None)
def noise_features(seed, size=512, grid=4, radial_bins=4):
    """Channel mean/std, grid x grid pooled means per channel, low-frequency radial spectrum energy per channel. Returns a dict."""
    z = initial_noise(seed, size)
    f = {}
    h, w = z.shape[1:]
    for c in range(z.shape[0]):
        f[f"n_mean_c{c}"] = float(z[c].mean())
        f[f"n_std_c{c}"] = float(z[c].std())
        for i in range(grid):
            for j in range(grid):
                f[f"n_grid_c{c}_{i}{j}"] = float(z[c, i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid].mean())
        P = np.abs(np.fft.fftshift(np.fft.fft2(z[c]))) ** 2
        yy, xx = np.indices(P.shape)
        r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
        for b in range(radial_bins):
            sel = (r >= b / radial_bins * 0.5) & (r < (b + 1) / radial_bins * 0.5)
            f[f"n_spec_c{c}_b{b}"] = float(P[sel].mean() / P.mean())
    return f


def prompt_features(constraints, prompt_text=""):
    """Structure-only features from the constraint list and prompt string (no text encoder)."""
    types = [c["type"] for c in constraints]
    f = {"p_n_constraints": len(constraints), "p_n_objects": len({c.get("object") for c in constraints if "object" in c}),
         "p_n_count": types.count("count"), "p_n_binding": types.count("binding"), "p_has_spatial": int("spatial" in types)}
    low = prompt_text.lower()
    bind = [c for c in constraints if c["type"] == "binding"]
    if bind:
        pos = [(low.find(c["object"]), low.find(c["color"])) for c in bind]
        f["p_attr_noun_dist"] = float(np.mean([abs(a - b) for a, b in pos if a >= 0 and b >= 0] or [np.nan]))
        f["p_first_coloured_is_first_object"] = int(low.find(bind[0]["color"]) <= min((low.find(c["object"]) for c in constraints if "object" in c), default=0) + len(bind[0]["color"]) + 3)
    else:
        f["p_attr_noun_dist"] = np.nan
        f["p_first_coloured_is_first_object"] = np.nan
    f["p_has_background"] = int(" on a " in low)
    return f
