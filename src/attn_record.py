"""Revision 3 cross-attention recording for HF: per denoising step, per constraint word, maps averaged over heads and over the layers of each
resolution group (16x16 and 32x32), conditional branch only. Reuses the pilot processor (attention.StoreCrossAttention), whose output path is the
fused SDPA kernel of the default processor, so recording must not change the image; this is checked on the GPU by the preflight
(hook-on vs hook-off identity), not assumed. attention.py (pilot, 32x32 average over all layers) is unchanged.
Schema of the saved .npz: words [n_words] (str), maps_16 [steps, n_words, 16, 16] fp16, maps_32 [steps, n_words, 32, 32] fp16, plus meta json
(job_id, scene_id, prompt_id, seed, cfg, steps, token_positions, layer_counts)."""
from __future__ import annotations

import json

import numpy as np

SIZE_WORDS = {"small", "large"}


def constraint_words(constraints, prompt):
    """Words whose maps HF needs: nouns of the constraints, colour words, and the size words present in the prompt."""
    w = []
    for c in constraints:
        for k in ("object", "left", "right", "larger", "smaller"):
            if k in c and c[k] not in w:
                w.append(c[k])
        if c.get("color") and c["color"] not in w:
            w.append(c["color"])
    for s in SIZE_WORDS:
        if s in prompt.lower().split() and s not in w:
            w.append(s)
    return w


def group_maps(store, resolutions=(16, 32)):
    """store: list of [tokens, side, side] tensors (one per attn2 layer call of the step). Returns {res: [tokens, res, res]} and layer counts."""
    import torch
    out, counts = {}, {}
    for r in resolutions:
        layers = [m for m in store if m.shape[-1] == r]
        counts[r] = len(layers)
        if layers:
            out[r] = torch.stack(layers).mean(0).cpu()
    return out, counts


def generate_with_attention_groups(pipe, prompt, constraints, cfg, steps, init_latents, size, resolutions=(16, 32), latents_save_at=()):
    """init_latents: the seed's initial noise (see r3_run.init_latents). Returns (image, maps {res: fp16 [steps, n_words, res, res]}, words, meta, kept latents)."""
    import torch
    import attention as A
    original = dict(pipe.unet.attn_processors)
    store = A.install(pipe)
    words = constraint_words(constraints, prompt)
    idx = A.word_token_indices(pipe.tokenizer, prompt)
    tok = {w: idx.get(w, []) for w in words}
    per_step = {r: [] for r in resolutions}
    counts_seen = {}
    kept = {}

    def cb(_pipe, i, _t, kw):
        maps, counts = group_maps(store, resolutions)
        counts_seen.update(counts)
        for r in resolutions:
            m = maps.get(r)
            per_step[r].append(torch.stack([m[tok[w]].mean(0) if tok[w] else torch.full(m.shape[1:], float("nan")) for w in words]) if m is not None else None)
        store.clear()
        if (i + 1) in latents_save_at:
            kept[i + 1] = kw["latents"].detach().to("cpu", torch.float16).clone()
        return kw
    try:
        store.clear()
        image = pipe(prompt, guidance_scale=cfg, num_inference_steps=steps, height=size, width=size, latents=init_latents,
                     negative_prompt="", callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
    finally:
        pipe.unet.set_attn_processor(original)
    maps = {r: np.stack([x.numpy() for x in per_step[r]]).astype(np.float16) for r in resolutions if per_step[r] and per_step[r][0] is not None}
    meta = dict(token_positions=tok, layer_counts={str(k): v for k, v in counts_seen.items()}, words=words)
    return image, maps, words, meta, kept


def save_npz(path, maps, words, meta):
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, words=np.array(words), meta=np.array(json.dumps(meta)), **{f"maps_{r}": m for r, m in maps.items()})
    import os
    os.replace(tmp, path)


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    maps = {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("maps_")}
    return maps, [str(w) for w in z["words"]], json.loads(str(z["meta"]))
