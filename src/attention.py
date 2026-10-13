"""Cross-attention extraction for Stable Diffusion 1.5 (DAAM-style, written for diffusers 0.41).

Every `attn2` layer of the UNet (text cross-attention) gets a processor that computes the same attention as the
default one but also keeps the softmax probabilities of the *conditional* half of the CFG batch
(batch = [unconditional, conditional]). At the end of each denoising step the per-layer maps (resolutions 8..64)
are resized to 32x32 and averaged over heads and layers.

Usage: python src/attention.py --run outputs/main --steps 40 [--check 3]
Writes <run>/attention/<sample name>.npz with `words` and `maps` [n_words, n_steps, 32, 32].
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from generate import ROOT, generate_image, load_pipeline

MAP_RES = 32
STOP_WORDS = {"a", "an", "the", "to", "of", "and", "on"}


class StoreCrossAttention:
    def __init__(self, store):
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch = hidden_states.shape[0]
        query, key, value = attn.to_q(hidden_states), attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        split = lambda x: x.view(batch, -1, attn.heads, x.shape[-1] // attn.heads).transpose(1, 2)
        # the output uses the same fused kernel as the default processor, so generation is unchanged
        out = F.scaled_dot_product_attention(split(query), split(key), split(value))
        out = out.transpose(1, 2).reshape(batch, -1, query.shape[-1])
        out = attn.to_out[1](attn.to_out[0](out))

        # probabilities are recomputed only for the conditional half, purely for recording
        q_cond = attn.head_to_batch_dim(query[-1:])
        probs = attn.get_attention_scores(q_cond, attn.head_to_batch_dim(key[-1:]), None)  # [heads, pixels, tokens]
        side = int(probs.shape[1] ** 0.5)
        cond = probs.view(attn.heads, side, side, -1).mean(0)
        self.store.append(cond.permute(2, 0, 1).float())  # [tokens, side, side]
        return out


def install(pipe):
    store = []
    pipe.unet.set_attn_processor({
        name: StoreCrossAttention(store) if name.endswith("attn2.processor") else proc
        for name, proc in pipe.unet.attn_processors.items()})
    return store


def word_token_indices(tokenizer, prompt):
    """Map each non-stop word to its token positions. Words are tokenised on their own; CLIP BPE never merges
    across whitespace, so the concatenation equals the tokenisation of the full prompt (checked)."""
    words = prompt.lower().replace(",", "").split()
    pos, indices = 1, {}  # position 0 is <|startoftext|>
    for w in words:
        n = len(tokenizer(w, add_special_tokens=False).input_ids)
        if w not in STOP_WORDS:
            indices.setdefault(w, []).extend(range(pos, pos + n))
        pos += n
    full = tokenizer(prompt).input_ids
    assert pos == full.index(tokenizer.eos_token_id), "word/token alignment broke"
    return indices


def generate_with_attention(pipe, store, prompt, cfg, steps, seed, size):
    """Returns the image and {word: [steps, 32, 32] array} of cross-attention probability maps."""
    per_step = []

    def end_of_step(pipeline, step, timestep, kwargs):
        layer_maps = [F.interpolate(m[None], size=(MAP_RES, MAP_RES), mode="bilinear", align_corners=False)[0]
                      for m in store]
        per_step.append(torch.stack(layer_maps).mean(0).cpu())
        store.clear()
        return kwargs

    store.clear()
    image = generate_image(pipe, prompt, cfg, steps, seed, size, callback_on_step_end=end_of_step)
    maps = torch.stack(per_step)  # [steps, 77, 32, 32]
    indices = word_token_indices(pipe.tokenizer, prompt)
    return image, {w: maps[:, idx].mean(1).numpy().astype(np.float16) for w, idx in indices.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--check", type=int, default=0, help="compare the first N regenerated images with the saved ones")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run = Path(args.run)
    config = json.loads((ROOT / "configs" / "experiment.json").read_text())
    samples = [json.loads(l) for l in (run / "samples.jsonl").read_text().splitlines()]
    samples = [s for s in samples if s["steps"] == args.steps][: args.limit]
    out_dir = run / "attention"
    out_dir.mkdir(exist_ok=True)
    todo = [s for s in samples if not (out_dir / f"{s['name']}.npz").exists()]
    print(f"{len(todo)} attention extractions", flush=True)

    pipe = load_pipeline(config["model_id"], config.get("revision"))
    store = install(pipe)
    for i, s in enumerate(todo, 1):
        image, maps = generate_with_attention(pipe, store, s["prompt"], s["cfg"], s["steps"], s["seed"], s["size"])
        np.savez_compressed(out_dir / f"{s['name']}.npz", words=np.array(list(maps)), maps=np.stack(list(maps.values())))
        if i <= args.check:
            saved = np.asarray(Image.open(run / "images" / f"{s['name']}.png"), dtype=int)
            diff = np.abs(saved - np.asarray(image, dtype=int))
            print(f"check {s['name']}: max pixel diff {diff.max()}, mean {diff.mean():.4f}", flush=True)
        if i % 25 == 0:
            print(f"[{i}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    main()
