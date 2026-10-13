"""Attention re-weighting intervention (see report/protocol.md, Addendum A).

For every text cross-attention layer the pre-softmax scores of the content tokens (constraint objects and colours) of the
conditional branch get +log(s), which multiplies their attention weight by s before the softmax re-normalises.
s = 1 goes through the same code path and serves as the control.
Usage: python src/intervene.py --out outputs/intervention --scales 1 2 4
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from attention import word_token_indices
from generate import ROOT, done_names, generate_image, load_pipeline

CFG, STEPS = 7.5, 40


class ReweightCrossAttention:
    def __init__(self):
        self.bias = None  # [batch, 1, 1, 77] additive score bias, set before each generation

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch = hidden_states.shape[0]
        query, key, value = attn.to_q(hidden_states), attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        split = lambda x: x.view(batch, -1, attn.heads, x.shape[-1] // attn.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(query), split(key), split(value), attn_mask=self.bias)
        out = out.transpose(1, 2).reshape(batch, -1, query.shape[-1])
        return attn.to_out[1](attn.to_out[0](out))


def install(pipe):
    processor = ReweightCrossAttention()
    pipe.unet.set_attn_processor({
        name: processor if name.endswith("attn2.processor") else proc for name, proc in pipe.unet.attn_processors.items()})
    return processor


def content_words(constraints):
    objects = {c["object"] for c in constraints if "object" in c} | {c[k] for c in constraints for k in ("left", "right") if k in c}
    colours = {c["color"] for c in constraints if c["type"] == "binding"}
    return objects, colours


def build_bias(tokenizer, prompt, constraints, scale, device="cuda"):
    objects, colours = content_words(constraints)
    wanted = objects | {o + "s" for o in objects} | colours
    indices = [i for w, idx in word_token_indices(tokenizer, prompt).items() if w in wanted for i in idx]
    bias = torch.zeros(2, 1, 1, 77, dtype=torch.float16, device=device)  # batch = [unconditional, conditional]
    bias[1, 0, 0, indices] = math.log(scale)
    return bias, indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[1, 2, 4])
    ap.add_argument("--limit-prompts", default=None)
    args = ap.parse_args()

    config = json.loads((ROOT / "configs" / "experiment.json").read_text())
    prompts = json.loads((ROOT / "data" / "prompts.json").read_text())
    if args.limit_prompts:
        prompts = [p for p in prompts if p["id"] in set(args.limit_prompts.split(","))]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta_path = out / "samples.jsonl"
    finished = done_names(meta_path)
    jobs = [(p, s, seed) for s in args.scales for p in prompts for seed in config["seeds"]
            if f"{p['id']}__cfg{CFG:g}__st{STEPS}__seed{seed}__rw{s:g}" not in finished]
    print(f"{len(jobs)} images to generate ({len(finished)} already done)", flush=True)
    if not jobs:
        return
    pipe = load_pipeline(config["model_id"], config.get("revision"))
    processor = install(pipe)
    with open(meta_path, "a") as meta:
        for i, (p, scale, seed) in enumerate(jobs, 1):
            name = f"{p['id']}__cfg{CFG:g}__st{STEPS}__seed{seed}__rw{scale:g}"
            processor.bias, indices = build_bias(pipe.tokenizer, p["prompt"], p["constraints"], scale)
            t0 = time.time()
            image = generate_image(pipe, p["prompt"], CFG, STEPS, seed, config["image_size"])
            image.save(out / "images" / f"{name}.png")
            meta.write(json.dumps({"name": name, "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"],
                                   "cfg": CFG, "steps": STEPS, "seed": seed, "size": config["image_size"], "reweight": scale,
                                   "boosted_token_positions": indices, "scheduler": "DDIM(eta=0)", "negative_prompt": "",
                                   "model_id": config["model_id"], "seconds": round(time.time() - t0, 2)}) + "\n")
            meta.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {name}", flush=True)


if __name__ == "__main__":
    main()
