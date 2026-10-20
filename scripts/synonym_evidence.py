"""Text-side evidence for the PAR colour synonyms (variant review H3). CPU only; no image is generated.
For every PAR prompt and every colour synonym in it, the synonym is replaced by each of the judge's 10 basic colours (src/evaluate.py COLORS) and the
nearest substituted prompt is found in two text spaces: SD 1.5's own text encoder (the generator's conditioning; mean of the non-padding hidden
states) and the judge's CLIP ViT-B/32 (pooled text embedding). A synonym "maps" to a basic colour when that colour's substitution is nearest.
Also a word-level probe ("a photo of a {w} object", the judge's colour template) and a noun/material-reading probe.
This is a PROXY: it shows how the models read the word, not what the generator draws. Output: docs/evidence/variant_review/synonym_evidence.json
Usage: python scripts/synonym_evidence.py"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model_cache import model_cache_dir  # noqa: E402

JUDGE_COLOURS = ["red", "blue", "yellow", "green", "orange", "purple", "white", "black", "brown", "pink"]   # src/evaluate.py COLORS
READINGS = {"golden": ["made of gold", "gold metal"], "ivory": ["made of ivory", "carved ivory"], "ebony": ["made of ebony wood", "ebony wood"],
            "emerald": ["made of emerald gemstone", "emerald gemstone"], "tangerine": ["next to a tangerine fruit", "tangerine fruit"],
            "violet": ["next to a violet flower", "violet flower"], "azure": ["azure sky"], "crimson": ["crimson velvet"]}


def main():
    from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
    cfg = json.loads((ROOT / "configs" / "experiment_r3.json").read_text())
    cache = model_cache_dir()
    sd_tok = CLIPTokenizer.from_pretrained(cfg["model_id"], subfolder="tokenizer", revision=cfg["revision"], cache_dir=cache, local_files_only=True)
    sd_te = CLIPTextModel.from_pretrained(cfg["model_id"], subfolder="text_encoder", revision=cfg["revision"], variant="fp16", torch_dtype=torch.float32,
                                          cache_dir=cache, local_files_only=True).eval()
    jr = cfg["evaluator_revisions"]["openai/clip-vit-base-patch32"]
    j_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", revision=jr, cache_dir=cache, local_files_only=True)
    j_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision=jr, cache_dir=cache, local_files_only=True).eval()

    @torch.no_grad()
    def sd_emb(texts):
        t = sd_tok(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        h = sd_te(**t).last_hidden_state
        m = t["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(e, dim=-1)

    @torch.no_grad()
    def j_emb(texts):
        t = j_tok(texts, padding=True, return_tensors="pt")
        f = j_clip.get_text_features(**t)
        if not torch.is_tensor(f):                    # transformers 5 returns an output object
            f = f.pooler_output
        assert f.shape[-1] == j_clip.config.projection_dim, f.shape
        return torch.nn.functional.normalize(f, dim=-1)

    lex = json.loads((ROOT / "data" / "benchmark_v2" / "lexicon.json").read_text())
    syn_of = lex["colour_synonyms"]
    canon_of = {v: k for k, v in syn_of.items()}
    prompts = [p for f in ("prompts_v2.json", "prompts_ps.json") for p in json.loads((ROOT / "data" / "benchmark_v2" / f).read_text()) if p["variant"] == "PAR"]
    res = defaultdict(lambda: dict(n=0, sd_nearest=Counter(), judge_nearest=Counter(), sd_margin=[], judge_margin=[], nouns=Counter()))
    for p in prompts:
        for syn in canon_of:
            for m in re.finditer(rf"\b{syn}\b", p["prompt"]):
                alts = [p["prompt"][:m.start()] + c + p["prompt"][m.end():] for c in JUDGE_COLOURS]
                noun = re.findall(r"[a-z]+", p["prompt"][m.end():])[:1]
                r = res[syn]
                r["n"] += 1
                r["nouns"][noun[0] if noun else "?"] += 1
                for space, fn in (("sd", sd_emb), ("judge", j_emb)):
                    e = fn([p["prompt"]] + alts)
                    s = (e[1:] @ e[0]).tolist()
                    order = sorted(range(len(s)), key=lambda i: -s[i])
                    r[f"{space}_nearest"][JUDGE_COLOURS[order[0]]] += 1
                    ci = JUDGE_COLOURS.index(canon_of[syn])
                    best_other = max(s[i] for i in range(len(s)) if i != ci)
                    r[f"{space}_margin"].append(round(s[ci] - best_other, 4))
    out = {}
    for syn, r in sorted(res.items()):
        canon = canon_of[syn]
        word = {}
        for space, fn in (("sd", sd_emb), ("judge", j_emb)):
            e = fn([f"a photo of a {syn} object"] + [f"a photo of a {c} object" for c in JUDGE_COLOURS])
            s = (e[1:] @ e[0]).tolist()
            word[space] = dict(sorted(((c, round(v, 4)) for c, v in zip(JUDGE_COLOURS, s)), key=lambda kv: -kv[1])[:3])
        reading = {}
        for space, fn in (("sd", sd_emb), ("judge", j_emb)):
            noun = r["nouns"].most_common(1)[0][0]
            texts = [f"a {syn} {noun}", f"a {canon} {noun}"] + [f"a {noun} {x}" for x in READINGS.get(syn, [])]
            e = fn(texts)
            s = (e[1:] @ e[0]).tolist()
            reading[space] = dict(probe_noun=noun, colour_reading=round(s[0], 4), **{x: round(v, 4) for x, v in zip(READINGS.get(syn, []), s[1:])})
        out[syn] = dict(canonical=canon, occurrences=r["n"],
                        sd_nearest_colour=dict(r["sd_nearest"]), sd_share_canonical=round(r["sd_nearest"][canon] / r["n"], 3),
                        sd_margin_min=min(r["sd_margin"]), sd_margin_median=sorted(r["sd_margin"])[len(r["sd_margin"]) // 2],
                        judge_nearest_colour=dict(r["judge_nearest"]), judge_share_canonical=round(r["judge_nearest"][canon] / r["n"], 3),
                        judge_margin_min=min(r["judge_margin"]), judge_margin_median=sorted(r["judge_margin"])[len(r["judge_margin"]) // 2],
                        word_level_top3=word, reading_probe=reading)
    doc = dict(method=__doc__.split("Usage:")[0].strip(), judge_colours=JUDGE_COLOURS, sd_model=f"{cfg['model_id']}@{cfg['revision']}",
               judge_clip=f"openai/clip-vit-base-patch32@{jr}", n_par_prompts=len(prompts), synonyms=out)
    dest = ROOT / "docs" / "evidence" / "variant_review" / "synonym_evidence.json"
    dest.write_text(json.dumps(doc, indent=1))
    for syn, v in out.items():
        print(f"{syn:10s}->{v['canonical']:7s} n={v['occurrences']:3d}  SD nearest {v['sd_nearest_colour']} (median margin {v['sd_margin_median']:+.4f})"
              f"  | judge nearest {v['judge_nearest_colour']} (median {v['judge_margin_median']:+.4f})")


if __name__ == "__main__":
    main()
