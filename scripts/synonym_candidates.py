"""Text-side screening of REPLACEMENT candidates for lexicon synonyms that failed review (CPU only; proxy evidence, no images).
Colour: for every PAR prompt that uses the current synonym, the candidate is substituted and the nearest of the judge's 10 basic colours (substituted at
the same position) is found in SD 1.5's text encoder and in the judge's CLIP. Relation: similarity of the candidate relation phrase to the canonical
phrase versus the reversed phrase, in the PAR prompts that use the relation synonym.
Output: docs/evidence/variant_review/synonym_candidates.json
Usage: python -u scripts/synonym_candidates.py"""
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model_cache import model_cache_dir  # noqa: E402

C = ["red", "blue", "yellow", "green", "orange", "purple", "white", "black", "brown", "pink"]
YELLOW = ["golden", "lemon", "canary", "mustard", "sunflower", "saffron", "bright yellow", "lemon-yellow"]
REL = {"left": ["on the left side of", "left of", "on the left of", "to the left side of"], "right": ["on the right side of", "right of", "on the right of", "to the right side of"]}


def main():
    from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = json.loads((ROOT / "configs" / "experiment_r3.json").read_text())
    cache = model_cache_dir()
    st = CLIPTokenizer.from_pretrained(cfg["model_id"], subfolder="tokenizer", revision=cfg["revision"], cache_dir=cache, local_files_only=True)
    se = CLIPTextModel.from_pretrained(cfg["model_id"], subfolder="text_encoder", revision=cfg["revision"], variant="fp16", torch_dtype=torch.float32,
                                       cache_dir=cache, local_files_only=True).eval()
    jr = cfg["evaluator_revisions"]["openai/clip-vit-base-patch32"]
    jt = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", revision=jr, cache_dir=cache, local_files_only=True)
    jc = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision=jr, cache_dir=cache, local_files_only=True).eval()

    @torch.no_grad()
    def sd(x):
        t = st(x, padding="max_length", max_length=77, return_tensors="pt")
        h = se(**t).last_hidden_state
        m = t["attention_mask"].unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)

    @torch.no_grad()
    def jd(x):
        f = jc.get_text_features(**jt(x, padding=True, return_tensors="pt"))
        f = f if torch.is_tensor(f) else f.pooler_output
        return torch.nn.functional.normalize(f, dim=-1)

    def sub(prompt, word, new):
        """Replace the word, fixing a preceding article; also bare occurrences after a number ("two golden kites")."""
        out = re.sub(rf"\b(a|an) {word}\b", lambda m: ("an" if new[0] in "aeiou" else "a") + " " + new, prompt)
        out = re.sub(rf"\b{word}\b", new, out)
        assert out != prompt or word == new, (prompt, word, new)
        return out

    par = [p for f in ("prompts_v2.json", "prompts_ps.json") for p in json.loads((ROOT / "data" / "benchmark_v2" / f).read_text()) if p["variant"] == "PAR"]
    gold = [p["prompt"] for p in par if re.search(r"\bgolden\b", p["prompt"])]
    out = dict(colour={}, relation={})
    for cand in YELLOW:
        row = {}
        for name, fn in (("sd", sd), ("judge", jd)):
            hits, margins = 0, []
            for pr in gold:
                q = sub(pr, "golden", cand)
                e = fn([q] + [sub(pr, "golden", c) for c in C])
                s = (e[1:] @ e[0]).tolist()
                hits += max(range(len(C)), key=lambda i: s[i]) == C.index("yellow")
                margins.append(s[C.index("yellow")] - max(s[i] for i in range(len(C)) if i != C.index("yellow")))
            margins.sort()
            row[name] = dict(yellow_nearest=hits, n=len(gold), margin_median=round(margins[len(margins) // 2], 4), margin_min=round(margins[0], 4))
        out["colour"][cand] = row
        print("colour", cand, json.dumps(row), flush=True)
    relp = [p["prompt"] for p in par if "side of" in p["prompt"]]
    for side, cands in REL.items():
        canon, rev = f"to the {side} of", f"to the {'right' if side == 'left' else 'left'} of"
        mine = [pr for pr in relp if f"on the {side} side of" in pr]
        for cand in cands:
            row = {}
            for name, fn in (("sd", sd), ("judge", jd)):
                closer = 0
                for pr in mine:
                    q = pr.replace(f"on the {side} side of", cand)
                    e = fn([q, pr.replace(f"on the {side} side of", canon), pr.replace(f"on the {side} side of", rev)])
                    closer += float(e[0] @ e[1]) > float(e[0] @ e[2])
                row[name] = dict(closer_to_canonical_than_reversed=closer, n=len(mine))
            out["relation"][cand] = row
            print("relation", cand, json.dumps(row), flush=True)
    (ROOT / "docs" / "evidence" / "variant_review" / "synonym_candidates.json").write_text(json.dumps(dict(method=__doc__.split("Usage:")[0].strip(), **out), indent=1))


if __name__ == "__main__":
    main()
