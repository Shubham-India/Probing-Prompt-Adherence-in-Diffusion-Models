"""Every constraint word of every attention-recorded prompt in the full manifests has token positions (HF needs a map per constraint word).
Regression: counting prompts use plural nouns ("four black kites") while constraints name the singular ("kite"), which left 55 noun maps NaN."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def tokenizer():
    from transformers import CLIPTokenizer
    from model_cache import model_cache_dir
    cfg = json.loads((ROOT / "configs" / "experiment_r3.json").read_text())
    try:
        return CLIPTokenizer.from_pretrained(cfg["model_id"], subfolder="tokenizer", revision=cfg["revision"], cache_dir=model_cache_dir(), local_files_only=True)
    except OSError:
        pytest.skip("SD 1.5 tokenizer not cached locally")


def test_every_attention_constraint_word_has_tokens():
    import attention as A
    import attn_record as AR
    import manifests as M
    tok = tokenizer()
    jobs, _, _ = M.build()
    prompts = {p["id"]: p for f in ("prompts_v2.json", "prompts_ps.json") for p in json.loads((ROOT / "data" / "benchmark_v2" / f).read_text())}
    ids = {j["prompt_id"] for j in jobs if j.get("attention")}
    assert ids
    missing = []
    for pid in sorted(ids):
        p = prompts[pid]
        idx = A.word_token_indices(tok, p["prompt"])
        missing += [(pid, w) for w in AR.constraint_words(p["constraints"], p["prompt"]) if not AR.token_positions(idx, w)]
    assert not missing, missing[:10]


def test_plural_fallback_keeps_the_constraint_key():
    import attn_record as AR
    idx = {"four": [1], "black": [2], "kites": [3], "yellow": [4], "teapot": [5]}
    assert AR.token_positions(idx, "kite") == [3] and AR.token_positions(idx, "teapot") == [5] and AR.token_positions(idx, "bus") == []
