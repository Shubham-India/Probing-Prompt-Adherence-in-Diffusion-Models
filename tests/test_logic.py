import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from evaluate import score


def det(score_, box, color="red"):
    return {"score": score_, "box": box, "colors": {"red": 0.0, "blue": 0.0, color: 1.0}}


def test_binding_swapped_colours_fail():
    cons = [{"type": "presence", "object": "cube"}, {"type": "presence", "object": "sphere"},
            {"type": "binding", "object": "cube", "color": "red"}, {"type": "binding", "object": "sphere", "color": "blue"}]
    right = {"cube": [det(0.5, [0, 0, 100, 100], "red")], "sphere": [det(0.5, [200, 0, 300, 100], "blue")]}
    swapped = {"cube": [det(0.5, [0, 0, 100, 100], "blue")], "sphere": [det(0.5, [200, 0, 300, 100], "red")]}
    assert score(cons, right) == [1, 1, 1, 1]
    assert score(cons, swapped) == [1, 1, 0, 0]


def test_same_box_cannot_satisfy_two_objects():
    cons = [{"type": "presence", "object": "cat"}, {"type": "presence", "object": "dog"}]
    dets = {"cat": [det(0.5, [0, 0, 100, 100])], "dog": [det(0.4, [2, 2, 100, 100])]}
    assert score(cons, dets) == [1, 0]


def test_spatial_uses_box_centres():
    cons = [{"type": "presence", "object": "cup"}, {"type": "presence", "object": "book"},
            {"type": "spatial", "left": "cup", "right": "book"}]
    left = {"cup": [det(0.5, [0, 0, 50, 50])], "book": [det(0.5, [300, 0, 400, 50])]}
    flipped = {"cup": [det(0.5, [300, 0, 400, 50])], "book": [det(0.5, [0, 0, 50, 50])]}
    assert score(cons, left)[2] == 1
    assert score(cons, flipped)[2] == 0


def test_count_exact_match_only():
    cons = [{"type": "count", "object": "apple", "n": 3}]
    boxes = [det(0.4, [i * 100, 0, i * 100 + 50, 50]) for i in range(3)]
    assert score(cons, {"apple": boxes}) == [1]
    assert score(cons, {"apple": boxes[:2]}) == [0]


def test_benchmark_is_consistent():
    prompts = json.loads((Path(__file__).resolve().parent.parent / "data" / "prompts.json").read_text())
    assert len({p["id"] for p in prompts}) == len(prompts)
    assert {p["category"] for p in prompts} == {"presence", "binding", "counting", "spatial", "multi"}
    assert all(p["n_constraints"] == len(p["constraints"]) for p in prompts)


def test_word_token_alignment():
    from transformers import CLIPTokenizer
    from attention import word_token_indices
    tok = CLIPTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer",
                                        cache_dir=Path(__file__).resolve().parent.parent / "models")
    idx = word_token_indices(tok, "a red cube and a blue sphere on a wooden table")
    ids = tok("a red cube and a blue sphere on a wooden table").input_ids
    assert [tok.decode(ids[i]) for i in idx["red"]] == ["red"]
    assert [tok.decode(ids[i]) for i in idx["sphere"]] == ["sphere"]
