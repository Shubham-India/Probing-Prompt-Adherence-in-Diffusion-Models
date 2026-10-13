"""GPU test (needs the cached SD 1.5 weights): the hooked cross-attention must be a valid distribution over the 77 text tokens
and must not change the generated image. Run with: pytest tests/test_attention_gpu.py"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_attention_is_a_distribution_and_image_is_unchanged():
    from attention import generate_with_attention, install
    from generate import generate_image, load_pipeline

    pipe = load_pipeline("stable-diffusion-v1-5/stable-diffusion-v1-5")
    plain = np.asarray(generate_image(pipe, "a red cube and a blue sphere on a wooden table", 7.5, 6, 3, 512), dtype=int)

    store = install(pipe)
    seen = {}

    def inspect(pipeline, step, timestep, kwargs):
        if step == 0:
            seen["layer_sums"] = [m.sum(0) for m in store]  # per layer: sum over the 77 tokens at each pixel
            seen["n_layers"] = len(store)
        store.clear()
        return kwargs

    hooked = np.asarray(generate_image(pipe, "a red cube and a blue sphere on a wooden table", 7.5, 6, 3, 512,
                                       callback_on_step_end=inspect), dtype=int)
    assert seen["n_layers"] == 16  # SD 1.5 has 16 text cross-attention layers
    for s in seen["layer_sums"]:
        assert torch.allclose(s, torch.ones_like(s), atol=2e-2)
    assert np.abs(plain - hooked).max() == 0

    image, maps = generate_with_attention(pipe, store, "a red cube and a blue sphere on a wooden table", 7.5, 6, 3, 512)
    assert set(maps) == {"red", "cube", "blue", "sphere", "wooden", "table"}
    assert all(m.shape == (6, 32, 32) and np.isfinite(m.astype(float)).all() for m in maps.values())
    assert np.abs(plain - np.asarray(image, dtype=int)).max() == 0
