"""Seed diversity for HE (GRAND_RESEARCH_BLUEPRINT.md section 24.5): mean pairwise cosine distance between image embeddings of the seeds of
one prompt and arm. Primary embedder: DINOv2 ViT-S/14 (hub id facebook/dinov2-small, CLS token); secondary: LPIPS (AlexNet), optional.
Both models are downloaded on first use; the hub revision used is recorded in the output. The pure statistics are torch-free and unit-tested;
model loading is NOT validated on the target hardware until the Kaggle preflight measures it."""
from __future__ import annotations

import itertools

import numpy as np

DINO_ID = "facebook/dinov2-small"


def mean_pairwise_cosine_distance(emb):
    """emb [n, d] -> mean over pairs of (1 - cosine similarity). n < 2 -> nan."""
    e = np.asarray(emb, dtype=np.float64)
    if len(e) < 2:
        return float("nan")
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return float(np.mean([1.0 - float(e[i] @ e[j]) for i, j in itertools.combinations(range(len(e)), 2)]))


def per_prompt_diversity(embeddings_by_key):
    """{(prompt_id, arm): [n_seeds, d]} -> {(prompt_id, arm): diversity}."""
    return {k: mean_pairwise_cosine_distance(v) for k, v in embeddings_by_key.items()}


class DinoEmbedder:
    def __init__(self, device="cuda", cache_dir=None, revision=None):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(DINO_ID, cache_dir=cache_dir, revision=revision)
        self.model = AutoModel.from_pretrained(DINO_ID, cache_dir=cache_dir, revision=revision).to(device).eval()
        self.device = device
        self.revision = revision or getattr(self.model.config, "_commit_hash", None)

    def __call__(self, images):
        with self.torch.no_grad():
            x = self.proc(images=images, return_tensors="pt").to(self.device)
            return self.model(**x).last_hidden_state[:, 0].float().cpu().numpy()


def lpips_pairwise(images, device="cuda"):
    """Optional secondary metric. Requires the `lpips` package; returns None if it is not installed."""
    try:
        import lpips
        import torch
    except ImportError:
        return None
    net = lpips.LPIPS(net="alex").to(device)
    t = [torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device) for im in images]
    with torch.no_grad():
        d = [float(net(t[i], t[j])) for i, j in itertools.combinations(range(len(t)), 2)]
    return float(np.mean(d)) if d else float("nan")
