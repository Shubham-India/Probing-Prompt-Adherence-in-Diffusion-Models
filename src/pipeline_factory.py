"""Pipeline construction for the Revision 3 worker.
real: Stable Diffusion 1.5 fp16 at the pinned revision, DDIM, on CUDA (generate.load_pipeline).
fake: a tiny RANDOM UNet/VAE/text encoder on CPU with the real CLIP tokenizer from the local model cache, for dry-running the control flow of
      the notebooks and scripts (manifests, resume, hashing, merging, analysis plumbing). Outputs made with it are flagged `fake_pipeline: true`
      and every merge/analysis step refuses them unless explicitly allowed. It proves nothing about Stable Diffusion."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(cfg, fake=False):
    if not fake:
        import generate as G
        return G.load_pipeline(cfg["model_id"], cfg.get("revision")), dict(device="cuda", dtype="float16", fake=False, image_size=cfg["image_size"])
    import torch
    import diffusers
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
                                          down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                                          cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8)
    vae = diffusers.AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
                                  block_out_channels=(32,), latent_channels=4, norm_num_groups=8, sample_size=8)
    sched = diffusers.DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                    clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    from model_cache import model_cache_dir
    cache = model_cache_dir()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache, local_files_only=True)
    tok.model_max_length = 77                     # CLIP text length; some cached tokenizer configs omit it
    te = CLIPTextModel(CLIPTextConfig(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=4,
                                      max_position_embeddings=77, projection_dim=32))
    pipe = diffusers.StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sched, safety_checker=None,
                                             feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe, dict(device="cpu", dtype="float32", fake=True, image_size=8)
