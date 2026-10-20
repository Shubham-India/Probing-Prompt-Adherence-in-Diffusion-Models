"""Loop-vs-pipeline identity on a tiny RANDOM diffusers pipeline (CPU, fp32, no weights). Validates indexing/resume logic of
branch.denoise_from / branch_generate, not Stable Diffusion 1.5."""
import sys
from pathlib import Path
import numpy as np, pytest, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import branch as B
diffusers = pytest.importorskip("diffusers")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
                                          down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                                          cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8)
    vae = diffusers.AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
                                  block_out_channels=(32,), latent_channels=4, norm_num_groups=8, sample_size=8)
    sched = diffusers.DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                    clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    pipe = diffusers.StableDiffusionPipeline(vae=vae, text_encoder=None, tokenizer=None, unet=unet, scheduler=sched, safety_checker=None,
                                             feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator().manual_seed(1)
    neg, pos = torch.randn(1, 7, 32, generator=g), torch.randn(1, 7, 32, generator=g)
    return pipe, (neg, pos)


def arr(img):
    return np.asarray(img).astype(int)


def pipeline_image(pipe, embeds, x0, cfg=7.5, steps=40, cb=None):
    kw = {} if cb is None else dict(callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    return pipe(prompt_embeds=embeds[1], negative_prompt_embeds=embeds[0], guidance_scale=cfg, num_inference_steps=steps, height=8, width=8,
                latents=x0, **kw).images[0]


def x0():
    return torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(5))


def test_loop_from_step_zero_equals_pipeline(tiny):
    pipe, emb = tiny
    z = x0()
    assert np.array_equal(arr(pipeline_image(pipe, emb, z.clone())), arr(B.denoise_from(pipe, emb, z.clone(), 0)))


def test_resume_from_saved_latent_equals_uninterrupted_run(tiny):
    pipe, emb = tiny
    z = x0()
    saved = {}
    def cb(_p, i, _t, kw):
        if i + 1 in (6, 18, 36):
            saved[i + 1] = kw["latents"].clone()
        return kw
    base = pipeline_image(pipe, emb, z.clone(), cb=cb)
    for t in (6, 18, 36):
        assert np.array_equal(arr(base), arr(B.denoise_from(pipe, emb, saved[t], t))), t


def test_D0_branch_equals_base_pixel_for_pixel(tiny):
    pipe, emb = tiny
    z = x0(); saved = {}
    def cb(_p, i, _t, kw):
        if i + 1 == 12:
            saved[12] = kw["latents"].clone()
        return kw
    base = pipeline_image(pipe, emb, z.clone(), cb=cb)
    sch = B.Schedule(40)
    out = B.branch_generate(pipe, emb, saved[12], 12, B.restart_plan(sch, 12, 0), noise_seed=3)
    assert np.array_equal(arr(base), arr(out))


def test_restart_branch_differs_and_is_deterministic(tiny):
    pipe, emb = tiny
    z = x0(); saved = {}
    def cb(_p, i, _t, kw):
        if i + 1 == 24:
            saved[24] = kw["latents"].clone()
        return kw
    base = pipeline_image(pipe, emb, z.clone(), cb=cb)
    plan = B.restart_plan(B.Schedule(40), 24, 8)
    a, b = (B.branch_generate(pipe, emb, saved[24], 24, plan, noise_seed=s) for s in (3, 3))
    c = B.branch_generate(pipe, emb, saved[24], 24, plan, noise_seed=4)
    assert np.array_equal(arr(a), arr(b)) and not np.array_equal(arr(a), arr(c)) and not np.array_equal(arr(a), arr(base))
    with pytest.raises(AssertionError):
        B.branch_generate(pipe, emb, saved[24], 18, plan, noise_seed=3)    # plan/latent mismatch is rejected


def test_branch_noise_is_cpu_seeded_float32(tiny):
    n1, n2 = B.branch_noise((1, 4, 8, 8), 7, torch.float16, "cpu"), B.branch_noise((1, 4, 8, 8), 7, torch.float16, "cpu")
    assert torch.equal(n1, n2) and n1.dtype == torch.float16
