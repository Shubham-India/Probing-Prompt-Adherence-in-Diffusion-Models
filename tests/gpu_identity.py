"""GPU identity checks on the REAL SD 1.5 fp16 pipeline (M7 control, M8 loop / reload / D=0). Requires CUDA and the model weights.
Run on the owner's machine or on Kaggle:  python tests/gpu_identity.py --out outputs/acceptance/gpu_identity.json [--n 3]
Each check compares decoded images pixel-for-pixel (max abs difference); it exits 1 if any required identity fails. Nothing here is
claimed to pass until this script has been executed on a GPU and its JSON output is saved."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import branch as B                       # noqa: E402
import edit_probe as EP                  # noqa: E402
import generate as G                     # noqa: E402

CFG, STEPS = 7.5, 40


def diff(a, b):
    return int(np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.json"))
    a = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    cfg = json.loads(Path(a.config).read_text())
    prompts = json.loads((ROOT / "data" / "prompts.json").read_text())[:: max(1, 28 // a.n)][: a.n]
    pipe = G.load_pipeline(cfg["model_id"], cfg.get("revision"))
    size, results, sch = cfg["image_size"], [], B.Schedule(STEPS)
    for p in prompts:
        seed = 0
        name = G.sample_name(p["id"], CFG, STEPS, seed)
        r = {"name": name, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
        base, kept = G.generate_with_latents(pipe, p["prompt"], CFG, STEPS, seed, size, {12})
        pilot = ROOT / "outputs" / "main" / "images" / f"{name}.png"
        r["vs_pilot_image_maxdiff"] = diff(base, Image.open(pilot).convert("RGB")) if pilot.exists() else None
        base2, kept2 = G.generate_with_latents(pipe, p["prompt"], CFG, STEPS, seed, size, {12})
        r["determinism_maxdiff"] = diff(base, base2)
        r["saved_latent_recompute_equal"] = bool(torch.equal(kept[12], kept2[12]))
        embeds = B.encode_prompt_pair(pipe, p["prompt"], "cuda")
        x0 = G.initial_latents(seed, size)
        r["loop_from_step0_maxdiff"] = diff(base, B.denoise_from(pipe, embeds, x0, 0, STEPS, CFG))
        tmp = Path(a.out).with_suffix(".latent.pt")
        torch.save(kept[12], tmp)
        reloaded = torch.load(tmp).to("cuda")
        r["reload_resume_maxdiff"] = diff(base, B.denoise_from(pipe, embeds, reloaded, 12, STEPS, CFG))
        tmp.unlink()
        r["d0_branch_maxdiff"] = diff(base, B.branch_generate(pipe, embeds, reloaded, 12, B.restart_plan(sch, 12, 0), 0, STEPS, CFG))
        st = EP.EditState(mode="off")
        EP.install_edit(pipe.unet, st)
        r["hook_off_control_maxdiff"] = diff(base, G.generate_image(pipe, p["prompt"], CFG, STEPS, seed, size))
        pipe.unet.set_default_attn_processor()
        results.append(r)
        print(json.dumps(r), flush=True)
    required = ["determinism_maxdiff", "loop_from_step0_maxdiff", "reload_resume_maxdiff", "d0_branch_maxdiff", "hook_off_control_maxdiff"]
    ok = all(r[k] == 0 for r in results for k in required) and all(r["saved_latent_recompute_equal"] for r in results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"all_required_identities_hold": ok, "required": required, "results": results}, indent=1))
    print("ALL REQUIRED IDENTITIES HOLD" if ok else "IDENTITY FAILURE: inspect the JSON (a nonzero maxdiff must be recorded and decided, D-07)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
