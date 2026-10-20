"""Record the execution environment (R0). Prints and writes JSON. Never prints environment variables or tokens.
Usage: python kaggle/env_probe.py --out /kaggle/working/env.json [--commit SHA] [--time-one]"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:                    # tool absent
        return f"unavailable: {type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--time-one", action="store_true", help="also time one 512px, 40-step image (loads the model)")
    a = ap.parse_args()
    env = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "platform": platform.platform(),
           "commit": a.commit or sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]), "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
           "disk_free_gb": round(shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").exists() else ".").free / 1e9, 1),
           "libs": {}}
    for lib in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "numpy", "pandas", "PIL", "scipy", "sklearn"):
        try:
            env["libs"][lib] = __import__(lib).__version__
        except Exception as e:
            env["libs"][lib] = f"missing: {type(e).__name__}"
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        env["devices"] = [{"name": torch.cuda.get_device_name(i), "total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)} for i in range(torch.cuda.device_count())]
        if a.time_one and torch.cuda.is_available():
            sys.path.insert(0, str(ROOT / "src"))
            import generate as G
            cfg = json.loads((ROOT / "configs" / "experiment.json").read_text())
            t0 = time.time()
            pipe = G.load_pipeline(cfg["model_id"], cfg.get("revision"))
            env["model_load_s"] = round(time.time() - t0, 1)
            torch.cuda.reset_peak_memory_stats()
            G.generate_image(pipe, "a red cup and a blue book", 7.5, 40, 0, 512)      # warm-up
            t0 = time.time()
            G.generate_image(pipe, "a red cup and a blue book", 7.5, 40, 1, 512)
            env["t_gen_one_s"] = round(time.time() - t0, 2)
            env["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception as e:
        env["torch_error"] = f"{type(e).__name__}: {e}"[:300]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(env, indent=1))
    print(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
