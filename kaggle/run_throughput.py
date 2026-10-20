"""R1 throughput probe: seconds per image with and without the recording hook, latent-save overhead, and 1- vs 2-worker scaling.
Writes throughput.json. Uses 6 DEV prompts; decides nothing by itself (FULL vs LIMITED is a recorded decision, RB 16).
Usage: python kaggle/run_throughput.py --out /kaggle/working/outputs/<run_id>/throughput.json [--n 6]"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def timed_worker(out, n, extra, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "src" / "generate.py"), "--config", str(ROOT / "configs" / "experiment_v2.json"), "--prompts",
                        str(ROOT / "data" / "benchmark_v2" / "prompts_v2.json"), "--out", str(out), "--limit", str(n), *extra],
                       env=env, capture_output=True, text=True)
    return time.time() - t0, r.returncode, r.stderr[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    import torch
    res = {"n_gpus": torch.cuda.device_count(), "n_images_per_probe": a.n}
    base = Path(a.out).parent / "throughput_work"
    # NB: wall time includes model load; per-image seconds are read from samples.jsonl (per-image timer) for the clean number.
    for label, extra in (("plain", []), ("latents", ["--save-latents-at", "6,12,18,24,30,36"])):
        w, rc, err = timed_worker(base / label, a.n, ["--split", "DEV", *extra], 0)
        recs = [json.loads(l) for l in (base / label / "samples.jsonl").read_text().splitlines()] if (base / label / "samples.jsonl").exists() else []
        secs = [r["seconds"] for r in recs][1:] or [r["seconds"] for r in recs]
        res[label] = {"returncode": rc, "wall_s": round(w, 1), "per_image_s_median": statistics.median(secs) if secs else None, "per_image_s_all": secs, "stderr_tail": err}
    if res["n_gpus"] >= 2:
        procs = [subprocess.Popen([sys.executable, str(ROOT / "src" / "generate.py"), "--config", str(ROOT / "configs" / "experiment_v2.json"), "--prompts",
                                   str(ROOT / "data" / "benchmark_v2" / "prompts_v2.json"), "--out", str(base / "two"), "--split", "DEV", "--limit", str(a.n), "--shard", f"{g}/2"],
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))) for g in (0, 1)]
        t0 = time.time(); [p.wait() for p in procs]
        res["two_workers_wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
