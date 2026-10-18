"""Owner-side helper: packs a verified core run (e.g. R2a dev_core) into a folder ready to upload as a PRIVATE Kaggle dataset, with LF line endings in the
text files (a Windows restore converts them to CRLF) and a manifest of SHA-256 hashes. R3 needs images (verifier), latents and samples*.jsonl.
Usage: python kaggle/make_core_dataset.py --core-run <.../dev_core> --out kaggle_core_dataset --run-id r2a-20261014T130457Z-daef6c4
       -> <out>/dev_core/{samples*.jsonl, images/, latents/}  and  <out>/core_manifest.json   (the out folder is git-ignored by name 'kaggle_core_dataset')"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(core, out, run_id):
    core, out = Path(core), Path(out)
    dst = out / "dev_core"
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "latents").mkdir(parents=True, exist_ok=True)
    manifest = dict(run_id=run_id, files={})
    for f in sorted(core.glob("samples*.jsonl")):
        (dst / f.name).write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    for sub in ("images", "latents"):
        for f in sorted((core / sub).iterdir()):
            shutil.copy2(f, dst / sub / f.name)
    for f in sorted(dst.rglob("*")):
        if f.is_file():
            manifest["files"][f.relative_to(out).as_posix()] = sha(f)
    manifest["n_images"] = sum(k.startswith("dev_core/images/") for k in manifest["files"])
    manifest["n_latents"] = sum(k.startswith("dev_core/latents/") for k in manifest["files"])
    (out / "core_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-run", required=True)
    ap.add_argument("--out", default="kaggle_core_dataset")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    m = main(a.core_run, a.out, a.run_id)
    print("packed", m["n_images"], "images,", m["n_latents"], "latent files ->", a.out)
