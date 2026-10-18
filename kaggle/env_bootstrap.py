"""Reproducible dependency bootstrap for every Guide-Before-Commitment notebook (standard library only; call it BEFORE importing torch/diffusers).

Steps (each recorded in <out>/env_bootstrap.json; any failure raises BootstrapError and nothing downstream may run):
1. refuse to run when diffusers / transformers / torchao are already imported in this kernel (a stale module would hide the fix: restart the kernel);
2. `pip check` before installing (pre-existing conflicts of the Kaggle image are recorded, not hidden);
3. `pip install -r requirements-kaggle.txt` (torch / torchvision are deliberately not pinned: Kaggle's CUDA build is kept);
4. remove torchao if it is installed (see TORCHAO below);
5. a FRESH Python process imports torch, DDIMScheduler, StableDiffusionPipeline and the judge / diversity model classes (the worker processes are
   fresh processes too, so this is the check that matters); the torch build must be the same one that was there before the install;
6. `pip check` again: a conflict that involves a package this project imports is a failure; other conflicts are recorded with whether they existed
   before the install.

TORCHAO. diffusers 0.41.0 (diffusers/quantizers/torchao/torchao_quantizer.py) runs `from torchao.quantization import FqnToConfig, quantize_` whenever
ANY torchao distribution is installed: `is_torchao_available()` checks presence only, not the `torchao>=0.15.0` minimum listed in its own
dependency table. The quantizer module is imported on the normal path StableDiffusionPipeline -> loaders.single_file -> quantizers.auto, so an older
preinstalled torchao (no FqnToConfig) makes `from diffusers import StableDiffusionPipeline` fail. This project never quantizes (fp16 SD 1.5, fp32
judges); torchao is an optional integration it does not use, so it is uninstalled rather than upgraded (an upgrade would have to match the image's
torch/CUDA build, which is not something this project controls). transformers 5.19.0 checks a minimum torchao version itself and is unaffected."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_PACKAGES = ("torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "huggingface-hub", "huggingface_hub", "tokenizers",
                     "numpy", "scipy", "pandas", "pillow", "scikit-learn", "statsmodels")
STALE_MODULES = ("diffusers", "transformers", "torchao")
REMOVE = ("torchao",)
PROBE = r"""
import json, sys
out = {}
import torch
out["torch"] = torch.__version__
out["cuda"] = torch.version.cuda
out["cuda_available"] = torch.cuda.is_available()
out["gpu_count"] = torch.cuda.device_count()
import diffusers
from diffusers import DDIMScheduler, StableDiffusionPipeline
out["diffusers"] = diffusers.__version__
import transformers
from transformers import CLIPModel, CLIPProcessor, OwlViTForObjectDetection, OwlViTProcessor, AutoModel, AutoImageProcessor
out["transformers"] = transformers.__version__
import accelerate, safetensors
out["accelerate"] = accelerate.__version__
out["safetensors"] = safetensors.__version__
import importlib.util
out["torchao_importable"] = importlib.util.find_spec("torchao") is not None
print("GBC_PROBE " + json.dumps(out))
"""


class BootstrapError(RuntimeError):
    pass


def _run(cmd, timeout=1800, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return dict(cmd=[str(c) for c in cmd], returncode=p.returncode, stdout=p.stdout[-6000:], stderr=p.stderr[-6000:])


def installed_version(dist, python=sys.executable):
    r = _run([python, "-c", f"import importlib.metadata as m\ntry:\n print(m.version({dist!r}))\nexcept m.PackageNotFoundError:\n print('')"], timeout=120)
    return r["stdout"].strip() or None


def parse_pip_check(text):
    """pip check lines -> list of {line, packages}. 'No broken requirements found.' -> []."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no broken requirements"):
            continue
        names = re.findall(r"^([A-Za-z0-9_.\-]+)|(?:requires|has requirement|which is not installed\.?)\s*([A-Za-z0-9_.\-]+)", line)
        pk = {n.lower() for pair in names for n in pair if n}
        pk |= {m.lower() for m in re.findall(r"\b(" + "|".join(re.escape(p) for p in REQUIRED_PACKAGES) + r")\b", line, flags=re.I)}
        rows.append(dict(line=line, packages=sorted(pk)))
    return rows


def classify_conflicts(before, after):
    """Conflicts after the install that involve a required package are failures; the rest are recorded (pre-existing or introduced)."""
    before_lines = {r["line"] for r in before}
    req = {p.lower() for p in REQUIRED_PACKAGES}
    blocking, other = [], []
    for r in after:
        entry = dict(r, preexisting=r["line"] in before_lines)
        (blocking if req & set(r["packages"]) else other).append(entry)
    return blocking, other


def import_probe(python=sys.executable, env=None):
    """Imports the required classes in a fresh process. Returns (ok, info, raw)."""
    r = _run([python, "-c", PROBE], timeout=900, env=env)
    info = None
    for line in r["stdout"].splitlines():
        if line.startswith("GBC_PROBE "):
            info = json.loads(line[len("GBC_PROBE "):])
    ok = r["returncode"] == 0 and info is not None
    return ok, info, r


def bootstrap(requirements, out_dir, dry_run=False, python=sys.executable, loaded_modules=None):
    """Install, repair and verify the environment. Writes env_bootstrap.json; raises BootstrapError on any failure (after writing the record)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = sorted(m for m in (sys.modules if loaded_modules is None else loaded_modules) if m.split(".")[0] in STALE_MODULES)
    rec = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), python=sys.version.split()[0], dry_run=dry_run, requirements=str(requirements),
               steps={}, problems=[], ok=False)

    def fail(msg):
        rec["problems"].append(msg)
        rec["ok"] = False
        (out_dir / "env_bootstrap.json").write_text(json.dumps(rec, indent=1))
        raise BootstrapError(msg + f"  (details: {out_dir / 'env_bootstrap.json'})")

    prior = out_dir / "env_bootstrap.json"
    if loaded and prior.exists():
        try:
            old = json.loads(prior.read_text())
        except ValueError:
            old = {}
        if old.get("ok") is True and old.get("python") == rec["python"] and old.get("dry_run") == dry_run:
            return dict(old, reused_in_same_kernel=True)      # this kernel already bootstrapped successfully (cell re-run)
    if loaded:
        fail(f"modules already imported in this kernel: {loaded[:5]}; restart the kernel (Run > Restart) and run the notebook from the top")
    rec["torch_before"] = installed_version("torch", python)
    if not dry_run:
        pc0 = _run([python, "-m", "pip", "check"], timeout=600)
        rec["steps"]["pip_check_before"] = dict(returncode=pc0["returncode"], conflicts=parse_pip_check(pc0["stdout"]))
        inst = _run([python, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", str(requirements)])
        rec["steps"]["pip_install"] = inst
        if inst["returncode"] != 0:
            fail(f"pip install -r {Path(requirements).name} failed (exit {inst['returncode']}): {inst['stderr'][-800:]}")
        for dist in REMOVE:
            v = installed_version(dist, python)
            step = dict(installed_version=v, action="none (not installed)")
            if v:
                un = _run([python, "-m", "pip", "uninstall", "-y", dist], timeout=600)
                step.update(action="uninstalled (unused optional integration that breaks diffusers 0.41 imports)", returncode=un["returncode"], stderr=un["stderr"][-800:])
                if un["returncode"] != 0 or installed_version(dist, python):
                    rec["steps"][f"remove_{dist}"] = step
                    fail(f"could not uninstall {dist} {v}")
            rec["steps"][f"remove_{dist}"] = step
    rec["torch_after"] = installed_version("torch", python)
    if rec["torch_before"] != rec["torch_after"]:
        fail(f"the install replaced torch {rec['torch_before']} with {rec['torch_after']}; Kaggle's CUDA build must be kept")
    ok, info, raw = import_probe(python)
    rec["steps"]["fresh_import_probe"] = dict(ok=ok, info=info, returncode=raw["returncode"], stderr_tail=raw["stderr"][-3000:])
    if not ok:
        last = [l for l in raw["stderr"].strip().splitlines() if l.strip()][-1:] or ["(no stderr)"]
        fail(f"fresh-process import of DDIMScheduler / StableDiffusionPipeline / judge classes failed: {last[0]}")
    rec["versions"] = info
    if not dry_run:
        pc1 = _run([python, "-m", "pip", "check"], timeout=600)
        blocking, other = classify_conflicts(rec["steps"]["pip_check_before"]["conflicts"], parse_pip_check(pc1["stdout"]))
        rec["steps"]["pip_check_after"] = dict(returncode=pc1["returncode"], blocking=blocking, other=other)
        if blocking:
            fail("pip check reports conflicts involving required packages: " + " | ".join(b["line"] for b in blocking[:5]))
    rec["ok"] = True
    (out_dir / "env_bootstrap.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", default=str(Path(__file__).resolve().parent.parent / "requirements-kaggle.txt"))
    ap.add_argument("--out", default=".")
    ap.add_argument("--dry-run", action="store_true", help="no pip changes; only the fresh-process import probe")
    a = ap.parse_args()
    try:
        r = bootstrap(a.requirements, a.out, a.dry_run)
        print(json.dumps(dict(ok=True, versions=r["versions"]), indent=1))
    except BootstrapError as e:
        print("BOOTSTRAP FAILED:", e)
        sys.exit(1)
