"""Writable cache directory for SD 1.5, OWL-ViT, CLIP and DINOv2 downloads (standard library only).
Order: GBC_MODEL_CACHE if set; on Kaggle /kaggle/working/models; locally <repo>/models. On Kaggle the source package is a read-only dataset under
/kaggle/input, so a cache inside it (or anywhere under /kaggle/input) is refused rather than failing later with Errno 30."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_WORKING = Path("/kaggle/working")
READ_ONLY_PREFIXES = ("/kaggle/input",)


def on_kaggle(root=ROOT, working=KAGGLE_WORKING):
    return Path(working).is_dir() or Path(root).as_posix().startswith(READ_ONLY_PREFIXES)


def model_cache_dir(create=True, root=ROOT, working=KAGGLE_WORKING, env=None):
    env = os.environ if env is None else env
    kaggle = on_kaggle(root, working)
    if env.get("GBC_MODEL_CACHE"):
        d = Path(env["GBC_MODEL_CACHE"])
    elif kaggle:
        d = Path(working) / "models"
    else:
        d = Path(root) / "models"
    posix = d.as_posix()
    if posix.startswith(READ_ONLY_PREFIXES) or (kaggle and (d == Path(root) or Path(root) in d.parents)):
        raise OSError(f"model cache {d} is inside the read-only input / source package; set GBC_MODEL_CACHE to a writable folder (e.g. /kaggle/working/models)")
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d
