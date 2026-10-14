"""Owner-side helper (runs on the machine that has the git checkout): packs a COMMIT into a folder ready to be uploaded as a PRIVATE Kaggle dataset.
Usage: python kaggle/make_bundle.py --commit <SHA> --out kaggle_bundle      -> kaggle_bundle/{repo/, COMMIT}
Then upload `kaggle_bundle` via the Kaggle web UI (Datasets -> New Dataset -> Private). Model weights, outputs/ and the pilot validation images (data/validation_subsets_images) are not included."""
import argparse
import subprocess
import tarfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--commit", required=True)
ap.add_argument("--out", default="kaggle_bundle")
a = ap.parse_args()
out = Path(a.out)
(out / "repo").mkdir(parents=True, exist_ok=True)
sha = subprocess.run(["git", "rev-parse", a.commit], capture_output=True, text=True, check=True).stdout.strip()
tmp = out / "repo.tar"
subprocess.run(["git", "archive", "--format=tar", "-o", str(tmp), sha, "--", "src", "tests", "configs", "data", "kaggle", "docs", "requirements-kaggle.txt", "requirements.txt", ":(exclude)data/validation_subsets_images"], check=True)
with tarfile.open(tmp) as t:
    t.extractall(out / "repo", filter="data")
tmp.unlink()
(out / "COMMIT").write_text(sha + "\n")
print("bundle ready:", out, "commit", sha)
