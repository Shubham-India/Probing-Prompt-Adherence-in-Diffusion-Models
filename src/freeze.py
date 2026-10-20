"""Pre-registration freeze tooling (D-06). Hashes every file that defines the confirmatory analysis (code, configs, split manifest,
metric definitions in evaluate.py, analysis plan/thresholds, Addendum B text), writes a manifest, and verifies it later.
TEST generation/analysis scripts must call require_verified() first.

FREEZE PROCEDURE
  1. Finalise code and docs; pin configs/experiment_v2.json revision (40-hex) and split_manifest; leave freeze.addendum_hash null.
  2. python src/freeze.py --write     writes docs/freeze/freeze_manifest.json and docs/freeze/ADDENDUM_HASH (SHA-256 of the file hashes).
  3. Copy that hash into configs/experiment_v2.json freeze.addendum_hash. This does not invalidate the manifest because the config is
     hashed with that one field blanked (canonical_config_bytes). Commit; record the hash in the Addendum B text.
  4. python src/freeze.py --verify    exit 0 only if nothing changed, the hash matches in all three places and the revision is pinned.
  python src/freeze.py --deviation "text"   appends a dated line to docs/freeze/deviations.md (history is never edited).
Anything changed after step 2 is a deviation: log it, relabel affected analyses EXPLORATORY, and never present them as untouched confirmation."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FREEZE_DIR = "docs/freeze"
TRACKED = [
    "src/*.py", "configs/experiment_v2.json", "configs/benchmark_v2.json", "configs/analysis_plan.json",
    "data/benchmark_v2/scene_graphs.json", "data/benchmark_v2/prompts_v2.json", "data/benchmark_v2/splits.json", "data/benchmark_v2/lexicon.json",
    "docs/ADDENDUM_B.md", "requirements-kaggle.txt",
]


def canonical_config_bytes(path):
    """experiment_v2.json with freeze.addendum_hash blanked, so storing the hash in the config does not change the hashed content."""
    cfg = json.loads(Path(path).read_text())
    cfg.get("freeze", {})["addendum_hash"] = None
    return json.dumps(cfg, sort_keys=True).encode()


def file_hash(root, rel):
    p = Path(root) / rel
    data = canonical_config_bytes(p) if rel.endswith("experiment_v2.json") else p.read_bytes()
    if rel.endswith((".py", ".json", ".md", ".txt", ".jsonl")):
        data = data.replace(b"\r\n", b"\n")   # a Windows checkout (autocrlf) must hash like the Linux clone used on Kaggle
    return hashlib.sha256(data).hexdigest()


def collect(root):
    files = []
    for pat in TRACKED:
        files += sorted(p.relative_to(root).as_posix() for p in Path(root).glob(pat))   # POSIX keys: manifests must be portable
    return sorted(set(files))


def build_manifest(root=ROOT):
    root = Path(root)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "files": {rel: file_hash(root, rel) for rel in collect(root)}}


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()


def verify(root=ROOT, manifest_path=None):
    """Returns a list of problems (empty = verified)."""
    root = Path(root)
    mp = Path(manifest_path or root / FREEZE_DIR / "freeze_manifest.json")
    if not mp.exists():
        return ["no freeze manifest"]
    man = json.loads(mp.read_text())
    problems = []
    now = {rel: file_hash(root, rel) for rel in collect(root)}
    for rel, h in man["files"].items():
        if rel not in now:
            problems.append(f"missing: {rel}")
        elif now[rel] != h:
            problems.append(f"changed after freeze: {rel}")
    problems += [f"new file not in manifest: {r}" for r in now if r not in man["files"]]
    ah = root / FREEZE_DIR / "ADDENDUM_HASH"
    if not ah.exists() or ah.read_text().strip() != manifest_hash(man):
        problems.append("ADDENDUM_HASH missing or differs from the manifest hash")
    cfg = json.loads((root / "configs/experiment_v2.json").read_text())
    if cfg["freeze"].get("addendum_hash") != manifest_hash(man):
        problems.append("configs/experiment_v2.json freeze.addendum_hash differs from the manifest hash")
    if cfg["revision"] == "UNPINNED-DRAFT" or not re.fullmatch(r"[0-9a-f]{40}", str(cfg["revision"])):
        problems.append("model revision not pinned")
    if not cfg.get("split_manifest"):
        problems.append("split_manifest not set")
    return problems


def require_verified(root=ROOT):
    p = verify(root)
    if p:
        sys.exit("TEST access refused (D-06): " + "; ".join(p[:5]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--deviation", default=None)
    a = ap.parse_args()
    if a.write:
        man = build_manifest()
        Path(ROOT / FREEZE_DIR).mkdir(parents=True, exist_ok=True)
        (ROOT / FREEZE_DIR / "freeze_manifest.json").write_text(json.dumps(man, indent=1))
        (ROOT / FREEZE_DIR / "ADDENDUM_HASH").write_text(manifest_hash(man) + "\n")
        print("manifest hash", manifest_hash(man), "- now copy it to configs/experiment_v2.json freeze.addendum_hash")
    elif a.verify:
        p = verify()
        print("VERIFIED" if not p else "NOT VERIFIED:\n  " + "\n  ".join(p))
        sys.exit(1 if p else 0)
    elif a.deviation:
        d = ROOT / FREEZE_DIR / "deviations.md"
        d.parent.mkdir(parents=True, exist_ok=True)
        with open(d, "a") as f:
            f.write(f"- {datetime.now(timezone.utc).isoformat()}: {a.deviation}\n")
