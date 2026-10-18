"""Build release/guide_before_commitment/ from the working tree: source_package.zip, benchmark_upload.zip, the five notebooks (.ipynb + .py),
helper scripts, manifests, guides and checksums; then validate everything it built (ZIPs extract and their internal manifests verify, notebooks are
valid nbformat-4 JSON whose code cells compile, every expected file exists).
Only git-TRACKED files are packaged (untracked or ignored files inside the source folders never reach Kaggle). `dirty` is true when any packaged path,
or any file the release itself copies (notebooks, guides, requirements), differs from HEAD or is untracked; uncommitted paths elsewhere in the working
tree are listed separately (`worktree_uncommitted_outside_release`) and do not change the release content. Validation FAILS on a dirty release unless
--allow-dirty is given (development builds only: never upload a dirty source package for research stages).
Usage: python scripts/build_release.py [--out release/guide_before_commitment] [--no-validate] [--allow-dirty]"""
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = (".py", ".json", ".md", ".txt", ".jsonl", ".csv")
NOTEBOOKS = [("nb00_preflight_dev_freeze.py", "00_PREFLIGHT_DEV_FREEZE"), ("nb01_test_core_cfg_worker.py", "01_TEST_CORE_CFG_WORKER"),
             ("nb02_test_commitment_worker.py", "02_TEST_COMMITMENT_WORKER"), ("nb03_test_cag_size_worker.py", "03_TEST_CAG_SIZE_WORKER"),
             ("nb04_merge_analyze_export.py", "04_MERGE_ANALYZE_EXPORT")]
SOURCE_DIRS = ["src", "kaggle", "configs", "scripts", "tests", "data/benchmark_v2", "protocol"]
SOURCE_FILES = ["requirements-kaggle.txt", "requirements.txt", "data/prompts.json", "docs/ADDENDUM_C.md", "docs/R3_DECISION_LOG.md", "docs/AUDIT_PROTOCOL_DEV.md",
                "GRAND_RESEARCH_BLUEPRINT.md"]
DOCS = {"README_FIRST.md": "docs/release/README_FIRST.md", "LOCAL_SETUP.md": "docs/release/LOCAL_SETUP.md", "KAGGLE_RUNBOOK.md": "docs/release/KAGGLE_RUNBOOK.md",
        "EXPERIMENT_STATUS.md": "docs/release/EXPERIMENT_STATUS.md", "KNOWN_LIMITATIONS.md": "docs/release/KNOWN_LIMITATIONS.md",
        "implementation_audit.md": "docs/release/implementation_audit.md", "RESULTS_PROVENANCE.md": "docs/RESULTS_PROVENANCE.md",
        "DECISION_LOG.md": "docs/R3_DECISION_LOG.md", "ADDENDUM_C.md": "docs/ADDENDUM_C.md", "validation_report.md": "docs/release/validation_report.md"}


def sha(p, text=True):
    b = Path(p).read_bytes()
    if text and Path(p).suffix in TEXT:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def to_notebook(py_text, title):
    cells, cur, kind = [], [], None
    for line in py_text.splitlines():
        if line.startswith("# %%"):
            if cur:
                cells.append((kind, cur))
            kind, cur = ("markdown" if "[markdown]" in line else "code"), ([] if "[markdown]" in line else [line])
            continue
        cur.append(line)
    if cur:
        cells.append((kind, cur))
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}, "gbc": {"title": title}}, "cells": []}
    for i, (k, lines) in enumerate(cells):
        if k == "markdown":
            src = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
            nb["cells"].append({"cell_type": "markdown", "id": f"c{i:02d}", "metadata": {}, "source": [s + "\n" for s in src]})
        else:
            while lines and not lines[-1].strip():
                lines.pop()
            nb["cells"].append({"cell_type": "code", "id": f"c{i:02d}", "metadata": {}, "execution_count": None, "outputs": [], "source": [s + "\n" for s in lines]})
    for c in nb["cells"]:
        if c["source"]:
            c["source"][-1] = c["source"][-1].rstrip("\n")
    return nb


RELEASE_INPUTS = ["requirements.txt", "requirements-kaggle.txt", "docs/release", "docs/RESULTS_PROVENANCE.md"]


def tracked_files(paths):
    """Git-tracked files under `paths` (relative POSIX paths). None if git is unavailable."""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", *paths], capture_output=True, text=True)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return sorted(f for f in r.stdout.split("\0") if f)


def uncommitted(paths=None):
    """Paths with uncommitted changes (modified, staged, deleted or untracked), optionally restricted to `paths`."""
    r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain=v1", "-z", "--untracked-files=all", *(["--", *paths] if paths else [])],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git status failed: {r.stderr.strip()}")
    entries, out = r.stdout.split("\0"), set()
    i = 0
    while i < len(entries):
        e = entries[i]
        if len(e) > 3:
            out.add(e[3:])
            if e[0] in "RC":            # rename / copy: the next entry is the original path
                i += 1
                out.add(entries[i])
        i += 1
    return sorted(out)


def untracked_files(paths):
    """Untracked, NOT ignored files under `paths` (development builds only)."""
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--others", "--exclude-standard", "--", *paths], capture_output=True, text=True)
    return sorted(f for f in r.stdout.split("\0") if f) if r.returncode == 0 else []


def stage_source(dest, include_untracked=False):
    dest.mkdir(parents=True, exist_ok=True)
    files = tracked_files(SOURCE_DIRS + SOURCE_FILES)
    if files is None:
        raise SystemExit("git is required to build the release (only tracked files are packaged)")
    if include_untracked:                     # development / dry-run builds: new files not yet committed (the package is then dirty)
        files = sorted(set(files) | set(untracked_files(SOURCE_DIRS + SOURCE_FILES)))
    for f in files:
        if f.endswith((".pyc", ".tmp")) or "__pycache__" in f or not (ROOT / f).is_file():
            continue
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest / f)
    return write_source_manifest(dest)


def write_source_manifest(dest):
    dest = Path(dest)
    scope = SOURCE_DIRS + SOURCE_FILES + RELEASE_INPUTS
    dirty = uncommitted(scope)
    outside = sorted(set(uncommitted()) - set(dirty))
    files = {p.relative_to(dest).as_posix(): sha(p) for p in sorted(dest.rglob("*")) if p.is_file() and p.name != "SOURCE_MANIFEST.json"}
    man = dict(package="guide_before_commitment source", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), git_head=git("rev-parse", "HEAD"),
               git_branch=git("branch", "--show-current"), dirty=bool(dirty), dirty_scope=scope, uncommitted_paths=dirty[:500],
               worktree_uncommitted_outside_release=outside[:500], files=files,
               note="dirty=true means a packaged path (or a release input) differs from HEAD or is untracked; commit and rebuild before any research run. "
                    "Only git-tracked files are packaged; worktree_uncommitted_outside_release lists changes that do not enter the release.")
    (dest / "SOURCE_MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def stage_benchmark(dest):
    b = dest / "benchmark_v2"
    b.mkdir(parents=True, exist_ok=True)
    for f in sorted((ROOT / "data" / "benchmark_v2").iterdir()):
        if f.is_file():
            shutil.copy2(f, b / f.name)
    (dest / "configs").mkdir(exist_ok=True)
    for f in ("experiment_r3.json", "analysis_plan_r3.json", "benchmark_v2.json"):
        shutil.copy2(ROOT / "configs" / f, dest / "configs" / f)
    shutil.copy2(ROOT / "docs" / "ADDENDUM_C.md", dest / "ADDENDUM_C.md")
    (dest / "README.md").write_text(
        "# Guide Before Commitment - benchmark input dataset\n\nSmall input artefacts only (no images, latents or weights).\n"
        "- benchmark_v2/: scene graphs, prompts (main benchmark and the PS size family), DEV/TEST splits, lexicon, datasheet, design counts, hashes\n"
        "- configs/: experiment_r3.json (generation design), analysis_plan_r3.json (frozen analysis rules once signed), benchmark_v2.json\n"
        "- ADDENDUM_C.md: the human-readable protocol (DRAFT until signed)\n- BENCHMARK_MANIFEST.json: SHA-256 (LF-normalised) of every file + split hashes\n"
        "Upload this ZIP (or its extracted folder) as a PRIVATE Kaggle dataset named e.g. `gbc-benchmark`. The notebooks verify it against the copy inside the source package.\n")
    files = {p.relative_to(dest).as_posix(): sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    splits = json.loads((b / "splits.json").read_text())
    ps_splits = json.loads((b / "splits_ps.json").read_text())
    man = dict(dataset="guide_before_commitment benchmark", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), files=files,
               split_sha256=hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
               ps_split_sha256=hashlib.sha256(json.dumps(ps_splits, sort_keys=True).encode()).hexdigest(),
               counts=dict(dev_scenes=len(splits["DEV"]), test_scenes=len(splits["TEST"]), ps_pairs_dev=len(ps_splits["DEV"]), ps_pairs_test=len(ps_splits["TEST"])))
    (dest / "BENCHMARK_MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def zipdir(folder, zpath, arc_root):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, f"{arc_root}/{p.relative_to(folder).as_posix()}")


def build(out, include_untracked=False):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "kaggle_notebooks").mkdir(parents=True)
    (out / "scripts").mkdir()
    (out / "manifests").mkdir()
    tmp = Path(tempfile.mkdtemp())
    src_man = stage_source(tmp / "gbc_source", include_untracked)
    zipdir(tmp / "gbc_source", out / "source_package.zip", "gbc_source")
    bm = stage_benchmark(tmp / "gbc_benchmark")
    zipdir(tmp / "gbc_benchmark", out / "benchmark_upload.zip", "gbc_benchmark")
    for py, name in NOTEBOOKS:
        text = (ROOT / "kaggle" / py).read_text(encoding="utf-8")
        (out / "kaggle_notebooks" / f"{name}.py").write_text(text, encoding="utf-8")
        (out / "kaggle_notebooks" / f"{name}.ipynb").write_text(json.dumps(to_notebook(text, name), indent=1), encoding="utf-8")
    for p in sorted((ROOT / "scripts").glob("*.py")):
        shutil.copy2(p, out / "scripts" / p.name)
    for dst, src in DOCS.items():
        if (ROOT / src).exists():
            shutil.copy2(ROOT / src, out / dst)
    shutil.copy2(ROOT / "requirements.txt", out / "requirements-local.txt")
    shutil.copy2(ROOT / "requirements-kaggle.txt", out / "requirements-kaggle.txt")
    sys.path.insert(0, str(ROOT / "src"))
    import manifests as M
    jobs, notes, extra = M.build()
    per = {}
    for st in M.STAGES:
        per[st] = {f"{i}/3": dict(jobs=len(s), image_equivalents=round(sum(j["unet_calls"] for j in s) / 80, 1), manifest_sha256=M.manifest_hash(s))
                   for i in range(3) for s in [M.select(jobs, st, (i, 3))]}
    (out / "manifests" / "experiment_manifest.json").write_text(json.dumps(dict(summary=M.summary(jobs, extra), per_stage_shard=per, notes=notes), indent=1))
    (out / "manifests" / "dataset_manifest.json").write_text(json.dumps(bm, indent=1))
    splits = json.loads((ROOT / "data" / "benchmark_v2" / "splits.json").read_text())
    (out / "manifests" / "scene_split_manifest.json").write_text(json.dumps(dict(main=splits, ps_family=json.loads((ROOT / "data" / "benchmark_v2" / "splits_ps.json").read_text()),
                                                                               split_sha256=bm["split_sha256"], ps_split_sha256=bm["ps_split_sha256"]), indent=1))
    (out / "manifests" / "required_artifacts.json").write_text(json.dumps(REQUIRED, indent=1))
    (out / "manifests" / "source_manifest_summary.json").write_text(json.dumps({k: v for k, v in src_man.items() if k != "files"} | {"n_files": len(src_man["files"])}, indent=1))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    (out / "checksums.sha256").write_text("".join(f"{sha(p, False)}  {p.relative_to(out).as_posix()}\n" for p in files), newline="\n")
    shutil.rmtree(tmp, ignore_errors=True)
    return src_man


REQUIRED = {
    "00_PREFLIGHT_DEV_FREEZE": {"inputs": ["source_package", "benchmark_upload", "DEV_COMMIT: r2a core dataset (kaggle_core_dataset) + DEV_CORE archive", "DEV_CAG_GUARD: dev_constants.json"],
                                "outputs": ["archive_<run>: preflight.json | DEV stage outputs"], "next": "04 (PHASE=DEV) after all DEV stages; then local audit + freeze"},
    "01_TEST_CORE_CFG_WORKER": {"inputs": ["source_package with signed protocol", "benchmark_upload"], "outputs": ["archive per shard: images, latents, attention, meta"],
                                "next": "02 (latents + meta groups) and 04"},
    "02_TEST_COMMITMENT_WORKER": {"inputs": ["source_package with signed protocol", "benchmark_upload", "01 output datasets"], "outputs": ["archive per shard"], "next": "04"},
    "03_TEST_CAG_SIZE_WORKER": {"inputs": ["source_package with signed protocol", "benchmark_upload"], "outputs": ["archive per shard"], "next": "04"},
    "04_MERGE_ANALYZE_EXPORT": {"inputs": ["source_package", "benchmark_upload", "all worker datasets of the phase", "DEV: r2a core dataset"],
                                "outputs": ["DEV: dev_constants.json", "TEST: confirmatory.json, tables, figures, RESULTS_CATALOGUE.json"], "next": "local: build_draft.py / paper"},
}


def validate(out, allow_dirty=False):
    out = Path(out)
    probs = []
    summary = json.loads((out / "manifests" / "source_manifest_summary.json").read_text())
    if summary.get("dirty") is not False and not allow_dirty:
        probs.append(f"source package is dirty ({len(summary.get('uncommitted_paths', []))} uncommitted packaged paths, e.g. "
                     f"{summary.get('uncommitted_paths', [])[:5]}); commit them and rebuild, or pass --allow-dirty for a development build")
    expected = ["README_FIRST.md", "implementation_audit.md", "KNOWN_LIMITATIONS.md", "requirements-local.txt", "requirements-kaggle.txt", "benchmark_upload.zip",
                "source_package.zip", "checksums.sha256", "LOCAL_SETUP.md", "KAGGLE_RUNBOOK.md", "EXPERIMENT_STATUS.md", "RESULTS_PROVENANCE.md"]
    expected += [f"kaggle_notebooks/{n}.{e}" for _, n in NOTEBOOKS for e in ("ipynb", "py")]
    expected += [f"manifests/{m}" for m in ("dataset_manifest.json", "scene_split_manifest.json", "experiment_manifest.json", "required_artifacts.json")]
    probs += [f"missing {e}" for e in expected if not (out / e).exists()]
    for line in (out / "checksums.sha256").read_text().splitlines():
        h, rel = line.split("  ", 1)
        if sha(out / rel, False) != h:
            probs.append(f"checksum mismatch {rel}")
    sys.path.insert(0, str(ROOT / "kaggle"))
    import nbtools
    with tempfile.TemporaryDirectory() as td:
        for zname, marker in (("source_package.zip", "SOURCE_MANIFEST.json"), ("benchmark_upload.zip", "BENCHMARK_MANIFEST.json")):
            zipfile.ZipFile(out / zname).extractall(td)
            hit = nbtools.find_marker([td], marker)
            if len(hit) != 1:
                probs.append(f"{zname}: marker {marker} not found once")
            else:
                probs += [f"{zname}: {p}" for p in nbtools.verify_manifest(hit[0], marker)]
        src = Path(td) / "gbc_source"
        sm = json.loads((src / "SOURCE_MANIFEST.json").read_text())
        if sm.get("dirty") != summary.get("dirty") or sm.get("git_head") != summary.get("git_head"):
            probs.append("source_manifest_summary.json disagrees with SOURCE_MANIFEST.json inside source_package.zip")
        for f in ("kaggle/env_bootstrap.py", "kaggle/nbtools.py", "kaggle/r3_preflight.py", "tests/gpu_identity.py", "requirements-kaggle.txt", "src/protocol_r3.py"):
            if not (src / f).exists():
                probs.append(f"source_package.zip lacks {f}")
        if (src / "requirements-kaggle.txt").exists() and sha(src / "requirements-kaggle.txt") != sha(out / "requirements-kaggle.txt"):
            probs.append("requirements-kaggle.txt differs between the release folder and the source package")
        for py, n in NOTEBOOKS:
            rel_py = (out / "kaggle_notebooks" / f"{n}.py").read_text(encoding="utf-8")
            if not (src / "kaggle" / py).exists() or (src / "kaggle" / py).read_text(encoding="utf-8").replace("\r\n", "\n") != rel_py.replace("\r\n", "\n"):
                probs.append(f"{n}.py differs from kaggle/{py} inside source_package.zip")
            nb = json.loads((out / "kaggle_notebooks" / f"{n}.ipynb").read_text(encoding="utf-8"))
            code_nb = [l for c in nb["cells"] if c["cell_type"] == "code" for l in "".join(c["source"]).splitlines() if l.strip()]
            code_py = [l for l in rel_py.replace("\r\n", "\n").splitlines() if l.strip() and not l.startswith("# ")]
            code_py_ref = [l for l in code_nb if not l.startswith("# ")]
            if code_py_ref != code_py:
                probs.append(f"{n}.ipynb code cells do not match {n}.py")
    for _, n in NOTEBOOKS:
        nb = json.loads((out / "kaggle_notebooks" / f"{n}.ipynb").read_text(encoding="utf-8"))
        if nb.get("nbformat") != 4 or not nb["cells"]:
            probs.append(f"{n}: not nbformat 4")
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                try:
                    compile("".join(c["source"]), n, "exec")
                except SyntaxError as e:
                    probs.append(f"{n}: cell {c['id']} does not compile: {e}")
        ast.parse((out / "kaggle_notebooks" / f"{n}.py").read_text(encoding="utf-8"))
    return probs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "release" / "guide_before_commitment"))
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="development build only: do not fail validation on uncommitted packaged paths")
    a = ap.parse_args()
    man = build(a.out, include_untracked=a.allow_dirty)
    print(f"built {a.out}; source commit {man['git_head'][:7]} dirty={man['dirty']} ({len(man['uncommitted_paths'])} uncommitted packaged paths; "
          f"{len(man['worktree_uncommitted_outside_release'])} uncommitted paths outside the release)")
    if man["uncommitted_paths"]:
        print("uncommitted packaged paths:\n  " + "\n  ".join(man["uncommitted_paths"][:80]))
    if not a.no_validate:
        p = validate(a.out, a.allow_dirty)
        print("VALIDATION:", "PASSED" if not p else "FAILED\n  " + "\n  ".join(p[:40]))
        sys.exit(1 if p else 0)
