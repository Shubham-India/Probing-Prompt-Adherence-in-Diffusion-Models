# %% [markdown]
# # 00 — Preflight, DEVELOPMENT experiments and freeze check (Guide Before Commitment)
# Run this notebook several times, once per STAGE, in this order (each stage refuses to start while its prerequisites are missing):
# 1. `PREFLIGHT` — hardware, versions, timing, VRAM, storage per image, hook-on/off identity, CAG loop identities, 1 vs 2 worker throughput, judge and
#    DINOv2 timing, pilot GPU identity test. Writes `preflight.json`. Generates no experiment data.
# 2. `DEV_CORE` — DEV non-canonical variants (+ CF bases with latents) and the DEV PS family. Gate: `variant_review_complete`.
# 3. `DEV_COMMIT` — DEV restart branches (D8, ER30, FULL) from the R2a latents, DEV bases and branch rehearsal at CFG 2 and 15, DEV prompt switches.
#    Gates: `evaluator_audit_complete`, `d13_resolved_for_r3`, `r3_authorised`. Needs the R2a core dataset (`kaggle_core_dataset`) and the DEV_CORE archive.
# 4. `DEV_CAG_GUARD` — CAG with w_lo = 1 and 3 on DEV canonical prompts. Needs `dev_constants.json` (from the DEV analysis, notebook 04 in DEV mode).
# 5. `FREEZE_CHECK` — verifies `protocol/protocol_freeze.json` inside the source package. The freeze itself is signed by the owner locally
#    (`python src/protocol_r3.py --sign`), after the human evaluator audit; then a NEW source package is uploaded and the TEST notebooks can run.
# **Inputs:** `source_package`, `benchmark_upload` (+ stage inputs above). **Settings:** GPU, Internet ON. **Outputs:** `/kaggle/working/archive_<RUN_ID>/`.
# %%  CONFIG — edit only this cell
SHARD_INDEX = 0          # DEV stages may also be split into shards (0..NUM_SHARDS-1)
NUM_SHARDS = 1
STAGE = "PREFLIGHT"     # PREFLIGHT | DEV_CORE | DEV_COMMIT | DEV_CAG_GUARD | FREEZE_CHECK
RUN_TAG = "dev"    # free text; the run id adds UTC time, shard and code commit
PART_GB = 4.0            # size of each output ZIP part
LIMIT = None             # None = whole shard; an integer = first N jobs (smoke run only, NOT a research run)
import os
DRY_RUN = os.environ.get("GBC_DRYRUN") == "1"   # local CPU dry-run of this notebook (fake pipeline); never set on Kaggle

# %%  1. Locate the source package and the benchmark dataset
import json, shutil, sys, time
from pathlib import Path
INPUT_ROOTS = [Path(p) for p in os.environ.get("GBC_INPUT_ROOTS", "/kaggle/input").split(os.pathsep)]
WORK = Path(os.environ.get("GBC_WORKING", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)

def _find(marker):
    hits = []
    for root in INPUT_ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            if len(Path(dirpath).relative_to(root).parts) > 6:
                dirnames[:] = []
            elif marker in filenames:
                hits.append(Path(dirpath))
    return sorted(set(hits))

src_hits = _find("SOURCE_MANIFEST.json")
if not src_hits:                                  # attached as a ZIP? extract it once into the working dir
    import zipfile
    for root in INPUT_ROOTS:
        for z in root.rglob("source_package*.zip"):
            zipfile.ZipFile(z).extractall(WORK / "_inputs" / z.stem)
    INPUT_ROOTS.append(WORK / "_inputs")
    src_hits = _find("SOURCE_MANIFEST.json")
assert len(src_hits) == 1, f"expected exactly one source package, found {src_hits}"
SRC = src_hits[0]
sys.path[:0] = [str(SRC / "src"), str(SRC / "kaggle")]
import nbtools
problems = nbtools.verify_manifest(SRC, "SOURCE_MANIFEST.json")
assert not problems, f"source package corrupt: {problems[:5]}"
SOURCE = json.loads((SRC / "SOURCE_MANIFEST.json").read_text())
bench = nbtools.find_marker(INPUT_ROOTS, "BENCHMARK_MANIFEST.json")
assert bench, "benchmark_upload dataset not attached"
bprob = nbtools.verify_manifest(bench[0], "BENCHMARK_MANIFEST.json")
assert not bprob, f"benchmark dataset corrupt: {bprob[:5]}"
for rel, h in json.loads((bench[0] / "BENCHMARK_MANIFEST.json").read_text())["files"].items():
    mirror = SRC / "data" / "benchmark_v2" / Path(rel).name if rel.startswith("benchmark_v2/") else None
    if mirror is not None and mirror.exists():
        assert nbtools.sha(mirror, True) == h, f"benchmark file {rel} differs from the copy inside the source package"
print("source commit:", SOURCE.get("git_head"), "| benchmark verified:", bench[0])

# %%  2. Dependency bootstrap (pinned libraries; keeps Kaggle's torch/CUDA build; removes the unused torchao that breaks diffusers 0.41 imports;
#     fresh-process import check of DDIMScheduler / StableDiffusionPipeline / judge classes; pip check). Raises on any failure.
import env_bootstrap
BOOT_DIR = WORK / "_bootstrap"
BOOT = env_bootstrap.bootstrap(SRC / "requirements-kaggle.txt", BOOT_DIR, dry_run=DRY_RUN)
print(json.dumps(BOOT["versions"], indent=1))

# %%  3. Environment, GPU and storage record
ENV = nbtools.env_report()
print(json.dumps(ENV, indent=1))
RUN_ID = f"{RUN_TAG}-s{SHARD_INDEX}of{NUM_SHARDS}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{str(SOURCE.get('git_head'))[:7]}"
# Resume across sessions: if a PARTIAL archive of this same stage and shard is attached, continue inside it (same run id)
for arch in nbtools.find_marker(INPUT_ROOTS, "ARCHIVE_MANIFEST.json"):
    am = json.loads((arch / "ARCHIVE_MANIFEST.json").read_text())
    if am.get("stage") == STAGE and am.get("shard") == f"{SHARD_INDEX}/{NUM_SHARDS}" and not nbtools.archive_is_verified(am):
        RUN_ID = am["run_id"]
        _, probs = nbtools.extract_and_verify(arch, WORK / f"out_{RUN_ID}")
        assert not probs, f"partial archive corrupt: {probs[:5]}"
        print("resuming from partial archive", arch)
OUT = WORK / f"out_{RUN_ID}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "env.json").write_text(json.dumps(ENV, indent=1))
(OUT / "env_bootstrap.json").write_text((BOOT_DIR / "env_bootstrap.json").read_text())
if SOURCE.get("dirty") is not False:
    print("WARNING: the source package was built from uncommitted work (dirty != false); it is not acceptable for DEV/TEST research stages")
    if STAGE != "PREFLIGHT" and not DRY_RUN:
        raise SystemExit("source package is dirty: commit, rebuild with scripts/build_release.py and upload a new gbc-source version")
print("RUN_ID =", RUN_ID)

# %%  4. PREFLIGHT / FREEZE_CHECK (no experiment data). Fail-closed: STAGE_OK stays False unless every required check passed.
import subprocess
STAGE_OK, STAGE_PROBLEMS = False, []
if STAGE == "PREFLIGHT":
    p = subprocess.run([sys.executable, str(SRC / "kaggle" / "r3_preflight.py"), "--out", str(OUT / "preflight.json"), "--n", "4"] + ([] if DRY_RUN else ["--two-workers"])
                       + (["--fake-pipeline"] if DRY_RUN else []), capture_output=True, text=True)
    (OUT / "preflight_process.log").write_text(f"[exit {p.returncode}]\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}\n")
    print(p.stdout[-4000:], p.stderr[-3000:])
    STAGE_PROBLEMS += nbtools.validate_preflight(OUT / "preflight.json", p.returncode, fake=DRY_RUN)
    if not DRY_RUN:
        gi = subprocess.run([sys.executable, str(SRC / "tests" / "gpu_identity.py"), "--out", str(OUT / "gpu_identity.json"), "--n", "3"], capture_output=True, text=True)
        (OUT / "gpu_identity_process.log").write_text(f"[exit {gi.returncode}]\n--- stdout\n{gi.stdout}\n--- stderr\n{gi.stderr}\n")
        print(gi.stdout[-2000:], gi.stderr[-2000:])
        STAGE_PROBLEMS += nbtools.validate_gpu_identity(OUT / "gpu_identity.json", gi.returncode)
    STAGE_OK = not STAGE_PROBLEMS
    print("PREFLIGHT", "PASSED" if STAGE_OK else "FAILED:\n  " + "\n  ".join(STAGE_PROBLEMS))
elif STAGE == "FREEZE_CHECK":
    f = subprocess.run([sys.executable, str(SRC / "src" / "protocol_r3.py"), "--verify", "--path", str(SRC / "protocol" / "protocol_freeze.json")], capture_output=True, text=True)
    print(f.stdout, f.stderr[-2000:])
    (OUT / "freeze_check.txt").write_text(f"[exit {f.returncode}]\n{f.stdout}\n--- stderr\n{f.stderr}\n")
    STAGE_PROBLEMS += nbtools.validate_freeze_check(f.returncode, f.stdout)
    STAGE_OK = not STAGE_PROBLEMS
    print("FREEZE_CHECK", "PASSED" if STAGE_OK else "FAILED:\n  " + "\n  ".join(STAGE_PROBLEMS))
elif not STAGE.startswith("DEV_"):
    raise SystemExit(f"unknown STAGE {STAGE!r}")

# %%  5. DEV stages: guard, manifest, generation, verification
CONSTANTS = None
for c in nbtools.find_marker(INPUT_ROOTS, "dev_constants.json") + ([SRC / "protocol"] if (SRC / "protocol" / "dev_constants.json").exists() else []):
    CONSTANTS = c / "dev_constants.json"
MINFO = {"manifest_sha256": None}
if STAGE.startswith("DEV_"):
    guard_cmd = [sys.executable, str(SRC / "src" / "stage_guard.py"), "--stage", STAGE, "--shard", f"{SHARD_INDEX}/{NUM_SHARDS}"]
    UPSTREAM = []
    if STAGE == "DEV_COMMIT":
        core = nbtools.find_marker(INPUT_ROOTS, "core_manifest.json")
        assert core, "attach the R2a core dataset (kaggle_core_dataset with core_manifest.json)"
        UPSTREAM.append(core[0] / "dev_core")
        guard_cmd += ["--core-run", str(core[0] / "dev_core")]
        for arch in nbtools.find_marker(INPUT_ROOTS, "ARCHIVE_MANIFEST.json"):
            am = json.loads((arch / "ARCHIVE_MANIFEST.json").read_text())
            if am.get("stage") == "DEV_CORE" and (nbtools.archive_is_verified(am) or DRY_RUN):
                dest, probs = nbtools.extract_and_verify(arch, WORK / "upstream" / am["run_id"], groups=("latents", "meta"))
                assert not probs, probs[:5]
                UPSTREAM.append(dest)
    if CONSTANTS:
        guard_cmd += ["--constants", str(CONSTANTS)]
    if os.environ.get("GBC_GATES"):
        guard_cmd += ["--gates", os.environ["GBC_GATES"]]
    g_rc, GUARD, g = nbtools.run_json(guard_cmd, "stage_guard", OUT / "logs")
    if GUARD is None:
        raise SystemExit(f"stage guard crashed (exit {g_rc}): {g.stderr[-2000:]}")
    (OUT / "guard.json").write_text(json.dumps(GUARD, indent=1))
    print(json.dumps(GUARD.get("plan"), indent=1))
    if g_rc != 0 or GUARD.get("ok") is not True:
        raise SystemExit("BLOCKED:\n  " + "\n  ".join(GUARD.get("blockers") or [f"stage guard exit {g_rc}"]))
    MANIFEST = OUT / "manifest.jsonl"
    mcmd = [sys.executable, str(SRC / "src" / "manifests.py"), "--stage", STAGE, "--shard", f"{SHARD_INDEX}/{NUM_SHARDS}", "--out", str(MANIFEST)]
    if CONSTANTS:
        mcmd += ["--constants", str(CONSTANTS)]
    m = subprocess.run(mcmd, capture_output=True, text=True, check=True)
    MINFO = json.loads(m.stdout.strip().splitlines()[-1])
    assert MINFO["manifest_sha256"] == GUARD["plan"]["manifest_sha256"]
    codes = nbtools.run_workers(SRC, MANIFEST, OUT, RUN_ID, f"d{SHARD_INDEX}", upstream=UPSTREAM, fake=DRY_RUN, limit=LIMIT)
    print("worker exit codes:", codes)
    merge_cmd = [sys.executable, str(SRC / "src" / "r3_merge.py"), "--manifest", str(MANIFEST), "--inputs", str(OUT), "--out", str(OUT / "verify")]
    if DRY_RUN:
        merge_cmd += ["--allow-fake", "--allow-missing"]
    shutil.rmtree(OUT / "verify", ignore_errors=True)          # never read a merge report left over from an earlier session
    v = subprocess.run(merge_cmd, capture_output=True, text=True)
    print(v.stdout[-3000:], v.stderr[-2000:])
    VERIFY, verr = nbtools.load_json_object(OUT / "verify" / "merge_report.json")
    STAGE_OK, STAGE_PROBLEMS = nbtools.stage_verdict(codes, v.returncode, VERIFY, limit=LIMIT, dry_run=DRY_RUN)
    if not STAGE_OK:
        print("STAGE NOT VERIFIED:\n  " + "\n  ".join(STAGE_PROBLEMS[:20]))
        print("Re-run this cell (it resumes), or package now as a PARTIAL archive (verified=false) and resume in a new session.")

# %%  6. Package for download / dataset upload (always packaged so diagnostics survive; verified only with STAGE_OK; then fail loudly)
ARCH = WORK / f"archive_{RUN_ID}"
man = nbtools.package(OUT, ARCH, RUN_ID, PART_GB, meta=dict(stage=STAGE, shard=f"{SHARD_INDEX}/{NUM_SHARDS}", verified=STAGE_OK is True and not DRY_RUN,
                                                            verification_problems=STAGE_PROBLEMS[:50], dry_run=DRY_RUN, source_commit=SOURCE.get("git_head"),
                                                            source_dirty=SOURCE.get("dirty"), manifest_sha256=MINFO["manifest_sha256"], smoke_limit=LIMIT))
print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))
print(f"DONE stage={STAGE} run={RUN_ID} verified={man['verified']} archive={ARCH}")
if nbtools.blocking_problems(STAGE_PROBLEMS, DRY_RUN) or (not STAGE_OK and not DRY_RUN):
    raise nbtools.ValidationError(f"{STAGE} NOT VERIFIED (archive packaged with verified=false for diagnosis/resume):\n  " + "\n  ".join(STAGE_PROBLEMS[:20]))
