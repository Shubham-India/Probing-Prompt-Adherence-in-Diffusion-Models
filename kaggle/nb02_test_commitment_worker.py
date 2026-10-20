# %% [markdown]
# # 02 — TEST commitment worker: restart branches at CFG 2 / 7.5 / 15 and CF-C / CF-R prompt switches
# **Purpose.** For one scene shard: re-noising branches (E23: D8 + ER30 at w = 7.5; E29: D8 at w = 2 and 15) from the saved TEST latents, and
# prompt-switch runs (E31) from the saved counterfactual latents.
# **Depends on notebook 01**: attach the output dataset(s) of notebook 01 (at least the same SHARD_INDEX; the `latents` and `meta` ZIP groups are enough).
# Runs only after the protocol freeze. **Inputs:** `source_package`, `benchmark_upload`, `gbc-test-core-shard<i>`. **Settings:** GPU, Internet ON.
# **Outputs:** `/kaggle/working/archive_<RUN_ID>/` (images + meta). **Next:** private dataset `gbc-test-commit-shard<i>` -> notebook 04.
# %%  CONFIG — edit only this cell
SHARD_INDEX = 0          # 0, 1 or 2 — each shard is a disjoint set of scene groups
NUM_SHARDS = 3
STAGE = "TEST_COMMIT"
RUN_TAG = "test-commit"    # free text; the run id adds UTC time, shard and code commit
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
import subprocess
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
if SOURCE.get("dirty") is not False and not DRY_RUN:
    raise SystemExit("source package is dirty (built from uncommitted work): commit, rebuild with scripts/build_release.py and upload a new gbc-source version")
print("RUN_ID =", RUN_ID)

# %%  4. Stage guard: gates, frozen protocol, independent job count and time estimate (stops here if anything is missing)
PROTOCOL = SRC / "protocol" / "protocol_freeze.json"
guard_cmd = [sys.executable, str(SRC / "src" / "stage_guard.py"), "--stage", STAGE, "--shard", f"{SHARD_INDEX}/{NUM_SHARDS}", "--protocol", str(PROTOCOL),
             "--constants", str(PROTOCOL)]
if os.environ.get("GBC_GATES"):
    guard_cmd += ["--gates", os.environ["GBC_GATES"]]
g_rc, GUARD, g = nbtools.run_json(guard_cmd, "stage_guard", OUT / "logs")
if GUARD is None:
    raise SystemExit(f"stage guard crashed (exit {g_rc}): {g.stderr[-2000:]}")
(OUT / "guard.json").write_text(json.dumps(GUARD, indent=1))
print(json.dumps(GUARD.get("plan"), indent=1))
if g_rc != 0 or GUARD.get("ok") is not True:
    raise SystemExit("BLOCKED:\n  " + "\n  ".join(GUARD.get("blockers") or [f"stage guard exit {g_rc}"]))

# %%  5. Build this shard's manifest (deterministic; its hash must equal the guard's)
MANIFEST = OUT / "manifest.jsonl"
m = subprocess.run([sys.executable, str(SRC / "src" / "manifests.py"), "--stage", STAGE, "--shard", f"{SHARD_INDEX}/{NUM_SHARDS}", "--out", str(MANIFEST),
                    "--constants", str(PROTOCOL)], capture_output=True, text=True, check=True)
MINFO = json.loads(m.stdout.strip().splitlines()[-1])
assert MINFO["manifest_sha256"] == GUARD["plan"]["manifest_sha256"], "manifest differs from the one the guard approved"
print(MINFO)

# %%  6a. Attach the base latents produced by notebook 01 (verified extraction; only the latents + meta groups)
UPSTREAM = []
for arch in nbtools.find_marker(INPUT_ROOTS, "ARCHIVE_MANIFEST.json"):
    am = json.loads((arch / "ARCHIVE_MANIFEST.json").read_text())
    if am.get("stage") != "TEST_CORE":
        continue
    if not nbtools.archive_is_verified(am) and not DRY_RUN:
        raise SystemExit(f"{arch} is an unverified TEST_CORE archive; refusing to use it")
    dest, probs = nbtools.extract_and_verify(arch, WORK / "upstream" / am["run_id"], groups=("latents", "meta"))
    assert not probs, f"upstream archive {arch} failed verification: {probs[:5]}"
    UPSTREAM.append(dest)
print("upstream latent folders:", UPSTREAM)
assert UPSTREAM, "attach the notebook-01 output dataset(s)"

# %%  6b. Generate branches and switches (one process per GPU; resume-safe)
codes = nbtools.run_workers(SRC, MANIFEST, OUT, RUN_ID, f"s{SHARD_INDEX}", upstream=UPSTREAM, protocol=PROTOCOL, fake=DRY_RUN, limit=LIMIT)
print("worker exit codes:", codes)

# %%  7. Verify the shard (completeness, duplicates, hashes, PNG decoding, settings vs manifest)
merge_cmd = [sys.executable, str(SRC / "src" / "r3_merge.py"), "--manifest", str(MANIFEST), "--inputs", str(OUT), "--out", str(OUT / "verify")]
if DRY_RUN:
    merge_cmd += ["--allow-fake", "--allow-missing"]
shutil.rmtree(OUT / "verify", ignore_errors=True)          # never read a merge report left over from an earlier session
v = subprocess.run(merge_cmd, capture_output=True, text=True)
print(v.stdout[-3000:], v.stderr[-2000:])
VERIFY, verr = nbtools.load_json_object(OUT / "verify" / "merge_report.json")
STAGE_OK, STAGE_PROBLEMS = nbtools.stage_verdict(codes, v.returncode, VERIFY, limit=LIMIT, dry_run=DRY_RUN)
if not STAGE_OK:
    print("SHARD NOT VERIFIED:\n  " + "\n  ".join(STAGE_PROBLEMS[:20]))
    print("If the session is about to end, cell 8 still packages it as a PARTIAL archive (verified=false) that a")
    print("new session can resume from (attach it and re-run this notebook). Otherwise re-run cell 6 (it resumes) and then this cell.")

# %%  8. Package for download / dataset upload
ARCH = WORK / f"archive_{RUN_ID}"
man = nbtools.package(OUT, ARCH, RUN_ID, PART_GB, meta=dict(stage=STAGE, shard=f"{SHARD_INDEX}/{NUM_SHARDS}", verified=STAGE_OK is True and not DRY_RUN,
                                                            verification_problems=STAGE_PROBLEMS[:50], dry_run=DRY_RUN, source_commit=SOURCE.get("git_head"),
                                                            source_dirty=SOURCE.get("dirty"), manifest_sha256=MINFO["manifest_sha256"], smoke_limit=LIMIT))
print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1))

# %%  9. Summary
print(f"""DONE  stage={STAGE} shard={SHARD_INDEX}/{NUM_SHARDS} run={RUN_ID}
verified={man['verified']}  merged={(VERIFY or {}).get('merged')} of {(VERIFY or {}).get('manifest_jobs')} jobs
archive folder: {ARCH}
Next: Kaggle 'Save Version' (or download the folder), then create a PRIVATE dataset from {ARCH.name}.""")
if nbtools.blocking_problems(STAGE_PROBLEMS, DRY_RUN) or (not STAGE_OK and not DRY_RUN):
    raise nbtools.ValidationError("shard NOT VERIFIED (packaged as a PARTIAL archive, verified=false; resume it in a new session):\n  " + "\n  ".join(STAGE_PROBLEMS[:20]))
