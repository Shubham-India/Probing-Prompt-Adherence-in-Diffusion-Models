# %% [markdown]
# # 04 — Merge, judge, analyse and export (Guide Before Commitment)
# **PHASE = "DEV"**: merges the verified DEV archives (notebook 00) with the R2a core dataset, judges every image (J1 + DINOv2), and computes the DEV
# constants and gates (`dev_constants.json`) that the owner reviews before signing the freeze. No TEST data is touched.
# **PHASE = "TEST"**: requires the signed protocol in the source package; merges the 9 verified TEST archives (notebooks 01-03, shards 0-2) against the
# full frozen manifests (rejects missing, duplicate, corrupt or mismatching outputs), judges every image once, runs the frozen confirmatory analysis
# (HA, HD, HE, HF + Holm) and exports the final package.
# **Inputs:** `source_package`, `benchmark_upload`, every worker output dataset of the phase (+ for DEV: the R2a core dataset).
# **Settings:** GPU (judge + DINOv2), Internet ON (first use downloads OWL-ViT, CLIP and DINOv2 at pinned revisions).
# **Output:** `/kaggle/working/final_<RUN_ID>/` and its ZIP parts. Download it; bring `confirmatory.json` / `dev_constants.json` back to the local repo.

# %%  CONFIG — edit only this cell
PHASE = "DEV"          # "DEV" or "TEST"
RUN_TAG = "merge"
PART_GB = 4.0
import os
DRY_RUN = os.environ.get("GBC_DRYRUN") == "1"

# %%  1. Locate inputs (same verification as the worker notebooks)
import json, shutil, subprocess, sys, time
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
if not src_hits:
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
assert not nbtools.verify_manifest(SRC, "SOURCE_MANIFEST.json"), "source package corrupt"
SOURCE = json.loads((SRC / "SOURCE_MANIFEST.json").read_text())
bench = nbtools.find_marker(INPUT_ROOTS, "BENCHMARK_MANIFEST.json")
assert bench and not nbtools.verify_manifest(bench[0], "BENCHMARK_MANIFEST.json"), "benchmark dataset missing or corrupt"
for rel, h in json.loads((bench[0] / "BENCHMARK_MANIFEST.json").read_text())["files"].items():
    mirror = SRC / "data" / "benchmark_v2" / Path(rel).name if rel.startswith("benchmark_v2/") else None
    if mirror is not None and mirror.exists():
        assert nbtools.sha(mirror, True) == h, f"benchmark file {rel} differs from the copy inside the source package"
if PHASE not in ("DEV", "TEST"):
    raise SystemExit(f"PHASE must be 'DEV' or 'TEST', not {PHASE!r}")
if SOURCE.get("dirty") is not False and not DRY_RUN:
    raise SystemExit("source package is dirty (built from uncommitted work): commit, rebuild with scripts/build_release.py and upload a new gbc-source version")
PROTOCOL = SRC / "protocol" / "protocol_freeze.json"
RUN_ID = f"{RUN_TAG}-{PHASE.lower()}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{str(SOURCE.get('git_head'))[:7]}"
OUT = WORK / f"final_{RUN_ID}"
OUT.mkdir(parents=True, exist_ok=True)
print("RUN_ID =", RUN_ID)

# %%  2. Dependency bootstrap (pinned libraries; keeps Kaggle's torch/CUDA build; removes the unused torchao that breaks diffusers 0.41 imports;
#     fresh-process import check; pip check) and the environment record. Raises on any failure.
import env_bootstrap
BOOT_DIR = WORK / "_bootstrap"
BOOT = env_bootstrap.bootstrap(SRC / "requirements-kaggle.txt", BOOT_DIR, dry_run=DRY_RUN)
(OUT / "env_bootstrap.json").write_text((BOOT_DIR / "env_bootstrap.json").read_text())
ENV = nbtools.env_report()
(OUT / "env.json").write_text(json.dumps(ENV, indent=1))
print(json.dumps(ENV, indent=1))

# %%  3. TEST phase only: the frozen protocol must verify
if PHASE == "TEST":
    v = subprocess.run([sys.executable, str(SRC / "src" / "protocol_r3.py"), "--verify", "--path", str(PROTOCOL)], capture_output=True, text=True)
    print(v.stdout, v.stderr[-2000:])
    fprobs = nbtools.validate_freeze_check(v.returncode, v.stdout)
    if fprobs:
        raise SystemExit("protocol not frozen/verified: the TEST merge and the confirmatory analysis are refused:\n  " + "\n  ".join(fprobs))
STAGES = ["DEV_CORE", "DEV_COMMIT", "DEV_CAG_GUARD"] if PHASE == "DEV" else ["TEST_CORE", "TEST_COMMIT", "TEST_CAG_SIZE"]
# Stages that MUST be present (all shards, checked by the full-manifest merge). DEV_CAG_GUARD needs dev_constants.json from a first DEV pass,
# so the first DEV pass runs without it; its absence is recorded, never silently treated as complete.
REQUIRED_STAGES = ["DEV_CORE", "DEV_COMMIT"] if PHASE == "DEV" else list(STAGES)
CONSTANTS = PROTOCOL if PHASE == "TEST" else next((c / "dev_constants.json" for c in nbtools.find_marker(INPUT_ROOTS, "dev_constants.json")), None)

# %%  4. Extract and verify every attached worker archive of this phase
EXTRACTED = {s: [] for s in STAGES}
for arch in nbtools.find_marker(INPUT_ROOTS, "ARCHIVE_MANIFEST.json"):
    am = json.loads((arch / "ARCHIVE_MANIFEST.json").read_text())
    if am.get("stage") not in STAGES:
        continue
    if not nbtools.archive_is_verified(am) and not DRY_RUN:
        raise SystemExit(f"{arch}: archive was not verified by its worker (verified={am.get('verified')!r}, smoke_limit={am.get('smoke_limit')!r}); finish that shard first")
    dest, probs = nbtools.extract_and_verify(arch, WORK / "merge_in" / am["run_id"])
    assert not probs, f"{arch}: {probs[:5]}"
    EXTRACTED[am["stage"]].append(dest)
print({s: len(v) for s, v in EXTRACTED.items()})
missing_stages = [s for s in REQUIRED_STAGES if not EXTRACTED[s]]
if missing_stages:
    raise SystemExit(f"required stage archives not attached: {missing_stages}; nothing is merged or analysed")

# %%  5. Merge each stage against its FULL manifest (all shards)
MERGED = {}
for st in STAGES:
    if not EXTRACTED[st]:
        print(f"{st}: no archives attached -> skipped (PENDING)")
        continue
    mf = OUT / f"manifest_{st}.jsonl"
    cmd = [sys.executable, str(SRC / "src" / "manifests.py"), "--stage", st, "--out", str(mf)]
    if CONSTANTS:
        cmd += ["--constants", str(CONSTANTS)]
    mk = subprocess.run(cmd, capture_output=True, text=True)
    if mk.returncode != 0:
        raise SystemExit(f"{st}: manifest build failed (exit {mk.returncode}): {mk.stderr[-2000:]}")
    if DRY_RUN:                                   # dry-run workers ran a LIMITED subset: merge against what they were given
        mf = EXTRACTED[st][0] / "manifest.jsonl"
    mcmd = [sys.executable, str(SRC / "src" / "r3_merge.py"), "--manifest", str(mf), "--inputs", *map(str, EXTRACTED[st]), "--out", str(OUT / "merged" / st)]
    if DRY_RUN:
        mcmd += ["--allow-fake", "--allow-missing"]
    shutil.rmtree(OUT / "merged" / st, ignore_errors=True)
    r = subprocess.run(mcmd, capture_output=True, text=True)
    print(st, r.stdout[-1500:], r.stderr[-1500:])
    rep, rerr = nbtools.load_json_object(OUT / "merged" / st / "merge_report.json")
    ok, why = nbtools.stage_verdict({"merge": 0}, r.returncode, rep, dry_run=DRY_RUN)
    if not ok and not DRY_RUN:
        raise SystemExit(f"{st}: merge rejected; nothing is analysed:\n  " + "\n  ".join(why[:20]))
    if DRY_RUN and (r.returncode != 0 or not isinstance(rep, dict) or rep.get("ok") is not True):
        raise SystemExit(f"{st}: dry-run merge failed: {rerr or r.stderr[-1500:]}")
    MERGED[st] = OUT / "merged" / st / "merged_records.jsonl"

# %%  6. DEV phase: adopt the R2a canonical core (verified) as core records
RECORDS = [MERGED[s] for s in STAGES if s in MERGED]
if PHASE == "DEV":
    core = nbtools.find_marker(INPUT_ROOTS, "core_manifest.json")
    if not core and not DRY_RUN:
        raise SystemExit("PHASE=DEV needs the R2a core dataset (kaggle_core_dataset with core_manifest.json): the DEV canonical bases come from it")
    if core:
        cm = json.loads((core[0] / "core_manifest.json").read_text())
        rows = []
        for line in (core[0] / "dev_core" / "samples.jsonl").read_text().splitlines():
            s = json.loads(line)
            img = core[0] / "dev_core" / "images" / f"{s['name']}.png"
            assert nbtools.sha(img) == s["image_sha256"], f"R2a image hash mismatch {s['name']}"
            rows.append(dict(job_id=s["name"], kind="core", experiment="E04-DEV(R2a)", stage="R2a", split=s["split"], scene_id=s["scene_id"], shard_group=s["scene_id"],
                             prompt_id=s["prompt_id"], prompt=s["prompt"], variant=s["variant"], category=s["category"], cfg=s["cfg"], steps=s["steps"], seed=s["seed"],
                             abs_files={f"images/{s['name']}.png": str(img)}, fake_pipeline=False, run_id=cm["run_id"]))
        p = OUT / "merged" / "R2a_records.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        RECORDS.insert(0, p)
        print("R2a core records:", len(rows))
ALL = OUT / "merged" / "all_records.jsonl"
ALL.write_text("".join(Path(p).read_text() for p in RECORDS))

# %%  7. Judge every image once (J1 + conditional outcomes; DINOv2 embeddings for diversity)
EVAL, EMB = OUT / "eval_rows.jsonl", OUT / "dino_embeddings.npz"
cmd = [sys.executable, str(SRC / "src" / "r3_evaluate.py"), "--records", str(ALL), "--out", str(EVAL), "--diversity", str(EMB)]
if DRY_RUN:
    cmd.append("--fake-judge")
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
if r.returncode != 0:
    raise SystemExit(f"judging failed (exit {r.returncode}); nothing is analysed")

# %%  8. Analysis (DEV constants and gates, or the frozen confirmatory analysis)
extra = ["--allow-fake"] if DRY_RUN else []
if PHASE == "DEV":
    a = subprocess.run([sys.executable, str(SRC / "src" / "r3_analyze.py"), "dev", "--rows", str(EVAL), "--out", str(OUT / "dev_constants.json"), "--n-boot", "2000"] + extra,
                       capture_output=True, text=True)
else:
    att = OUT / "attention_all"
    att.mkdir(exist_ok=True)
    for line in ALL.read_text().splitlines():
        rec = json.loads(line)
        for rel, path in rec.get("abs_files", {}).items():
            if rel.startswith("attention/"):
                shutil.copy2(path, att / Path(rel).name)
    a = subprocess.run([sys.executable, str(SRC / "src" / "r3_analyze.py"), "confirm", "--rows", str(EVAL), "--attention-dir", str(att), "--embeddings", str(EMB),
                        "--out", str(OUT / "confirmatory.json"), "--protocol", str(PROTOCOL)] + extra, capture_output=True, text=True)
print(a.stdout[-3000:], a.stderr[-3000:])
(OUT / "analysis_process.log").write_text(f"[exit {a.returncode}]\n--- stdout\n{a.stdout}\n--- stderr\n{a.stderr}\n")
if a.returncode != 0:
    raise SystemExit(f"analysis failed or was refused (exit {a.returncode})")

# %%  9. Tables, figures, results catalogue and the export archive
if PHASE == "TEST":
    subprocess.run([sys.executable, str(SRC / "src" / "r3_report.py"), "--results", str(OUT / "confirmatory.json"), "--out", str(OUT / "report")], check=True)
CATALOGUE = dict(run_id=RUN_ID, phase=PHASE, source_commit=SOURCE.get("git_head"), protocol=str(PROTOCOL) if PHASE == "TEST" else None,
                 stages_merged=list(MERGED), files={p.relative_to(OUT).as_posix(): nbtools.sha(p) for p in OUT.rglob("*") if p.is_file() and "merge_in" not in p.parts},
                 reproduce="python src/r3_analyze.py {dev|confirm} --rows eval_rows.jsonl ... (see KAGGLE_RUNBOOK.md)")
(OUT / "RESULTS_CATALOGUE.json").write_text(json.dumps(CATALOGUE, indent=1))
if (SRC / "protocol").exists():
    shutil.copytree(SRC / "protocol", OUT / "protocol", dirs_exist_ok=True)
ARCH = WORK / f"archive_{RUN_ID}"
# Every check above raises on failure, so reaching this line with a real (non-dry) run is the evidence for marking it verified.
man = nbtools.package(OUT, ARCH, RUN_ID, PART_GB, meta=dict(stage=f"FINAL_{PHASE}", verified=not DRY_RUN, dry_run=DRY_RUN, stages_merged=list(MERGED),
                                                            stages_absent=[s for s in STAGES if s not in MERGED], source_commit=SOURCE.get("git_head"),
                                                            source_dirty=SOURCE.get("dirty")))
print(f"DONE phase={PHASE} run={RUN_ID}\nresults: {OUT}\narchive: {ARCH} ({len(man['files'])} files)")
