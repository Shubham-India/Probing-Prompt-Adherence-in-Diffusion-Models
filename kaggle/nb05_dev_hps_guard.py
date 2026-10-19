# %% [markdown]
# # 05 — DEV CAG quality guard with HPS v2 (diagnostic first; no image generation)
# **Purpose.** Scores EXISTING DEV images with HPS v2: the DEV_CAG_GUARD images (A2 at w_lo = 1 and 3, seeds 0-1) and their A1 partners (constant
# CFG 7.5, same prompt and seed: the R2a DEV core). Step 3 is a small diagnostic (install, pinned checkpoint, 4 images); only if it passes, step 4
# scores all 600 images and applies the predeclared guard (mean drop <= 0.5 HPS points; `src/hps_guard.py`). No TEST data.
# **Inputs:** `gbc-source` (version with this notebook), `gbc-benchmark`, `gbc-r2a-core`, `gbc-dev-cag-v2`. **Settings:** GPU T4 (one is enough), Internet ON.
# **Output:** `/kaggle/working/archive_<RUN_ID>/` with `hps_diagnostic.json`, `hps_scores.jsonl`, `hps_guard.json` -> private dataset `gbc-dev-hps`.
# %%  CONFIG — edit only this cell
RUN_TAG = "r3-dev-hps-guard"
PART_GB = 4.0
DIAGNOSTIC_ONLY = False      # True: stop after the 4-image diagnostic
import os
DRY_RUN = os.environ.get("GBC_DRYRUN") == "1"

# %%  1. Locate inputs (same verification as the other notebooks)
import json, shutil, subprocess, sys, time, hashlib
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
if SOURCE.get("dirty") is not False and not DRY_RUN:
    raise SystemExit("source package is dirty: commit, rebuild with scripts/build_release.py and upload a new gbc-source version")
CFG = json.loads((SRC / "configs" / "experiment_r3.json").read_text())
HPS = CFG["hps_guard"]
RUN_ID = f"{RUN_TAG}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{str(SOURCE.get('git_head'))[:7]}"
OUT = WORK / f"out_{RUN_ID}"
OUT.mkdir(parents=True, exist_ok=True)
DIAG = dict(run_id=RUN_ID, source_commit=SOURCE.get("git_head"), steps={}, ok=False)

def save_diag():
    (OUT / "hps_diagnostic.json").write_text(json.dumps(DIAG, indent=1, default=str))

# CAG guard archive (verified) and the R2a core (A1 images)
cag_arch = [a for a in nbtools.find_marker(INPUT_ROOTS, "ARCHIVE_MANIFEST.json")
            if json.loads((a / "ARCHIVE_MANIFEST.json").read_text()).get("stage") == "DEV_CAG_GUARD"]
assert len(cag_arch) == 1, f"attach exactly one DEV_CAG_GUARD archive (gbc-dev-cag-v2), found {cag_arch}"
am = json.loads((cag_arch[0] / "ARCHIVE_MANIFEST.json").read_text())
assert nbtools.archive_is_verified(am) or DRY_RUN, "DEV_CAG_GUARD archive is not verified"
assert am.get("manifest_sha256") == HPS["dev_cag_guard_manifest_sha256"] or DRY_RUN, \
    f"DEV_CAG_GUARD archive manifest {am.get('manifest_sha256')} is not the corrected-constants run {HPS['dev_cag_guard_manifest_sha256']}"
CAG_DIR, probs = nbtools.extract_and_verify(cag_arch[0], WORK / "cag", groups=("images", "meta"))
assert not probs, probs[:5]
core = nbtools.find_marker(INPUT_ROOTS, "core_manifest.json")
assert core, "attach gbc-r2a-core (A1 images)"
A1_DIR = core[0] / "dev_core" / "images"
CAG = [json.loads(l) for f in sorted(CAG_DIR.glob("records_*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
import hps_guard as HG
for r in CAG:
    assert r["split"] == "DEV", "TEST rows present"
    p = A1_DIR / f"{HG.a1_job(r['prompt_id'], r['seed'])}.png"
    assert p.exists(), f"A1 image missing: {p.name}"
print(f"source {SOURCE.get('git_head')} | DEV_CAG_GUARD records {len(CAG)} | A1 images in {A1_DIR}")

# %%  2. Dependency bootstrap (same as every notebook), then HPS v2 without its dependency pins (they would downgrade Kaggle packages)
import env_bootstrap
BOOT = env_bootstrap.bootstrap(SRC / "requirements-kaggle.txt", WORK / "_bootstrap", dry_run=DRY_RUN)
(OUT / "env_bootstrap.json").write_text((WORK / "_bootstrap" / "env_bootstrap.json").read_text())
torch_before = env_bootstrap.installed_version("torch")
if not DRY_RUN:
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", f"hpsv2=={HPS['hpsv2_version']}"] + HPS["extra_packages"], capture_output=True, text=True)
    DIAG["steps"]["pip_install_hpsv2"] = dict(returncode=pip.returncode, stderr=pip.stderr[-2000:])
    DIAG["steps"]["torch_unchanged"] = env_bootstrap.installed_version("torch") == torch_before
    save_diag()
    if pip.returncode != 0 or not DIAG["steps"]["torch_unchanged"]:
        raise nbtools.ValidationError(f"HPS v2 install failed or changed torch: see {OUT / 'hps_diagnostic.json'}")

# %%  3. DIAGNOSTIC: import, tokenizer vocabulary, pinned checkpoint, 4 images
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

try:
    import inspect, urllib.request
    # hpsv2 1.2.0 calls torch.load(cp) without weights_only; torch >= 2.6 defaults to weights_only=True. The checkpoint is a pinned file of a
    # verified repo revision (sha256 recorded below), so the package's own load behaviour is restored for this process only.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import hpsv2
    pkg = Path(hpsv2.__file__).parent
    vocab = pkg / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
    DIAG["steps"]["hpsv2"] = dict(version=getattr(hpsv2, "__version__", HPS["hpsv2_version"]), vocab_present=vocab.exists(), score_signature=str(inspect.signature(hpsv2.score)))
    if not vocab.exists():                       # known packaging gap of the PyPI wheel: fetch the tokenizer vocabulary from the pinned open_clip commit
        vocab.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HPS["bpe_vocab_url"], vocab)
        DIAG["steps"]["hpsv2"]["vocab_fetched_sha256"] = sha_file(vocab)
        assert DIAG["steps"]["hpsv2"]["vocab_fetched_sha256"] == HPS["bpe_vocab_sha256"], "tokenizer vocabulary hash differs from the pinned one"
    assert "cp" in inspect.signature(hpsv2.score).parameters, "hpsv2.score has no checkpoint argument: cannot pin the checkpoint"
    from huggingface_hub import hf_hub_download
    CKPT = hf_hub_download(HPS["repo"], HPS["checkpoint"], revision=HPS["revision"])
    DIAG["steps"]["checkpoint"] = dict(repo=HPS["repo"], file=HPS["checkpoint"], revision=HPS["revision"], sha256=sha_file(CKPT))
    t = time.time()
    test = CAG[:2]
    s_cag = [float(x) for x in hpsv2.score([str(CAG_DIR / "images" / f"{r['job_id']}.png") for r in test[:1]], test[0]["prompt"], cp=CKPT, hps_version=HPS["hps_version"])]
    s_cag += [float(x) for x in hpsv2.score([str(CAG_DIR / "images" / f"{r['job_id']}.png") for r in test[1:2]], test[1]["prompt"], cp=CKPT, hps_version=HPS["hps_version"])]
    s_a1 = [float(x) for x in hpsv2.score([str(A1_DIR / f"{HG.a1_job(r['prompt_id'], r['seed'])}.png") for r in test[:1]], test[0]["prompt"], cp=CKPT, hps_version=HPS["hps_version"])]
    s_a1 += [float(x) for x in hpsv2.score([str(A1_DIR / f"{HG.a1_job(r['prompt_id'], r['seed'])}.png") for r in test[1:2]], test[1]["prompt"], cp=CKPT, hps_version=HPS["hps_version"])]
    finite = all(0.0 < x < 1.0 for x in s_cag + s_a1)
    DIAG["steps"]["diagnostic_scores"] = dict(cag=s_cag, a1=s_a1, all_in_0_1=finite, seconds=round(time.time() - t, 1))
    DIAG["ok"] = bool(finite)
except Exception as e:
    DIAG["steps"]["error"] = f"{type(e).__name__}: {e}"
    DIAG["ok"] = False
save_diag()
print(json.dumps(DIAG, indent=1, default=str)[:3000])
if not DIAG["ok"]:
    nbtools.package(OUT, WORK / f"archive_{RUN_ID}", RUN_ID, PART_GB, meta=dict(stage="DEV_HPS_GUARD", verified=False, diagnostic_only=True))
    raise nbtools.ValidationError("HPS v2 diagnostic FAILED (packaged with verified=false): " + str(DIAG["steps"].get("error")))
print("HPS v2 DIAGNOSTIC PASSED")

# %%  4. Score every guard image and its A1 partner; apply the predeclared rule
STAGE_OK, PROBLEMS, G = False, [], None
if not DIAGNOSTIC_ONLY:
    items = {}
    for r in CAG:
        items[r["job_id"]] = (CAG_DIR / "images" / f"{r['job_id']}.png", r["prompt"])
        a1 = HG.a1_job(r["prompt_id"], r["seed"])
        items[a1] = (A1_DIR / f"{a1}.png", r["prompt"])
    by_prompt = {}
    for jid, (p, pr) in items.items():
        by_prompt.setdefault(pr, []).append((jid, p))
    scores = {}
    t = time.time()
    for k, (pr, lst) in enumerate(by_prompt.items()):
        vals = hpsv2.score([str(p) for _, p in lst], pr, cp=CKPT, hps_version=HPS["hps_version"])
        for (jid, _), v in zip(lst, vals):
            scores[jid] = float(v)
        if k % 20 == 0:
            print(f"[{k}/{len(by_prompt)}] prompts scored, {time.time() - t:.0f}s", flush=True)
    with open(OUT / "hps_scores.jsonl", "w") as f:
        for jid, v in sorted(scores.items()):
            f.write(json.dumps(dict(job_id=jid, hps=v, sha256=nbtools.sha(items[jid][0]))) + "\n")
    if len(scores) != len(items):
        PROBLEMS.append(f"scored {len(scores)} of {len(items)} images")
    if any(not (0.0 < v < 1.0) for v in scores.values()):
        PROBLEMS.append("HPS score outside (0, 1)")
    if not PROBLEMS:
        G = HG.guard(CAG, scores)
        G.update(hps=dict(version=HPS["hps_version"], package=f"hpsv2=={HPS['hpsv2_version']}", checkpoint=DIAG["steps"]["checkpoint"]),
                 inputs=dict(dev_cag_guard_run=am["run_id"], dev_cag_guard_manifest=am["manifest_sha256"], n_images=len(scores)))
        (OUT / "hps_guard.json").write_text(json.dumps(G, indent=1))
        print(json.dumps(G, indent=1))
    STAGE_OK = not PROBLEMS

# %%  5. Package (verified only when the full scoring ran without problems)
man = nbtools.package(OUT, WORK / f"archive_{RUN_ID}", RUN_ID, PART_GB,
                      meta=dict(stage="DEV_HPS_GUARD", verified=bool(STAGE_OK and not DRY_RUN and not DIAGNOSTIC_ONLY), dry_run=DRY_RUN, diagnostic_only=DIAGNOSTIC_ONLY,
                                verification_problems=PROBLEMS, source_commit=SOURCE.get("git_head"), source_dirty=SOURCE.get("dirty")))
print(f"DONE run={RUN_ID} verified={man['verified']} decision={(G or {}).get('decision')}")
if PROBLEMS:
    raise nbtools.ValidationError("HPS guard NOT VERIFIED: " + "; ".join(PROBLEMS))
