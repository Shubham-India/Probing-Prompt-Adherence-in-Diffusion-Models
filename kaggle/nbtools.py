"""Helpers used by the Guide-Before-Commitment notebooks (kaggle/nb*.py -> release/.../kaggle_notebooks/*.ipynb).
Input discovery under /kaggle/input (folders or attached ZIPs), manifest verification, environment report, one worker process per visible GPU,
and multipart output archives with an internal manifest. Pure standard library + optional torch; no credentials are read or written."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

TEXT = (".py", ".json", ".md", ".txt", ".jsonl", ".csv")


def sha(path, text_normalise=False):
    p = Path(path)
    if text_normalise and p.suffix in TEXT:
        return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def find_marker(roots, marker, max_depth=6):
    """All folders under `roots` (depth-limited) that contain a file named `marker`. ZIPs found on the way are extracted to
    /kaggle/working/_inputs (or <cwd>/_inputs) once and searched too."""
    found, zips = [], []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            if depth > max_depth:
                dirnames[:] = []
                continue
            if marker in filenames:
                found.append(Path(dirpath))
            zips += [Path(dirpath) / f for f in filenames if f.endswith(".zip")]
    if not found and zips:
        dest = Path("/kaggle/working/_inputs") if Path("/kaggle/working").exists() else Path.cwd() / "_inputs"
        for z in zips:
            target = dest / z.stem
            if not target.exists():
                with zipfile.ZipFile(z) as zz:
                    if any(n.endswith(marker) for n in zz.namelist()):
                        zz.extractall(target)
        found = find_marker([dest], marker, max_depth) if dest.exists() else []
    return sorted(set(found))


def verify_manifest(folder, manifest_name):
    """Manifest format: {"files": {relative path: sha256 (text files LF-normalised)}}. Returns list of problems."""
    folder = Path(folder)
    m = json.loads((folder / manifest_name).read_text())
    probs = []
    for rel, h in m["files"].items():
        p = folder / rel
        if not p.exists():
            probs.append(f"missing {rel}")
        elif sha(p, text_normalise=True) != h:
            probs.append(f"hash mismatch {rel}")
    return probs


def env_report():
    rep = dict(python=sys.version.split()[0], platform=sys.platform, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        rep["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        rep["nvidia_smi"] = f"unavailable: {type(e).__name__}"
    try:
        import torch
        rep.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                   gpus=[dict(name=torch.cuda.get_device_name(i), total_gb=round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)) for i in range(torch.cuda.device_count())])
    except Exception as e:
        rep["torch"] = f"unavailable: {type(e).__name__}"
    for lib in ("diffusers", "transformers", "accelerate", "numpy", "scipy", "pandas", "sklearn", "PIL"):
        try:
            rep[lib] = __import__(lib).__version__
        except Exception:
            rep[lib] = None
    u = shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").exists() else ".")
    rep["disk_free_gb"] = round(u.free / 1e9, 1)
    return rep


class ValidationError(RuntimeError):
    """Raised by the notebooks when validation evidence is missing, malformed or negative. Never caught to continue a stage."""


def run_json(cmd, name, out_dir=None, allowed_codes=(0,)):
    """Run a subprocess whose whole stdout (or its last line) is a JSON object. Returns (returncode, obj or None, CompletedProcess).
    stdout/stderr are saved to <out_dir>/<name>.log so a failure keeps its diagnostics."""
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"{name}.log").write_text(f"$ {' '.join(map(str, cmd))}\n[exit {p.returncode}]\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}\n")
    obj = None
    for candidate in (p.stdout, (p.stdout.strip().splitlines() or [""])[-1]):
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                obj = v
                break
        except (ValueError, TypeError):
            pass
    return p.returncode, obj, p


def load_json_object(path):
    """(obj, problem). A missing, unreadable or non-object file is a problem, never an empty success."""
    p = Path(path)
    if not p.exists():
        return None, f"{p.name} missing"
    try:
        v = json.loads(p.read_text())
    except (ValueError, OSError) as e:
        return None, f"{p.name} unreadable: {type(e).__name__}: {e}"
    if not isinstance(v, dict):
        return None, f"{p.name} is not a JSON object"
    return v, None


PREFLIGHT_REQUIRED_TRUE = ("all_identities_hold", "attention_hook_image_identical", "cag_constant_equals_pipeline", "cag_w1_equals_pipeline_w1")


def validate_preflight(path, returncode, fake=False):
    """Problems with a preflight run (empty list = accepted). Every condition must hold; a default never stands in for evidence."""
    probs = [] if returncode == 0 else [f"r3_preflight.py exited with code {returncode}"]
    pre, err = load_json_object(path)
    if err:
        return probs + [err]
    checks, timing = pre.get("checks"), pre.get("timing")
    if not isinstance(checks, dict) or not isinstance(timing, dict) or not isinstance(pre.get("env"), dict):
        return probs + ["preflight.json lacks the env / checks / timing objects"]
    if pre.get("fake_pipeline") is not bool(fake):
        probs.append(f"preflight.json fake_pipeline={pre.get('fake_pipeline')!r}, expected {bool(fake)}")
    for k in PREFLIGHT_REQUIRED_TRUE:
        if checks.get(k) is not True:
            probs.append(f"checks.{k} is {checks.get(k)!r}, required true")
    if checks.get("cag_w1_unet_evaluations") != 40:
        probs.append(f"checks.cag_w1_unet_evaluations is {checks.get('cag_w1_unet_evaluations')!r}, required 40")
    for k in ("model_load_s", "plain_s_median", "with_attention_and_latents_s_median", "branch_t12_D8_s"):
        if not isinstance(timing.get(k), (int, float)):
            probs.append(f"timing.{k} missing")
    if not fake:
        if not pre["env"].get("cuda_available"):
            probs.append("CUDA not available in the preflight process")
        if "evaluator_error" in checks:
            probs.append(f"judge / DINOv2 check failed: {checks['evaluator_error']}")
        for k in ("judge_s_per_image", "dino_s_per_image"):
            if not isinstance(timing.get(k), (int, float)):
                probs.append(f"timing.{k} missing (judge / DINOv2 not measured)")
        if not isinstance(pre.get("peak_vram_gb"), (int, float)):
            probs.append("peak_vram_gb missing")
        n_gpu = len(pre["env"].get("gpus") or [])
        if n_gpu >= 2:
            codes = checks.get("two_worker_exit_codes")
            if not isinstance(codes, dict) or len(codes) != n_gpu or any(c != 0 for c in codes.values()):
                probs.append(f"two-worker probe exit codes {codes!r}: every GPU worker must exit 0")
    if checks.get("preflight_ok") is not True:
        probs.append(f"checks.preflight_ok is {checks.get('preflight_ok')!r}, required true")
    return probs


def validate_gpu_identity(path, returncode):
    probs = [] if returncode == 0 else [f"gpu_identity.py exited with code {returncode}"]
    gi, err = load_json_object(path)
    if err:
        return probs + [err]
    if gi.get("all_required_identities_hold") is not True:
        probs.append(f"all_required_identities_hold is {gi.get('all_required_identities_hold')!r}, required true")
    req, res = gi.get("required"), gi.get("results")
    if not isinstance(req, list) or not req or not isinstance(res, list) or not res:
        return probs + ["gpu_identity.json has no required checks or no results"]
    for r in res:
        bad = [k for k in req if r.get(k) != 0]
        if bad or r.get("saved_latent_recompute_equal") is not True:
            probs.append(f"{r.get('name')}: identity failed for {bad or ['saved_latent_recompute_equal']}")
    return probs


def validate_freeze_check(returncode, stdout):
    """protocol_r3.py --verify prints exactly 'VERIFIED' on success and exits 0; anything else is a failure."""
    first = (stdout or "").strip().splitlines()[:1]
    probs = [] if returncode == 0 else [f"protocol_r3.py --verify exited with code {returncode}"]
    if first != ["VERIFIED"]:
        probs.append("protocol verification did not report VERIFIED: " + " | ".join((stdout or "(no output)").strip().splitlines()[:6]))
    return probs


def stage_verdict(worker_codes, merge_returncode, merge_report, limit=None, dry_run=False):
    """(verified, reasons). verified is True only when every worker exited 0, the merge exited 0 and its report says the whole manifest was merged
    with nothing missing, unexpected, conflicting, corrupt, mismatched, fake or of the wrong revision, and the run was neither a smoke LIMIT run nor
    a dry run. Partial outputs are packaged with verified=false so a later session can resume them."""
    reasons = []
    if not isinstance(worker_codes, dict) or not worker_codes:
        reasons.append("no worker exit codes recorded")
    else:
        reasons += [f"worker {w} exited with code {c}" for w, c in sorted(worker_codes.items()) if c != 0]
    if merge_returncode != 0:
        reasons.append(f"r3_merge.py exited with code {merge_returncode}")
    if not isinstance(merge_report, dict):
        reasons.append("merge_report.json missing or unreadable")
    else:
        if merge_report.get("ok") is not True:
            reasons.append("merge report not ok")
        for k in ("missing", "unexpected", "conflicts", "corrupt", "mismatch", "fake", "wrong_revision"):
            v = merge_report.get(k)
            if not isinstance(v, list):
                reasons.append(f"merge report lacks the '{k}' list")
            elif v:
                reasons.append(f"merge report: {len(v)} {k}")
        mj, mg = merge_report.get("manifest_jobs"), merge_report.get("merged")
        if not isinstance(mj, int) or mj <= 0 or mg != mj:
            reasons.append(f"merged {mg} of {mj} manifest jobs")
        if merge_report.get("allow_missing_dry_run"):
            reasons.append("merge ran with --allow-missing (dry run)")
    if limit is not None:
        reasons.append(f"smoke run (LIMIT={limit}) is never a verified research archive")
    if dry_run:
        reasons.append("(dry run) fake pipeline output is never verified")
    return (not reasons), reasons


DRY_RUN_EXPECTED = (re.compile(r"^merge report: \d+ missing$"), re.compile(r"^merged \d+ of \d+ manifest jobs$"), re.compile(r"\(dry run"))


def blocking_problems(problems, dry_run=False):
    """Problems that must stop the notebook. In a CPU dry run (limited subset, fake pipeline) the expected 'missing' / 'dry run' reasons do not
    stop it, but they still keep verified=false; on a real run every problem blocks."""
    if not dry_run:
        return list(problems)
    return [p for p in problems if not any(rx.search(p) for rx in DRY_RUN_EXPECTED)]


def archive_is_verified(am):
    """An attached archive counts as verified only with explicit evidence: verified is True, it is not a smoke/dry run and it lists its files."""
    return (isinstance(am, dict) and am.get("verified") is True and am.get("smoke_limit") is None and not am.get("dry_run")
            and isinstance(am.get("files"), dict) and bool(am["files"]))


def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def run_workers(src_root, manifest, out, run_id, worker_prefix, upstream=(), protocol=None, fake=False, limit=None, extra=()):
    """One r3_run.py process per visible GPU (scene-group sub-shards), started together and waited for. Logs: <out>/logs/<worker>.log."""
    n = max(1, gpu_count()) if not fake else 1
    out = Path(out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for g in range(n):
        wid = f"{worker_prefix}g{g}"
        cmd = [sys.executable, str(Path(src_root) / "src" / "r3_run.py"), "--manifest", str(manifest), "--out", str(out), "--run-id", run_id, "--worker-id", wid]
        if n > 1:
            cmd += ["--gpu-subshard", f"{g}/{n}"]
        if upstream:
            cmd += ["--upstream", *map(str, upstream)]
        if protocol:
            cmd += ["--protocol", str(protocol)]
        if fake:
            cmd += ["--fake-pipeline"]
        if limit:
            cmd += ["--limit", str(limit)]
        if os.environ.get("GBC_DRYRUN_LIMIT_PER_KIND"):
            cmd += ["--limit-per-kind", os.environ["GBC_DRYRUN_LIMIT_PER_KIND"]]
        cmd += list(extra)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)) if not fake else dict(os.environ)
        log = open(out / "logs" / f"{wid}.log", "a")
        procs.append((wid, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    codes = {}
    while procs:
        time.sleep(30 if not fake else 1)
        for item in list(procs):
            wid, p, log = item
            if p.poll() is not None:
                codes[wid] = p.returncode
                log.close()
                procs.remove(item)
        for wid, p, log in procs:
            tail = (out / "logs" / f"{wid}.log").read_text(errors="replace").strip().splitlines()[-1:]   # progress line
            print(f"{wid}: {tail[0] if tail else '(starting)'}", flush=True)
    return codes


def package(out_dir, archive_dir, run_id, part_gb=4.0, meta=None):
    """Multipart ZIPs of a worker output folder, grouped by top-level folder (images, latents, attention, logs; root files = 'meta'), plus
    ARCHIVE_MANIFEST.json (sha256, group and part of every file; stage/shard/run metadata). Files are never split across parts, so each part extracts
    on its own; a downstream notebook may attach only the groups it needs (e.g. 'latents' + 'meta'). Reassembly: extract_and_verify()."""
    out_dir, archive_dir = Path(out_dir), Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for p in sorted(x for x in out_dir.rglob("*") if x.is_file() and not x.name.endswith(".tmp")):
        rel = p.relative_to(out_dir)
        g = rel.parts[0] if len(rel.parts) > 1 else "meta"
        groups.setdefault(g, []).append(p)
    meta = dict(meta or {})
    if meta.get("verified") is not True:
        meta["verified"] = False               # only an explicit True (from stage_verdict / the notebook's checks) marks an archive verified
    manifest = dict(run_id=run_id, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta, groups={}, files={})
    limit = part_gb * 1e9
    for g, files in groups.items():
        parts, cur, size = [], [], 0
        for p in files:
            s = p.stat().st_size
            if cur and size + s > limit:
                parts.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += s
        if cur:
            parts.append(cur)
        names = []
        for i, plist in enumerate(parts, 1):
            name = f"{run_id}.{g}.part{i:02d}of{len(parts):02d}.zip"
            names.append(name)
            with zipfile.ZipFile(archive_dir / name, "w", zipfile.ZIP_STORED) as z:
                for p in plist:
                    rel = p.relative_to(out_dir).as_posix()
                    z.write(p, rel)
                    manifest["files"][rel] = dict(sha256=sha(p), part=name, group=g, bytes=p.stat().st_size)
        manifest["groups"][g] = names
    probs = verify_parts(archive_dir, manifest)
    if probs:
        manifest["verified"] = False
        manifest["packaging_problems"] = probs[:50]
    (archive_dir / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    (archive_dir / "checksums.sha256").write_text("".join(f"{sha(archive_dir / f)}  {f}\n" for f in sorted(os.listdir(archive_dir)) if f.endswith(".zip")), newline="\n")
    return manifest


def verify_parts(archive_dir, manifest):
    """Every file of the manifest is inside its part with the recorded size; every part opens as a ZIP. Cheap (central directory only)."""
    archive_dir, probs, listing = Path(archive_dir), [], {}
    for part in sorted({v["part"] for v in manifest["files"].values()}):
        try:
            with zipfile.ZipFile(archive_dir / part) as z:
                listing[part] = {i.filename: i.file_size for i in z.infolist()}
        except (OSError, zipfile.BadZipFile) as e:
            probs.append(f"{part}: {type(e).__name__}")
    for rel, v in manifest["files"].items():
        if v["part"] in listing and listing[v["part"]].get(rel) != v["bytes"]:
            probs.append(f"{rel}: not in {v['part']} with {v['bytes']} bytes")
    return probs


def extract_and_verify(archive_dir, dest, groups=None):
    """Extract the parts listed in ARCHIVE_MANIFEST.json (all groups, or only `groups`) into dest and verify every extracted file's hash.
    Returns (dest, problems). A missing part is a problem, never silently skipped."""
    archive_dir, dest = Path(archive_dir), Path(dest)
    m = json.loads((archive_dir / "ARCHIVE_MANIFEST.json").read_text())
    want = {rel: v for rel, v in m["files"].items() if groups is None or v.get("group") in groups}
    if not want:
        return dest, [f"archive lists no files{' in groups ' + str(tuple(groups)) if groups else ''}"]
    sums = {}
    if (archive_dir / "checksums.sha256").exists():
        for line in (archive_dir / "checksums.sha256").read_text().splitlines():
            if "  " in line:
                h, name = line.split("  ", 1)
                sums[name.strip()] = h
    for part in sorted({v["part"] for v in want.values()}):
        p = archive_dir / part
        if not p.exists():
            return dest, [f"archive part missing: {part}"]
        if part in sums and sha(p) != sums[part]:
            return dest, [f"archive part checksum mismatch: {part}"]
        with zipfile.ZipFile(p) as z:
            z.extractall(dest)
    probs = [f"{rel}: {'missing' if not (dest / rel).exists() else 'hash mismatch'}" for rel, v in want.items()
             if not (dest / rel).exists() or sha(dest / rel) != v["sha256"]]
    return dest, probs
