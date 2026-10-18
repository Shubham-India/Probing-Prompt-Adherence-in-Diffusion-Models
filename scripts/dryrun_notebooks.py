"""CPU dry-run of the complete notebook chain with the FAKE pipeline and FAKE judge, in a temporary sandbox (default local_audit/dryrun).
Order: 00 PREFLIGHT -> 00 DEV_CORE -> 00 DEV_COMMIT -> 04 DEV -> 00 DEV_CAG_GUARD -> dry-run freeze (marked dry_run, accepted only with
GBC_ALLOW_DRYRUN_PROTOCOL=1) -> 01 / 02 / 03 shard 0 -> 04 TEST. Every job kind runs (2 per kind per stage). Outputs are flagged fake and every
real-run path refuses them. This proves control flow, manifests, resume/verification, archives and analysis plumbing - nothing about SD 1.5.
Usage: python scripts/dryrun_notebooks.py [--sandbox DIR] [--keep]"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "kaggle"), str(ROOT / "src")]
import build_release as BR   # noqa: E402

R2A = ROOT / "local_audit" / "R2a_checkpoint" / "r2a-20261014T130457Z-daef6c4" / "dev_core"


def configured(nb_py, **cfg):
    text = (ROOT / "kaggle" / nb_py).read_text(encoding="utf-8")
    for k, v in cfg.items():
        text, n = re.subn(rf"^{k} = .*$", f"{k} = {v!r}", text, count=1, flags=re.M)
        assert n == 1, f"{k} not found in {nb_py}"
    return text


def run_nb(sandbox, name, nb_py, inputs, **cfg):
    work = sandbox / f"work_{name}"
    work.mkdir(parents=True, exist_ok=True)
    script = sandbox / f"{name}.py"
    script.write_text(configured(nb_py, **cfg), encoding="utf-8")
    env = dict(os.environ, GBC_DRYRUN="1", GBC_INPUT_ROOTS=os.pathsep.join(map(str, inputs)), GBC_WORKING=str(work), GBC_GATES=str(sandbox / "gates.json"),
               GBC_DRYRUN_LIMIT_PER_KIND="2", GBC_ALLOW_DRYRUN_PROTOCOL="1", PYTHONIOENCODING="utf-8", GBC_MODEL_CACHE=str(ROOT / "models"))
    t = time.time()
    r = subprocess.run([sys.executable, str(script)], cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    (sandbox / f"{name}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    status = "OK" if r.returncode == 0 else f"FAILED ({r.returncode})"
    print(f"{name:22s} {status:12s} {time.time() - t:6.1f}s  log: {sandbox / (name + '.log')}", flush=True)
    if r.returncode != 0:
        print(r.stdout[-2500:], r.stderr[-2500:])
        raise SystemExit(f"dry-run step {name} failed")
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", default=str(ROOT / "local_audit" / "dryrun"))
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    sb = Path(a.sandbox).resolve()
    if sb.exists() and not a.keep:
        shutil.rmtree(sb)
    inp = sb / "inputs"
    BR.stage_source(inp / "src" / "gbc_source", include_untracked=True)
    BR.stage_benchmark(inp / "bench" / "gbc_benchmark")
    gates = json.loads((ROOT / "configs" / "stage_gates.json").read_text())
    (sb / "gates.json").write_text(json.dumps({k: (True if not k.startswith("_") else v) for k, v in gates.items()}, indent=1))
    base = [inp / "src", inp / "bench"]
    if not R2A.exists():
        raise SystemExit(f"R2a core run not found at {R2A}; the DEV_COMMIT dry-run needs its latents")
    import make_core_dataset as MC
    MC.main(R2A, inp / "core" / "kaggle_core_dataset", "r2a-20261014T130457Z-daef6c4")
    run_nb(sb, "00_preflight", "nb00_preflight_dev_freeze.py", base, STAGE="PREFLIGHT")
    w_dc = run_nb(sb, "00_dev_core", "nb00_preflight_dev_freeze.py", base, STAGE="DEV_CORE")
    w_dm = run_nb(sb, "00_dev_commit", "nb00_preflight_dev_freeze.py", base + [inp / "core", w_dc], STAGE="DEV_COMMIT")
    w_md = run_nb(sb, "04_dev", "nb04_merge_analyze_export.py", base + [inp / "core", w_dc, w_dm], PHASE="DEV")
    devc = next(w_md.rglob("final_*/dev_constants.json"))
    dc = json.loads(devc.read_text())
    consts = dc["constants"]
    for k in ("presence", "count", "bind_given_pres", "spatial_given_pres"):
        consts["t50_dev_w7.5"].setdefault(k, ">36")                   # dry-run: kinds absent from the tiny fake subset are set unresolved
    (sb / "constants").mkdir(exist_ok=True)
    (sb / "constants" / "dev_constants.json").write_text(json.dumps(dict(dc, constants=consts), indent=1))
    run_nb(sb, "00_dev_cag_guard", "nb00_preflight_dev_freeze.py", base + [sb / "constants"], STAGE="DEV_CAG_GUARD")
    # dry-run freeze inside the sandbox copy of the source package (never in the repository)
    src = inp / "src" / "gbc_source"
    audit = {t: {"criterion_met": True, "accuracy": None, "note": "DRY RUN"} for t in ("presence", "count", "binding", "spatial", "size")}
    (sb / "audit_dryrun.json").write_text(json.dumps(audit))
    env = dict(os.environ, GBC_ALLOW_DRYRUN_PROTOCOL="1")
    proto = src / "protocol" / "protocol_freeze.json"
    subprocess.run([sys.executable, str(src / "src" / "protocol_r3.py"), "--draft", "--dry-run", "--constants", str(sb / "constants" / "dev_constants.json"),
                    "--audit", str(sb / "audit_dryrun.json"), "--path", str(proto)], check=True, env=env, cwd=src)
    d = json.loads(proto.read_text())
    d["constants"] = consts                                            # draft stores the file content; keep only the constants block
    proto.write_text(json.dumps(d, indent=1, sort_keys=True))
    subprocess.run([sys.executable, str(src / "src" / "protocol_r3.py"), "--draft", "--dry-run", "--constants", str(_write(sb / "c_only.json", consts)),
                    "--audit", str(sb / "audit_dryrun.json"), "--path", str(proto)], check=True, env=env, cwd=src)
    s = subprocess.run([sys.executable, str(src / "src" / "protocol_r3.py"), "--sign", "DRY-RUN (not a research freeze)", "--path", str(proto)], env=env, cwd=src,
                       capture_output=True, text=True)
    print("dry-run freeze:", s.stdout.strip(), s.stderr.strip()[-500:])
    assert s.returncode == 0
    BR.write_source_manifest(src)
    w1 = run_nb(sb, "01_test_core_s0", "nb01_test_core_cfg_worker.py", base, SHARD_INDEX=0)
    w2 = run_nb(sb, "02_test_commit_s0", "nb02_test_commitment_worker.py", base + [w1], SHARD_INDEX=0)
    w3 = run_nb(sb, "03_test_cagsize_s0", "nb03_test_cag_size_worker.py", base, SHARD_INDEX=0)
    w4 = run_nb(sb, "04_test", "nb04_merge_analyze_export.py", base + [w1, w2, w3], PHASE="TEST")
    conf = next(w4.rglob("final_*/confirmatory.json"))
    res = json.loads(conf.read_text())
    print("dry-run analysis status:", res.get("status"))
    assert str(res.get("status", "")).startswith("DRY-RUN"), "dry-run output must be labelled DRY-RUN"
    print(json.dumps({k: v.get("final") for k, v in res["results"].items()}, indent=1))
    # a real-run check must refuse the dry-run protocol
    env2 = {k: v for k, v in os.environ.items() if k != "GBC_ALLOW_DRYRUN_PROTOCOL"}
    v = subprocess.run([sys.executable, str(src / "src" / "protocol_r3.py"), "--verify", "--path", str(proto)], env=env2, capture_output=True, text=True)
    assert v.returncode != 0 and "dry-run protocol" in v.stdout, "dry-run protocol was accepted without the dry-run flag"
    print("real-run verification correctly refuses the dry-run protocol")
    print("DRY RUN COMPLETE")


def _write(p, obj):
    Path(p).write_text(json.dumps(obj))
    return p


if __name__ == "__main__":
    main()
