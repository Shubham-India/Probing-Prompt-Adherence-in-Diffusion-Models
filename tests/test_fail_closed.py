"""Fail-closed validation: missing, malformed, negative or contradictory evidence can never mark PREFLIGHT, FREEZE_CHECK, a worker stage, a merge
or an archive as verified. Includes end-to-end runs of notebook 00 (CPU, fake pipeline) with an injected PREFLIGHT failure and a missing freeze."""
import ast
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "kaggle"), str(ROOT / "src"), str(ROOT / "scripts")]
import nbtools as N  # noqa: E402

NOTEBOOKS = ["nb00_preflight_dev_freeze.py", "nb01_test_core_cfg_worker.py", "nb02_test_commitment_worker.py", "nb03_test_cag_size_worker.py",
             "nb04_merge_analyze_export.py"]


# ---------- PREFLIGHT ----------
def good_preflight(fake=False, gpus=2):
    checks = dict(all_identities_hold=True, attention_hook_image_identical=True, cag_constant_equals_pipeline=True, cag_w1_equals_pipeline_w1=True,
                  cag_w1_unet_evaluations=40, preflight_ok=True, preflight_failures=[])
    timing = dict(model_load_s=100.0, plain_s_median=7.0, with_attention_and_latents_s_median=7.1, branch_t12_D8_s=6.0)
    rep = dict(env=dict(cuda_available=not fake, gpus=[dict(name="Tesla T4")] * (0 if fake else gpus)), fake_pipeline=fake, checks=checks, timing=timing)
    if not fake:
        timing.update(judge_s_per_image=0.5, dino_s_per_image=0.1)
        rep["peak_vram_gb"] = 3.1
        if gpus >= 2:
            checks["two_worker_exit_codes"] = {f"pg{i}": 0 for i in range(gpus)}
    return rep


def write(tmp_path, obj, name="preflight.json"):
    p = tmp_path / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return p


@pytest.mark.parametrize("fake", [False, True])
def test_good_preflight_is_accepted(tmp_path, fake):
    assert N.validate_preflight(write(tmp_path, good_preflight(fake)), 0, fake=fake) == []


def mutate(path_keys, value):
    def f(rep):
        d = rep
        for k in path_keys[:-1]:
            d = d[k]
        if value is KeyError:
            d.pop(path_keys[-1])
        else:
            d[path_keys[-1]] = value
        return rep
    return f


@pytest.mark.parametrize("change, needle", [
    (mutate(["checks", "all_identities_hold"], False), "all_identities_hold"),
    (mutate(["checks", "all_identities_hold"], "true"), "all_identities_hold"),       # a string is not exactly true
    (mutate(["checks", "all_identities_hold"], 1), "all_identities_hold"),
    (mutate(["checks", "all_identities_hold"], KeyError), "all_identities_hold"),
    (mutate(["checks", "attention_hook_image_identical"], False), "attention_hook"),
    (mutate(["checks", "cag_w1_unet_evaluations"], 80), "unet_evaluations"),
    (mutate(["checks", "preflight_ok"], KeyError), "preflight_ok"),
    (mutate(["checks", "evaluator_error"], "OSError: no internet"), "judge"),
    (mutate(["timing", "judge_s_per_image"], KeyError), "judge_s_per_image"),
    (mutate(["checks", "two_worker_exit_codes"], {"pg0": 0, "pg1": 1}), "two-worker"),
    (mutate(["checks", "two_worker_exit_codes"], KeyError), "two-worker"),
    (mutate(["env", "cuda_available"], False), "CUDA"),
    (mutate(["fake_pipeline"], True), "fake_pipeline"),
    (mutate(["checks"], KeyError), "checks"),
])
def test_bad_preflight_is_rejected(tmp_path, change, needle):
    probs = N.validate_preflight(write(tmp_path, change(good_preflight())), 0)
    assert probs and any(needle in p for p in probs), probs


@pytest.mark.parametrize("content", [None, "", "{not json", "[1, 2]", "null"])
def test_missing_or_malformed_preflight_json_is_rejected(tmp_path, content):
    p = tmp_path / "preflight.json" if content is None else write(tmp_path, content)
    assert N.validate_preflight(p, 0)


def test_nonzero_preflight_exit_code_is_rejected_even_with_a_good_json(tmp_path):
    probs = N.validate_preflight(write(tmp_path, good_preflight()), 1)
    assert probs == ["r3_preflight.py exited with code 1"]


# ---------- GPU identity ----------
def good_identity():
    req = ["determinism_maxdiff", "loop_from_step0_maxdiff", "reload_resume_maxdiff", "d0_branch_maxdiff", "hook_off_control_maxdiff", "restored_baseline_maxdiff"]
    return dict(all_required_identities_hold=True, required=req, results=[dict({k: 0 for k in req}, name=f"p{i}", saved_latent_recompute_equal=True) for i in range(3)])


def test_good_gpu_identity_is_accepted(tmp_path):
    assert N.validate_gpu_identity(write(tmp_path, good_identity(), "gi.json"), 0) == []


@pytest.mark.parametrize("change, rc", [
    (lambda g: g, 1),
    (lambda g: dict(g, all_required_identities_hold=False), 0),
    (lambda g: dict(g, all_required_identities_hold="True"), 0),
    (lambda g: dict(g, results=[]), 0),                                              # vacuous truth is not evidence
    (lambda g: dict(g, required=[]), 0),
    (lambda g: dict(g, results=[dict(g["results"][0], d0_branch_maxdiff=3)]), 0),     # contradicts the summary flag
    (lambda g: dict(g, results=[dict(g["results"][0], saved_latent_recompute_equal=False)]), 0),
])
def test_bad_gpu_identity_is_rejected(tmp_path, change, rc):
    assert N.validate_gpu_identity(write(tmp_path, change(good_identity()), "gi.json"), rc)


def test_missing_gpu_identity_is_rejected(tmp_path):
    assert N.validate_gpu_identity(tmp_path / "gi.json", 0) == ["gi.json missing"]


# ---------- FREEZE_CHECK ----------
@pytest.mark.parametrize("rc, out, ok", [
    (0, "VERIFIED\n", True),
    (1, "VERIFIED\n", False),
    (0, "NOT VERIFIED:\n  protocol not signed\n", False),
    (1, "NOT VERIFIED:\n  missing\n", False),
    (0, "", False),
    (0, None, False),
    (0, "Traceback (most recent call last):\nVERIFIED\n", False),
])
def test_freeze_check_requires_exit_0_and_verified(rc, out, ok):
    assert (N.validate_freeze_check(rc, out) == []) is ok


def test_protocol_verify_rejects_missing_malformed_and_unsigned(tmp_path):
    import protocol_r3 as P
    assert P.verify(tmp_path / "nope.json")
    assert any("not valid JSON" in p for p in P.verify(write(tmp_path, "{broken", "pf.json")))
    assert any("not a JSON object" in p for p in P.verify(write(tmp_path, "[]", "pf2.json")))
    assert any("not signed" in p for p in P.verify(write(tmp_path, {"status": "DRAFT", "files": {}}, "pf3.json")))
    r = subprocess.run([sys.executable, str(ROOT / "src" / "protocol_r3.py"), "--verify", "--path", str(tmp_path / "pf3.json")], capture_output=True, text=True)
    assert r.returncode == 1 and N.validate_freeze_check(r.returncode, r.stdout)


# ---------- worker stages / merges ----------
def good_report(n=10):
    return dict(ok=True, manifest_jobs=n, merged=n, missing=[], unexpected=[], conflicts=[], corrupt=[], mismatch=[], fake=[], wrong_revision=[],
                allow_missing_dry_run=False)


def test_complete_stage_is_verified():
    assert N.stage_verdict({"s0g0": 0, "s0g1": 0}, 0, good_report()) == (True, [])


@pytest.mark.parametrize("codes, rc, report, kw", [
    ({}, 0, good_report(), {}),
    (None, 0, good_report(), {}),
    ({"s0g0": 0, "s0g1": 1}, 0, good_report(), {}),                 # a persistent job failure on one GPU
    ({"s0g0": 0}, 1, good_report(), {}),                              # merge exit code wins over its report
    ({"s0g0": 0}, 0, None, {}),
    ({"s0g0": 0}, 0, dict(good_report(), ok=False), {}),
    ({"s0g0": 0}, 0, dict(good_report(), ok="yes"), {}),
    ({"s0g0": 0}, 0, dict(good_report(), missing=["j1"]), {}),       # contradictory: ok but jobs missing
    ({"s0g0": 0}, 0, dict(good_report(), merged=9), {}),
    ({"s0g0": 0}, 0, dict(good_report(0), manifest_jobs=0, merged=0), {}),
    ({"s0g0": 0}, 0, {k: v for k, v in good_report().items() if k != "corrupt"}, {}),
    ({"s0g0": 0}, 0, dict(good_report(), allow_missing_dry_run=True), {}),
    ({"s0g0": 0}, 0, good_report(), dict(limit=5)),                   # smoke run
    ({"s0g0": 0}, 0, good_report(), dict(dry_run=True)),
])
def test_incomplete_or_contradictory_stage_is_not_verified(codes, rc, report, kw):
    ok, reasons = N.stage_verdict(codes, rc, report, **kw)
    assert ok is False and reasons


def test_dry_run_tolerates_only_the_expected_dry_run_reasons():
    ok, reasons = N.stage_verdict({"d0g0": 0}, 0, dict(good_report(), missing=["a", "b"], merged=8, allow_missing_dry_run=True), dry_run=True)
    assert ok is False and N.blocking_problems(reasons, dry_run=True) == []
    assert N.blocking_problems(reasons, dry_run=False) == reasons
    ok, reasons = N.stage_verdict({"d0g0": 2}, 1, None, dry_run=True)
    assert len(N.blocking_problems(reasons, dry_run=True)) == 3


def test_merge_rejects_a_record_missing_a_required_file(tmp_path):
    import r3_merge as M
    from PIL import Image
    out = tmp_path / "w"
    (out / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(out / "images" / "j1.png")
    h = M.sha(out / "images" / "j1.png")
    job = dict(job_id="j1", kind="core", prompt_id="p", seed=0, cfg=7.5, steps=40, split="DEV", scene_id="s", save_latents=True, attention=False)
    (tmp_path / "m.jsonl").write_text(json.dumps(job) + "\n")
    (out / "records_w.jsonl").write_text(json.dumps(dict(job, files={"images/j1.png": h}, image_sha256=h, revision="r")) + "\n")
    rep = M.merge(tmp_path / "m.jsonl", [out], tmp_path / "v", revision="r")
    assert rep["ok"] is False and rep["mismatch"] == [{"j1": ["files"]}]
    (out / "records_w.jsonl").write_text(json.dumps(dict(job, files={"images/j1.png": h}, revision="r")) + "\n")      # no image_sha256
    rep = M.merge(tmp_path / "m.jsonl", [out], tmp_path / "v2", revision="r")
    assert rep["ok"] is False and rep["corrupt"] == ["j1"]


# ---------- archives ----------
def make_out(tmp_path):
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    (out / "images" / "a.png").write_bytes(b"x" * 100)
    (out / "meta.json").write_text("{}")
    return out


@pytest.mark.parametrize("given", [None, False, "yes", 1, "True"])
def test_package_is_unverified_without_an_explicit_true(tmp_path, given):
    meta = {} if given is None else dict(verified=given)
    man = N.package(make_out(tmp_path), tmp_path / "arch", "r1", meta=meta)
    assert man["verified"] is False and N.archive_is_verified(man) is False


def test_package_verified_true_and_round_trip(tmp_path):
    man = N.package(make_out(tmp_path), tmp_path / "arch", "r1", meta=dict(verified=True, smoke_limit=None))
    assert man["verified"] is True and N.archive_is_verified(man)
    dest, probs = N.extract_and_verify(tmp_path / "arch", tmp_path / "x")
    assert probs == [] and (dest / "images" / "a.png").exists()


def test_tampered_or_missing_parts_are_detected(tmp_path):
    man = N.package(make_out(tmp_path), tmp_path / "arch", "r1", meta=dict(verified=True))
    part = tmp_path / "arch" / man["groups"]["images"][0]
    with zipfile.ZipFile(part, "a") as z:                       # changes the part's bytes -> checksum mismatch
        z.writestr("extra.txt", "x")
    _, probs = N.extract_and_verify(tmp_path / "arch", tmp_path / "x")
    assert probs and "checksum" in probs[0]
    part.unlink()
    _, probs = N.extract_and_verify(tmp_path / "arch", tmp_path / "y")
    assert probs and "missing" in probs[0]
    assert N.verify_parts(tmp_path / "arch", man)


@pytest.mark.parametrize("am", [dict(verified=True, smoke_limit=5, files={"a": 1}), dict(verified=True, dry_run=True, files={"a": 1}),
                                dict(verified="true", files={"a": 1}), dict(verified=True, files={}), dict(files={"a": 1}), None])
def test_archive_is_verified_needs_explicit_complete_evidence(am):
    assert N.archive_is_verified(am) is False


# ---------- notebook sources ----------
@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_notebook_bootstraps_before_importing_the_stack(nb):
    text = (ROOT / "kaggle" / nb).read_text(encoding="utf-8")
    ast.parse(text)
    boot = text.index("env_bootstrap.bootstrap(")
    for mod in ("import torch", "import diffusers", "from diffusers", "import transformers", "env_report()"):
        i = text.find(mod)
        assert i == -1 or i > boot, f"{nb}: '{mod}' before the dependency bootstrap"
    assert '"-m", "pip", "install"' not in text, f"{nb}: installs packages outside env_bootstrap"


@pytest.mark.parametrize("nb", NOTEBOOKS)
def test_notebook_has_no_optimistic_defaults(nb):
    text = (ROOT / "kaggle" / nb).read_text(encoding="utf-8")
    assert not re.search(r'["\']ok["\']\s*:\s*True', text)
    assert not re.search(r"verified\s*=\s*True\b", text)
    assert 'am.get("verified")' not in text, "use nbtools.archive_is_verified()"
    assert "json.loads(g.stdout)" not in text
    if "r3_merge.py" in text:
        assert "stage_verdict(" in text and 'shutil.rmtree(OUT / "verify"' in text or 'shutil.rmtree(OUT / "merged" / st' in text


def test_preflight_and_freeze_cells_use_the_validators():
    text = (ROOT / "kaggle" / "nb00_preflight_dev_freeze.py").read_text(encoding="utf-8")
    for fn in ("validate_preflight(", "validate_gpu_identity(", "validate_freeze_check(", "raise nbtools.ValidationError("):
        assert fn in text


# ---------- end to end: notebook 00 on CPU with injected failures ----------
@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    import build_release as BR
    sb = tmp_path_factory.mktemp("nb00")
    BR.stage_source(sb / "inputs" / "src" / "gbc_source", include_untracked=True)
    BR.stage_benchmark(sb / "inputs" / "bench" / "gbc_benchmark")
    return sb


def run_nb00(sb, name, stage, preflight_stub=None):
    import build_release as BR
    import dryrun_notebooks as D
    src = sb / "inputs" / "src" / "gbc_source"
    target = src / "kaggle" / "r3_preflight.py"
    original = target.read_text(encoding="utf-8")
    try:
        if preflight_stub is not None:
            target.write_text(preflight_stub, encoding="utf-8")
        BR.write_source_manifest(src)                            # the stubbed package still verifies, so only the validators can stop it
        work = sb / f"work_{name}"
        work.mkdir()
        script = sb / f"{name}.py"
        script.write_text(D.configured("nb00_preflight_dev_freeze.py", STAGE=stage), encoding="utf-8")
        env = dict(os.environ, GBC_DRYRUN="1", GBC_INPUT_ROOTS=os.pathsep.join([str(sb / "inputs" / "src"), str(sb / "inputs" / "bench")]),
                   GBC_WORKING=str(work), PYTHONIOENCODING="utf-8", GBC_MODEL_CACHE=str(ROOT / "models"))
        env.pop("GBC_ALLOW_DRYRUN_PROTOCOL", None)
        r = subprocess.run([sys.executable, str(script)], cwd=work, env=env, capture_output=True, text=True, timeout=1800)
    finally:
        target.write_text(original, encoding="utf-8")
    archives = [json.loads(p.read_text()) for p in work.rglob("ARCHIVE_MANIFEST.json")]
    return r, archives


STUB_FALSE = '''import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--n"); ap.add_argument("--two-workers", action="store_true")
ap.add_argument("--fake-pipeline", action="store_true"); a = ap.parse_args()
rep = dict(env=dict(cuda_available=False, gpus=[]), fake_pipeline=True, timing=dict(model_load_s=1.0, plain_s_median=1.0,
           with_attention_and_latents_s_median=1.0, branch_t12_D8_s=1.0),
           checks=dict(all_identities_hold=False, attention_hook_image_identical=False, cag_constant_equals_pipeline=True, cag_w1_equals_pipeline_w1=True,
                       cag_w1_unet_evaluations=40, preflight_ok=True))
open(a.out, "w").write(json.dumps(rep))
print("identities printed but exit code 0")
'''
STUB_CRASH = "import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n"


@pytest.mark.parametrize("stub, needle", [(STUB_FALSE, "all_identities_hold"), (STUB_CRASH, "exited with code 3")])
def test_nb00_preflight_fails_closed_end_to_end(sandbox, stub, needle):
    r, archives = run_nb00(sandbox, f"pf_{needle.split()[0]}", "PREFLIGHT", stub)
    assert r.returncode != 0, r.stdout[-2000:]
    assert needle in r.stdout + r.stderr
    assert "PREFLIGHT FAILED" in r.stdout
    assert archives and all(a["verified"] is False for a in archives)          # diagnostics kept, never verified


def test_nb00_freeze_check_fails_closed_without_a_signed_protocol(sandbox):
    r, archives = run_nb00(sandbox, "freeze_missing", "FREEZE_CHECK")
    assert r.returncode != 0 and "FREEZE_CHECK FAILED" in r.stdout
    assert archives and all(a["verified"] is False for a in archives)


def test_nb00_fake_preflight_passes_but_is_never_verified(sandbox):
    r, archives = run_nb00(sandbox, "pf_ok", "PREFLIGHT")
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    assert "PREFLIGHT PASSED" in r.stdout
    assert archives and all(a["verified"] is False and a["dry_run"] is True for a in archives)
