import json, subprocess, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "kaggle"))
import test_verify_branches as TV
import make_core_dataset as MC

CFG = ROOT / "configs" / "experiment_v2.json"
PROMPTS = ROOT / "data" / "benchmark_v2" / "prompts_v2.json"


def add_latents(core):
    (core / "latents").mkdir(exist_ok=True)
    for line in (core / "samples.jsonl").read_text().splitlines():
        (core / "latents" / (json.loads(line)["name"] + ".pt")).write_bytes(b"x")


def finish(run_dir, core, zip_out=None):
    cmd = [sys.executable, str(ROOT / "kaggle" / "finish_run.py"), "--stage", "r3", "--run-dir", str(run_dir), "--config", str(CFG), "--prompts", str(PROMPTS),
           "--run-id", "t-run", "--core-run", str(core), "--M", "1", "--split", "DEV"]
    if zip_out:
        cmd += ["--zip-out", str(zip_out)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_finish_run_marks_complete_only_after_verification(tmp_path):
    core, out, rows = TV.make_run(tmp_path)
    add_latents(core)
    r = finish(out, core, tmp_path / "x.zip")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "COMPLETE").exists() and (out / "checksums.sha256").exists()
    rep = json.loads((out / "run_report.json").read_text())
    assert rep["ok"] and rep["verifier_stats"]["records"] == rep["verifier_stats"]["expected"] == 36
    assert "samples.jsonl" in zipfile.ZipFile(tmp_path / "x.zip").namelist()


def test_finish_run_refuses_a_damaged_run(tmp_path):
    core, out, rows = TV.make_run(tmp_path)
    add_latents(core)
    (out / "images" / f"{rows[0]['name']}.png").unlink()
    r = finish(out, core)
    assert r.returncode == 1
    assert not (out / "COMPLETE").exists() and not (out / "checksums.sha256").exists()
    assert json.loads((out / "run_report.json").read_text())["ok"] is False


def test_core_dataset_has_lf_text_and_manifest(tmp_path):
    core = tmp_path / "core"
    (core / "images").mkdir(parents=True); (core / "latents").mkdir()
    (core / "samples.jsonl").write_bytes(b'{"a": 1}\r\n{"a": 2}\r\n')
    (core / "images" / "a.png").write_bytes(b"png"); (core / "latents" / "a.pt").write_bytes(b"pt")
    m = MC.main(core, tmp_path / "out", "r2a-test")
    assert b"\r" not in (tmp_path / "out" / "dev_core" / "samples.jsonl").read_bytes()
    assert m["n_images"] == 1 and m["n_latents"] == 1 and "dev_core/samples.jsonl" in m["files"]
