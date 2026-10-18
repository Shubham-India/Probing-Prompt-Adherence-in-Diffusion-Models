"""CPU end-to-end of the Revision 3 worker with the FAKE (tiny random) pipeline: core with latents + attention, a restart branch, a prompt switch
and a CAG job; then resume, corruption recovery and the TEST guard. Validates control flow and integrity rules, not Stable Diffusion."""
import json, subprocess, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import manifests as M
import cag as CAG

pytest.importorskip("diffusers")
CLIP_CACHE = ROOT / "models" / "models--openai--clip-vit-base-patch32"
pytestmark = pytest.mark.skipif(not CLIP_CACHE.exists(), reason="fake pipeline needs the local CLIP tokenizer cache")


def mini_manifest():
    cfg = M.load_cfg()
    main, ps = M.load_prompts(cfg)
    can = next(p for p in main if p["split"] == "DEV" and p["variant"] == "CAN" and any(q["scene_id"] == p["scene_id"] and q["variant"] == "CF-C" for q in main))
    cf = next(q for q in main if q["scene_id"] == can["scene_id"] and q["variant"] == "CF-C")
    base = M.core_job(can, 7.5, 40, 0, "test", "DEV_COMMIT", latents=True, attention=True)
    cfb = M.core_job(cf, 7.5, 40, 0, "test", "DEV_COMMIT", latents=True)
    br = [j for j in M.branch_jobs(base, cfg, ["D8"], "test", "DEV_COMMIT", 1) if j["t"] == 12]
    sw = [j for j in M.switch_jobs(cfb, can, cfg, "test", "DEV_COMMIT") if j["t"] == 18]
    sch, _ = CAG.schedules(10, 7.5, 1.0)
    cagj = dict(job_id=f"{can['id']}__cag_A2__seed0", kind="cag", experiment="test", stage="DEV_CAG_GUARD", split="DEV", shard_group=can["scene_id"],
                scene_id=can["scene_id"], prompt_id=can["id"], prompt=can["prompt"], variant="CAN", category=can["category"], steps=40, seed=0, arm="A2",
                schedule=sch["A2"], unet_calls=CAG.unet_calls(sch["A2"]), cfg=None)
    return [base, cfb] + br + sw + [cagj]


def run(manifest, out, *extra):
    return subprocess.run([sys.executable, str(ROOT / "src" / "r3_run.py"), "--manifest", str(manifest), "--out", str(out), "--run-id", "fake-test",
                           "--worker-id", "w0", "--fake-pipeline", *extra], capture_output=True, text=True, timeout=900)


def test_worker_end_to_end_resume_and_recovery(tmp_path):
    jobs = mini_manifest()
    mf = tmp_path / "m.jsonl"
    mf.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    out = tmp_path / "out"
    r = run(mf, out)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    recs = [json.loads(l) for l in (out / "records_w0.jsonl").read_text().splitlines()]
    assert {x["job_id"] for x in recs} == {j["job_id"] for j in jobs}
    by = {x["job_id"]: x for x in recs}
    base = jobs[0]
    assert (out / "latents" / f"{base['job_id']}.pt").exists() and (out / "attention" / f"{base['job_id']}.npz").exists()
    assert by[base["job_id"]]["unet_calls_measured"] == 80                       # pipeline: 40 steps x [uncond, cond]
    br = next(x for x in recs if x["kind"] == "branch")
    assert br["unet_calls_measured"] == br["unet_calls"]                          # only the remaining steps are run
    cg = next(x for x in recs if x["kind"] == "cag")
    assert cg["unet_calls_measured"] == 2 * 10 + 30                              # w = 1 steps skip the unconditional pass
    assert all(x["fake_pipeline"] for x in recs)
    import attn_record as AR
    maps, words, meta = AR.load_npz(out / "attention" / f"{base['job_id']}.npz")
    assert 8 in maps and maps[8].shape[0] == 40 and maps[8].shape[1] == len(words)
    r2 = run(mf, out)
    assert r2.returncode == 0 and '"to_run": 0' in r2.stdout                     # idempotent resume
    victim = out / "images" / f"{br['job_id']}.png"
    victim.write_bytes(b"corrupt")
    r3 = run(mf, out)
    assert '"to_run": 1' in r3.stdout and r3.returncode == 0                     # only the corrupted job is regenerated


def test_test_jobs_are_refused_without_a_frozen_protocol(tmp_path):
    jobs = mini_manifest()[:1]
    jobs[0] = dict(jobs[0], split="TEST")
    mf = tmp_path / "m.jsonl"
    mf.write_text(json.dumps(jobs[0]) + "\n")
    r = run(mf, tmp_path / "out", "--protocol", str(tmp_path / "missing.json"))
    assert r.returncode != 0 and "TEST jobs refused" in (r.stdout + r.stderr)
