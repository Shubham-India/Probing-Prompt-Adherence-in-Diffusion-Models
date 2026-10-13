import json, sys
from pathlib import Path
import pytest
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config_schema, jobs as J, merge_samples, verify_outputs
import generate as G

PILOT_CFG = json.loads((ROOT / "configs/experiment.json").read_text())
PROMPTS = [p for p in json.loads((ROOT / "data/prompts.json").read_text())][:3]


def test_default_names_match_pilot_samples():
    pilot = [json.loads(l)["name"] for l in (ROOT / "outputs/main/samples.jsonl").read_text().splitlines()]
    mine = [J.sample_name(p["id"], c, s, sd) for p, c, s, sd in J.enumerate_jobs(json.loads((ROOT / "data/prompts.json").read_text()), PILOT_CFG)]
    assert sorted(mine) == sorted(pilot) and len(set(mine)) == 1008


def test_shards_disjoint_and_complete():
    jobs = J.enumerate_jobs(PROMPTS, PILOT_CFG)
    parts = [J.shard_jobs(jobs, (i, 3)) for i in range(3)]
    names = [[J.sample_name(p["id"], c, s, sd) for p, c, s, sd in part] for part in parts]
    flat = sum(names, [])
    assert len(flat) == len(set(flat)) == len(jobs)
    assert sorted(flat) == sorted(J.sample_name(p["id"], c, s, sd) for p, c, s, sd in jobs)
    with pytest.raises(ValueError): J.parse_shard("3/3")


def test_test_prompts_need_frozen_config():
    with pytest.raises(ValueError): J.guard_test_access([{"id": "x", "split": "TEST"}], PILOT_CFG)
    v2 = json.loads((ROOT / "configs/experiment_v2.json").read_text())
    with pytest.raises(ValueError): J.guard_test_access([{"id": "x", "split": "TEST"}], v2)
    J.guard_test_access([{"id": "x", "split": "DEV"}], v2)   # DEV is allowed on a draft config


@pytest.fixture
def fake(monkeypatch):
    import torch
    calls = []
    def gen(pipe, prompt, cfg, steps, seed, size, **kw):
        calls.append((prompt, cfg, steps, seed))
        return Image.new("RGB", (size, size), (seed * 10 % 255, 40, 90))
    def gen_lat(pipe, prompt, cfg, steps, seed, size, save_at):
        return gen(pipe, prompt, cfg, steps, seed, size), {t: torch.zeros(1, 4, 8, 8, dtype=torch.float16) for t in save_at}
    monkeypatch.setattr(G, "load_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(G, "generate_image", gen)
    monkeypatch.setattr(G, "generate_with_latents", gen_lat)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "fake-gpu")
    return calls


def run_main(monkeypatch, tmp_path, *extra):
    cfg = dict(PILOT_CFG, image_size=64, cfg_scales=[7.5], steps=[40], seeds=[0, 1])
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    (tmp_path / "prompts.json").write_text(json.dumps(PROMPTS))
    monkeypatch.setattr(sys, "argv", ["generate.py", "--config", str(tmp_path / "cfg.json"), "--prompts", str(tmp_path / "prompts.json"),
                                      "--out", str(tmp_path / "run"), *extra])
    G.main()
    return cfg


def test_generate_resume_shards_verify_merge(monkeypatch, tmp_path, fake):
    cfg = run_main(monkeypatch, tmp_path, "--shard", "0/2", "--run-id", "t")
    n0 = len(fake)
    run_main(monkeypatch, tmp_path, "--shard", "0/2", "--run-id", "t")      # restart: nothing to do
    assert len(fake) == n0
    run_main(monkeypatch, tmp_path, "--shard", "1/2", "--run-id", "t")
    assert len(fake) == 6                                                    # 3 prompts x 1 cfg x 1 step x 2 seeds, each once
    run = tmp_path / "run"
    errs, st = verify_outputs.verify(run, cfg, PROMPTS)
    assert errs == [] and st["unique"] == 6
    assert merge_samples.merge(run) == (6, 2)
    # seeded faults
    recs = [json.loads(l) for l in (run / "samples_shard0of2.jsonl").read_text().splitlines()]
    with open(run / "samples_shard0of2.jsonl", "a") as f:
        f.write(json.dumps(recs[0]) + "\n")                                  # duplicate record
    assert any("duplicate record" in e for e in verify_outputs.verify(run, cfg, PROMPTS)[0])
    (run / "samples_shard0of2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    img = run / "images" / f"{recs[0]['name']}.png"
    img.write_bytes(img.read_bytes()[:40])                                   # truncated PNG
    assert any("corrupt image" in e for e in verify_outputs.verify(run, cfg, PROMPTS)[0])


def test_conflicting_duplicates_abort_merge(monkeypatch, tmp_path, fake):
    run_main(monkeypatch, tmp_path, "--shard", "0/2")
    run = tmp_path / "run"
    r = json.loads((run / "samples_shard0of2.jsonl").read_text().splitlines()[0])
    r["image_sha256"] = "0" * 64
    (run / "samples_shard1of2.jsonl").write_text(json.dumps(r) + "\n")
    with pytest.raises(SystemExit): merge_samples.merge(run)


def test_latent_saving_and_default_record_has_no_extra_fields(monkeypatch, tmp_path, fake):
    cfg = run_main(monkeypatch, tmp_path, "--save-latents-at", "6,12")
    run = tmp_path / "run"
    rec = json.loads((run / "samples.jsonl").read_text().splitlines()[0])
    assert "latents_sha256" in rec and (run / "latents" / f"{rec['name']}.pt").exists()
    assert verify_outputs.verify(run, cfg, PROMPTS, need_latents=True)[0] == []
    import torch
    d = torch.load(run / "latents" / f"{rec['name']}.pt")
    assert d["steps_completed"] == [6, 12]


def test_default_flags_keep_pilot_record_schema(monkeypatch, tmp_path, fake):
    run_main(monkeypatch, tmp_path)
    rec = json.loads((tmp_path / "run" / "samples.jsonl").read_text().splitlines()[0])
    pilot = json.loads((ROOT / "outputs/main/samples.jsonl").read_text().splitlines()[0])
    assert set(rec) == set(pilot)
