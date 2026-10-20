import json, sys
from pathlib import Path
import pytest, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import run_branches as RB, branch as B, generate as G

SCH = B.Schedule(40)


def test_plans_and_clipping_flags():
    p6, p12 = RB.plan_for("D8", SCH, 6, None), RB.plan_for("D8", SCH, 12, None)
    assert p6["clipped"] and p6["effective_jump"] == 6 and not p12["clipped"] and p12["effective_jump"] == 8
    assert abs(RB.plan_for("ER30", SCH, 24, None)["ratio"] - 0.30) < 0.06
    f = RB.plan_for("FULL", SCH, 24, None)
    assert f["target_idx"] == 0 and f["ratio"] == 0.0
    assert RB.plan_for("D0", SCH, 12, None)["ratio"] == 1.0
    with pytest.raises(ValueError): RB.plan_for("X", SCH, 6, None)


def test_noise_seed_is_deterministic_and_distinct():
    assert RB.noise_seed("a", 6, "D8", 0) == RB.noise_seed("a", 6, "D8", 0)
    assert len({RB.noise_seed("a", 6, "D8", m) for m in range(2)} | {RB.noise_seed("a", 6, "ER30", 0)}) == 3


def test_job_enumeration_counts():
    recs = [{"name": f"s{k}__seed{sd}", "seed": sd, "variant": "CAN"} for k in range(3) for sd in range(4)] + [{"name": "x", "seed": 0, "variant": "PAR"}]
    jobs = RB.enumerate_branch_jobs(recs, [6, 12], ["D8", "ER30"], 2, [0, 1])
    assert len(jobs) == 3 * 2 * 2 * 2 * 2          # 3 scenes x 2 base seeds x 2 grid x 2 variants x M=2; PAR record excluded


def test_runner_end_to_end_with_fakes(monkeypatch, tmp_path):
    core = tmp_path / "core"; (core / "latents").mkdir(parents=True)
    prompts = [{"id": "sg_0001_CAN", "scene_id": "sg_0001", "split": "DEV", "variant": "CAN", "category": "binding", "prompt": "a red cup and a blue book",
                "constraints": [{"type": "presence", "object": "cup"}]}]
    (tmp_path / "p.json").write_text(json.dumps(prompts))
    recs = []
    for sd in (0, 1):
        name = f"sg_0001_CAN__cfg7.5__st40__seed{sd}"
        recs.append({"name": name, "prompt_id": "sg_0001_CAN", "category": "binding", "prompt": prompts[0]["prompt"], "seed": sd, "scene_id": "sg_0001", "split": "DEV", "variant": "CAN"})
        torch.save({"steps_completed": [6, 12], "latents_fp16": {t: torch.zeros(1, 4, 8, 8, dtype=torch.float16) for t in (6, 12)}}, core / "latents" / f"{name}.pt")
    (core / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    cfg = json.loads((ROOT / "configs/experiment_v2.json").read_text()); cfg.update(image_size=64)
    cfg["restart"] = dict(cfg["restart"], grid=[6, 12])
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    calls = []
    monkeypatch.setattr(G, "load_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(B, "encode_prompt_pair", lambda *a, **k: (None, None))
    monkeypatch.setattr(B, "branch_generate", lambda pipe, e, lat, t, plan, ns, steps, cfg_: calls.append(("br", t, plan["ratio"])) or Image.new("RGB", (64, 64), (ns % 255, 0, 0)))
    monkeypatch.setattr(B, "denoise_from", lambda pipe, e, x, start, steps, cfg_, decode=True: calls.append(("full", start)) or Image.new("RGB", (64, 64), (9, 9, 9)))
    class Fake:                       # stands in for .to("cuda") on a CPU-only machine
        pass
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)
    argv = ["run_branches.py", "--core-run", str(core), "--config", str(tmp_path / "cfg.json"), "--prompts", str(tmp_path / "p.json"), "--out", str(tmp_path / "br"),
            "--variants", "D8", "FULL", "--M", "2", "--run-id", "t"]
    monkeypatch.setattr(sys, "argv", argv)
    RB.main()
    out = [json.loads(l) for l in (tmp_path / "br" / "samples.jsonl").read_text().splitlines()]
    assert len(out) == 2 * 2 * 2 * 2 == len(calls)                       # seeds x grid x variants x M
    d8_6 = [r for r in out if r["variant"] == "D8" and r["t"] == 6][0]
    assert d8_6["clipped"] and d8_6["effective_jump"] == 6 and d8_6["requested_jump"] == 8 and d8_6["scene_id"] == "sg_0001"
    assert {c for c in calls if c[0] == "full"} == {("full", 0)}
    n = len(calls)
    RB.main()                                                           # resume: nothing regenerated
    assert len(calls) == n
