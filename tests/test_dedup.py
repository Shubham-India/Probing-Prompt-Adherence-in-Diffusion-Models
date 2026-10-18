"""Q2 duplicate rules (src/dedup.py, DL-33): one generation counts once, the duplicate DEV canonical pair contributes once to DEV constants,
TEST items whose prompt also occurs in DEV are excluded from TEST S2 estimates; on the real benchmark and full manifests."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import dedup as D  # noqa: E402

P = {"a_CAN": dict(id="a_CAN", scene_id="a", split="TEST", variant="CAN", prompt="a cup on a wooden table"),
     "b_CFO": dict(id="b_CFO", scene_id="b", split="TEST", variant="CF-O", prompt="a cup on a wooden table"),
     "psX_CAN": dict(id="psX_CAN", scene_id="psX", split="TEST", variant="CAN", prompt="A cup, on a wooden table"),
     "c_CFO": dict(id="c_CFO", scene_id="c", split="TEST", variant="CF-O", prompt="a shoe"),
     "d_CFO": dict(id="d_CFO", scene_id="d", split="DEV", variant="CF-O", prompt="a shoe"),
     "e_CAN": dict(id="e_CAN", scene_id="e", split="TEST", variant="CAN", prompt="a red bus")}


def core(pid, seed=0, cfg=7.5, **kw):
    p = P[pid]
    return dict(job_id=f"{pid}__s{seed}__w{cfg}", kind="core", prompt_id=pid, scene_id=p["scene_id"], split=p["split"], variant=p["variant"], cfg=cfg,
                steps=40, seed=seed, **kw)


def test_one_generation_counts_once_and_the_canonical_main_item_is_kept():
    rows = [core("b_CFO"), core("psX_CAN"), core("a_CAN"), core("a_CAN", seed=1), core("b_CFO", cfg=2)]
    rows.append(dict(job_id="br1", kind="branch", base_job="b_CFO__s0__w7.5", scene_id="b", split="TEST", cfg=7.5, steps=40, seed=0))
    rows.append(dict(job_id="br2", kind="branch", base_job="a_CAN__s0__w7.5", scene_id="a", split="TEST", cfg=7.5, steps=40, seed=0))
    kept, log = D.dedup(rows, P)
    ids = {r["job_id"] for r in kept}
    assert "a_CAN__s0__w7.5" in ids                                          # CAN, main benchmark
    assert "psX_CAN__s0__w7.5" not in ids and "b_CFO__s0__w7.5" not in ids   # same normalised text, CFG, steps, seed
    assert {"a_CAN__s1__w7.5", "b_CFO__s0__w2"} <= ids                         # different seed / CFG: distinct generations
    assert "br1" not in ids and "br2" in ids and log["dropped_dependents"] == ["br1"]
    assert log["dropped_generations"] == ["b_CFO__s0__w7.5"]                  # R1
    assert log["dropped_scenes"] == {"psX": "a"}                              # R2: duplicate canonical scene (PS family) dropped first


def test_rows_without_a_known_prompt_are_never_merged():
    rows = [dict(job_id=f"x{i}", kind="core", scene_id=f"s{i}", cfg=7.5, steps=40, seed=0, variant="CAN") for i in range(5)]
    kept, log = D.dedup(rows, P)
    assert len(kept) == 5 and not log["dropped_generations"]


def test_dev_test_overlap_is_excluded_from_s2():
    import r3_analyze as RA
    rows = [core("c_CFO", seed=s, objects_present=1) for s in range(4)] + [core("e_CAN", seed=s, objects_present=0) for s in range(4)]
    kept, excl = D.exclude_dev_overlap(rows, P)
    assert sorted(excl) == [f"c_CFO__s{s}__w7.5" for s in range(4)] and len(kept) == 4
    D._INDEX = None
    import unittest.mock as um
    with um.patch.object(D, "prompt_index", lambda root=None: P):
        s2 = RA.analyse_s2(rows)
    assert s2["excluded_dev_overlap"] == 4 and set(s2["rates"]) == {"CAN"} and s2["rates"]["CAN"]["objects_present"]["mean"] == 0.0


def test_real_benchmark_duplicate_scene_and_counts():
    D._INDEX = None
    prompts = D.prompt_index()
    assert D.duplicate_scenes(prompts) == {"psD05_P": "sg_0091"}
    import manifests as M
    jobs, _, _ = M.build()
    kept, log = D.dedup(jobs, prompts)
    assert len(log["dropped_generations"]) == 124        # DL-33: of 156 redundant generations, 124 lie within a split (R1); 32 span DEV/TEST (R3)
    test_core = [j for j in kept if j["kind"] == "core" and j["split"] == "TEST" and j["cfg"] == 7.5 and j["steps"] == 40]
    _, excl = D.exclude_dev_overlap(test_core, prompts)
    assert len(excl) == 32 and all(prompts[j.split("__")[0]]["variant"] == "CF-O" for j in excl)    # after R1: 8 TEST CF-O items x 4 seeds
    raw = [j for j in jobs if j["kind"] == "core" and j["split"] == "TEST" and j["cfg"] == 7.5 and j["steps"] == 40]
    assert len(D.exclude_dev_overlap(raw, prompts)[1]) == 44                  # before R1: 11 TEST CF-O items x 4 seeds (DL-33)


def test_duplicate_dev_canonical_pair_contributes_once_to_dev_constants(tmp_path):
    from test_r3_analyze import TAU, world
    c, b = world(TAU, n_scenes=40, seed=3)
    rows = []
    for i, r in enumerate(c + b):                                            # map synthetic scenes sc0 / sc1 onto sg_0091 / psD05_P, others on DEV ids
        sc = {"sc0": "sg_0091", "sc1": "psD05_P"}.get(r["scene_id"], f"sg_{int(r['scene_id'][2:]) + 1:04d}")
        pid = "psD05_P_CAN" if sc == "psD05_P" else f"{sc}_CAN"
        rows.append(dict(r, scene_id=sc, prompt_id=pid, fake_pipeline=True, job_id=r["job_id"].replace(r["scene_id"] + "__", sc + "__", 1),
                         base_job=r.get("base_job", "").replace(r["scene_id"] + "__", sc + "__", 1) or None))
    f = tmp_path / "rows.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "dev.json"
    r = subprocess.run([sys.executable, str(ROOT / "src" / "r3_analyze.py"), "dev", "--rows", str(f), "--out", str(out), "--n-boot", "100", "--allow-fake"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["dedup"]["dropped_scenes"] == {"psD05_P": "sg_0091"}
    assert res["ha_dev"]["variants"]["D8"]["n_scenes"] == 39                 # 40 synthetic scenes, the PS twin counted once (with its twin)
