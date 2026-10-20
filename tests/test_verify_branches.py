import copy, json, sys
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import branch as B, run_branches as R, verify_branches as VB
from verify_outputs import sha

CFG = json.loads((ROOT / "configs" / "experiment_v2.json").read_text())


def make_run(tmp_path, variants=("D8", "ER30", "FULL"), M=1):
    core = tmp_path / "core"
    core.mkdir()
    recs = [{"name": f"sg_0001_CAN__cfg7.5__st40__seed{s}", "prompt_id": "sg_0001_CAN", "category": "binding", "prompt": "a red cup and a blue book",
             "scene_id": "sg_0001", "split": "DEV", "seed": s, "variant": "CAN"} for s in (0, 1, 2)]
    (core / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    out = tmp_path / "br"
    (out / "images").mkdir(parents=True)
    sch = B.Schedule(40)
    rows = []
    for r, t, v, m in R.enumerate_branch_jobs(recs, CFG["restart"]["grid"], list(variants), M, CFG["seeds_branch_bases"]):
        name = f"{r['name']}__br_{v}_t{t}_m{m}"
        Image.new("RGB", (512, 512), (t * 5, 10, 10)).save(out / "images" / f"{name}.png")
        plan = R.plan_for(v, sch, t, CFG["restart"])
        rows.append({"name": name, "core_name": r["name"], "t": t, "m": m, "variant": v, "size": 512, "revision": CFG["revision"],
                     "noise_seed": R.noise_seed(r["name"], t, v, m), "requested_jump": plan["requested_jump"], "effective_jump": plan["effective_jump"],
                     "ratio": plan["ratio"], "clipped": plan["clipped"], "ts_target": plan["ts_target"], "image_sha256": sha(out / "images" / f"{name}.png")})
    (out / "samples.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    return core, out, rows


def run(core, out, variants=("D8", "ER30", "FULL")):
    return VB.verify(out, core, CFG, None, list(variants), 1, "DEV")


def test_clean_run_verifies(tmp_path):
    core, out, rows = make_run(tmp_path)
    errs, stats = run(core, out)
    assert errs == [] and stats["expected"] == stats["records"] == 2 * 6 * 3      # only seeds 0,1 are branch bases


def test_missing_corrupt_and_orphan_are_detected(tmp_path):
    core, out, rows = make_run(tmp_path)
    (out / "images" / f"{rows[0]['name']}.png").unlink()
    (out / "images" / f"{rows[1]['name']}.png").write_bytes(b"not a png")
    Image.new("RGB", (512, 512)).save(out / "images" / "orphan.png")
    errs, _ = run(core, out)
    text = " | ".join(errs)
    assert "image missing" in text and "corrupt image" in text and "orphan image: orphan" in text


def test_plan_and_noise_seed_tampering_is_detected(tmp_path):
    core, out, rows = make_run(tmp_path)
    bad = copy.deepcopy(rows)
    bad[0]["effective_jump"] += 1
    bad[1]["noise_seed"] += 1
    (out / "samples.jsonl").write_text("\n".join(json.dumps(x) for x in bad) + "\n")
    errs, _ = run(core, out)
    assert any("plan mismatch effective_jump" in e for e in errs) and any("noise seed rule" in e for e in errs)


def test_duplicate_and_missing_records(tmp_path):
    core, out, rows = make_run(tmp_path)
    (out / "samples.jsonl").write_text("\n".join(json.dumps(x) for x in rows[1:] + [rows[1]]) + "\n")
    errs, _ = run(core, out)
    assert any("duplicate record" in e for e in errs) and any("missing branch" in e for e in errs)
