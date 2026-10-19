"""Revision 3 confirmatory analysis code on SYNTHETIC worlds with known answers (not research results): HA ordering, HD shift with
per-CFG chance correction, HE intersection-union rule, HF concordance, Holm, DEV constants and gates."""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import r3_analyze as RA

GRID = (6, 12, 18, 24, 30, 36)


def world(tau, n_scenes=60, seed=0, levels=(2, 7.5, 15), restarts=("D8", "ER30")):
    """tau[(kind, w)] = hard persistence step. Kinds: presence (unconditional) and bind (conditional on presence)."""
    rng = np.random.default_rng(seed)
    core, br = [], []
    for s in range(n_scenes):
        pp, pb = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
        for w in levels:
            for sd in range(4):
                pres = int(rng.random() < pp)
                bind = int(rng.random() < pb) if pres else None
                jid = f"sc{s}__cfg{w}__seed{sd}"
                core.append(dict(job_id=jid, kind="core", scene_id=f"sc{s}", seed=sd, cfg=w, steps=40, variant="CAN", split="DEV",
                                 objects_present=pres, bind_given_pres=bind, joint_bind=int(bool(pres and bind))))
                if sd > 1:
                    continue
                for v in restarts if w == 7.5 else ("D8",):
                    for t in GRID:
                        for m in range(2):
                            kp = t >= tau[("presence", w)]
                            p2 = pres if kp else int(rng.random() < pp)
                            kb = t >= tau[("bind", w)]
                            b2 = (bind if (kb and bind is not None) else int(rng.random() < pb)) if p2 else None
                            br.append(dict(job_id=f"{jid}__br_{v}_t{t}_m{m}", kind="branch", scene_id=f"sc{s}", seed=sd, cfg=w, steps=40, variant="CAN",
                                           restart=v, t=t, m=m, base_job=jid, split="DEV", objects_present=p2, bind_given_pres=b2, joint_bind=int(bool(p2 and b2))))
    return core, br


TAU = {("presence", 2): 18, ("presence", 7.5): 12, ("presence", 15): 7, ("bind", 2): 31, ("bind", 7.5): 25, ("bind", 15): 24}


def test_ha_detects_binding_later_than_presence():
    core, br = world(TAU)
    ha = RA.analyse_ha([r for r in core if r["cfg"] == 7.5], [r for r in br if r["cfg"] == 7.5], n_boot=300)
    d = ha["variants"]["D8"]["delta_t50"]
    assert d["estimate"] > 6 and d["ci95"][0] > 0
    assert ha["decision"] == "supported"


def test_ha_null_world_is_not_supported():
    tau = {k: 12 for k in TAU}
    core, br = world(tau, seed=3)
    ha = RA.analyse_ha([r for r in core if r["cfg"] == 7.5], [r for r in br if r["cfg"] == 7.5], n_boot=300)
    assert ha["decision"] != "supported"


def test_hd_shift_with_chance_level_per_cfg():
    core, br = world(TAU, seed=1)
    hd = RA.analyse_hd(core, br, n_boot=300)
    pres = hd["estimands"]["presence"]
    assert pres["delta"] < -6 and pres["ci95"][1] < 0 and pres["decision"] == "supported"
    assert hd["decision"] == "supported" and hd["p"] is not None


def test_he_intersection_union_rule():
    rng = np.random.default_rng(2)
    core, cag, ids, emb = [], [], [], []
    for s in range(400):                                                    # enough scenes that the 0.03 margin is resolvable
        base_f = rng.uniform(0.3, 0.8)
        for arm, df, spread in (("A1", 0.0, 0.2), ("A2", 0.0, 0.6), ("A4", -0.15, 0.6)):
            for sd in range(4):
                jid = f"p{s}_{arm}_{sd}"
                f = float(np.clip(base_f + df + rng.normal(0, 0.02), 0, 1))
                r = dict(job_id=jid, scene_id=f"p{s}", seed=sd, cfg=7.5, steps=40, variant="CAN", objects_present=1, bind_given_pres=int(rng.random() < f), arm=arm)
                (core if arm == "A1" else cag).append(dict(r, kind="core" if arm == "A1" else "cag"))
                ids.append(jid)
                emb.append(np.r_[1.0, rng.normal(0, spread, 7)])
    he = RA.analyse_he(core, cag, (ids, np.asarray(emb)), n_boot=500)
    assert he["conditions"]["more_diverse"] and he["conditions"]["beats_reversed"]
    assert he["conditions"]["non_inferior"] and he["decision"] == "supported"
    worse = [dict(r, bind_given_pres=0) if r["arm"] == "A2" else r for r in cag]
    assert RA.analyse_he(core, worse, (ids, np.asarray(emb)), n_boot=300)["conditions"]["non_inferior"] is False


def test_holm_and_final_labels():
    res = {"HA": dict(p=0.001, decision="supported"), "HD": dict(p=0.04, decision="supported"), "HE": dict(p=0.2, decision="not supported"), "HF": dict(p=None, decision="not estimable")}
    out = RA.finalise(res)
    assert out["HA"]["final"] == "supported" and abs(out["HA"]["p_holm"] - 0.003) < 1e-12
    assert out["HD"]["final"].startswith("inconclusive")                     # 0.04 * 2 = 0.08 >= 0.05 after Holm
    adj = RA.holm({"a": 0.01, "b": 0.02, "c": 0.03})
    assert adj == {"a": 0.03, "b": 0.04, "c": 0.04}


def test_dev_constants_and_gates_from_dev_rows():
    core, br = world(TAU, seed=4)
    full = [dict(r, restart="FULL") for r in br if r["cfg"] == 7.5 and r["restart"] == "D8"]
    for r in full:                                                           # FULL restart: outcome independent of the base
        r["job_id"] += "F"
    rng = np.random.default_rng(9)
    for r in full:
        r["objects_present"] = int(rng.random() < 0.5)
        r["bind_given_pres"] = int(rng.random() < 0.5) if r["objects_present"] else None
    out = RA.dev_constants(core, br + full, n_boot=200)
    t50 = out["constants"]["t50_dev_w7.5"]
    assert 6 <= t50["presence"] <= 18 and t50["bind_given_pres"] > t50["presence"]
    assert "full_resample_control_max_abs_C_le_0.15" in out["gates"] and out["gates"]["CAG_quality_guard"]["passed"] is None
    json.dumps(RA.clean(out))


def test_hf_concordance_from_synthetic_attention(tmp_path):
    import attn_record as AR
    core, br = world(TAU, seed=5, levels=(7.5,))
    rng = np.random.default_rng(0)
    prompts, att = {}, []
    for r in [x for x in core if x["seed"] < 2]:
        pid = f"P_{r['scene_id']}"
        prompts[pid] = {"constraints": [{"type": "presence", "object": "cube"}, {"type": "binding", "object": "cube", "color": "red"}]}
        def settling(step, n=40):
            final = rng.random((16, 16))
            return np.array([final + (rng.random((16, 16)) if i < step else 0) for i in range(n)])
        maps = {16: np.stack([settling(10), settling(23)], axis=1).astype(np.float16)}
        p = tmp_path / f"{r['job_id']}.npz"
        AR.save_npz(p, maps, ["cube", "red"], {})
        att.append(dict(r, prompt_id=pid, path=str(p)))
    ts = RA.t_stab_table(att, prompts)
    assert set(ts.kind) == {"presence", "bind_given_pres"}
    assert ts[ts.kind == "presence"].t_stab.mean() < ts[ts.kind == "bind_given_pres"].t_stab.mean()
    hf = RA.analyse_hf(core, br, ts, n_boot=100)
    assert hf["concordance"] == 1.0 and hf["pairs"] == 1
    assert hf["decision"] in ("inconclusive", "supported", "rejected (bounded null)")


def test_chance_agreement_uses_only_the_branch_base_prompt_and_settings():
    """DL-40: A0 must come from the independent seeds of the branches' own base prompt at the base CFG/steps, not from every core row of the
    scene (other variants, CFG levels or step counts)."""
    core = [dict(job_id=f"s_CAN__cfg7.5__seed{i}", kind="core", scene_id="s", prompt_id="s_CAN", variant="CAN", cfg=7.5, steps=40, seed=i,
                 objects_present=y) for i, y in enumerate([1, 1, 1, 0])]
    core += [dict(job_id=f"s_CFO__cfg7.5__seed{i}", kind="core", scene_id="s", prompt_id="s_CFO", variant="CF-O", cfg=7.5, steps=40, seed=i,
                  objects_present=0) for i in range(4)]
    core += [dict(job_id=f"s_CAN__cfg2__seed{i}", kind="core", scene_id="s", prompt_id="s_CAN", variant="CAN", cfg=2, steps=40, seed=i,
                  objects_present=0) for i in range(2)]
    core += [dict(job_id=f"s_CAN__cfg7.5__st20__seed{i}", kind="core", scene_id="s", prompt_id="s_CAN", variant="CAN", cfg=7.5, steps=20, seed=i,
                  objects_present=0) for i in range(2)]
    br = [dict(job_id=f"b{t}{m}{b}", kind="branch", scene_id="s", prompt_id="s_CAN", cfg=7.5, steps=40, seed=b, restart="FULL", t=t, m=m,
               base_job=f"s_CAN__cfg7.5__seed{b}", objects_present=1) for t in GRID for m in range(2) for b in range(2)]
    adf, a0 = RA.prepare(core, br, "FULL")
    row = a0[(a0.scene == "s") & (a0.kind == "presence")].iloc[0]
    assert row.n_seeds == 4 and abs(row.A0 - 0.5) < 1e-12        # seeds [1,1,1,0]: 6 of 12 ordered pairs agree
