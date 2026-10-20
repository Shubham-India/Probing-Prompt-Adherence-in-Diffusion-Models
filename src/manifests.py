"""Executable experiment manifests for the Revision 3 programme. A manifest is the single source of truth for which jobs exist, which worker shard
runs them, which upstream job they depend on, and how much compute they cost. Workers (src/r3_run.py), verification and merging (src/r3_merge.py)
and every count in the documentation are derived from it; nothing is counted by hand.
Job kinds: core (full generation, optional latents/attention), branch (re-noising restart from a saved latent), switch (counterfactual prompt until
step t, canonical prompt afterwards), cag (per-step guidance schedule).
Stages (one Kaggle notebook section each):
  DEV_CORE       DEV non-canonical variants, CF bases for the DEV switch rehearsal, DEV PS family (R2a already holds the DEV canonical images)
  DEV_COMMIT     DEV restart branches at CFG 7.5 (D8, ER30, FULL) from R2a; DEV bases and branch rehearsal at CFG 2 and 15; DEV switch rehearsal
  DEV_CAG_GUARD  CAG with w_lo = 1 and 3 on DEV canonical prompts (needs t50 constants from the DEV analysis)
  TEST_CORE      TEST main benchmark: all variants at CFG 7.5 + the CFG x steps study (latents and attention where needed)
  TEST_COMMIT    TEST restart branches at CFG 2 / 7.5 / 15 and CF-C / CF-R prompt switches (needs TEST_CORE latents)
  TEST_CAG_SIZE  TEST CAG arms + PS family core, its restart branches and CF-S switches (self-contained)
Usage: python src/manifests.py --summary [--constants protocol/protocol_freeze.json]
       python src/manifests.py --stage TEST_COMMIT --shard 0/3 --out manifest.jsonl [--constants ...]"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import branch as B          # noqa: E402  (pure functions; torch is imported lazily inside)
import cag as CAG           # noqa: E402  (schedule functions are torch-free)
import jobs as J            # noqa: E402

STAGES = ["DEV_CORE", "DEV_COMMIT", "DEV_CAG_GUARD", "TEST_CORE", "TEST_COMMIT", "TEST_CAG_SIZE"]
R2A_RUN = "r2a-20261014T130457Z-daef6c4"


def load_cfg(path=None):
    return json.loads(Path(path or ROOT / "configs" / "experiment_r3.json").read_text())


def load_prompts(cfg):
    main = J.load_prompts(ROOT / cfg["benchmark"]["prompts"])
    ps = J.load_prompts(ROOT / cfg["benchmark"]["ps_prompts"])
    for p in main:
        p.setdefault("shard_group", p["scene_id"])
    return main, ps


def stratified(prompts, split, quota):
    """First scenes per category in scene_id order (deterministic, outcome-blind)."""
    can = sorted((p for p in prompts if p["split"] == split and p["variant"] == "CAN"), key=lambda p: p["scene_id"])
    out = []
    for cat, n in quota.items():
        out += [p for p in can if p["category"] == cat][:n]
    return sorted(out, key=lambda p: p["scene_id"])


def commitment_scenes(main, split, cfg):
    if split == "DEV":
        return sorted((p for p in main if p["split"] == "DEV" and p["variant"] == "CAN"), key=lambda p: p["scene_id"])
    return stratified(main, "TEST", cfg["commitment"]["test_quota"])


def core_job(p, cfg_w, steps, seed, experiment, stage, latents=False, attention=False, upstream=None):
    return dict(job_id=J.sample_name(p["id"], cfg_w, steps, seed), kind="core", experiment=experiment, stage=stage, split=p["split"],
                shard_group=p.get("shard_group", p["scene_id"]), scene_id=p["scene_id"], prompt_id=p["id"], prompt=p["prompt"], variant=p["variant"],
                category=p["category"], cfg=cfg_w, steps=steps, seed=seed, save_latents=bool(latents), attention=bool(attention), upstream=upstream,
                unet_calls=steps * (2 if cfg_w != 1 else 1))


def branch_jobs(base, cfg, variants, experiment, stage, M):
    import run_branches as RB
    sch = B.Schedule(base["steps"])
    out = []
    for t in cfg["commitment"]["grid"]:
        for v in variants:
            plan = RB.plan_for(v, sch, t, cfg["commitment"])
            for m in range(M):
                out.append(dict(job_id=f"{base['job_id']}__br_{v}_t{t}_m{m}", kind="branch", experiment=experiment, stage=stage, split=base["split"],
                                shard_group=base["shard_group"], scene_id=base["scene_id"], prompt_id=base["prompt_id"], prompt=base["prompt"],
                                variant=base["variant"], category=base["category"], cfg=base["cfg"], steps=base["steps"], seed=base["seed"], base_job=base["job_id"],
                                base_upstream=base.get("upstream"), t=t, restart=v, m=m, noise_seed=RB.noise_seed(base["job_id"], t, v, m),
                                target_idx=plan["target_idx"], effective_jump=plan["effective_jump"], ratio=plan["ratio"], clipped=plan["clipped"],
                                unet_calls=(base["steps"] - plan["target_idx"]) * (2 if base["cfg"] != 1 else 1)))
    return out


def switch_jobs(cf_base, canonical, cfg, experiment, stage):
    return [dict(job_id=f"{cf_base['job_id']}__sw_t{t}", kind="switch", experiment=experiment, stage=stage, split=cf_base["split"],
                 shard_group=cf_base["shard_group"], scene_id=cf_base["scene_id"], prompt_id=canonical["id"], prompt=canonical["prompt"],
                 cf_prompt_id=cf_base["prompt_id"], cf_prompt=cf_base["prompt"], cf_type=cf_base["variant"], variant="SWITCH", category=cf_base["category"],
                 cfg=cf_base["cfg"], steps=cf_base["steps"], seed=cf_base["seed"], base_job=cf_base["job_id"], t=t,
                 unet_calls=(cf_base["steps"] - t) * 2) for t in cfg["commitment"]["grid"]]


# ---------------------------------------------------------------- CAG schedules (src/cag.py; constants come from the DEV analysis via the freeze file)
def cag_jobs(p, cfg, constants, stage, seeds, w_lo, arms, tag=""):
    c = cfg["cag"]
    types = [x["type"] for x in p["constraints"]]
    s, note = CAG.s_star(types, constants["t50_dev_w7.5"], c["margin"], cfg["steps"])
    sg = constants.get("s_star_global", CAG.s_star_global(constants["t50_dev_w7.5"], c["margin"], cfg["steps"]))
    sch, mean = CAG.schedules(s, c["w_hi"], w_lo, cfg["steps"])
    sch["A3"] = CAG.schedules(sg, c["w_hi"], w_lo, cfg["steps"])[0]["A2"]
    out = []
    for arm in arms:
        w = sch[arm]
        for seed in seeds:
            jid = f"{p['id']}__cag_{arm}{tag}__seed{seed}"
            base = dict(job_id=jid, kind="cag", experiment="E30" if stage.startswith("TEST") else "E30-DEV-guard", stage=stage, split=p["split"],
                        shard_group=p.get("shard_group", p["scene_id"]), scene_id=p["scene_id"], prompt_id=p["id"], prompt=p["prompt"], variant=p["variant"],
                        category=p["category"], steps=cfg["steps"], seed=seed, arm=arm, w_lo=w_lo, s_star=s if arm != "A3" else sg, s_star_note=note)
            if w is None:
                out.append(dict(base, schedule=None, not_estimable="reversed schedule undefined because s* = 40", unet_calls=0))
            else:
                out.append(dict(base, schedule=w, mean_guidance=sum(w) / len(w), max_guidance=max(w), unet_calls=CAG.unet_calls(w)))
    return out


# ---------------------------------------------------------------- the programme
def build(cfg=None, constants=None):
    cfg = cfg or load_cfg()
    main, ps = load_prompts(cfg)
    S, steps, w0 = cfg["seeds_core"], cfg["steps"], cfg["cfg_core"]
    bases_seeds = cfg["seeds_branch_bases"]
    jobs, notes = [], []
    by_scene = defaultdict(list)
    for p in main + ps:
        by_scene[p["scene_id"]].append(p)
    can_of = {sid: next(q for q in v if q["variant"] == "CAN") for sid, v in by_scene.items()}

    # ---------------- DEV ----------------
    dev_can = commitment_scenes(main, "DEV", cfg)
    r2a_bases = [core_job(p, w0, steps, sd, "E04-DEV(R2a)", "R2a", latents=True, upstream=R2A_RUN) for p in dev_can for sd in bases_seeds]
    sw = cfg["switch"]
    dev_cf = {t: [p for p in main if p["split"] == "DEV" and p["variant"] == t] for t in ("CF-C", "CF-R")}
    dev_cf["CF-S"] = [p for p in ps if p["split"] == "DEV" and p["variant"] == "CF-S"]
    dev_cf_sel = {t: sorted(v, key=lambda p: p["scene_id"])[: sw["scenes_per_type"]["DEV"]] for t, v in dev_cf.items()}
    dev_cf_ids = {p["id"] for v in dev_cf_sel.values() for p in v}
    for p in main:
        if p["split"] == "DEV" and p["variant"] != "CAN":
            for sd in S:
                jobs.append(core_job(p, w0, steps, sd, "E04-DEV-variants", "DEV_CORE", latents=(p["id"] in dev_cf_ids and sd in bases_seeds)))
    for p in ps:
        if p["split"] == "DEV":
            seeds = S if p["variant"] == "CAN" else bases_seeds
            for sd in seeds:
                jobs.append(core_job(p, w0, steps, sd, "E32-DEV-core", "DEV_CORE", latents=(p["id"] in dev_cf_ids and sd in bases_seeds)))
    for b in r2a_bases:
        jobs += branch_jobs(b, cfg, cfg["commitment"]["variants_dev"], "E23-DEV", "DEV_COMMIT", cfg["commitment"]["M"])
    reh = {p["scene_id"] for p in dev_can[: cfg["commitment"]["dev_extra_cfg_rehearsal_scenes"]]}
    for w in [x for x in cfg["commitment"]["cfg_levels"] if x != w0]:
        for p in dev_can:
            for sd in bases_seeds:
                b = core_job(p, w, steps, sd, "E29-DEV-base", "DEV_COMMIT", latents=True)
                jobs.append(b)
                if p["scene_id"] in reh:
                    jobs += branch_jobs(b, cfg, cfg["commitment"]["variants_extra_cfg"], "E29-DEV", "DEV_COMMIT", cfg["commitment"]["M"])
    cf_core = {j["prompt_id"] + f"|{j['seed']}": j for j in jobs if j["kind"] == "core" and j["save_latents"] and j["variant"] in ("CF-C", "CF-R", "CF-S")}
    for t, plist in dev_cf_sel.items():
        for p in plist:
            for sd in bases_seeds:
                jobs += switch_jobs(cf_core[f"{p['id']}|{sd}"], can_of[p["scene_id"]], cfg, "E31-DEV", "DEV_COMMIT")
    if constants:
        for p in dev_can:
            for wl in cfg["cag"]["dev_quality_guard"]["w_lo"]:
                jobs += cag_jobs(p, cfg, constants, "DEV_CAG_GUARD", cfg["cag"]["dev_quality_guard"]["seeds"], wl, ["A2"], tag=f"_wlo{wl:g}")
    else:
        notes.append("DEV_CAG_GUARD not buildable yet: needs t50 constants from the DEV analysis (protocol/protocol_freeze.json or dev_constants.json)")

    # ---------------- TEST ----------------
    test_commit = commitment_scenes(main, "TEST", cfg)
    commit_ids = {p["id"] for p in test_commit}
    cs = cfg["cfg_steps_study"]
    cfg_scenes = stratified(main, "TEST", cs["quota"])
    if not commit_ids <= {p["id"] for p in cfg_scenes}:
        raise SystemExit("design error: TEST commitment scenes must be a subset of the CFG x steps scenes (shared E29 bases)")
    test_cf = {t: sorted((p for p in main if p["split"] == "TEST" and p["variant"] == t), key=lambda p: p["scene_id"])[: sw["scenes_per_type"]["TEST"]] for t in ("CF-C", "CF-R")}
    test_cf_ids = {p["id"] for v in test_cf.values() for p in v}
    test_core = {}
    for p in main:
        if p["split"] != "TEST":
            continue
        for sd in S:
            j = core_job(p, w0, steps, sd, "E04-TEST", "TEST_CORE", latents=(sd in bases_seeds and (p["id"] in commit_ids or p["id"] in test_cf_ids)),
                         attention=p["variant"] == "CAN")
            jobs.append(j)
            test_core[j["job_id"]] = j
    for p in cfg_scenes:
        for w in cs["cfg"]:
            for st in cs["steps"]:
                if w == w0 and st == steps:
                    continue                       # shared with E04-TEST
                for sd in cs["seeds"]:
                    jobs.append(core_job(p, w, st, sd, "E11", "TEST_CORE", latents=(st == steps and w in cfg["commitment"]["cfg_levels"] and sd in bases_seeds and p["id"] in commit_ids),
                                         attention=(st == steps)))
    core_by_id = {j["job_id"]: j for j in jobs if j["kind"] == "core"}
    for p in test_commit:
        for w in cfg["commitment"]["cfg_levels"]:
            for sd in bases_seeds:
                b = core_by_id[J.sample_name(p["id"], w, steps, sd)]
                variants = cfg["commitment"]["variants_test"] if w == w0 else cfg["commitment"]["variants_extra_cfg"]
                jobs += branch_jobs(b, cfg, variants, "E23-TEST" if w == w0 else "E29-TEST", "TEST_COMMIT", cfg["commitment"]["M"])
    for t, plist in test_cf.items():
        for p in plist:
            for sd in bases_seeds:
                jobs += switch_jobs(core_by_id[J.sample_name(p["id"], w0, steps, sd)], can_of[p["scene_id"]], cfg, "E31-TEST", "TEST_COMMIT")
    pf = cfg["ps_family"]
    for p in ps:
        if p["split"] != "TEST":
            continue
        is_branch_base = p["variant"] == "CAN" and p["ps_subset"] in pf["branch_subsets"]
        is_cf = p["variant"] in ("CF-C", "CF-R", "CF-S")
        for sd in (pf["seeds"] if p["variant"] == "CAN" else bases_seeds):
            j = core_job(p, w0, steps, sd, "E32-TEST-core", "TEST_CAG_SIZE", latents=(sd in pf["branch_seeds"] and (is_branch_base or is_cf)), attention=p["variant"] == "CAN")
            jobs.append(j)
            if is_branch_base and sd in pf["branch_seeds"]:
                jobs += branch_jobs(j, cfg, pf["branch_variants"], "E32-TEST-branch", "TEST_CAG_SIZE", cfg["commitment"]["M"])
            if p["variant"] == "CF-S" and sd in bases_seeds:
                jobs += switch_jobs(j, can_of[p["scene_id"]], cfg, "E31-TEST", "TEST_CAG_SIZE")
    if constants:
        for p in main:
            if p["split"] == "TEST" and p["variant"] == "CAN":
                jobs += cag_jobs(p, cfg, constants, "TEST_CAG_SIZE", cfg["cag"]["seeds"], constants.get("w_lo", cfg["cag"]["w_lo_primary"]), cfg["cag"]["arms"])
    else:
        notes.append("TEST CAG arms (E30) not buildable yet: s* needs the frozen DEV t50 constants")
    ids = [j["job_id"] for j in jobs]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate job ids in manifest: {dup[:5]}")
    return jobs, notes, dict(n_cag_prompts_test=sum(1 for p in main if p["split"] == "TEST" and p["variant"] == "CAN"),
                             n_dev_can=len(dev_can), cag_arms_test=len(cfg["cag"]["arms"]), cag_seeds_test=len(cfg["cag"]["seeds"]),
                             dev_guard=len(dev_can) * len(cfg["cag"]["dev_quality_guard"]["w_lo"]) * len(cfg["cag"]["dev_quality_guard"]["seeds"]))


def shard_of_groups(jobs, n):
    """Scene-group sharding: groups in sorted order are dealt round-robin, so every variant, seed and branch of a scene stays in one shard."""
    groups = sorted({j["shard_group"] for j in jobs})
    return {g: i % n for i, g in enumerate(groups)}


def select(jobs, stage, shard=None):
    """Jobs of one stage, optionally restricted to shard i of n. Groups are dealt over the WHOLE programme (all stages), so a scene's TEST core
    images, its branches and its switches always land in the same shard number."""
    st = [j for j in jobs if j["stage"] == stage]
    if shard:
        i, n = shard
        m = shard_of_groups(jobs, n)
        st = [dict(j, shard=f"{i}/{n}") for j in st if m[j["shard_group"]] == i]
    return st


def manifest_hash(jobs):
    return hashlib.sha256("\n".join(json.dumps(j, sort_keys=True) for j in sorted(jobs, key=lambda j: j["job_id"])).encode()).hexdigest()


def summary(jobs, extra):
    rows = defaultdict(lambda: dict(jobs=0, unet_calls=0))
    for j in jobs:
        r = rows[(j["stage"], j["experiment"], j["kind"])]
        r["jobs"] += 1
        r["unet_calls"] += j["unet_calls"]
    table = [dict(stage=s, experiment=e, kind=k, jobs=v["jobs"], image_equivalents=round(v["unet_calls"] / 80, 1)) for (s, e, k), v in sorted(rows.items())]
    tot = lambda f: round(sum(r["image_equivalents"] for r in table if f(r)), 1)
    out = dict(rows=table, totals=dict(all=tot(lambda r: True), dev=tot(lambda r: r["stage"].startswith("DEV")), test=tot(lambda r: r["stage"].startswith("TEST"))))
    if extra:
        n = extra["n_cag_prompts_test"] * extra["cag_arms_test"] * extra["cag_seeds_test"]
        out["cag_not_yet_built"] = dict(test_jobs=n, image_equivalents_range=[round(n * (2 * 6 + 34) / 80, 1), n * 1.0],
                                         dev_guard_jobs=extra["dev_guard"],
                                         note="exact cost depends on s* (DEV-derived); bounds assume s* between 6 and 40")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--constants", default=None, help="JSON with t50_dev_w7.5, s_star_global and optionally w_lo (from the DEV analysis / freeze)")
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    constants = json.loads(Path(a.constants).read_text()) if a.constants else None
    if constants and "constants" in constants:
        constants = constants["constants"]
    jobs, notes, extra = build(cfg, constants)
    if a.summary or not a.stage:
        print(json.dumps(summary(jobs, None if constants else extra), indent=1))
        for n in notes:
            print("NOTE:", n)
    if a.stage:
        sel = select(jobs, a.stage, J.parse_shard(a.shard) if a.shard else None)
        if a.out:
            Path(a.out).write_text("".join(json.dumps(j) + "\n" for j in sel))
        print(json.dumps(dict(stage=a.stage, shard=a.shard, jobs=len(sel), image_equivalents=round(sum(j["unet_calls"] for j in sel) / 80, 1),
                              manifest_sha256=manifest_hash(sel))))
