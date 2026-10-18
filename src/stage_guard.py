"""Pre-flight guard for every expensive stage (Kaggle or local GPU). Run it FIRST; run the stage only if it exits 0.
It never generates anything. It (1) checks the owner-controlled gate flags in configs/stage_gates.json, (2) checks code/config provenance,
(3) verifies the input artifacts the stage consumes, and (4) prints the independently derived job count, disk and wall-time range.
A stage whose gates are false is refused even if the code could run it; flags are set to true ONLY by the owner, after the evidence exists.
Usage: python src/stage_guard.py --stage {r2b,r3,e22,r4} [--core-run DIR] [--config configs/experiment_v2.json] [--prompts data/benchmark_v2/prompts_v2.json]
                                  [--variants D8 ER30 FULL] [--M 2] [--expect-commit SHA] [--bundle-dir DIR] [--gates configs/stage_gates.json]
Exit code 0 = stage may start; 1 = blocked (reasons printed as JSON)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# measured evidence behind the time range (docs/R0_R1_AUDIT.md): seconds per full 40-step image on one T4, two workers running concurrently
RATE_LOW_S, RATE_HIGH_S = 6.2, 7.8
PNG_MB, LATENT_MB = 0.462, 0.200            # R2a averages (400 images)

REQUIRED_GATES = {
    "r2b": ["variant_review_complete"],
    "r3": ["evaluator_audit_complete", "d13_resolved_for_r3", "r3_authorised"],
    "e22": ["synthetic_probe_threshold_approved"],
    "r4": ["evaluator_audit_complete", "variant_review_complete", "addendum_b_signed"],
    # Revision 3 stages (src/manifests.py); TEST stages additionally require protocol/protocol_freeze.json to verify (src/protocol_r3.py)
    "DEV_CORE": ["variant_review_complete"],
    "DEV_COMMIT": ["evaluator_audit_complete", "d13_resolved_for_r3", "r3_authorised"],
    "DEV_CAG_GUARD": ["evaluator_audit_complete", "r3_authorised"],
    "TEST_CORE": ["evaluator_audit_complete", "variant_review_complete"],
    "TEST_COMMIT": ["evaluator_audit_complete", "variant_review_complete"],
    "TEST_CAG_SIZE": ["evaluator_audit_complete", "variant_review_complete"],
}
R3_STAGES = ("DEV_CORE", "DEV_COMMIT", "DEV_CAG_GUARD", "TEST_CORE", "TEST_COMMIT", "TEST_CAG_SIZE")


def plan_stage_r3(stage, shard, constants_path):
    import manifests as M
    import jobs as J
    constants = json.loads(Path(constants_path).read_text()) if constants_path else None
    if constants and "constants" in constants:
        constants = constants["constants"]
    jobs, notes, _ = M.build(constants=constants)
    sel = M.select(jobs, stage, J.parse_shard(shard) if shard else None)
    eq = sum(j["unet_calls"] for j in sel) / 80
    lat = sum(1 for j in sel if j.get("save_latents"))
    return dict(jobs=len(sel), image_equivalents=round(eq, 1), latent_files=lat, disk_gb=round((len(sel) * PNG_MB + lat * LATENT_MB) / 1000, 2),
                wall_hours_2_gpus=wall_hours(eq), wall_hours_with_25pct_allowance=[round(h * 1.25 + 0.1, 2) for h in wall_hours(eq)],
                manifest_sha256=M.manifest_hash(sel), notes=notes)


def load_gates(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def wall_hours(image_equivalents, workers=2):
    return [round(image_equivalents * r / workers / 3600, 2) for r in (RATE_LOW_S, RATE_HIGH_S)]


def plan_r3(core_recs, cfg, variants, M, bases=None):
    import branch as B
    import run_branches as R
    sch = B.Schedule(cfg["steps"][0])
    base_seeds = cfg["seeds_branch_bases"] if bases is None else list(range(bases))
    jobs = R.enumerate_branch_jobs(core_recs, cfg["restart"]["grid"], variants, M, base_seeds)
    eq = sum((cfg["steps"][0] - R.plan_for(v, sch, t, cfg["restart"])["target_idx"]) / cfg["steps"][0] for _, t, v, _ in jobs)
    return dict(jobs=len(jobs), image_equivalents=round(eq, 1), base_seeds=base_seeds, disk_gb=round(len(jobs) * PNG_MB / 1000, 2),
                wall_hours_2_workers=wall_hours(eq), wall_hours_with_25pct_allowance=[round(h * 1.25 + 0.1, 2) for h in wall_hours(eq)])


def plan_r2b(prompts, cfg):
    sel = [p for p in prompts if p.get("split") == "DEV" and p.get("variant") != "CAN"]       # CAN was generated in R2a: never repeat it
    jobs = len(sel) * len(cfg["cfg_scales"]) * len(cfg["steps"]) * len(cfg["seeds"])
    return dict(jobs=jobs, prompts=len(sel), image_equivalents=jobs, disk_gb=round(jobs * PNG_MB / 1000, 2), wall_hours_2_workers=wall_hours(jobs),
                note="DEV non-canonical variants only; the runbook row that said 470 prompts x 4 = 1,880 images would have regenerated the 400 canonical images of R2a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(REQUIRED_GATES))
    ap.add_argument("--core-run", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment_v2.json"))
    ap.add_argument("--prompts", default=str(ROOT / "data" / "benchmark_v2" / "prompts_v2.json"))
    ap.add_argument("--variants", nargs="+", default=["D8", "ER30", "FULL"])
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--bases", type=int, default=None, help="override the number of base seeds (planning only; the config decides the real run)")
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--bundle-dir", default=None, help="dir containing COMMIT (the mounted private bundle)")
    ap.add_argument("--gates", default=str(ROOT / "configs" / "stage_gates.json"))
    ap.add_argument("--shard", default=None, help="Revision 3 stages: i/n")
    ap.add_argument("--constants", default=None, help="Revision 3: DEV constants or protocol file (CAG stages)")
    ap.add_argument("--protocol", default=str(ROOT / "protocol" / "protocol_freeze.json"))
    a = ap.parse_args()
    if a.stage in R3_STAGES and a.config.endswith("experiment_v2.json"):
        a.config = str(ROOT / "configs" / "experiment_r3.json")
    import re
    import jobs as J
    blockers, plan = [], {}
    gates = load_gates(a.gates)
    for g in REQUIRED_GATES[a.stage]:
        if gates.get(g) is not True:
            blockers.append(f"gate '{g}' is not true in {a.gates} (set only by the owner after the evidence exists)")
    cfg = json.loads(Path(a.config).read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("revision"))):
        blockers.append("model revision in the config is not a pinned 40-hex commit")
    if a.bundle_dir:
        c = Path(a.bundle_dir) / "COMMIT"
        if not c.exists():
            blockers.append(f"{c} missing: not a bundle made by kaggle/make_bundle.py")
        elif a.expect_commit and c.read_text().strip() != a.expect_commit:
            blockers.append(f"bundle commit {c.read_text().strip()} differs from expected {a.expect_commit}")
    prompts = J.load_prompts(a.prompts)
    if a.stage in R3_STAGES:
        if a.stage.startswith("TEST"):
            import protocol_r3
            blockers += [f"protocol: {p}" for p in protocol_r3.verify(a.protocol)]
        if a.stage in ("DEV_CAG_GUARD", "TEST_CAG_SIZE") and not a.constants:
            blockers.append("--constants (DEV constants / frozen protocol) required for CAG schedules")
        if a.stage == "DEV_COMMIT" and a.core_run:
            import verify_outputs
            v2 = json.loads((ROOT / "configs" / "experiment_v2.json").read_text())
            dev_can = [p for p in prompts if p.get("split") == "DEV" and p.get("variant") == "CAN"]
            errs, stats = verify_outputs.verify(Path(a.core_run), v2, dev_can, need_latents=True)
            blockers += [f"R2a core run: {e}" for e in errs[:10]]
        plan = plan_stage_r3(a.stage, a.shard, a.constants)
    elif a.stage == "r4":
        import freeze
        problems = freeze.verify(ROOT)
        blockers += [f"freeze: {p}" for p in problems]
    elif a.stage == "r2b":
        plan = plan_r2b(prompts, cfg)
    elif a.stage == "e22":
        plan = dict(jobs=200, note="synthetic scenes judged by J1 (detector inference); python src/synth_probe.py ... --n 200")
    elif a.stage == "r3":
        if not a.core_run:
            blockers.append("--core-run is required for r3")
        else:
            import verify_outputs
            core = Path(a.core_run)
            dev_can = [p for p in prompts if p.get("split") == "DEV" and p.get("variant") == "CAN"]
            errs, stats = verify_outputs.verify(core, cfg, dev_can, need_latents=True)
            plan["core_run_check"] = stats
            blockers += [f"core run: {e}" for e in errs[:10]]
            if not errs:
                recs = [json.loads(l) for l in (core / "samples.jsonl").read_text().splitlines() if l.strip()]
                plan.update(plan_r3(recs, cfg, a.variants, a.M, a.bases))
    out = dict(stage=a.stage, ok=not blockers, blockers=blockers, plan=plan, gates_checked={g: gates.get(g) for g in REQUIRED_GATES[a.stage]})
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
