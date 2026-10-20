# %% [markdown]
# # 06 — PARTIAL TEST analysis (EXPLORATORY) from the verified TEST shards that exist
# **What it does.** Uses ONLY existing, verified TEST archives made by the frozen source (no image generation). For every stage it merges the shards
# that are present against THEIR OWN manifests (strict: a present shard must be complete, uncorrupted and match the frozen manifest), judges every
# image once (J1 + DINOv2; both GPUs), and runs the frozen analysis functions (HA, HD, HE, HF, Holm, pre-TEST classifications, S2) on the covered
# scenes. Every output is labelled **PARTIAL TEST — EXPLORATORY**: missing shards are listed, never imputed; it is NOT the registered confirmatory
# analysis (that needs all nine shards and notebook 04).
# **Inputs (Add Input):** `gbc-source` (the frozen version, commit 40ade4d), `gbc-benchmark`, and every TEST shard dataset you have
# (`gbc-test-core-s*`, `gbc-test-commit-s*`, `gbc-test-cagsize-s*`). Duplicates are detected and ignored. **Settings:** GPU T4 x2, Internet ON.
# **Output:** `/kaggle/working/archive_<RUN_ID>/` -> private dataset `gbc-test-partial`: partial_results.json, coverage.json, eval_rows.jsonl,
# dino_embeddings.npz, merge reports.
# %%  CONFIG — edit only this cell
RUN_TAG = "r3-test-partial"
PART_GB = 4.0
N_BOOT = 10000             # registered bootstrap draws
import os
DRY_RUN = os.environ.get("GBC_DRYRUN") == "1"
LOCAL = os.environ.get("GBC_LOCAL") == "1"     # local machine: real judge, no pip changes (bootstrap only checks imports)

# %%  1. Locate and verify the frozen source package and the benchmark
import json, shutil, subprocess, sys, time
from pathlib import Path
INPUT_ROOTS = [Path(p) for p in os.environ.get("GBC_INPUT_ROOTS", "/kaggle/input").split(os.pathsep)]
WORK = Path(os.environ.get("GBC_WORKING", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)

def _find(marker):
    hits = []
    for root in INPUT_ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            if len(Path(dirpath).relative_to(root).parts) > 6:
                dirnames[:] = []
            elif marker in filenames:
                hits.append(Path(dirpath))
    return sorted(set(hits))

src_hits = _find("SOURCE_MANIFEST.json")
assert len(src_hits) == 1, f"attach exactly one gbc-source version, found {src_hits}"
SRC = src_hits[0]
sys.path[:0] = [str(SRC / "src"), str(SRC / "kaggle")]
import nbtools
assert not nbtools.verify_manifest(SRC, "SOURCE_MANIFEST.json"), "source package corrupt"
SOURCE = json.loads((SRC / "SOURCE_MANIFEST.json").read_text())
assert SOURCE.get("dirty") is False or DRY_RUN, "source package is dirty"
bench = nbtools.find_marker(INPUT_ROOTS, "BENCHMARK_MANIFEST.json")
assert bench and not nbtools.verify_manifest(bench[0], "BENCHMARK_MANIFEST.json"), "benchmark dataset missing or corrupt"
import protocol_r3
PROTOCOL = SRC / "protocol" / "protocol_freeze.json"
pv = protocol_r3.verify(PROTOCOL)
assert not pv or DRY_RUN, f"frozen protocol does not verify in this source package: {pv[:3]}"
FROZEN = json.loads(PROTOCOL.read_text())
CONSTANTS = FROZEN["constants"]
RUN_ID = f"{RUN_TAG}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{str(SOURCE.get('git_head'))[:7]}"
OUT = WORK / f"final_{RUN_ID}"
OUT.mkdir(parents=True, exist_ok=True)
print("source", SOURCE.get("git_head"), "| protocol", "VERIFIED" if not pv else pv[:1], "| w_lo", CONSTANTS.get("w_lo"))

# %%  2. Dependency bootstrap (same as every notebook)
import env_bootstrap
BOOT = env_bootstrap.bootstrap(SRC / "requirements-kaggle.txt", WORK / "_bootstrap", dry_run=DRY_RUN or LOCAL)
(OUT / "env_bootstrap.json").write_text((WORK / "_bootstrap" / "env_bootstrap.json").read_text())

# %%  3. Find the verified TEST archives (made by the frozen source), drop duplicates, extract and verify every file
STAGES = ["TEST_CORE", "TEST_COMMIT", "TEST_CAG_SIZE"]
found, skipped = {}, []
for arch in nbtools.find_marker(INPUT_ROOTS, "ARCHIVE_MANIFEST.json"):
    am = json.loads((arch / "ARCHIVE_MANIFEST.json").read_text())
    if am.get("stage") not in STAGES:
        continue
    why = None
    if not (nbtools.archive_is_verified(am) or DRY_RUN):
        why = "not verified"
    elif am.get("source_commit") != SOURCE.get("git_head") and not DRY_RUN:
        why = f"made by source {str(am.get('source_commit'))[:7]}, not the frozen {str(SOURCE.get('git_head'))[:7]}"
    if why:
        skipped.append(dict(archive=str(arch), run_id=am.get("run_id"), reason=why))
        continue
    key = (am["stage"], am["shard"])
    if key in found:
        if found[key][1]["run_id"] != am["run_id"]:
            raise SystemExit(f"two different runs for {key}: {found[key][1]['run_id']} and {am['run_id']}; attach only one")
        skipped.append(dict(archive=str(arch), run_id=am["run_id"], reason="duplicate of an attached archive"))
        continue
    found[key] = (arch, am)
EXTRACTED = {}
for (st, sh), (arch, am) in sorted(found.items()):
    dest, probs = nbtools.extract_and_verify(arch, WORK / "merge_in" / am["run_id"])
    assert not probs, f"{arch}: {probs[:5]}"
    EXTRACTED[(st, sh)] = dest
SHARDS = {st: sorted(int(sh.split("/")[0]) for (s, sh) in EXTRACTED if s == st) for st in STAGES}
print("present shards:", SHARDS, "| skipped:", skipped)

# %%  4. Strict merge of each present shard against the frozen manifest of THAT shard (no --allow-missing)
import manifests as M
JOBS, _, _ = M.build(constants=CONSTANTS)
assert {st: M.manifest_hash(M.select(JOBS, st)) for st in STAGES} == FROZEN["test_manifests"] or DRY_RUN, "rebuilt TEST manifests differ from the frozen ones"
MERGED, COVER = [], {}
for st in STAGES:
    if not SHARDS[st]:
        COVER[st] = dict(present=[], missing=[0, 1, 2], merged=0, manifest_jobs=0)
        continue
    sel = [j for i in SHARDS[st] for j in M.select(JOBS, st, (i, 3))]
    if DRY_RUN:
        sel = [json.loads(l) for (s, sh), dest in EXTRACTED.items() if s == st for l in (dest / "manifest.jsonl").read_text().splitlines() if l.strip()]
    mf = OUT / f"manifest_{st}_present_shards.jsonl"
    mf.write_text("".join(json.dumps(j) + "\n" for j in sel))
    shutil.rmtree(OUT / "merged" / st, ignore_errors=True)
    cmd = [sys.executable, str(SRC / "src" / "r3_merge.py"), "--manifest", str(mf), "--inputs", *[str(d) for (s, _), d in EXTRACTED.items() if s == st], "--out", str(OUT / "merged" / st)]
    if DRY_RUN:
        cmd += ["--allow-fake", "--allow-missing"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rep, err = nbtools.load_json_object(OUT / "merged" / st / "merge_report.json")
    ok, why = nbtools.stage_verdict({"merge": 0}, r.returncode, rep, dry_run=DRY_RUN)
    if not ok and not DRY_RUN:
        raise SystemExit(f"{st}: present shards {SHARDS[st]} failed the strict merge: {why[:5]}")
    COVER[st] = dict(present=SHARDS[st], missing=[i for i in range(3) if i not in SHARDS[st]], merged=rep["merged"], manifest_jobs=rep["manifest_jobs"],
                     scenes=len({j["scene_id"] for j in sel}))
    MERGED.append(OUT / "merged" / st / "merged_records.jsonl")
ALL = OUT / "merged" / "all_records.jsonl"
ALL.write_text("".join(Path(p).read_text() for p in MERGED))
print(json.dumps(COVER, indent=1))

# %%  5. Judge every image once (J1 + DINOv2), one process per GPU
recs = [l for l in ALL.read_text().splitlines() if l.strip()]
ngpu = 1 if DRY_RUN else max(1, nbtools.gpu_count())
procs, outs = [], []
for g in range(ngpu):
    part = OUT / f"records_part{g}.jsonl"
    part.write_text("".join(x + "\n" for x in recs[g::ngpu]))
    ev, emb = OUT / f"eval_part{g}.jsonl", OUT / f"dino_part{g}.npz"
    cmd = [sys.executable, str(SRC / "src" / "r3_evaluate.py"), "--records", str(part), "--out", str(ev), "--diversity", str(emb)] + (["--fake-judge"] if DRY_RUN else [])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)) if not DRY_RUN else dict(os.environ)
    procs.append(subprocess.Popen(cmd, env=env, stdout=open(OUT / f"evaluate_{g}.log", "w"), stderr=subprocess.STDOUT))
    outs.append((ev, emb))
codes = [p.wait() for p in procs]
assert all(c == 0 for c in codes), f"judging failed: exit codes {codes} (see evaluate_*.log)"
import numpy as np
EVAL, EMB = OUT / "eval_rows.jsonl", OUT / "dino_embeddings.npz"
EVAL.write_text("".join(Path(e).read_text() for e, _ in outs))
ids, embs = [], []
for _, e in outs:
    if Path(e).exists():
        z = np.load(e)
        ids += [str(x) for x in z["ids"]]
        embs.append(z["emb"])
if embs:
    np.savez_compressed(EMB, ids=np.array(ids), emb=np.concatenate(embs))
print("judged rows:", sum(1 for _ in open(EVAL)), "| embeddings:", len(ids))

# %%  6. Frozen analysis functions on the covered scenes, labelled PARTIAL / EXPLORATORY
import r3_analyze as RA
import dedup as D
rows = RA.read(EVAL)
rows, dlog = D.dedup(rows)
core = [r for r in rows if r.get("kind", "core") == "core"]
br = [r for r in rows if r.get("kind") == "branch"]
cag = [r for r in rows if r.get("kind") == "cag"]
recmap = {json.loads(l)["job_id"]: json.loads(l) for l in recs}
att = [dict(r, path=p) for r in core for rel, p in recmap.get(r["job_id"], {}).get("abs_files", {}).items() if rel.startswith("attention/")]
cfg = json.loads((SRC / "configs" / "experiment_r3.json").read_text())
prompts = {p["id"]: p for f in (cfg["benchmark"]["prompts"], cfg["benchmark"]["ps_prompts"]) for p in json.loads((SRC / f).read_text())}
emb = (ids, np.concatenate(embs)) if embs else None
c75 = [r for r in core if float(r["cfg"]) == 7.5]
b75 = [r for r in br if float(r["cfg"]) == 7.5]
res = {"HA": RA.analyse_ha(c75, b75, N_BOOT), "HD": RA.analyse_hd(core, br, N_BOOT), "HE": RA.analyse_he(core, cag, emb, N_BOOT),
       "HF": RA.analyse_hf(c75, b75, RA.t_stab_table(att, prompts), N_BOOT) if att else dict(decision="not estimable (no attention maps)", p=None)}
res = RA.finalise(res)
for k, v in res.items():
    v["final"] = "PARTIAL TEST, EXPLORATORY (not the registered confirmatory test): " + str(v["final"])
he_scenes = sorted({r["scene_id"] for r in cag} & {r["scene_id"] for r in core if r.get("variant") == "CAN"})
COVER["hypotheses"] = dict(HA_HD_HF_scenes=len({r["scene_id"] for r in b75}), HE_scenes_with_A1_and_CAG=len(he_scenes), attention_maps=len(att))
out = dict(status="PARTIAL TEST (EXPLORATORY): verified shards only; missing shards listed in coverage; Holm shown over the registered family for completeness",
           protocol=dict(verified=not pv, freeze_sha256=FROZEN.get("freeze_sha256"), source_commit=SOURCE.get("git_head")), coverage=COVER, skipped_archives=skipped,
           results=RA.clean(res), supporting=dict(S2=RA.clean(RA.analyse_s2(core))),
           dedup=dict(dropped_scenes=dlog["dropped_scenes"], n_dropped_generations=len(dlog["dropped_generations"]), n_dropped_dependents=len(dlog["dropped_dependents"])))
(OUT / "partial_results.json").write_text(json.dumps(out, indent=1, default=str))
(OUT / "coverage.json").write_text(json.dumps(COVER, indent=1))
print(json.dumps({k: v["final"] for k, v in res.items()}, indent=1))

# %%  7. Package (integrity verified; content PARTIAL)
for p in OUT.glob("records_part*.jsonl"):
    p.unlink()
man = nbtools.package(OUT, WORK / f"archive_{RUN_ID}", RUN_ID, PART_GB, meta=dict(stage="PARTIAL_TEST", verified=not DRY_RUN, dry_run=DRY_RUN, partial=True,
                      coverage=COVER, source_commit=SOURCE.get("git_head"), source_dirty=SOURCE.get("dirty")))
print(f"DONE run={RUN_ID} archive=archive_{RUN_ID} -> save as private dataset gbc-test-partial")
