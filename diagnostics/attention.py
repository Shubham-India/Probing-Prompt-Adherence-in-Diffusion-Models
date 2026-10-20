"""Cross-attention analyses of the partial TEST split (EXPLORATORY, post hoc; no image is generated or re-judged).
Inputs: the saved per-step cross-attention maps of canonical TEST_CORE prompts (40 DDIM steps; w in {2, 5, 7.5, 10, 15}; 4 seeds; one map per
constraint word, heads and layers averaged within the 16x16 and 32x32 groups), the judged rows, and the benchmark prompt specifications.
Analyses (16x16 maps; each word's map normalised to unit mass at every step):
  1. Object level, w = 7.5: for every presence constraint, compare its noun's attention between seeds where the judge found the object and seeds
     where it did not, WITHIN the same scene and object (pairs with both outcomes), per step: localisation (mass in the top 10% of cells) and
     overlap with the other objects' nouns (histogram intersection). Scene-bootstrap 95% intervals.
  2. Guidance: the same metrics at w = 2, 7.5, 15 on identical initial noise (same scene, seed and object), paired differences.
  3. Attention stabilisation (registered t_stab definition, attn_stability) per constraint type and guidance scale, against the registered t50.
  4. Two illustrative panels chosen by a fixed rule stated below (no outcome-based or appearance-based choice).
Usage: python diagnostics/attention.py --attention DIR [DIR ...] --rows eval_rows.jsonl --partial partial_results.json [--images-zip Z ...]
Writes reports/diagnostics/attention.json, paper/figures/attn_*.pdf, paper/tables/attn_numbers.tex."""
import argparse
import glob
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import attn_stability as AS  # noqa: E402
import r3_analyze as RA  # noqa: E402

FIG, OUT, TAB = ROOT / "paper" / "figures", ROOT / "reports" / "diagnostics", ROOT / "paper" / "tables"
NB = 2000
TOPQ = 0.10


def rng():
    return np.random.default_rng(20261021)


def norm_maps(m):
    m = np.clip(m.astype(np.float64), 0, None)
    s = m.reshape(m.shape[0], -1).sum(1)[:, None, None]
    return m / np.where(s > 0, s, 1)


def localisation(m):
    f = m.reshape(m.shape[0], -1)
    k = max(1, int(round(TOPQ * f.shape[1])))
    return np.sort(f, axis=1)[:, -k:].sum(1)


def overlap(a, b):
    return np.minimum(a, b).reshape(a.shape[0], -1).sum(1)


def boot_ci(x, g, axis_len=None):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, (np.nan, np.nan)
    d = [x[g.integers(0, len(x), len(x))].mean(0) for _ in range(NB)]
    return x.mean(0), (np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention", nargs="+", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--partial", required=True)
    ap.add_argument("--images-zip", nargs="*", default=[])
    a = ap.parse_args()
    cfg = json.loads((ROOT / "configs" / "experiment_r3.json").read_text())
    prompts = {p["id"]: p for f in (cfg["benchmark"]["prompts"], cfg["benchmark"]["ps_prompts"]) for p in json.loads((ROOT / f).read_text())}
    rows = {}
    for line in open(a.rows, encoding="utf-8"):
        r = json.loads(line)
        rows[r["job_id"]] = r
    files = {Path(p).stem: p for d in a.attention for p in glob.glob(str(Path(d) / "**" / "*.npz"), recursive=True)}
    inv = defaultdict(int)
    for j in files:
        r = rows.get(j)
        if r:
            inv[(float(r["cfg"]), int(r["steps"]))] += 1
    R = dict(status="EXPLORATORY attention analyses (post hoc)", n_files=len(files), n_matched=sum(inv.values()),
             inventory={f"w={w:g}|steps={s}": n for (w, s), n in sorted(inv.items())},
             scenes=len({rows[j]["scene_id"] for j in files if j in rows}))

    # ---------------- per (image, object) metrics
    rec = []
    for j, path in files.items():
        r = rows.get(j)
        if r is None or r.get("variant") != "CAN":
            continue
        p = prompts[r["prompt_id"]]
        z = np.load(path)
        words = [str(w) for w in z["words"]]
        m = z["maps_16"]
        maps = {w: norm_maps(m[:, i]) for i, w in enumerate(words)}
        cr = r.get("constraint_results") or []
        objs = [(c["object"], cr[i]) for i, c in enumerate(p["constraints"]) if c["type"] == "presence" and i < len(cr) and c["object"] in maps]
        nouns = [o for o, _ in objs]
        for o, y in objs:
            others = [maps[x] for x in nouns if x != o]
            rec.append(dict(job=j, scene=r["scene_id"], seed=int(r["seed"]), w=float(r["cfg"]), obj=o, y=int(y), n_obj=len(nouns),
                            loc=localisation(maps[o]), ovl=np.max([overlap(maps[o], q) for q in others], axis=0) if others else None))
    R["object_records"] = len(rec)

    # 1. within-scene detected vs missed, w = 7.5
    out1 = {}
    for metric in ("loc", "ovl"):
        grp = defaultdict(lambda: {0: [], 1: []})
        for x in rec:
            if x["w"] == 7.5 and x[metric] is not None:
                grp[(x["scene"], x["obj"])][x["y"]].append(x[metric])
        diffs, scenes = [], []
        for (s, o), d in grp.items():
            if d[0] and d[1]:
                diffs.append(np.mean(d[1], 0) - np.mean(d[0], 0))
                scenes.append(s)
        # bootstrap over scenes (pairs of one scene stay together)
        by_scene = defaultdict(list)
        for s, dv in zip(scenes, diffs):
            by_scene[s].append(dv)
        S = list(by_scene)
        per = np.array([np.mean(by_scene[s], 0) for s in S]) if S else np.zeros((0, 40))
        g = rng()
        d = [per[g.integers(0, len(per), len(per))].mean(0) for _ in range(NB)] if len(per) else []
        lo, hi = (np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)) if d else (None, None)
        first = next((i + 1 for i in range(40) if lo is not None and (lo[i] > 0 or hi[i] < 0)), None)
        out1[metric] = dict(n_pairs=len(diffs), n_scenes=len(S), mean=per.mean(0).tolist() if len(per) else None,
                            lo=lo.tolist() if lo is not None else None, hi=hi.tolist() if hi is not None else None, first_step_ci_excludes_0=first)
    R["detected_vs_missed_w7.5"] = out1

    # 2. guidance on identical noise: paired by (scene, seed, object)
    out2 = {}
    key = defaultdict(dict)
    for x in rec:
        key[(x["scene"], x["seed"], x["obj"])][x["w"]] = x
    for metric in ("loc", "ovl"):
        res = {}
        for w in (2.0, 5.0, 7.5, 10.0, 15.0):
            vals = defaultdict(list)
            for k, d in key.items():
                if w in d and d[w][metric] is not None:
                    vals[k[0]].append(d[w][metric])
            per = np.array([np.mean(v, 0) for v in vals.values()])
            if len(per):
                g = rng()
                bs = [per[g.integers(0, len(per), len(per))].mean(0) for _ in range(NB)]
                res[str(w)] = dict(mean=per.mean(0).tolist(), lo=np.percentile(bs, 2.5, 0).tolist(), hi=np.percentile(bs, 97.5, 0).tolist(), n_scenes=len(per))
        pd_ = defaultdict(list)
        for k, d in key.items():
            if 2.0 in d and 15.0 in d and d[2.0][metric] is not None:
                pd_[k[0]].append(d[15.0][metric] - d[2.0][metric])
        per = np.array([np.mean(v, 0) for v in pd_.values()])
        g = rng()
        bs = [per[g.integers(0, len(per), len(per))].mean(0) for _ in range(NB)]
        res["w15_minus_w2"] = dict(mean=per.mean(0).tolist(), lo=np.percentile(bs, 2.5, 0).tolist(), hi=np.percentile(bs, 97.5, 0).tolist(), n_scenes=len(per))
        out2[metric] = res
    R["guidance"] = out2
    # presence rate on the same images (for the caption)
    pr = defaultdict(list)
    for x in rec:
        pr[x["w"]].append(x["y"])
    R["object_detection_rate_by_w"] = {str(w): float(np.mean(v)) for w, v in sorted(pr.items())}

    # 3. t_stab per type and guidance (registered definition), and against registered t50
    import pandas as pd  # noqa: F401
    att_rows = [dict(job_id=j, scene_id=rows[j]["scene_id"], prompt_id=rows[j]["prompt_id"], seed=rows[j]["seed"], path=p, w=float(rows[j]["cfg"]))
                for j, p in files.items() if j in rows and rows[j].get("variant") == "CAN"]
    ts_by_w = {}
    for w in (2.0, 7.5, 15.0):
        tt = RA.t_stab_table([r for r in att_rows if r["w"] == w], prompts)
        res = {}
        for k in sorted(set(tt.kind)):
            sc = tt[tt.kind == k].groupby("scene").t_stab.mean().to_numpy()
            g = rng()
            bs = [sc[g.integers(0, len(sc), len(sc))].mean() for _ in range(NB)]
            res[k] = dict(mean=float(sc.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_scenes=int(len(sc)),
                          sd_scene=float(sc.std(ddof=1)) if len(sc) > 1 else None)
        ts_by_w[str(w)] = res
    R["t_stab"] = ts_by_w
    tt2, tt15 = (RA.t_stab_table([r for r in att_rows if r["w"] == w], prompts) for w in (2.0, 15.0))
    pdiff = {}
    for k in sorted(set(tt2.kind) & set(tt15.kind)):
        a2 = tt2[tt2.kind == k].groupby(["scene", "seed"]).t_stab.mean()
        a15 = tt15[tt15.kind == k].groupby(["scene", "seed"]).t_stab.mean()
        d = (a15 - a2).dropna().groupby(level=0).mean().to_numpy()
        g = rng()
        bs = [d[g.integers(0, len(d), len(d))].mean() for _ in range(NB)]
        pdiff[k] = dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_scenes=int(len(d)))
    R["t_stab_w15_minus_w2_paired"] = pdiff
    PR = json.loads(Path(a.partial).read_text())["results"]
    t50 = {k: v for k, v in PR["HA"]["variants"]["D8"]["t50"].items()}
    R["t50_registered_D8"] = t50
    R["hf_registered"] = dict(concordance=PR["HF"]["concordance"], pairs=PR["HF"]["pairs"], ci95=PR["HF"]["ci95"], decision=PR["HF"]["decision"])

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    steps = np.arange(1, 41)
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.3))
    for i, (metric, lab) in enumerate((("loc", "localisation (top-10% mass)"), ("ovl", "overlap with other noun"))):
        o = out1[metric]
        if o["mean"] is not None:
            ax[i].plot(steps, o["mean"], color="C0"); ax[i].fill_between(steps, o["lo"], o["hi"], color="C0", alpha=0.2)
        ax[i].axhline(0, color="grey", lw=0.8)
        ax[i].set_xlabel("denoising step"); ax[i].set_title(f"({'ab'[i]}) {lab}\ndetected $-$ missed (w = 7.5)", fontsize=7)
    for w, c in ((2.0, "C1"), (7.5, "C0"), (15.0, "C2")):
        r = out2["ovl"].get(str(w))
        if r:
            ax[2].plot(steps, r["mean"], color=c, label=f"w = {w:g}"); ax[2].fill_between(steps, r["lo"], r["hi"], color=c, alpha=0.15)
    ax[2].set_xlabel("denoising step"); ax[2].set_title("(c) noun overlap by guidance\n(identical initial noise)", fontsize=7); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "attn_success_failure.pdf"); fig.savefig(OUT / "attn_success_failure.png", dpi=150); plt.close(fig)

    # HF: t_stab vs t50 by type
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ts = ts_by_w["7.5"]
    for i, k in enumerate(sorted(set(ts) & set(t50))):
        tv = t50[k][0]
        tv = 37 if tv in (None, "inf") or not math.isfinite(float(tv)) else float(tv)
        e = ts[k]
        ax[0].errorbar([tv], [e["mean"]], yerr=[[e["mean"] - e["ci95"][0]], [e["ci95"][1] - e["mean"]]], fmt="o", capsize=3, color=f"C{i}",
                       label=f"{k.replace('_given_pres', '|pres')} (n={e['n_scenes']})", mfc="white" if t50[k][1] != "ok" else f"C{i}")
    ax[0].set_xlabel("$t_{50}$ (restart D8, w=7.5)"); ax[0].set_ylabel("$t_{stab}$ (attention), scene mean"); ax[0].legend(fontsize=6)
    ax[0].set_title(f"(a) concordance {R['hf_registered']['concordance']:.2f} over {R['hf_registered']['pairs']} type pairs", fontsize=7)
    kinds = sorted(set().union(*[set(v) for v in ts_by_w.values()]))
    for i, k in enumerate(kinds):
        ws = [w for w in ("2.0", "7.5", "15.0") if k in ts_by_w[w]]
        ax[1].errorbar([float(w) + (i - len(kinds) / 2) * 0.25 for w in ws], [ts_by_w[w][k]["mean"] for w in ws],
                       yerr=[[ts_by_w[w][k]["mean"] - ts_by_w[w][k]["ci95"][0] for w in ws], [ts_by_w[w][k]["ci95"][1] - ts_by_w[w][k]["mean"] for w in ws]],
                       fmt="o-", ms=3, capsize=2, color=f"C{i}", label=k.replace("_given_pres", "|pres"))
    ax[1].set_xlabel("guidance w"); ax[1].set_ylabel("$t_{stab}$"); ax[1].set_title("(b) attention stabilisation by guidance", fontsize=7); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "attn_tstab.pdf"); fig.savefig(OUT / "attn_tstab.png", dpi=150); plt.close(fig)

    # ---------------- illustrative panels (fixed selection rules)
    imgs = {}
    for zp in a.images_zip:
        with zipfile.ZipFile(zp) as zf:
            for n in zf.namelist():
                imgs[Path(n).stem] = (zp, n)

    def image(job):
        if job not in imgs:
            return None
        from PIL import Image
        import io
        zp, n = imgs[job]
        with zipfile.ZipFile(zp) as zf:
            return np.asarray(Image.open(io.BytesIO(zf.read(n))).convert("RGB").resize((128, 128)))

    # Panel A rule: the lowest scene ID of category 'binding' whose seed-0 maps exist at w = 2, 7.5 and 15 (no outcome or appearance criterion).
    cand = sorted({rows[j]["scene_id"] for j in files if j in rows and rows[j].get("category") == "binding"})
    sel = next((s for s in cand if all(f"{s}_CAN__cfg{w}__st40__seed0" in files for w in ("2", "7.5", "15"))), None)
    R["panel_A_scene"] = sel
    if sel:
        show = [2, 5, 10, 20, 40]
        fig, ax = plt.subplots(3, len(show) + 1, figsize=(6.6, 3.9))
        for i, w in enumerate(("2", "7.5", "15")):
            j = f"{sel}_CAN__cfg{w}__st40__seed0"
            z = np.load(files[j]); words = [str(x) for x in z["words"]]
            p = prompts[rows[j]["prompt_id"]]
            nouns = [c["object"] for c in p["constraints"] if c["type"] == "presence"][:2]
            mA, mB = (norm_maps(z["maps_16"][:, words.index(n)].astype(float)) for n in nouns)
            for c, s in enumerate(show):
                A, B = mA[s - 1] / max(mA[s - 1].max(), 1e-9), mB[s - 1] / max(mB[s - 1].max(), 1e-9)
                ax[i, c].imshow(np.stack([A, 0.5 * B, B], -1), interpolation="nearest"); ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
                if i == 0:
                    ax[i, c].set_title(f"step {s}", fontsize=7)
            ax[i, 0].set_ylabel(f"w = {w}", fontsize=7)
            im = image(j)
            r = rows[j]
            if im is not None:
                ax[i, -1].imshow(im)
            ax[i, -1].set_xticks([]); ax[i, -1].set_yticks([])
            ax[i, -1].set_title("objects: " + ("both" if r["objects_present"] else "missing") + ", colours: " + {1: "correct", 0: "wrong", None: "n/a"}[r.get("bind_given_pres")], fontsize=6)
        fig.suptitle(f"\"{prompts[rows[j]['prompt_id']]['prompt']}\" (seed 0): red = '{nouns[0]}', cyan = '{nouns[1]}'", fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / "attn_evolution.pdf"); fig.savefig(OUT / "attn_evolution.png", dpi=150); plt.close(fig)

    # Panel B rule: at w = 7.5, the three lowest main-benchmark scene IDs with a two-object prompt in which one object is detected in some seeds and missed in
    # others; for each, the lowest-index seed of each outcome. The missed object's noun is shown in both seeds.
    pick = []
    grp = defaultdict(lambda: {0: [], 1: []})
    for x in rec:
        if x["w"] == 7.5 and x["n_obj"] == 2:
            grp[(x["scene"], x["obj"])][x["y"]].append(x)
    for (s, o), d in sorted(grp.items()):
        if s.startswith("sg_") and d[0] and d[1] and s not in [p_[0] for p_ in pick]:
            pick.append((s, o, min(d[1], key=lambda x: x["seed"]), min(d[0], key=lambda x: x["seed"])))
        if len(pick) == 3:
            break
    R["panel_B"] = [(s, o, a_["job"], b_["job"]) for s, o, a_, b_ in pick]
    if pick:
        fig, ax = plt.subplots(len(pick), 4, figsize=(5.2, 1.45 * len(pick) + 0.4))
        ax = np.atleast_2d(ax)
        for i, (s, o, det, mis) in enumerate(pick):
            for c, (x, lab) in enumerate(((det, "detected"), (mis, "missed"))):
                z = np.load(files[x["job"]]); words = [str(v) for v in z["words"]]
                mm = norm_maps(z["maps_16"][:, words.index(o)].astype(float))
                ax[i, 2 * c].imshow(mm[9] / mm[9].max(), cmap="magma", interpolation="nearest")
                ax[i, 2 * c].set_title(f"'{o}', step 10, {lab}", fontsize=6)
                im = image(x["job"])
                if im is not None:
                    ax[i, 2 * c + 1].imshow(im)
                ax[i, 2 * c + 1].set_title(f"seed {x['seed']}", fontsize=6)
                for k in (2 * c, 2 * c + 1):
                    ax[i, k].set_xticks([]); ax[i, k].set_yticks([])
            ax[i, 0].set_ylabel(s, fontsize=6)
        fig.tight_layout(); fig.savefig(FIG / "attn_examples.pdf"); fig.savefig(OUT / "attn_examples.png", dpi=150); plt.close(fig)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attention.json").write_text(json.dumps(R, indent=1, default=str))

    def mac(n, v):
        return rf"\newcommand{{\{n}}}{{{v}}}"
    o1, o2 = out1["loc"], out1["ovl"]
    M = ["% generated by diagnostics/attention.py from reports/diagnostics/attention.json",
         mac("atFiles", f"{R['n_matched']:,}"), mac("atScenes", R["scenes"]), mac("atPairs", o1["n_pairs"]), mac("atPairScenes", o1["n_scenes"]),
         mac("atLocFirst", o1["first_step_ci_excludes_0"] or "--"), mac("atOvlFirst", o2["first_step_ci_excludes_0"] or "--"),
         mac("atLocTen", f"{o1['mean'][9]:+.3f}"), mac("atLocTenLo", f"{o1['lo'][9]:.3f}"), mac("atLocTenHi", f"{o1['hi'][9]:.3f}"),
         mac("atOvlTen", f"{o2['mean'][9]:+.3f}"), mac("atOvlTenLo", f"{o2['lo'][9]:.3f}"), mac("atOvlTenHi", f"{o2['hi'][9]:.3f}"),
         mac("atOvlTwoTen", f"{out2['ovl']['2.0']['mean'][9]:.3f}"), mac("atOvlFifteenTen", f"{out2['ovl']['15.0']['mean'][9]:.3f}"),
         mac("atOvlDiffTen", f"{out2['ovl']['w15_minus_w2']['mean'][9]:+.3f}"), mac("atOvlDiffTenLo", f"{out2['ovl']['w15_minus_w2']['lo'][9]:.3f}"),
         mac("atOvlDiffTenHi", f"{out2['ovl']['w15_minus_w2']['hi'][9]:.3f}"),
         mac("atDetTwo", f"{R['object_detection_rate_by_w'].get('2.0', math.nan):.2f}"), mac("atDetFifteen", f"{R['object_detection_rate_by_w'].get('15.0', math.nan):.2f}")]
    ts75 = ts_by_w["7.5"]
    for k, nm in (("presence", "Pres"), ("bind_given_pres", "Bind"), ("count", "Count"), ("spatial_given_pres", "Spat")):
        if k in ts75:
            M.append(mac(f"atTs{nm}", f"{ts75[k]['mean']:.1f}"))
    for k, nm in (("presence", "Pres"), ("bind_given_pres", "Bind"), ("count", "Count")):
        if k in pdiff:
            M += [mac(f"atTsDiff{nm}", f"{pdiff[k]['mean']:.1f}"), mac(f"atTsDiff{nm}Lo", f"{pdiff[k]['ci95'][0]:.1f}"), mac(f"atTsDiff{nm}Hi", f"{pdiff[k]['ci95'][1]:.1f}")]
    for k, nm in (("presence", "Pres"), ("bind_given_pres", "Bind")):
        if k in ts_by_w["2.0"]:
            M += [mac(f"atTsTwo{nm}", f"{ts_by_w['2.0'][k]['mean']:.1f}"), mac(f"atTsFifteen{nm}", f"{ts_by_w['15.0'][k]['mean']:.1f}")]
    lo_ = min(v["mean"] for v in ts75.values()); hi_ = max(v["mean"] for v in ts75.values())
    M += [mac("atTsRangeLo", f"{lo_:.1f}"), mac("atTsRangeHi", f"{hi_:.1f}")]
    (TAB / "attn_numbers.tex").write_text("\n".join(M) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ("n_files", "n_matched", "inventory", "scenes", "object_records", "object_detection_rate_by_w", "t_stab", "panel_A_scene", "panel_B")}, indent=1, default=str))
    print("loc first", o1["first_step_ci_excludes_0"], "ovl first", o2["first_step_ci_excludes_0"], "pairs", o1["n_pairs"], o1["n_scenes"])
    print("loc diff steps 1,5,10,20,40", [round(o1["mean"][i], 3) for i in (0, 4, 9, 19, 39)], "ovl", [round(o2["mean"][i], 3) for i in (0, 4, 9, 19, 39)])
    print("ovl by w @10", {w: round(v["mean"][9], 3) for w, v in out2["ovl"].items()}, "loc by w @10", {w: round(v["mean"][9], 3) for w, v in out2["loc"].items()})


if __name__ == "__main__":
    main()
