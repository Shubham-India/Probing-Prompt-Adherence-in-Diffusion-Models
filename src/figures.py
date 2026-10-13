"""Builds every figure from saved results (<run>/results/*, <run>/attention/*, <run>/images/*).

Example-selection rules (fixed in advance, not chosen by appearance):
  contact sheet : per category, the prompt whose overall success rate is closest to the category median; seed 0.
  attention maps: the binding-category prompt whose success at CFG 7.5 / 40 steps is closest to 50% (ties -> lowest id);
                  seed 0 for the CFG comparison; for success-vs-failure the lowest seed that succeeded and the lowest
                  seed that failed under CFG 7.5 / 40 steps for that same prompt.
Usage: python src/figures.py --run outputs/main
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CATEGORIES = ["presence", "binding", "counting", "spatial", "multi"]
STAGES = {"early": slice(0, 13), "middle": slice(13, 27), "late": slice(27, 40)}
CFG_COLORS = {2.0: "#4c78a8", 7.5: "#f58518", 15.0: "#54a24b"}


def adherence_figure(res, fig_dir):
    cells = pd.read_csv(res / "adherence_by_cell.csv")
    fig, axes = plt.subplots(1, 6, figsize=(20, 3.6), sharey=True)
    for ax, cat in zip(axes, CATEGORIES + ["all"]):
        for steps, marker in ((20, "o"), (40, "s")):
            d = cells[(cells.category == cat) & (cells.steps == steps)].sort_values("cfg")
            x = np.arange(len(d)) + (0.06 if steps == 40 else -0.06)
            ax.errorbar(x, d["mean"], yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], marker=marker, capsize=3, label=f"{steps} steps")
        ax.set_xticks(range(3), [f"{c:g}" for c in sorted(cells.cfg.unique())])
        ax.set_title(f"{cat} (n prompts={int(d.n_prompts.iloc[0])})")
        ax.set_xlabel("CFG scale")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("primary success rate")
    axes[0].legend()
    fig.suptitle("Prompt adherence by CFG and steps (95% prompt-cluster bootstrap CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "adherence_by_cfg_steps.png", dpi=150)
    plt.close(fig)

    types = pd.read_csv(res / "constraint_type_by_cell.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    pooled = types.groupby(["constraint", "cfg"]).apply(lambda g: np.average(g["mean"], weights=g.n_images), include_groups=False).unstack()
    pooled.plot.bar(ax=ax, color=[CFG_COLORS[c] for c in pooled.columns])
    ax.set_ylabel("constraint accuracy (steps pooled)")
    ax.legend(title="CFG")
    fig.tight_layout()
    fig.savefig(fig_dir / "constraint_type_accuracy.png", dpi=150)
    plt.close(fig)


def contact_sheet(run, res, fig_dir):
    df = pd.read_csv(res / "per_image.csv")
    rate = df.groupby(["category", "prompt_id"]).primary.mean().reset_index()
    chosen = []
    for cat in CATEGORIES:
        r = rate[rate.category == cat]
        chosen.append(r.iloc[(r.primary - r.primary.median()).abs().argsort(kind="stable")].prompt_id.iloc[0])
    cfgs, steps = sorted(df.cfg.unique()), sorted(df.steps.unique())
    fig, axes = plt.subplots(len(chosen), 6, figsize=(18, 3.1 * len(chosen)))
    for i, pid in enumerate(chosen):
        for j, (cfg, st) in enumerate([(c, s) for c in cfgs for s in steps]):
            row = df[(df.prompt_id == pid) & (df.cfg == cfg) & (df.steps == st) & (df.seed == 0)].iloc[0]
            ax = axes[i, j]
            ax.imshow(Image.open(run / "images" / f"{row['name']}.png"))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("green" if row.primary else "red")
                s.set_linewidth(4)
            ax.set_title(f"CFG {cfg:g}, {st} steps", fontsize=9)
            if j == 0:
                ax.set_ylabel(pid, fontsize=8)
    fig.suptitle("Seed 0, median-difficulty prompt per category (green = primary success, red = failure)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_dir / "contact_sheet.png", dpi=110)
    plt.close(fig)


def pick_binding_prompt(df):
    d = df[(df.category == "binding") & (df.cfg == 7.5) & (df.steps == 40)]
    r = d.groupby("prompt_id").primary.mean()
    return (r - 0.5).abs().sort_values(kind="stable").index[0]


def load_maps(run, name):
    z = np.load(run / "attention" / f"{name}.npz")
    return {w: m.astype(float) for w, m in zip(z["words"], z["maps"])}


def show_map(ax, image, m, title=None):
    ax.imshow(image)
    ax.imshow(np.array(Image.fromarray((m / (m.max() + 1e-12) * 255).astype(np.uint8)).resize((512, 512), Image.BILINEAR)),
              cmap="inferno", alpha=0.6)
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def attention_figures(run, res, fig_dir):
    df = pd.read_csv(res / "per_image.csv")
    pid = pick_binding_prompt(df)
    words = [w for w in load_maps(run, df[(df.prompt_id == pid) & (df.steps == 40)].name.iloc[0]) if w not in ("wooden", "table")]
    cfgs = sorted(df.cfg.unique())

    fig, axes = plt.subplots(len(cfgs), 1 + len(words) * 3, figsize=(1.8 * (1 + len(words) * 3), 2.1 * len(cfgs)))
    for i, cfg in enumerate(cfgs):
        row = df[(df.prompt_id == pid) & (df.cfg == cfg) & (df.steps == 40) & (df.seed == 0)].iloc[0]
        image = Image.open(run / "images" / f"{row['name']}.png")
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"CFG {cfg:g}: {'success' if row.primary else 'failure'}", fontsize=8)
        axes[i, 0].set_xticks([]), axes[i, 0].set_yticks([])
        maps = load_maps(run, row["name"])
        for k, w in enumerate(words):
            for s, (stage, sl) in enumerate(STAGES.items()):
                show_map(axes[i, 1 + k * 3 + s], image, maps[w][sl].mean(0), f"'{w}' {stage}" if i == 0 else stage)
    fig.suptitle(f"{pid}, seed 0, 40 steps: token cross-attention at early/middle/late denoising, by CFG")
    fig.tight_layout()
    fig.savefig(fig_dir / "attention_maps_by_cfg.png", dpi=110)
    plt.close(fig)

    d = df[(df.prompt_id == pid) & (df.cfg == 7.5) & (df.steps == 40)].sort_values("seed")
    picks = [d[d.primary == 1].head(1), d[d.primary == 0].head(1)]
    picks = [p.iloc[0] for p in picks if len(p)]
    fig, axes = plt.subplots(len(picks), 1 + len(words), figsize=(2.3 * (1 + len(words)), 2.5 * len(picks)), squeeze=False)
    for i, row in enumerate(picks):
        image = Image.open(run / "images" / f"{row['name']}.png")
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"seed {row.seed}: {'success' if row.primary else 'failure'}", fontsize=8)
        axes[i, 0].set_xticks([]), axes[i, 0].set_yticks([])
        maps = load_maps(run, row["name"])
        for k, w in enumerate(words):
            show_map(axes[i, 1 + k], image, maps[w][STAGES["middle"]].mean(0), f"'{w}' (middle)")
    fig.suptitle(f"{pid}, CFG 7.5, 40 steps: success vs failure (lowest seed of each)")
    fig.tight_layout()
    fig.savefig(fig_dir / "attention_success_vs_failure.png", dpi=110)
    plt.close(fig)


def attention_summary_figure(res, fig_dir):
    curves = pd.read_csv(res / "attention_top10_by_step.csv")
    eff = pd.read_csv(res / "attention_cfg_effect.csv")
    auc = pd.read_csv(res / "attention_auc.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    for cfg, g in curves.groupby("cfg"):
        axes[0].plot(g.step, g["mean"], color=CFG_COLORS[float(cfg)], label=f"CFG {cfg:g}")
        axes[0].fill_between(g.step, g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"], color=CFG_COLORS[float(cfg)], alpha=0.2)
    axes[0].set(xlabel="denoising step (0 = noisiest)", ylabel="top-10% mass share of object-token maps", title="Localisation over denoising")
    axes[0].legend()
    lvl = eff[(eff.metric == "top10") & (eff.b == "level")]
    for cfg, g in lvl.groupby("a"):
        axes[1].plot(list(STAGES), g.set_index("stage").loc[list(STAGES), "diff"], marker="o", color=CFG_COLORS[float(cfg)], label=f"CFG {float(cfg):g}")
    axes[1].set(ylabel="mean top-10% share", title="Localisation by stage (mean over images)")
    sel = auc[auc.feature.str.contains("top10|entropy|strength|bind_margin|box_ratio")].copy()
    sel["label"] = sel.feature + np.where(sel.within_prompt, " (within prompt)", "")
    sel = sel.sort_values("label")
    y = np.arange(len(sel))
    axes[2].errorbar(sel.auc, y, xerr=[sel.auc - sel.ci_lo, sel.ci_hi - sel.auc], fmt="o", capsize=2)
    axes[2].axvline(0.5, color="grey", ls="--")
    axes[2].set_yticks(y, sel.label, fontsize=6)
    axes[2].set(xlabel="AUC for adherence success (95% prompt-cluster CI)", title="Does attention separate success from failure?")
    fig.tight_layout()
    fig.savefig(fig_dir / "attention_summary.png", dpi=130)
    plt.close(fig)


def methodology_figure(fig_dir):
    boxes = ["prompts.json\n28 prompts\n5 categories",
             "generate.py\nSD1.5 fp16, DDIM\n3 CFG x 2 steps\n6 seeds: 1008 images",
             "evaluate.py\nOWL-ViT boxes\n+ CLIP colour\n-> constraint 0/1",
             "analyze.py\npaired contrasts,\ncluster bootstrap,\nWilcoxon, GEE",
             "figures.py\n+ report"]
    fig, ax = plt.subplots(figsize=(19, 4.8))
    ax.axis("off")
    step = 3.9
    for i, t in enumerate(boxes):
        ax.text(i * step, 1.0, t, ha="center", va="center", fontsize=14, bbox=dict(boxstyle="round,pad=0.5", fc="#dde8f5", ec="#4c78a8"))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i * step + 2.5, 1.0), xytext=(i * step + 1.8, 1.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(7.8, -0.45, "attention.py: re-generates the 40-step images with a cross-attention hook (images bit-identical)\n"
            "-> 32x32 word maps per step -> analyze_attention.py: localisation, colour/object overlap, AUC vs success\n"
            "intervene.py: re-weights object/colour attention (control s=1) -> analyze_intervention.py", ha="center", va="center", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fbe7d0", ec="#f58518"))
    ax.annotate("", xy=(7.8, 0.35), xytext=(7.8, 0.0), arrowprops=dict(arrowstyle="<-", color="#f58518", lw=1.5))
    ax.set_xlim(-2.4, 18.0)
    ax.set_ylim(-1.1, 1.6)
    fig.savefig(fig_dir / "methodology.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def slide_contact_sheet(run, res, fig_dir):
    """Same prompts and rule as contact_sheet, 40-step images only, with large labels for the slide."""
    df = pd.read_csv(res / "per_image.csv")
    rate = df.groupby(["category", "prompt_id"]).primary.mean().reset_index()
    chosen = []
    for cat in CATEGORIES:
        r = rate[rate.category == cat]
        chosen.append(r.iloc[(r.primary - r.primary.median()).abs().argsort(kind="stable")].prompt_id.iloc[0])
    cfgs = sorted(df.cfg.unique())
    fig, axes = plt.subplots(len(chosen), len(cfgs), figsize=(6.6, 11.2))
    for i, pid in enumerate(chosen):
        for j, cfg in enumerate(cfgs):
            row = df[(df.prompt_id == pid) & (df.cfg == cfg) & (df.steps == 40) & (df.seed == 0)].iloc[0]
            ax = axes[i, j]
            ax.imshow(Image.open(run / "images" / f"{row['name']}.png"))
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("green" if row.primary else "red")
                sp.set_linewidth(6)
            if i == 0:
                ax.set_title(f"CFG {cfg:g}", fontsize=20)
            if j == 0:
                ax.set_ylabel(pid.replace("_", " ")[:24], fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "slide_contact_sheet.png", dpi=110)
    plt.close(fig)


def slide_adherence_figure(res, fig_dir):
    cells = pd.read_csv(res / "adherence_by_cell.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharey=True)
    for ax, cat in zip(axes.ravel(), CATEGORIES + ["all"]):
        for steps, marker in ((20, "o"), (40, "s")):
            d = cells[(cells.category == cat) & (cells.steps == steps)].sort_values("cfg")
            x = np.arange(len(d)) + (0.06 if steps == 40 else -0.06)
            ax.errorbar(x, d["mean"], yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], marker=marker, capsize=4, lw=2, ms=7, label=f"{steps} steps")
        ax.set_xticks(range(3), [f"{c:g}" for c in sorted(cells.cfg.unique())], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(f"{cat} ({int(d.n_prompts.iloc[0])} prompts)", fontsize=15)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("CFG scale", fontsize=13)
    for ax in axes[:, 0]:
        ax.set_ylabel("primary success", fontsize=13)
    axes[0, 0].legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "slide_adherence.png", dpi=130)
    plt.close(fig)


def slide_attention_maps(run, res, fig_dir):
    df = pd.read_csv(res / "per_image.csv")
    pid = pick_binding_prompt(df)
    words = [w for w in load_maps(run, df[(df.prompt_id == pid) & (df.steps == 40)].name.iloc[0]) if w not in ("wooden", "table")]
    cfgs = sorted(df.cfg.unique())
    fig, axes = plt.subplots(len(cfgs), 1 + len(words), figsize=(2.4 * (1 + len(words)), 2.5 * len(cfgs)))
    for i, cfg in enumerate(cfgs):
        row = df[(df.prompt_id == pid) & (df.cfg == cfg) & (df.steps == 40) & (df.seed == 0)].iloc[0]
        image = Image.open(run / "images" / f"{row['name']}.png")
        axes[i, 0].imshow(image)
        axes[i, 0].set_ylabel(f"CFG {cfg:g}", fontsize=14)
        axes[i, 0].set_xticks([]), axes[i, 0].set_yticks([])
        maps = load_maps(run, row["name"])
        for k, w in enumerate(words):
            show_map(axes[i, 1 + k], image, maps[w][STAGES["middle"]].mean(0))
            if i == 0:
                axes[i, 1 + k].set_title(f"'{w}'", fontsize=15)
    axes[0, 0].set_title("image", fontsize=15)
    fig.suptitle(f"{pid}, seed 0, 40 steps, middle denoising stage", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "slide_attention_maps.png", dpi=110)
    plt.close(fig)


def slide_attention_summary(res, fig_dir):
    curves = pd.read_csv(res / "attention_top10_by_step.csv")
    auc = pd.read_csv(res / "attention_auc.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    for cfg, g in curves.groupby("cfg"):
        axes[0].plot(g.step, g["mean"], color=CFG_COLORS[float(cfg)], lw=2.5, label=f"CFG {cfg:g}")
        axes[0].fill_between(g.step, g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"], color=CFG_COLORS[float(cfg)], alpha=0.2)
    axes[0].set_xlabel("denoising step (0 = noisiest)", fontsize=13)
    axes[0].set_ylabel("top-10% mass share (object words)", fontsize=13)
    axes[0].set_title("Localisation of object attention", fontsize=15)
    axes[0].legend(fontsize=12)
    axes[0].grid(alpha=0.3)
    picks = [("middle_bind_margin", False, "colour/object overlap margin\n(binding success, n=216)"),
             ("middle_top10", True, "top-10% share (within prompt)"),
             ("middle_entropy", True, "entropy (within prompt)"),
             ("middle_strength", True, "strength (within prompt)"),
             ("middle_box_ratio", True, "box ratio (within prompt)")]
    for y, (feat, within, label) in enumerate(picks[::-1]):
        r = auc[(auc.feature == feat) & (auc.within_prompt == within)].iloc[0]
        axes[1].errorbar(r.auc, y, xerr=[[r.auc - r.ci_lo], [r.ci_hi - r.auc]], fmt="o", capsize=4, ms=8, color="#4c78a8")
    axes[1].set_yticks(range(len(picks)), [p[2] for p in picks[::-1]], fontsize=11)
    axes[1].axvline(0.5, color="grey", ls="--")
    axes[1].set_xlabel("AUC for success (95% prompt-cluster CI), middle stage", fontsize=13)
    axes[1].set_title("Does attention separate success from failure?", fontsize=15)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "slide_attention_summary.png", dpi=130)
    plt.close(fig)


def intervention_figure(run, fig_dir):
    res = run / "results"
    cells = pd.read_csv(res / "intervention_by_condition.csv")
    df = pd.read_csv(res / "per_image.csv")
    fig = plt.figure(figsize=(14, 5.6))
    ax = fig.add_axes([0.06, 0.14, 0.44, 0.76])
    conds = ["rw1", "rw2", "rw4"]
    colors = {"rw1": "#9aa5b1", "rw2": "#f58518", "rw4": "#b3541e"}
    cats = CATEGORIES + ["all"]
    for j, c in enumerate(conds):
        d = cells[cells.condition == c].set_index("category").loc[cats]
        x = np.arange(len(cats)) + (j - 1) * 0.26
        ax.bar(x, d["mean"], 0.26, color=colors[c], label={"rw1": "control (s=1)", "rw2": "re-weight s=2", "rw4": "re-weight s=4"}[c])
        ax.errorbar(x, d["mean"], yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], fmt="none", ecolor="black", capsize=2, lw=1)
    ax.set_xticks(range(len(cats)), cats, fontsize=13, rotation=20)
    ax.set_ylabel("primary success (CFG 7.5, 40 steps)", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title("Attention re-weighting vs control", fontsize=16)
    ax.legend(fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    picks = [df[df.category == cat].prompt_id.sort_values().iloc[0] for cat in ("presence", "binding", "counting", "multi")]
    for i, pid in enumerate(picks):
        for j, c in enumerate(conds):
            row = df[(df.prompt_id == pid) & (df.seed == 0) & (df.condition == c)].iloc[0]
            a = fig.add_axes([0.56 + j * 0.14, 0.74 - i * 0.24, 0.1, 0.22])
            a.imshow(Image.open(run / "images" / f"{row['name']}.png"))
            a.set_xticks([]), a.set_yticks([])
            for s in a.spines.values():
                s.set_edgecolor("green" if row.primary else "red")
                s.set_linewidth(3)
            if i == 0:
                a.set_title({"rw1": "control", "rw2": "s=2", "rw4": "s=4"}[c], fontsize=15)
            if j == 0:
                category, rest = pid.split("_", 1)
                a.set_ylabel(f"{category}\n{rest.replace('_', ' ')}", fontsize=10)
    fig.savefig(fig_dir / "intervention.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--intervention", default=None, help="intervention run directory (optional)")
    ap_args = ap.parse_args()
    run = Path(ap_args.run)
    res = run / "results"
    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    adherence_figure(res, fig_dir)
    contact_sheet(run, res, fig_dir)
    slide_contact_sheet(run, res, fig_dir)
    attention_figures(run, res, fig_dir)
    attention_summary_figure(res, fig_dir)
    slide_adherence_figure(res, fig_dir)
    slide_attention_maps(run, res, fig_dir)
    slide_attention_summary(res, fig_dir)
    methodology_figure(fig_dir)
    intervention_dir = Path(ap_args.intervention) if ap_args.intervention else None
    if intervention_dir and (intervention_dir / "results" / "per_image.csv").exists():
        intervention_figure(intervention_dir, fig_dir)
    print("figures written to", fig_dir)
