"""Noise-only readout (E21/E25): does the initial noise (reproducible from the seed) predict the outcome beyond prompt-structure features?
Usage: python src/analyze_noise.py --eval outputs/main/eval.jsonl --prompts data/prompts.json --out outputs/m6_noise_pilot
Outcome: all_satisfied (legacy) unless --outcome names another column in the eval rows. Groups = prompt_id. EXPLORATORY on pilot data."""
import argparse
import json
from pathlib import Path

import pandas as pd

import features as F
import predict_failure as P


def build(rows, prompts):
    recs = []
    for r in rows:
        p = prompts[r["prompt_id"]]
        recs.append({"name": r["name"], "prompt_id": r["prompt_id"], "seed": r["seed"], "cfg": r["cfg"], "steps": r["steps"], "category": r["category"],
                     **F.prompt_features(p["constraints"], p["prompt"]), **F.noise_features(r["seed"])})
    return pd.DataFrame(recs)


def sets_for(df):
    pf = [c for c in df if c.startswith("p_")]
    base = df[pf].fillna(df[pf].median()).join(pd.get_dummies(df.cfg.astype(str), prefix="cfg").astype(float))
    base = base.join(pd.get_dummies(df.category, prefix="cat").astype(float))
    noise = df[[c for c in df if c.startswith("n_")]]
    return {"B1(prompt+cfg)": base, "B_noise(+x_T features)": base.join(noise)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outcome", default="all_satisfied")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.eval).read_text().splitlines()]
    if a.steps:
        rows = [r for r in rows if r["steps"] == a.steps]
    prompts = {p["id"]: p for p in json.loads(Path(a.prompts).read_text())}
    df = build(rows, prompts)
    y = [r[a.outcome] for r in rows]
    diffs = {"noise_minus_B1": ("B_noise(+x_T features)", "B1(prompt+cfg)")}
    res, _ = P.nested_compare(sets_for(df), y, df.prompt_id, diffs)
    res["status"] = "EXPLORATORY (pilot); only 6 distinct noise vectors exist, so noise features equal seed identity"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "noise_readout.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
