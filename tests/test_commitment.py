import math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import commitment as C
import branch as B

GRID = C.GRID
SCH = B.Schedule(40)


def simulate(kind, tau, n_scenes=60, k=4, m=2, seed=0, p_range=(0.3, 0.7), mech=None, mode="hard", scene_prefix="s"):
    """Synthetic world with KNOWN persistence time tau. Base outcomes: independent Bernoulli(p_s) across seeds.
    Branch at t keeps the base outcome with prob pi(t) (hard: 1[t >= tau]; soft: logistic) plus an optional MECHANICAL persistence
    term mech(t) shared by all kinds (models the D-02 confound: later restarts inject less noise), else redraws Bernoulli(p_s)."""
    rng = np.random.default_rng(seed)
    core, br = [], []
    for s in range(n_scenes):
        sid = f"{scene_prefix}{s}"
        p = rng.uniform(*p_range)
        base = (rng.random(k) < p).astype(int)
        for j in range(k):
            core.append(dict(scene=sid, kind=kind, seed=j, y=base[j], elig=1))
        for j in range(min(k, 2)):
            for t in GRID:
                pi = float(t >= tau) if mode == "hard" else 1 / (1 + math.exp(-(t - tau) / 3.0))
                if mech is not None:
                    pi = pi + (1 - pi) * mech(t)
                for mm in range(m):
                    keep = rng.random() < pi
                    y = base[j] if keep else int(rng.random() < p)
                    br.append(dict(scene=sid, kind=kind, seed=j, t=t, m=mm, y=y, elig=1, base_y=base[j], base_elig=1))
    return pd.DataFrame(core), pd.DataFrame(br)


def run(parts, kinds):
    core = pd.concat([c for c, _ in parts]); br = pd.concat([b for _, b in parts])
    return C.agreement(br), C.chance_agreement(core)


def test_known_commitment_recovered_within_one_grid_step():
    pa, pb = simulate("presence", 12, seed=1), simulate("bind", 24, seed=2)
    adf, a0 = run([pa, pb], ["presence", "bind"])
    cur, _ = C.curves(adf, a0, ["presence", "bind"])
    assert abs(cur["presence"]["t50"] - 12 + 3) <= 6 and abs(cur["bind"]["t50"] - 24 + 3) <= 6   # hard step at tau: crossing interpolated near tau
    res = C.bootstrap_delta_t50(adf, a0, "presence", "bind", n_boot=500)
    assert abs(res["delta_t50"] - 12) <= 6 and res["ci95"][0] > 0
    assert C.classify_HA(res) == "support"


def _independent_run(seed, S=300):
    rng = np.random.default_rng(seed)
    base = rng.random((S, 4)) < 0.5
    core = pd.DataFrame([dict(scene=s, kind="x", seed=j, y=int(base[s, j]), elig=1) for s in range(S) for j in range(4)])
    br = pd.DataFrame([dict(scene=s, kind="x", seed=j, t=t, m=m, y=int(rng.random() < 0.5), elig=1, base_y=int(base[s, j]), base_elig=1)
                       for s in range(S) for j in range(2) for t in GRID for m in range(2)])
    return C.curves(C.agreement(br), C.chance_agreement(core), ["x"])[0]["x"]


def test_independent_outcomes_give_C_near_zero():
    """Unbiased: averaged over replicates C is ~0 (a Monte Carlo of 40 replicates gave mean 0.0017, SE 0.005). A single run carries
    noise that is SHARED across grid steps (one A0 estimate), SD about 0.04 per step at 300 scenes, so the per-run bound is 0.15."""
    runs = [_independent_run(i) for i in range(10)]
    Cs = np.array([r["C"] for r in runs])
    assert abs(Cs.mean()) < 0.025
    assert np.nanmax(np.abs(Cs)) < 0.15
    assert all(r["t50_status"] == "right_censored" for r in runs)


def test_identical_outcomes_give_C_one():
    core, br = simulate("x", 0, mode="hard")           # tau=0: every branch keeps the base outcome
    adf, a0 = C.agreement(br), C.chance_agreement(core)
    cur, _ = C.curves(adf, a0, ["x"])
    assert np.allclose(cur["x"]["C"], 1.0) and cur["x"]["t50_status"] == "left_censored"


def test_stable_wrong_scene_has_zero_weight():
    core = pd.DataFrame([dict(scene="w", kind="x", seed=j, y=0, elig=1) for j in range(4)])
    br = pd.DataFrame([dict(scene="w", kind="x", seed=j, t=t, m=0, y=0, elig=1, base_y=0, base_elig=1) for j in range(2) for t in GRID])
    adf, a0 = C.agreement(br), C.chance_agreement(core)
    assert a0.A0.iloc[0] == 1.0
    cur, _ = C.curves(adf, a0, ["x"])
    assert cur["x"]["weight"].sum() == 0 and np.isnan(cur["x"]["C"]).all()      # carries no information; no fake commitment


def test_mechanical_persistence_biases_absolute_t50_but_not_the_difference():
    """D-02: if every kind gets the same schedule-driven persistence (later restarts inject less noise), absolute t50 shifts earlier
    but the difference between kinds survives. Uses the real ratio r(t) of the D=8 restart plan."""
    mech = lambda t: 0.6 * B.restart_plan(SCH, t, 8)["ratio"] ** 1.0
    pa, pb = simulate("presence", 12, seed=3, mech=mech), simulate("bind", 30, seed=4, mech=mech)
    adf, a0 = run([pa, pb], None)
    cur, _ = C.curves(adf, a0, ["presence", "bind"])
    clean = C.curves(*run([simulate("presence", 12, seed=3), simulate("bind", 30, seed=4)], None), ["presence", "bind"])[0]
    assert cur["bind"]["t50"] < clean["bind"]["t50"]                             # absolute t50 biased earlier by the mechanical term
    d_mech = cur["bind"]["t50"] - cur["presence"]["t50"]
    assert d_mech > 6 and abs(d_mech - 18) <= 10                                 # difference still positive and in the right range


def test_conditional_pairs_require_both_eligible_and_counts_reported():
    core, br = simulate("bind", 12, n_scenes=30, seed=5)
    br.loc[br.sample(frac=0.5, random_state=0).index, "elig"] = 0
    a = C.agreement(br)
    assert a.n_pairs.sum() == (br.elig == 1).sum() and (a.n_pairs > 0).all()


def test_t50_interpolation_and_flags():
    v, st = C.t50([0, 0.25, 0.75, 1, 1, 1]);  assert st == "ok" and abs(v - 15.0) < 1e-9
    assert C.t50([0.6] * 6)[1] == "left_censored"
    assert C.t50([0.1] * 6) == (math.inf, "right_censored")
    assert C.nonmonotone([0, 0.8, 0.2, 0.9]) and not C.nonmonotone([0, 0.1, 0.3, 0.3])


def test_sign_agreement_and_windows():
    assert C.sign_agreement({"delta_t50": 5.0}, {"delta_t50": 2.0}) and not C.sign_agreement({"delta_t50": 5.0}, {"delta_t50": -1.0})
    w = C.windows_from_t50_dev(10); assert w["tau_pre_label"] == "EXPLORATORY"
    assert C.windows_from_t50_dev(30)["tau_post"] is None and not C.windows_from_t50_dev(30)["G_estimable"]
    assert C.windows_from_t50_dev(math.inf)["onsets"] is None
    w = C.windows_from_t50_dev(24); assert w["tau_pre"] == 12 and w["tau_post"] == 36 and w["onsets"]["late"] == 28


def _rows_from_sim(parts, variant):
    """Turn the synthetic long tables into eval-style rows so the driver is tested end to end."""
    import analyze_commitment as AC
    core_rows, br_rows = [], []
    for core, br in parts:
        for r in core.itertuples():
            d = {"scene_id": r.scene, "seed": r.seed}
            if r.kind == "presence": d["objects_present"] = int(r.y)
            else: d.update({"bind_given_pres": int(r.y), "joint_bind": int(r.y)})
            core_rows.append(d)
        for r in br.itertuples():
            d = {"scene_id": r.scene, "seed": r.seed, "t": r.t, "m": r.m, "variant": variant}
            if r.kind == "presence": d["objects_present"] = int(r.y)
            else: d.update({"bind_given_pres": int(r.y), "joint_bind": int(r.y)})
            br_rows.append(d)
    return core_rows, br_rows


def test_driver_end_to_end_with_equal_ratio_gate(tmp_path):
    import analyze_commitment as AC
    # presence and binding share the SAME scenes (the driver keys by scene); persistence times 12 vs 24
    pa, pb = simulate("presence", 12, n_scenes=200, seed=11, p_range=(0.5, 0.9)), simulate("bind_given_pres", 24, n_scenes=200, seed=12)
    pa_rows = _rows_from_sim([pa], "D8"); pb_rows = _rows_from_sim([pb], "D8")
    # merge presence+bind rows per (scene, seed[, t, m]) into single eval rows
    def merge(r1, r2, key):
        d = {key(r): dict(r) for r in r1}
        for r in r2: d.setdefault(key(r), {}).update(r)
        return list(d.values())
    core = merge(pa_rows[0], pb_rows[0], lambda r: (r["scene_id"], r["seed"]))
    br = merge(pa_rows[1], pb_rows[1], lambda r: (r["scene_id"], r["seed"], r["t"], r["m"]))
    er_br = [dict(r, variant="ER30") for r in br]
    out, elig = AC.analyse(core, br + er_br, n_boot=300)
    ha = out["HA"]
    assert ha["decision"] == "support" and abs(ha["primary"]["delta_t50"] - 12) <= 6
    assert ((elig[elig.kind == "presence"].eligible_fraction) == 1).all() and (elig[elig.kind == "bind_given_pres"].eligible_fraction < 1).all()   # conditional kind loses pairs (D-04)
    # without an equal-ratio run support is not qualified (D-02)
    out2, _ = AC.analyse(core, br, n_boot=300)
    assert out2["HA"]["decision"].startswith("support NOT QUALIFIED")


def test_driver_refuses_test_rows_without_freeze(tmp_path):
    import json, subprocess
    row = {"scene_id": "s", "split": "TEST", "seed": 0, "objects_present": 1}
    (tmp_path / "c.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(dict(row, t=6, m=0, variant="D8")) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "src/analyze_commitment.py"), "--core", str(tmp_path / "c.jsonl"), "--branches", str(tmp_path / "b.jsonl"), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "frozen config" in (r.stderr + r.stdout)
