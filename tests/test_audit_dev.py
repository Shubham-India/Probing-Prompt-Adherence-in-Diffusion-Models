import csv, json, random, sys
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import audit_dev as A

CONS = [{"type": "presence", "object": "cup"}, {"type": "presence", "object": "book"},
        {"type": "binding", "object": "cup", "color": "red"}, {"type": "binding", "object": "book", "color": "blue"},
        {"type": "count", "object": "apple", "n": 3}, {"type": "spatial", "left": "cup", "right": "book"}]


def fake(tmp_path, n=60, seed=0):
    rng = random.Random(seed)
    img = tmp_path / "images"; img.mkdir()
    prompts, rows = {}, []
    for k in range(n):
        pid = f"sg_{k:04d}_CAN"
        prompts[pid] = {"id": pid, "constraints": CONS}
        name = f"{pid}__cfg7.5__st40__seed{k % 4}"
        Image.new("RGB", (32, 32), (k, 0, 0)).save(img / f"{name}.png")
        rows.append({"name": name, "prompt_id": pid, "uncertain": int(rng.random() < 0.1), "objects_present": rng.randint(0, 1),
                     "constraint_results": [rng.randint(0, 1) for _ in CONS]})
    return rows, prompts, img


def annotate(key_rows, rows_by_name, judge_error, rng, practice_too=True):
    """Simulated annotator who answers truthfully; the 'truth' equals the judge prediction flipped with probability judge_error."""
    labels = {}
    for r in key_rows:
        c = json.loads(r["constraint_json"])
        true_ok = int(r["judge_pred"]) if rng.random() > judge_error else 1 - int(r["judge_pred"])
        t = c["type"]
        if t == "presence":
            resp = "yes" if true_ok else "no"
        elif t == "count":
            resp = str(c["n"] if true_ok else c["n"] + 1)
        elif t == "binding":
            resp = f"col:{c['color']}" if true_ok else "col:green"
        else:
            resp = f"obj:{c['left']}" if true_ok else f"obj:{c['right']}"
        labels[r["item_id"]] = resp
    return labels


def test_package_is_blind_and_scoring_recovers_error_rate(tmp_path):
    rows, prompts, img = fake(tmp_path)
    out = tmp_path / "audit"
    man = A.build(rows, prompts, img, out, seed=3, quotas={"presence": 0, "binding": 0, "count": 0, "spatial": 0}, n_practice=0, n_double=0)
    assert man["n_items"] == 60 * len(CONS)                                  # census
    for f in (out / "annotator_A" / "label.html",):
        text = f.read_text(encoding="utf-8")
        assert "sg_00" not in text and "seed" not in text and "cfg" not in text and "judge_pred" not in text and "constraint_json" not in text
    assert not any(p.stem.startswith("sg_") for p in (out / "annotator_A" / "images").iterdir())
    key = A.read_key(out / "key" / "audit_key.csv")
    rng = random.Random(1)
    labs = annotate(key, None, 0.10, rng)
    rep = A.score(key, labs, n_boot=200)
    for t in ("presence", "count", "binding", "spatial"):
        assert rep[t]["n_used"] > 0
        assert 0.8 <= rep[t]["unweighted_accuracy"] <= 1.0                    # judge error 10% by construction
        assert rep[t]["accuracy"] == rep[t]["unweighted_accuracy"]            # census: weights are all 1


def test_stratified_sample_is_reweighted_to_the_population(tmp_path):
    rows, prompts, img = fake(tmp_path, n=120, seed=1)
    out = tmp_path / "audit"
    A.build(rows, prompts, img, out, seed=5, quotas={"presence": 40, "binding": 40, "count": 20, "spatial": 20}, n_practice=5, n_double=8)
    key = A.read_key(out / "key" / "audit_key.csv")
    assert sum(r["practice"] == "1" for r in key) == 5
    assert (out / "annotator_B").exists() and sum(r["double"] == "1" for r in key) <= 8
    labs = annotate(key, None, 0.0, random.Random(2))
    rep = A.score(key, labs, n_boot=100)
    assert all(abs(rep[t]["accuracy"] - 1.0) < 1e-9 for t in rep)             # error-free judge: weighted accuracy 1 regardless of allocation
    # practice items are never scored
    assert sum(rep[t]["n_items_labelled"] for t in rep) == len([r for r in key if r["practice"] == "0"])


def test_answer_mapping_and_ambiguity():
    c = {"type": "binding", "object": "cup", "color": "red"}
    assert A.truth(c, "col:red") == (1, 1) and A.truth(c, "col:blue") == (0, 1) and A.truth(c, "absent") == (0, 0) and A.truth(c, "amb") == (None, None)
    assert A.truth({"type": "count", "object": "x", "n": 2}, "2") == (1, 1) and A.truth({"type": "count", "object": "x", "n": 2}, "0") == (0, 0)
    assert A.truth({"type": "count", "object": "x", "n": 2}, "10+") == (0, 1)
    s = {"type": "spatial", "left": "cup", "right": "book"}
    assert A.truth(s, "obj:cup") == (1, 1) and A.truth(s, "obj:book") == (0, 1) and A.truth(s, "absent") == (0, 0)


def test_double_annotation_disagreements_are_excluded(tmp_path):
    rows, prompts, img = fake(tmp_path, n=40, seed=2)
    out = tmp_path / "audit"
    A.build(rows, prompts, img, out, seed=2, quotas={}, n_practice=0, n_double=40)
    key = A.read_key(out / "key" / "audit_key.csv")
    la = annotate(key, None, 0.0, random.Random(0))
    lb = dict(la)
    flipped = [r["item_id"] for r in key if r["double"] == "1" and r["type"] == "presence"][:3]
    for i in flipped:
        lb[i] = "no" if la[i] == "yes" else "yes"
    rep = A.score(key, la, lb, n_boot=50)
    assert rep["presence"]["n_excluded_ambiguous_or_disagree"] >= len(flipped)
    assert rep["presence"]["double_annotation"]["raw_agreement"] < 1.0


def test_size_question_is_neutral_and_maps_answers_to_truth():
    import random
    import audit_dev as AD
    c = {"type": "size", "larger": "vase", "smaller": "hat"}
    q = AD.question(c, random.Random(0))
    keys = [o[0] for o in q["options"]]
    assert set(keys) == {"obj:vase", "obj:hat", "same", "absent", "amb"} and "larger" not in q["text"].lower().split("which one looks")[0]
    assert AD.truth(c, "obj:vase") == (1, 1) and AD.truth(c, "obj:hat") == (0, 1) and AD.truth(c, "same") == (0, 1)
    assert AD.truth(c, "absent") == (0, 0) and AD.truth(c, "amb") == (None, None)
