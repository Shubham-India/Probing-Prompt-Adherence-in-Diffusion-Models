import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import audit_sample as A, validate_human as H

PROMPTS = {p["id"]: p for p in json.loads((ROOT / "data/prompts.json").read_text())}
ROWS = [json.loads(l) for l in (ROOT / "outputs/main/eval.jsonl").read_text().splitlines()]


def test_draw_reproducible_and_stratified():
    a, b = A.draw(ROWS, PROMPTS, 120, 7), A.draw(ROWS, PROMPTS, 120, 7)
    assert [(x["name"], x["i"]) for x in a] == [(x["name"], x["i"]) for x in b] and len(a) == 120
    assert {x["type"] for x in a} == {"presence", "count", "binding", "spatial"}
    assert [(x["name"], x["i"]) for x in A.draw(ROWS, PROMPTS, 120, 8)] != [(x["name"], x["i"]) for x in a]


def test_blinded_export_hides_arm_and_verdict(tmp_path):
    n = A.export(ROOT / "outputs/main", ROWS, PROMPTS, tmp_path, 50, 3)
    assert n == 50
    items = (tmp_path / "audit_items.csv").read_text()
    assert list(csv.reader(items.splitlines()))[0] == ["item_id", "image_file", "question", "label"]
    for leak in ("cfg", "__st", "seed", "judge"):
        assert leak not in items.lower()
    key = list(csv.DictReader(open(tmp_path / "audit_key.csv")))
    assert len(key) == 50 and "name" in key[0] and "name" not in items.splitlines()[0]
    assert not any(k["name"] in items for k in key)


def test_wilson_and_kappa_known_values():
    lo, hi = H.wilson(9, 10)
    assert abs(lo - 0.5958) < 1e-3 and abs(hi - 0.9821) < 1e-3
    assert abs(H.kappa([1, 1, 0, 0], [1, 1, 0, 0]) - 1) < 1e-9
    assert abs(H.kappa([1, 0, 1, 0], [1, 1, 0, 0]) - 0.0) < 1e-9


def test_analyse_flags_exploratory_and_ambiguous():
    key = [dict(item_id=f"i{k}", type="spatial", judge_pred=str(p), uncertain="0") for k, p in enumerate([1, 1, 0, 0, 1])]
    la = {"i0": "1", "i1": "0", "i2": "0", "i3": "amb", "i4": "0"}
    r = H.analyse(key, la)
    s = r["spatial"]
    assert s["n"] == 4 and s["tp"] == 1 and s["fp"] == 2 and s["tn"] == 1 and s["ambiguous_or_disagree_share"] == 0.2 and s["exploratory"] is True
