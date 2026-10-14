import copy, json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config_schema as cs

V2 = json.loads((ROOT / "configs/experiment_v2.json").read_text())


def test_pilot_and_v2_accepted():
    assert cs.validate(json.loads((ROOT / "configs/experiment.json").read_text())) == "pilot"
    assert cs.validate(V2) == "v2"


def test_unknown_and_missing_keys_rejected():
    bad = copy.deepcopy(V2); bad["surprise"] = 1
    with pytest.raises(ValueError): cs.validate(bad)
    bad = copy.deepcopy(V2); del bad["restart"]
    with pytest.raises(ValueError): cs.validate(bad)
    bad = copy.deepcopy(V2); bad["restart"]["extra"] = 1
    with pytest.raises(ValueError): cs.validate(bad)


def test_bad_values_rejected():
    bad = copy.deepcopy(V2); bad["restart"]["grid"] = [6, 99]
    with pytest.raises(ValueError): cs.validate(bad)
    bad = copy.deepcopy(V2); bad["seeds_branch_bases"] = [9]
    with pytest.raises(ValueError): cs.validate(bad)
    bad = copy.deepcopy(V2); bad["revision"] = "main"
    with pytest.raises(ValueError): cs.validate(bad)
    bad = copy.deepcopy(V2); bad["restart"]["D"] = True
    with pytest.raises(ValueError): cs.validate(bad)


def test_draft_cannot_be_frozen():
    with pytest.raises(ValueError): cs.check_frozen(V2)                      # pinned revision but no split manifest / Addendum hash yet
    draft = copy.deepcopy(V2); draft["revision"] = cs.UNPINNED
    cs.validate(draft)
    with pytest.raises(ValueError): cs.check_frozen(draft)
    ok = copy.deepcopy(V2); ok["revision"] = "a" * 40; ok["split_manifest"] = "x.json"; ok["freeze"]["addendum_hash"] = "b" * 64
    cs.check_frozen(ok)
