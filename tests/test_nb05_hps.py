"""Notebook 05 (HPS v2 guard): hpsv2 1.2.0's top-level score() has no `cp` argument; only hpsv2.img_score.score() accepts the pinned checkpoint.
Regression for the Kaggle diagnostic failure of 2026-10-20: every scoring call must go through img_score.score with cp=CKPT, and the checkpoint
pin, its SHA-256, the tokenizer hash check and the fail-closed paths must stay."""
import inspect
import json
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "kaggle" / "nb05_dev_hps_guard.py"


def stub_hpsv2():
    """Module shapes of hpsv2 1.2.0 (wheel inspected 2026-10-20)."""
    top = types.ModuleType("hpsv2")
    img = types.ModuleType("hpsv2.img_score")

    def wrapper(imgs_path, prompt, hps_version="v2.0"):
        return []

    def scorer(img_path, prompt, cp=None, hps_version="v2.0"):
        return []
    top.score, img.score, top.img_score = wrapper, scorer, img
    return top, img


def test_only_img_score_accepts_the_checkpoint():
    top, img = stub_hpsv2()
    assert "cp" not in inspect.signature(top.score).parameters
    assert "cp" in inspect.signature(img.score).parameters


def test_notebook_scores_through_img_score_with_the_pinned_checkpoint():
    s = NB.read_text(encoding="utf-8")
    assert "from hpsv2 import img_score as hps_scoring" in s
    assert "hpsv2.score(" not in s
    calls = re.findall(r"hps_scoring\.score\((.*)\)", s)
    assert len(calls) == 5 and all("cp=CKPT" in c and 'hps_version=HPS["hps_version"]' in c for c in calls)
    assert 'assert "cp" in inspect.signature(hps_scoring.score).parameters' in s
    for kept in ('revision=HPS["revision"]', 'sha256=sha_file(CKPT)', 'HPS["bpe_vocab_sha256"]', "raise nbtools.ValidationError(\"HPS v2 diagnostic FAILED",
                 "verified=bool(STAGE_OK and not DRY_RUN and not DIAGNOSTIC_ONLY)"):
        assert kept in s, kept


def test_signature_check_passes_on_img_score_and_fails_on_the_wrapper():
    top, img = stub_hpsv2()
    check = lambda f: "cp" in inspect.signature(f).parameters
    assert check(img.score) and not check(top.score)


def test_released_notebook_matches_the_source():
    rel = ROOT / "release" / "guide_before_commitment" / "kaggle_notebooks" / "05_DEV_HPS_GUARD.ipynb"
    if not rel.exists():
        return
    nb = json.loads(rel.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    if "hps_scoring" in code:                     # a release built after the fix
        assert "hpsv2.score(" not in code


def test_vocabulary_hash_is_checked_whether_pre_existing_or_fetched():
    """DL-46: the tokenizer vocabulary's sha256 is asserted against the pin on every run, not only after a download."""
    s = NB.read_text(encoding="utf-8")
    i = s.index('DIAG["steps"]["hpsv2"]["vocab_sha256"] = sha_file(vocab)')
    j = s.index('if not vocab.exists():')
    assert i > j                                                        # computed after the optional fetch, outside its branch
    line = s[s.rfind("\n", 0, i) + 1:s.index("\n", i)]
    assert line.startswith("    DIAG") and not line.startswith("        ")    # not indented under the `if not vocab.exists()` branch
    assert 'assert DIAG["steps"]["hpsv2"]["vocab_sha256"] == HPS["bpe_vocab_sha256"]' in s
    assert '"vocab_source"' in s
    cfg = json.loads((ROOT / "configs" / "experiment_r3.json").read_text())
    assert cfg["hps_guard"]["bpe_vocab_sha256"] == "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
