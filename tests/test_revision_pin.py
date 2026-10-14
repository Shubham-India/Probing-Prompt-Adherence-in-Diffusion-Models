import json, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config_schema as CS, generate as G

SHA = "f03de327dd89b501a01da37fc5240cf4fdba85a1"
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CFGS = {n: json.loads((ROOT / f"configs/{n}.json").read_text()) for n in ("experiment", "experiment_v2")}


def test_both_configs_pin_same_model_and_revision_and_validate():
    for n, c in CFGS.items():
        assert c["model_id"] == MODEL and c["revision"] == SHA, n
    assert CS.validate(CFGS["experiment"]) == "pilot" and CS.validate(CFGS["experiment_v2"]) == "v2"
    assert CFGS["experiment"]["model_id"] == CFGS["experiment_v2"]["model_id"] and CFGS["experiment"]["revision"] == CFGS["experiment_v2"]["revision"]


def test_load_pipeline_forwards_revision_to_from_pretrained(monkeypatch):
    seen = {}

    class FakePipe:
        scheduler = type("S", (), {"config": {}})()
        def set_progress_bar_config(self, **k): pass
        def to(self, d): return self
    def fake_from_pretrained(model_id, **kw):
        seen.update(model_id=model_id, **kw)
        return FakePipe()
    monkeypatch.setattr(G.StableDiffusionPipeline, "from_pretrained", staticmethod(fake_from_pretrained))
    monkeypatch.setattr(G.DDIMScheduler, "from_config", staticmethod(lambda c: object()))
    G.load_pipeline(CFGS["experiment_v2"]["model_id"], CFGS["experiment_v2"]["revision"])
    assert seen["model_id"] == MODEL and seen["revision"] == SHA and seen["variant"] == "fp16"


@pytest.mark.parametrize("cfg_name", ["experiment", "experiment_v2"])
def test_every_entry_point_passes_config_revision(monkeypatch, tmp_path, cfg_name):
    """generate.main passes the configured revision unchanged to load_pipeline (only the draft marker is mapped to None)."""
    seen = {}
    monkeypatch.setattr(G, "load_pipeline", lambda model_id, revision=None: seen.update(m=model_id, r=revision) or (_ for _ in ()).throw(SystemExit(0)))
    cfg = dict(CFGS[cfg_name], seeds=[0])
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    prompts = json.loads((ROOT / "data/prompts.json").read_text())[:1]
    (tmp_path / "p.json").write_text(json.dumps(prompts))
    monkeypatch.setattr(sys, "argv", ["generate.py", "--config", str(tmp_path / "c.json"), "--prompts", str(tmp_path / "p.json"), "--out", str(tmp_path / "o"), "--limit", "1"])
    with pytest.raises(SystemExit):
        G.main()
    assert seen == {"m": MODEL, "r": SHA}


def test_other_entry_points_read_revision_from_config():
    """Static check: intervene.py, attention.py, run_branches.py and kaggle/env_probe.py all route the config revision into load_pipeline."""
    for rel in ("src/intervene.py", "src/attention.py", "src/run_branches.py", "kaggle/env_probe.py"):
        text = (ROOT / rel).read_text()
        assert 'get("revision")' in text and "load_pipeline(" in text, rel
