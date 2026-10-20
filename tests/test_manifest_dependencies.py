"""Every branch / switch job of the full manifests has a base latent that an upstream stage (or, for DEV canonical CFG 7.5 bases, the R2a core run)
produces. A violation would surface only as worker failures on Kaggle, so it is checked here on the complete design."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import manifests as M  # noqa: E402

UPSTREAM = {"DEV_COMMIT": ("DEV_CORE", "DEV_COMMIT"), "TEST_COMMIT": ("TEST_CORE",), "TEST_CAG_SIZE": ("TEST_CAG_SIZE",)}
R2A_BASE = re.compile(r"^sg_\d{4}_CAN__cfg7\.5__st40__seed[01]$")      # R2a DEV core: canonical prompts, CFG 7.5, 40 steps, seeds 0-1 (200 scenes)


def test_every_base_latent_is_produced_upstream():
    jobs, _, _ = M.build()
    by = {j["job_id"]: j for j in jobs}
    external = set()
    for j in jobs:
        if not j.get("base_job"):
            continue
        assert j["stage"] in UPSTREAM, j["job_id"]
        b = by.get(j["base_job"])
        if b is None:
            assert j["stage"] == "DEV_COMMIT" and R2A_BASE.match(j["base_job"]), f"{j['job_id']}: base {j['base_job']} is produced by no stage"
            external.add(j["base_job"])
            continue
        assert b["stage"] in UPSTREAM[j["stage"]], f"{j['job_id']}: base made in {b['stage']}"
        assert b.get("save_latents") is True, f"{j['job_id']}: base {b['job_id']} does not save latents"
        assert b["split"] == j["split"] and b["scene_id"] == j["scene_id"]
    assert len(external) == 200
