"""Model cache paths are writable: on Kaggle never inside /kaggle/input or the read-only source package (Errno 30 at PREFLIGHT)."""
import re
import sys
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_cache as MC  # noqa: E402

KAGGLE_SRC = PurePosixPath("/kaggle/input/gbc-source/gbc_source")


def test_kaggle_default_is_working_dir(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    d = MC.model_cache_dir(create=True, root=tmp_path / "src_pkg", working=working, env={})
    assert d == working / "models" and d.is_dir()


def test_kaggle_input_root_never_used_even_without_working_dir(tmp_path):
    assert MC.on_kaggle(root=KAGGLE_SRC, working=tmp_path / "absent")
    d = MC.model_cache_dir(create=False, root=KAGGLE_SRC, working=Path("/kaggle/working"), env={})
    assert d.as_posix() == "/kaggle/working/models"


@pytest.mark.parametrize("bad", ["/kaggle/input/gbc-source/gbc_source/models", "/kaggle/input/x"])
def test_env_override_into_read_only_input_is_refused(bad):
    with pytest.raises(OSError, match="read-only"):
        MC.model_cache_dir(create=False, root=KAGGLE_SRC, env={"GBC_MODEL_CACHE": bad})


def test_env_override_into_source_package_on_kaggle_is_refused(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    pkg = tmp_path / "pkg"
    with pytest.raises(OSError, match="read-only"):
        MC.model_cache_dir(create=False, root=pkg, working=working, env={"GBC_MODEL_CACHE": str(pkg / "models")})


def test_env_override_and_local_default(tmp_path):
    assert MC.model_cache_dir(create=True, root=tmp_path, working=tmp_path / "absent", env={"GBC_MODEL_CACHE": str(tmp_path / "c")}) == tmp_path / "c"
    assert MC.model_cache_dir(create=False, root=tmp_path, working=tmp_path / "absent", env={}) == tmp_path / "models"


def test_no_loader_hardcodes_a_cache_inside_the_package():
    for d in ("src", "kaggle", "tests"):
        for f in (ROOT / d).glob("*.py"):
            if f.name.startswith("test_"):
                continue
            text = f.read_text(encoding="utf-8")
            assert not re.search(r"cache_dir\s*=\s*ROOT\s*/", text), f"{f}: cache_dir inside the package"
            assert 'ROOT / "models"' not in text, f"{f}: model cache inside the package"
            for m in re.finditer(r"\.from_pretrained\(([^)]*)\)", text):
                if "cache_dir" not in m.group(1) and "**" not in m.group(1):
                    assert "local_files_only" in m.group(1) or "DDIMScheduler" in m.group(0), f"{f}: from_pretrained without cache_dir: {m.group(0)[:80]}"
