"""Dependency bootstrap (kaggle/env_bootstrap.py): the diffusers 0.41 / old-torchao import failure is reproduced offline with a stub torchao
distribution, and every failure path of bootstrap() raises and records ok=false."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))
import env_bootstrap as E  # noqa: E402


def stub_torchao(d, version="0.12.0"):
    """A torchao distribution WITHOUT FqnToConfig (as in torchao < 0.15), discoverable by importlib.metadata."""
    (d / "torchao" / "quantization").mkdir(parents=True)
    (d / "torchao" / "__init__.py").write_text(f"__version__ = {version!r}\n")
    (d / "torchao" / "quantization" / "__init__.py").write_text("def quantize_(*a, **k):\n    raise NotImplementedError\n")
    di = d / f"torchao-{version}.dist-info"
    di.mkdir()
    (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: torchao\nVersion: {version}\n")
    (di / "top_level.txt").write_text("torchao\n")
    (di / "RECORD").write_text("")
    return d


diffusers_installed = pytest.mark.skipif(not __import__("importlib").util.find_spec("diffusers"), reason="diffusers not installed")


@diffusers_installed
def test_old_torchao_reproduces_the_kaggle_import_failure_and_absence_fixes_it(tmp_path):
    stub = stub_torchao(tmp_path / "site")
    ok, info, raw = E.import_probe(env=dict(os.environ, PYTHONPATH=str(stub)))
    assert not ok and info is None
    assert "FqnToConfig" in raw["stderr"]
    ok, info, raw = E.import_probe(env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    if E.installed_version("torchao"):
        pytest.skip("this environment has torchao installed; the clean-import half needs an environment without it")
    assert ok, raw["stderr"][-1500:]
    assert info["diffusers"] == "0.41.0" and info["torchao_importable"] is False


def test_parse_and_classify_pip_check():
    before = E.parse_pip_check("kaggle-env 1.0 requires foo, which is not installed.\n")
    after = E.parse_pip_check("kaggle-env 1.0 requires foo, which is not installed.\n"
                              "diffusers 0.41.0 has requirement huggingface-hub>=0.34, but you have huggingface-hub 0.20.0.\n"
                              "otherpkg 2.0 requires torchao, which is not installed.\n")
    blocking, other = E.classify_conflicts(before, after)
    assert [b["line"].split()[0] for b in blocking] == ["diffusers"]
    assert {o["line"].split()[0]: o["preexisting"] for o in other} == {"kaggle-env": True, "otherpkg": False}
    assert E.parse_pip_check("No broken requirements found.\n") == []


class Fake:
    """Scripted pip / probe results for bootstrap() failure paths."""
    def __init__(self, install_rc=0, torch=("2.11.0", "2.11.0"), torchao=(None,), uninstall_rc=0, probe_ok=True, pip_check_after=""):
        self.install_rc, self.torch, self.torchao, self.uninstall_rc, self.probe_ok, self.after = install_rc, list(torch), list(torchao), uninstall_rc, probe_ok, pip_check_after
        self.calls, self.checks = [], 0

    def run(self, cmd, timeout=0, env=None):
        self.calls.append(cmd)
        out = ""
        if cmd[-1] == "check":
            self.checks += 1
            out = "No broken requirements found.\n" if self.checks == 1 else (self.after or "No broken requirements found.\n")
            return dict(cmd=cmd, returncode=0, stdout=out, stderr="")
        if "install" in cmd:
            return dict(cmd=cmd, returncode=self.install_rc, stdout="", stderr="ERROR: resolver failure" if self.install_rc else "")
        if "uninstall" in cmd:
            return dict(cmd=cmd, returncode=self.uninstall_rc, stdout="", stderr="")
        raise AssertionError(cmd)

    def version(self, dist, python=None):
        seq = self.torch if dist == "torch" else self.torchao
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def probe(self, python=None, env=None):
        info = dict(torch="2.11.0", diffusers="0.41.0", torchao_importable=False) if self.probe_ok else None
        return self.probe_ok, info, dict(returncode=0 if self.probe_ok else 1, stdout="", stderr="" if self.probe_ok else "ImportError: cannot import name 'FqnToConfig'")


def run(monkeypatch, tmp_path, fake, **kw):
    monkeypatch.setattr(E, "_run", fake.run)
    monkeypatch.setattr(E, "installed_version", fake.version)
    monkeypatch.setattr(E, "import_probe", fake.probe)
    return E.bootstrap(tmp_path / "req.txt", tmp_path / "boot", loaded_modules=[], **kw)


def record(tmp_path):
    return json.loads((tmp_path / "boot" / "env_bootstrap.json").read_text())


def test_success_removes_old_torchao_and_records_ok(monkeypatch, tmp_path):
    fake = Fake(torchao=("0.12.0", None))
    rec = run(monkeypatch, tmp_path, fake)
    assert rec["ok"] is True and record(tmp_path)["ok"] is True
    assert rec["steps"]["remove_torchao"]["installed_version"] == "0.12.0"
    assert any("uninstall" in c for c in fake.calls)


@pytest.mark.parametrize("fake, needle", [
    (Fake(install_rc=1), "pip install"),
    (Fake(torchao=("0.12.0",)), "could not uninstall torchao"),             # still installed after uninstall
    (Fake(torchao=("0.12.0", None), uninstall_rc=1), "could not uninstall torchao"),
    (Fake(torch=("2.11.0+cu128", "2.6.0")), "replaced torch"),
    (Fake(probe_ok=False), "fresh-process import"),
    (Fake(pip_check_after="transformers 5.19.0 has requirement tokenizers>=0.22, but you have tokenizers 0.10.\n"), "pip check reports conflicts"),
])
def test_every_failure_raises_and_records_not_ok(monkeypatch, tmp_path, fake, needle):
    with pytest.raises(E.BootstrapError, match=needle):
        run(monkeypatch, tmp_path, fake)
    assert record(tmp_path)["ok"] is False


def test_unrelated_pip_conflict_is_recorded_not_hidden(monkeypatch, tmp_path):
    rec = run(monkeypatch, tmp_path, Fake(pip_check_after="otherpkg 2.0 requires torchao, which is not installed.\n"))
    assert rec["ok"] is True and rec["steps"]["pip_check_after"]["other"][0]["preexisting"] is False


def test_stale_modules_require_a_kernel_restart_unless_this_kernel_already_bootstrapped(monkeypatch, tmp_path):
    with pytest.raises(E.BootstrapError, match="restart the kernel"):
        E.bootstrap(tmp_path / "req.txt", tmp_path / "boot", loaded_modules=["diffusers", "torch"])
    assert record(tmp_path)["ok"] is False
    run(monkeypatch, tmp_path, Fake())                                      # a successful bootstrap in this kernel ...
    again = E.bootstrap(tmp_path / "req.txt", tmp_path / "boot", loaded_modules=["diffusers"])
    assert again["ok"] is True and again["reused_in_same_kernel"] is True    # ... may be re-run without a restart


def test_requirements_keep_kaggle_torch_and_pin_the_stack():
    req = (ROOT / "requirements-kaggle.txt").read_text()
    lines = [l.strip() for l in req.splitlines() if l.strip() and not l.startswith("#")]
    assert not any(l.split("=")[0].split("<")[0].split(">")[0] in ("torch", "torchvision", "torchao") for l in lines)
    for pin in ("diffusers==0.41.0", "transformers==5.19.0", "accelerate==1.15.0", "safetensors==0.8.0"):
        assert pin in lines
