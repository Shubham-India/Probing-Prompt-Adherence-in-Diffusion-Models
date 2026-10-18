"""Verify protocol/protocol_freeze.json (signed, nothing changed since the freeze)."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.argv[0] = str(ROOT / "src" / "protocol_r3.py")
runpy.run_path(str(ROOT / "src" / "protocol_r3.py"), run_name="__main__")
