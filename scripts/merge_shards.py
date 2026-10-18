"""Merge all shards of a stage against the full stage manifest (rejects missing/duplicate/corrupt/mismatching outputs)."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.argv[0] = str(ROOT / "src" / "r3_merge.py")
runpy.run_path(str(ROOT / "src" / "r3_merge.py"), run_name="__main__")
