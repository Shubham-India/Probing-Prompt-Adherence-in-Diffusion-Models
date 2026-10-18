"""Frozen analyses: python scripts/final_analysis.py {dev|confirm} --rows eval_rows.jsonl ..."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.argv[0] = str(ROOT / "src" / "r3_analyze.py")
runpy.run_path(str(ROOT / "src" / "r3_analyze.py"), run_name="__main__")
