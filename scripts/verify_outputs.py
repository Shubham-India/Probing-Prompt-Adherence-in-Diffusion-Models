"""Verify one worker output folder against its shard manifest: python scripts/verify_outputs.py --manifest M --inputs DIR --out DIR/verify"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.argv[0] = str(ROOT / "src" / "r3_merge.py")
runpy.run_path(str(ROOT / "src" / "r3_merge.py"), run_name="__main__")
