"""Print or write executable experiment manifests (src/manifests.py)."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.argv[0] = str(ROOT / "src" / "manifests.py")
runpy.run_path(str(ROOT / "src" / "manifests.py"), run_name="__main__")
