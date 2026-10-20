"""Tables/figures from confirmatory.json (r3_report.py); for the paper draft run python src/build_draft.py."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.argv[0] = str(ROOT / "src" / "r3_report.py")
runpy.run_path(str(ROOT / "src" / "r3_report.py"), run_name="__main__")
