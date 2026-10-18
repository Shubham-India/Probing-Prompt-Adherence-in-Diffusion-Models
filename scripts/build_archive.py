"""Package a folder into multipart ZIPs with ARCHIVE_MANIFEST.json: python scripts/build_archive.py SRC_DIR ARCHIVE_DIR RUN_ID [PART_GB]"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import sys
sys.path.insert(0, str(ROOT / "kaggle"))
import nbtools
src, dst, rid = sys.argv[1:4]
m = nbtools.package(src, dst, rid, float(sys.argv[4]) if len(sys.argv) > 4 else 4.0)
print(len(m["files"]), "files in", sum(len(v) for v in m["groups"].values()), "parts")
