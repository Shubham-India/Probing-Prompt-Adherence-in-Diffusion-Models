"""Validate Benchmark V2 and the PS family (regeneration, leakage, one-field counterfactuals, hashes)."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import subprocess, sys
r1 = subprocess.run([sys.executable, str(ROOT / "src" / "benchmark_v2.py"), "--validate"])
r2 = subprocess.run([sys.executable, str(ROOT / "src" / "benchmark_v2.py"), "--validate-ps"])
sys.exit(r1.returncode or r2.returncode)
