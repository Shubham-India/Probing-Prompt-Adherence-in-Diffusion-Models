"""Build path for the PDF without a local TeX installation: regenerate assets, run the static checks, and pack everything main.tex needs
(sections, tables, figures, bibliography) into paper/paper_overleaf.zip. Upload the ZIP to Overleaf (New Project -> Upload Project) and compile
with pdfLaTeX + BibTeX. With a local TeX installation instead: `cd paper && latexmk -pdf main.tex`.
Usage: python paper/make_overleaf_zip.py"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

P = Path(__file__).resolve().parent


def deps():
    seen, stack, files = set(), ["main.tex"], {"main.tex", "references.bib"}
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        path = f if f.endswith(".tex") else f + ".tex"
        files.add(path)
        t = (P / path).read_text(encoding="utf-8")
        stack += re.findall(r"\\input\{([^}]+)\}", t)
        files |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t))
    return sorted(files)


def main():
    subprocess.run([sys.executable, str(P / "make_paper_assets.py")], check=True)
    chk = subprocess.run([sys.executable, str(P / "check_paper.py")], capture_output=True, text=True)
    print(chk.stdout[-400:])
    if chk.returncode != 0:
        sys.exit("static checks failed; fix before packing")
    out = P / "paper_overleaf.zip"
    files = deps()
    missing = [f for f in files if not (P / f).exists()]
    if missing:
        sys.exit(f"missing files: {missing}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(P / f, f)
    print(f"wrote {out} ({len(files)} files)")


if __name__ == "__main__":
    main()
