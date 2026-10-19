r"""Static checks for the LaTeX draft (no LaTeX installation needed) and an optional compile.
Checks: every \input / \includegraphics file exists; every \cite key is in references.bib; every \ref has a \label; every \newcommand macro used from
tables/numbers.tex is defined; PENDING placeholders are counted; forbidden strong claims are absent from the section text.
Usage: python paper/check_paper.py [--compile]      exit 0 only if all static checks pass."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
BANNED = ["state-of-the-art", "novel ", "publication-ready", "accepted", "significantly", "proves", "we show that"]


def tex(rel):
    return (P / rel).read_text(encoding="utf-8")


def main():
    errs, notes = [], []
    seen, stack, texts = set(), ["main.tex"], {}
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        path = f if f.endswith(".tex") else f + ".tex"
        if not (P / path).exists():
            errs.append(f"missing input: {f}")
            continue
        t = tex(path)
        texts[path] = t
        stack += re.findall(r"\\input\{([^}]+)\}", t)
    alltext = "\n".join(texts.values())
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", alltext):
        if not (P / g).exists():
            errs.append(f"missing figure: {g}")
    bib = tex("references.bib")
    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cited = set(k.strip() for c in re.findall(r"\\cite[pt]?\{([^}]+)\}", alltext) for k in c.split(","))
    errs += [f"unresolved citation: {k}" for k in sorted(cited - keys)]
    notes.append(f"{len(cited)} cited keys, {len(keys)} bib entries, {len(keys - cited)} uncited")
    labels = set(re.findall(r"\\label\{([^}]+)\}", alltext))
    refs = set(r.strip() for c in re.findall(r"\\ref\{([^}]+)\}", alltext) for r in c.split(","))
    errs += [f"unresolved reference: {r}" for r in sorted(refs - labels)]
    nums = tex("tables/numbers.tex")
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", nums)) | set(re.findall(r"\\newcommand\{\\(\w+)\}", alltext))
    builtin = {"pending", "tfifty", "textbackslash"}
    used = set(re.findall(r"\\([A-Za-z]+)\{\}", alltext))
    errs += [f"macro used but not defined: \\{m}" for m in sorted(used - defined - builtin) if m[0].islower() and m not in {"textbf", "emph", "item"}]
    pend = len(re.findall(r"PENDING|\\pending", alltext + "".join(p.read_text(encoding="utf-8") for p in (P / "tables").glob("*.tex"))))
    notes.append(f"PENDING placeholders: {pend}")
    for ph in BANNED:
        for name, t in texts.items():
            if re.search(r"(?<![a-z])" + re.escape(ph.strip()) + r"(?![a-z])", t.lower()):      # whole words only ("improves" is not "proves")
                errs.append(f"banned phrase '{ph.strip()}' in {name}")
    unverified = len(re.findall(r"x-verification=\{ABS\}", bib))
    notes.append(f"bibliography entries verified at abstract level only: {unverified}")
    if "--compile" in sys.argv:
        tool = shutil.which("latexmk") or shutil.which("pdflatex") or shutil.which("tectonic")
        if not tool:
            notes.append("COMPILE SKIPPED: no LaTeX toolchain on PATH (install TeX Live or MiKTeX, then run: latexmk -pdf main.tex)")
        else:
            cmd = [tool, "-pdf", "-interaction=nonstopmode", "main.tex"] if tool.endswith(("latexmk", "latexmk.exe")) else [tool, "-interaction=nonstopmode", "main.tex"]
            r = subprocess.run(cmd, cwd=P, capture_output=True, text=True)
            notes.append(f"compile exit code {r.returncode}")
            if r.returncode:
                errs.append("LaTeX compilation failed (see paper/main.log)")
    print("\n".join(notes))
    print("\n".join(errs) if errs else "STATIC CHECKS PASSED")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
