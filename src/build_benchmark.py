"""Builds data/prompts.json from explicit templates.

Every prompt is stored with the constraints it must satisfy, so evaluation never has to
parse the prompt text. Constraint types:
  presence : {"type": "presence", "object": o}
  count    : {"type": "count", "object": o, "n": n}
  binding  : {"type": "binding", "object": o, "color": c}
  spatial  : {"type": "spatial", "left": a, "right": b}
Complexity = number of constraints (each presence/count/binding/spatial item counts once).
"""
import json
from pathlib import Path

NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four"}
SURFACE = " on a wooden table"


def presence(obj):
    return {"type": "presence", "object": obj}


def prompt_entry(pid, category, text, constraints, group, n_objects, **extra):
    return {"id": pid, "category": category, "prompt": text, "constraints": constraints,
            "pair_group": group, "n_objects": n_objects, "n_constraints": len(constraints), **extra}


def presence_prompts():
    pairs = [("dog", "bicycle"), ("cat", "umbrella"), ("horse", "bench"), ("bird", "clock")]
    return [prompt_entry(f"presence_{a}_{b}", "presence", f"a {a} and a {b}",
                         [presence(a), presence(b)], f"presence_{a}_{b}", 2)
            for a, b in pairs]


def binding_prompts():
    """Each object pair appears twice with the colours swapped (the controlled variant)."""
    pairs = [("cube", "sphere"), ("cup", "book"), ("chair", "vase")]
    out = []
    for a, b in pairs:
        for tag, (ca, cb) in {"v1": ("red", "blue"), "v2": ("blue", "red")}.items():
            cons = [presence(a), presence(b),
                    {"type": "binding", "object": a, "color": ca},
                    {"type": "binding", "object": b, "color": cb}]
            out.append(prompt_entry(f"binding_{a}_{b}_{tag}", "binding",
                                    f"a {ca} {a} and a {cb} {b}{SURFACE}", cons,
                                    f"binding_{a}_{b}", 2))
    return out


def counting_prompts():
    items = [("apple", "yellow"), ("ball", "red")]
    out = []
    for obj, color in items:
        for n in (2, 3, 4):
            cons = [{"type": "count", "object": obj, "n": n}, {"type": "binding", "object": obj, "color": color}]
            out.append(prompt_entry(f"count_{obj}_{n}", "counting",
                                    f"{NUMBER_WORD[n]} {color} {obj}s{SURFACE}", cons,
                                    f"count_{obj}", 1, count=n))
    return out


def spatial_prompts():
    pairs = [("cat", "dog"), ("cup", "book"), ("car", "bicycle")]
    out = []
    for a, b in pairs:
        for tag, (l, r) in {"lr": (a, b), "rl": (b, a)}.items():
            cons = [presence(a), presence(b), {"type": "spatial", "left": l, "right": r}]
            out.append(prompt_entry(f"spatial_{a}_{b}_{tag}", "spatial",
                                    f"a {l} to the left of a {r}", cons, f"spatial_{a}_{b}", 2))
    return out


def multi_prompts():
    """Binding + spatial together (and count + binding + presence); complexity 6-7."""
    out = []
    for a, b in [("cup", "book"), ("chair", "vase")]:
        for tag, (l, r, cl, cr) in {"v1": (a, b, "red", "blue"), "v2": (b, a, "red", "blue")}.items():
            cons = [presence(a), presence(b),
                    {"type": "binding", "object": l, "color": cl},
                    {"type": "binding", "object": r, "color": cr},
                    {"type": "spatial", "left": l, "right": r}]
            out.append(prompt_entry(f"multi_{a}_{b}_{tag}", "multi",
                                    f"a {cl} {l} to the left of a {cr} {r}{SURFACE}", cons,
                                    f"multi_{a}_{b}", 2))
    for n in (2, 3):
        cons = [{"type": "count", "object": "ball", "n": n}, {"type": "binding", "object": "ball", "color": "red"},
                presence("cube"), {"type": "binding", "object": "cube", "color": "blue"}]
        out.append(prompt_entry(f"multi_count_ball_{n}", "multi",
                                f"{NUMBER_WORD[n]} red balls and a blue cube{SURFACE}", cons,
                                "multi_count_ball", 2, count=n))
    return out


if __name__ == "__main__":
    prompts = presence_prompts() + binding_prompts() + counting_prompts() + spatial_prompts() + multi_prompts()
    path = Path(__file__).resolve().parent.parent / "data" / "prompts.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(prompts, indent=1))
    print(f"wrote {len(prompts)} prompts to {path}")
