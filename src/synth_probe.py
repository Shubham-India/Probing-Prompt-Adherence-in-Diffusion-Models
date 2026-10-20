"""E22 synthetic probe: flat coloured shapes with KNOWN ground truth, to measure judge accuracy where the answer is not in doubt.
Shapes stand in for objects (square=cube, circle=sphere, triangle=cone). Constraints use the legacy format so any judge can be run on
the images; `expected()` computes the truth independently from the scene description, not from pixels.
Usage: python src/synth_probe.py --out outputs/synth_probe --n 200 --seed 0   (writes images/ and truth.jsonl)"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

RGB = {"red": (220, 30, 30), "blue": (30, 60, 220), "yellow": (240, 220, 30), "green": (30, 170, 60), "orange": (250, 140, 20),
       "purple": (140, 40, 170), "white": (255, 255, 255), "black": (10, 10, 10)}
SHAPES = {"cube": "square", "sphere": "circle", "cone": "triangle"}
SIZE = 512


def make_scene(rng, n_objects=2, with_count=False):
    """Objects get distinct shapes/colours and are placed in distinct horizontal bands so centres never tie."""
    shapes = rng.sample(list(SHAPES), n_objects)
    colours = rng.sample([c for c in RGB if c != "white"], n_objects)
    xs = rng.sample([90, 200, 310, 420], n_objects)   # spacing 110 > shape diameter 80: objects never overlap
    objs = []
    for sh, col, x in zip(shapes, colours, xs):
        n = rng.randint(2, 3) if (with_count and not objs) else 1
        objs.append(dict(noun=sh, colour=col, count=n, x=x, y=rng.randint(60, 150) if n > 1 else rng.randint(60, SIZE - 100)))
    return objs


def render(objs):
    img = Image.new("RGB", (SIZE, SIZE), RGB["white"] if all(o["colour"] != "white" for o in objs) else (200, 200, 200))
    d = ImageDraw.Draw(img)
    boxes = []
    for o in objs:
        for k in range(o["count"]):
            cx, cy, r = o["x"], o["y"] + k * 110, 40   # instances of a counted object are stacked vertically
            box = [cx - r, cy - r, cx + r, cy + r]
            shape = SHAPES[o["noun"]]
            if shape == "square":
                d.rectangle(box, fill=RGB[o["colour"]])
            elif shape == "circle":
                d.ellipse(box, fill=RGB[o["colour"]])
            else:
                d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=RGB[o["colour"]])
            boxes.append((o["noun"], (cx, cy)))
    return img, boxes


def constraints(objs):
    cons = []
    for o in objs:
        cons.append({"type": "count", "object": o["noun"], "n": o["count"]} if o["count"] > 1 else {"type": "presence", "object": o["noun"]})
    cons += [{"type": "binding", "object": o["noun"], "color": o["colour"]} for o in objs]
    if len(objs) >= 2:
        a, b = objs[0], objs[1]
        cons.append({"type": "spatial", "left": a["noun"], "right": b["noun"]})
    return cons


def expected(objs, cons):
    """Truth for each constraint, computed from the scene description only."""
    by = {o["noun"]: o for o in objs}
    out = []
    for c in cons:
        if c["type"] == "presence":
            out.append(1)
        elif c["type"] == "count":
            out.append(int(by[c["object"]]["count"] == c["n"]))
        elif c["type"] == "binding":
            out.append(int(by[c["object"]]["colour"] == c["color"]))
        else:
            out.append(int(by[c["left"]]["x"] < by[c["right"]]["x"]))
    return out


def corrupt(cons, rng):
    """Make some constraints false on purpose (swap a colour, flip a relation, change a count) so both labels occur."""
    cons = json.loads(json.dumps(cons))
    for c in cons:
        if rng.random() < 0.4:
            if c["type"] == "binding":
                c["color"] = rng.choice([x for x in RGB if x != c["color"] and x != "white"])
            elif c["type"] == "spatial":
                c["left"], c["right"] = c["right"], c["left"]
            elif c["type"] == "count":
                c["n"] = c["n"] + 1
    return cons


def export_for_judge(out):
    """Writes samples.jsonl + prompts_synth.json next to truth.jsonl so that `python src/evaluate.py --run <out> --prompts <out>/prompts_synth.json` can score the images."""
    out = Path(out)
    truth = [json.loads(l) for l in (out / "truth.jsonl").read_text().splitlines() if l.strip()]
    (out / "prompts_synth.json").write_text(json.dumps([dict(id=r["name"], category="synthetic", prompt="", constraints=r["constraints"]) for r in truth]))
    with open(out / "samples.jsonl", "w") as f:
        for r in truth:
            f.write(json.dumps(dict(name=r["name"], prompt_id=r["name"], category="synthetic", prompt="", cfg=0, steps=0, seed=0, size=SIZE)) + "\n")
    return len(truth)


def score_eval(out, eval_name="eval.jsonl"):
    """Judge accuracy per constraint type against the exact truth in truth.jsonl (E22). Returns {type: {n, accuracy, tp, fp, fn, tn}} plus ALL."""
    out = Path(out)
    truth = {r["name"]: r for r in (json.loads(l) for l in (out / "truth.jsonl").read_text().splitlines() if l.strip())}
    stats = {}
    for line in (out / eval_name).read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        tr = truth[e["name"]]
        for c, exp, pred in zip(tr["constraints"], tr["expected"], e["constraint_results"]):
            for key in (c["type"], "ALL"):
                s = stats.setdefault(key, dict(tp=0, fp=0, fn=0, tn=0))
                s["tp" if exp and pred else "fp" if pred else "fn" if exp else "tn"] += 1
    for s in stats.values():
        n = sum(s.values())
        s["n"], s["accuracy"] = n, (s["tp"] + s["tn"]) / n if n else None
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-judge-run", action="store_true", help="after generating, also write samples.jsonl and prompts_synth.json for evaluate.py")
    ap.add_argument("--score", default=None, metavar="EVAL_JSONL", help="do not generate: score an existing <out>/<EVAL_JSONL> against truth.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    if a.score:
        print(json.dumps(score_eval(out, a.score), indent=1))
        raise SystemExit(0)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    with open(out / "truth.jsonl", "w") as f:
        for i in range(a.n):
            objs = make_scene(rng, rng.choice([2, 3]), with_count=(i % 4 == 0))
            img, boxes = render(objs)
            cons = corrupt(constraints(objs), rng)
            name = f"synth_{i:04d}"
            img.save(out / "images" / f"{name}.png")
            f.write(json.dumps(dict(name=name, prompt_id=name, objects=objs, constraints=cons, expected=expected(objs, cons))) + "\n")
    print("wrote", a.n, "scenes to", out)
    if a.export_judge_run:
        print("exported judge run files for", export_for_judge(out), "scenes")
