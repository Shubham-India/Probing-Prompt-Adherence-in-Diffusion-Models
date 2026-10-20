import random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import synth_probe as S, judge_vqa as Q


def test_render_matches_truth_at_object_centres():
    rng = random.Random(0)
    for i in range(60):
        objs = S.make_scene(rng, rng.choice([2, 3]), with_count=(i % 4 == 0))
        img, boxes = S.render(objs)
        by = {o["noun"]: o for o in objs}
        for noun, (cx, cy) in boxes:
            px = img.getpixel((cx, min(cy + 8, S.SIZE - 1))) if S.SHAPES[noun] == "triangle" else img.getpixel((cx, cy))
            assert px == S.RGB[by[noun]["colour"]], (noun, px)
        centres = {n: c[0] for n, c in boxes}
        assert len(set(centres.values())) == len(centres)      # no ties in the horizontal order


def test_expected_matches_a_pixel_based_recount():
    rng = random.Random(1)
    n_true = n_false = 0
    for i in range(80):
        objs = S.make_scene(rng, 2, with_count=(i % 4 == 0))
        img, boxes = S.render(objs)
        cons = S.corrupt(S.constraints(objs), rng)
        exp = S.expected(objs, cons)
        xs = {n: c[0] for n, c in boxes}
        for c, e in zip(cons, exp):
            if c["type"] == "spatial":
                assert e == int(xs[c["left"]] < xs[c["right"]])
            if c["type"] == "count":
                assert e == int(sum(1 for n, _ in boxes if n == c["object"]) == c["n"])
        n_true += sum(exp); n_false += len(exp) - sum(exp)
    assert n_true > 20 and n_false > 20                          # both labels occur


def test_judge_interface_flags_uncertainty():
    class B(Q.Backend):
        def ask(self, image, q): return 0.5 if "left" in q.lower() else 0.95
    res, probs, unc = Q.score_constraints(None, [{"type": "presence", "object": "cup"}, {"type": "spatial", "left": "a", "right": "b"}], B())
    assert res == [1, 1] and unc == 1
    try:
        Q.Backend().ask(None, "x"); assert False
    except NotImplementedError:
        pass


def test_export_and_score_roundtrip(tmp_path):
    import json
    truth = [dict(name="synth_0000", prompt_id="synth_0000", objects=[], constraints=[{"type": "presence", "object": "cube"}, {"type": "binding", "object": "cube", "color": "red"}], expected=[1, 0]),
             dict(name="synth_0001", prompt_id="synth_0001", objects=[], constraints=[{"type": "presence", "object": "cone"}], expected=[0])]
    (tmp_path / "truth.jsonl").write_text("\n".join(json.dumps(t) for t in truth) + "\n")
    assert S.export_for_judge(tmp_path) == 2
    assert json.loads((tmp_path / "prompts_synth.json").read_text())[0]["constraints"] == truth[0]["constraints"]
    (tmp_path / "eval.jsonl").write_text(json.dumps(dict(name="synth_0000", constraint_results=[1, 1])) + "\n" + json.dumps(dict(name="synth_0001", constraint_results=[0])) + "\n")
    st = S.score_eval(tmp_path)
    assert st["presence"]["accuracy"] == 1.0 and st["binding"]["accuracy"] == 0.0 and st["ALL"]["n"] == 3
