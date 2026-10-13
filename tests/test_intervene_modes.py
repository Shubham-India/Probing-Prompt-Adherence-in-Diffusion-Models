import math, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import edit_probe as E

torch.manual_seed(0)
C, H, SIDE = 16, 2, 4
N = SIDE * SIDE


class ToyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = H
        self.to_q, self.to_k, self.to_v = nn.Linear(C, C), nn.Linear(C, C), nn.Linear(C, C)
        self.to_out = nn.ModuleList([nn.Linear(C, C), nn.Identity()])


ATTN = ToyAttn()
HS = torch.randn(2, N, C)
ENC = torch.randn(2, 77, C)
COLS = dict(I_a=[3], I_n=[4], I_o=[7])


def vanilla(attn, hs, enc):
    b = hs.shape[0]
    sp = lambda x: x.view(b, -1, attn.heads, x.shape[-1] // attn.heads).transpose(1, 2)
    out = F.scaled_dot_product_attention(sp(attn.to_q(hs)), sp(attn.to_k(enc)), sp(attn.to_v(enc)))
    return attn.to_out[1](attn.to_out[0](out.transpose(1, 2).reshape(b, -1, C)))


def proc(state, name="up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor"):
    return E.EditProcessor(state, name, SIDE)


def test_control_is_identical_to_plain_attention():
    for mode in ("off", "targeted"):
        st = E.EditState(mode=mode, onset=5, length=3, cols=COLS, sides=(SIDE,))   # targeted but step 0 is outside the window
        assert torch.equal(proc(st)(ATTN, HS, ENC), vanilla(ATTN, HS, ENC))
        assert st.mass == 0 and st.n_edits == 0


def test_targeted_edit_changes_only_conditional_element():
    st = E.EditState(mode="always_on", onset=0, length=3, cols=COLS, sides=(SIDE,), q=0.25)
    out = proc(st)(ATTN, HS, ENC)
    ref = vanilla(ATTN, HS, ENC)
    assert torch.equal(out[0], ref[0]) and not torch.allclose(out[1], ref[1], atol=1e-6)
    assert st.layer_calls_edited == 1 and st.mass > 0


def test_bias_confined_to_attribute_columns_with_equal_area_and_beta():
    S_n = E.top_q_mask(torch.rand(SIDE, SIDE), 0.25)
    S_o = E.top_q_mask(torch.rand(SIDE, SIDE), 0.25)
    B = E.build_pixel_bias(N, [3], S_n, S_o, math.log(2))
    nz = (B.abs() > 0).any(dim=2).any(dim=0).any(dim=0).nonzero().flatten().tolist()
    assert nz == [3] or nz == []                                   # only the attribute column (may cancel where masks overlap)
    assert all(min(abs(v), abs(abs(v) - math.log(2))) < 1e-6 for v in B.unique().tolist())
    rng = np.random.default_rng(0)
    r1 = E.blob_mask(SIDE, int(S_n.sum()), rng)
    assert int(r1.sum()) == int(S_n.sum()) == 4


def test_random_pair_uses_other_columns_with_same_area_and_magnitude():
    st_t = E.EditState(mode="always_on", cols=COLS, sides=(SIDE,), length=1)
    st_r = E.EditState(mode="random_pair", cols=COLS, sides=(SIDE,), length=1, rand_seed=3)
    proc(st_t)(ATTN, HS, ENC); proc(st_r)(ATTN, HS, ENC)
    assert st_r.rand_cols and not set(st_r.rand_cols) & set(COLS["I_a"]) and len(st_r.rand_cols) == len(COLS["I_a"])
    # equal k pixels per mask and equal beta; random blobs are disjoint so they never cancel, while overlapping targeted pixels (+beta-beta) can:
    assert st_r.mass >= st_t.mass and abs(st_r.mass - 2 * 4 * st_r.beta) < 1e-4      # k=4 pixels x 2 masks x 1 column x beta
    assert E.mass_ratio_ok(st_t.mass, st_t.mass * 1.04) and not E.mass_ratio_ok(st_t.mass, st_t.mass * 1.2)   # analysis-time acceptance (RB 13.8)


def run_steps(state, groups=("down", "up"), steps=40):
    names = {"down": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor", "up": "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor"}
    ps = [proc(state, names[g]) for g in groups]
    for _ in range(steps):
        for p in ps:
            p(ATTN, HS, ENC)
        state.end_step()
    return state


def test_bias_mass_equal_across_onsets_and_window_respected():
    masses = {}
    for onset in (0, 14, 28):
        st = run_steps(E.EditState(mode="always_on", onset=onset, length=12, cols=COLS, sides=(SIDE,), q=0.25, beta=math.log(2)))
        assert st.fired_steps == list(range(onset, onset + 12))
        masses[onset] = st.mass
    assert max(masses.values()) > 0 and (max(masses.values()) - min(masses.values())) / max(masses.values()) <= 0.05 + 1e-9, masses


def test_knockout_only_in_chosen_group():
    st = E.EditState(mode="knockout", onset=0, length=5, cols=COLS, knock_group="up", sides=(SIDE,))
    out_down = proc(st, "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor")(ATTN, HS, ENC)
    assert torch.equal(out_down, vanilla(ATTN, HS, ENC)) and st.mass == 0
    proc(st, "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor")(ATTN, HS, ENC)
    assert st.mass == abs(E.KNOCKOUT_BIAS) * N * len(COLS["I_a"])
    B = proc(st, "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor")._bias(ATTN, ATTN.to_q(HS).view(2, N, H, -1).transpose(1, 2), ATTN.to_k(ENC).view(2, 77, H, -1).transpose(1, 2), N)
    assert (B[0] == 0).all() and (B[1, :, :, [i for i in range(77) if i not in COLS["I_a"]]] == 0).all()


def test_max_edits_cap_and_logging():
    st = run_steps(E.EditState(mode="always_on", onset=0, length=20, max_edits=5, cols=COLS, sides=(SIDE,)))
    log = st.log()
    assert log["n_edits"] == 5 and log["fired_steps"] == [0, 1, 2, 3, 4] and log["mode"] == "always_on"


def test_skip_reasons_logged_and_missing_columns():
    st = E.EditState(mode="always_on", cols=dict(I_a=[3], I_n=[], I_o=[]), sides=(SIDE,), length=1)
    out = proc(st)(ATTN, HS, ENC)
    assert torch.equal(out, vanilla(ATTN, HS, ENC)) and st.log()["skipped_reason"] == ["missing_noun_tokens"]
    st = E.EditState(mode="always_on", cols=dict(I_a=[], I_n=[], I_o=[]), sides=(SIDE,), length=1)
    proc(st)(ATTN, HS, ENC)
    assert "no_attribute_tokens" in st.log()["skipped_reason"]


def test_determinism_same_seed_same_outcome_and_shuffled_uses_foreign_masks():
    a = E.EditState(mode="random_pair", cols=COLS, sides=(SIDE,), length=1, rand_seed=9)
    b = E.EditState(mode="random_pair", cols=COLS, sides=(SIDE,), length=1, rand_seed=9)
    assert torch.equal(proc(a)(ATTN, HS, ENC), proc(b)(ATTN, HS, ENC)) and a.rand_cols == b.rand_cols
    S_n = torch.zeros(SIDE, SIDE, dtype=torch.bool); S_n[0, :] = True
    S_o = torch.zeros(SIDE, SIDE, dtype=torch.bool); S_o[3, :] = True
    s = E.EditState(mode="shuffled_mask", cols=COLS, sides=(SIDE,), length=1, foreign_masks={SIDE: (S_n, S_o)})
    out = proc(s)(ATTN, HS, ENC)
    assert s.layer_calls_edited == 1 and not torch.allclose(out[1], vanilla(ATTN, HS, ENC)[1], atol=1e-6)


def test_plan_columns_and_mask_helpers():
    wi = {"red": [2], "cup": [3], "blue": [6], "book": [7]}
    cons = [{"type": "binding", "object": "cup", "color": "red"}, {"type": "binding", "object": "book", "color": "blue"}]
    c = E.plan_columns(wi, cons)
    assert c["I_a"] == [2, 6] and c["I_n"] == [3, 7] and c["I_o"] == []
    m = torch.zeros(4, 4); m[1, 1] = 5
    assert E.top_q_mask(m, 0.25).sum() == 4 and E.top_q_mask(m, 0.25)[1, 1]
    assert E.iou(E.top_q_mask(m, 0.25), E.top_q_mask(m, 0.25)) == 1.0
