"""Targeted cross-attention edit probe with matched controls (RB 13; milestone M7). Pure torch: no model weights needed to test.

Per-pixel additive logit bias on the ATTRIBUTE-token columns of the conditional batch element (batch layout [uncond, cond]):
    B[pix in S_n, I_a] += beta,   B[pix in S_o, I_a] -= beta,
S_n / S_o = top-q% pixels of the smoothed own-noun / other-noun cross-attention maps, computed at the same layer and step.
Arms (mode): targeted | random_pair | shuffled_mask | always_on (= targeted without trigger; the confirmatory HC arm, RB 13.8) |
             knockout (large negative bias on the attribute columns, only in one layer group) | off (control: bias identically absent).
Window: the edit is active for steps onset <= step < onset + length. The total bias mass  sum_{steps} sum_{layers} ||B||_1  is logged.
Trigger thresholds are not implemented: HC uses the always-on variant (RB 13.8); a trigger is DEV-only and out of scope here.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

MODES = ("off", "targeted", "random_pair", "shuffled_mask", "always_on", "knockout")
KNOCKOUT_BIAS = -1.0e4


# ----- pure helpers ---------------------------------------------------------------------------------------------------
def smooth3(m):
    """3x3 binomial smoothing of a [side, side] map (replicate padding)."""
    k = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], dtype=m.dtype, device=m.device) / 16
    return F.conv2d(F.pad(m[None, None], (1, 1, 1, 1), mode="replicate"), k[None, None])[0, 0]


def top_q_mask(m, q):
    """Boolean [side, side] mask of exactly k = max(1, round(q * N)) pixels with the largest values (ties broken by index)."""
    flat = m.reshape(-1)
    k = max(1, int(round(q * flat.numel())))
    idx = torch.argsort(flat, descending=True, stable=True)[:k]
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    return mask.view_as(m)


def iou(a, b):
    return float((a & b).sum()) / max(float((a | b).sum()), 1.0)


def blob_mask(side, k, rng, avoid=None):
    """k pixels nearest to a random centre (a contiguous blob of exactly k pixels), optionally disjoint from `avoid`."""
    cy, cx = rng.integers(0, side, 2)
    yy, xx = np.indices((side, side))
    d = ((yy - cy) ** 2 + (xx - cx) ** 2).astype(float) + rng.random((side, side)) * 1e-3
    if avoid is not None:
        d[avoid.cpu().numpy()] = np.inf
    order = np.argsort(d.reshape(-1), kind="stable")[:k]
    m = np.zeros(side * side, bool)
    m[order] = True
    return torch.from_numpy(m.reshape(side, side))


def plan_columns(word_idx, constraints):
    """Token columns from the scene's binding constraints. word_idx: word -> list of token positions (attention.word_token_indices).
    Returns dict(I_a, I_n, I_o) (sorted lists). Raises KeyError if a word is missing, which callers log as skipped_reason."""
    I_a, I_n, objs = [], [], []
    for c in constraints:
        if c["type"] == "binding":
            I_a += word_idx[c["color"]]
            I_n += word_idx.get(c["object"], word_idx.get(c["object"] + "s", []))
            objs.append(c["object"])
    others = {c["object"] for c in constraints if "object" in c} - set(objs)
    I_o = [i for o in others for i in word_idx.get(o, word_idx.get(o + "s", []))]
    return dict(I_a=sorted(set(I_a)), I_n=sorted(set(I_n)), I_o=sorted(set(I_o)))


def build_pixel_bias(N, I_a, S_n, S_o, beta, n_tok=77, dtype=torch.float32):
    """[1, 1, N, n_tok] additive bias for the conditional element. S_n, S_o: boolean [side, side]."""
    B = torch.zeros(1, 1, N, n_tok, dtype=dtype)
    for i in I_a:
        B[0, 0, S_n.reshape(-1), i] += beta
        B[0, 0, S_o.reshape(-1), i] -= beta
    return B


# ----- state and processor --------------------------------------------------------------------------------------------
class EditState:
    def __init__(self, mode="off", onset=0, length=12, beta=math.log(2), q=0.25, max_edits=40, cols=None, sides=(16,),
                 knock_group=None, rand_seed=0, foreign_masks=None, n_tok=77):
        assert mode in MODES, mode
        self.mode, self.onset, self.length, self.beta, self.q, self.max_edits = mode, onset, length, beta, q, max_edits
        self.cols, self.sides, self.knock_group, self.foreign, self.n_tok = cols or {}, tuple(sides), knock_group, foreign_masks, n_tok
        self.rng = np.random.default_rng(rand_seed)
        self.reset()

    def reset(self):
        self.step = 0
        self.mass = 0.0
        self.layer_calls_edited = 0
        self.n_edits = 0
        self.fired_steps = []
        self.skipped = []
        self._fired_this_step = False
        self.rand_cols = None

    @property
    def active(self):
        return self.mode != "off" and self.onset <= self.step < self.onset + self.length and self.n_edits < self.max_edits

    def end_step(self):
        """Call once per denoising step (callback_on_step_end)."""
        if self._fired_this_step:
            self.n_edits += 1
            self.fired_steps.append(self.step)
        self._fired_this_step = False
        self.step += 1

    def log(self):
        return dict(mode=self.mode, onset=self.onset, length=self.length, beta=self.beta, q=self.q, n_edits=self.n_edits,
                    fired_steps=self.fired_steps, skipped_reason=sorted(set(self.skipped)), bias_mass_l1=self.mass,
                    layer_calls_edited=self.layer_calls_edited)


def layer_group(name):
    return "down" if name.startswith("down_blocks") else "mid" if name.startswith("mid_block") else "up"


class EditProcessor:
    """Drop-in cross-attention processor (diffusers Attention API). One instance per layer; all share one EditState.
    With mode == 'off' or outside the window the computation is exactly scaled_dot_product_attention without a mask."""

    def __init__(self, state, layer_name, side):
        self.state, self.name, self.side = state, layer_name, side

    def _heads(self, attn, x, batch):
        return x.view(batch, -1, attn.heads, x.shape[-1] // attn.heads).transpose(1, 2)

    @torch.no_grad()
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        st = self.state
        batch, N = hidden_states.shape[0], hidden_states.shape[1]
        q, k, v = attn.to_q(hidden_states), attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        q, k, v = self._heads(attn, q, batch), self._heads(attn, k, batch), self._heads(attn, v, batch)
        bias = None
        if st.active and batch == 2:
            bias = self._bias(attn, q, k, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(batch, -1, q.shape[-1] * attn.heads)
        return attn.to_out[1](attn.to_out[0](out))

    def _bias(self, attn, q, k, N):
        st, side = self.state, self.side
        dt = q.dtype
        cols = st.cols
        if st.mode == "knockout":
            if layer_group(self.name) != st.knock_group:
                return None
            B = torch.zeros(2, 1, N, st.n_tok, dtype=dt, device=q.device)
            B[1, 0, :, cols["I_a"]] = KNOCKOUT_BIAS
            st._fired_this_step = True
            st.layer_calls_edited += 1
            st.mass += float(B[1].abs().sum())
            return B
        if side not in st.sides or not cols.get("I_a"):
            if not cols.get("I_a"):
                st.skipped.append("no_attribute_tokens")
            return None
        kk = max(1, int(round(st.q * N)))
        if st.mode == "random_pair":
            if st.rand_cols is None:
                pool = [i for i in range(1, st.n_tok - 1) if i not in set(cols["I_a"])]
                n_used = len(cols["I_a"])
                st.rand_cols = sorted(int(x) for x in st.rng.choice(pool, n_used, replace=False))
            S_n = blob_mask(side, kk, st.rng)
            S_o = blob_mask(side, kk, st.rng, avoid=S_n)
            use_cols = st.rand_cols
        else:
            use_cols = cols["I_a"]
            if st.mode == "shuffled_mask":
                S_n, S_o = (m.to(q.device) for m in st.foreign[side])
            else:   # targeted / always_on: masks from this layer's own conditional attention at this step
                if not cols.get("I_n") or not cols.get("I_o"):
                    st.skipped.append("missing_noun_tokens")
                    return None
                probs = torch.softmax((q[1:2].float() @ k[1:2].float().transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)[0]   # [h, N, 77]
                def pmap(I):
                    m = probs[:, :, I].mean(dim=(0, 2))
                    m = smooth3((m / m.sum().clamp_min(1e-12)).view(side, side))
                    return m
                p_n, p_o = pmap(cols["I_n"]), pmap(cols["I_o"])
                if float(p_n.detach().sum()) < 1e-6:
                    st.skipped.append("noun_map_mass_below_eps")
                    return None
                S_n, S_o = top_q_mask(p_n, st.q), top_q_mask(p_o, st.q)
                if iou(S_n, S_o) >= 0.5:
                    st.skipped.append("masks_overlap")
                    return None
        Bc = build_pixel_bias(N, use_cols, S_n.cpu(), S_o.cpu(), st.beta, st.n_tok).to(q.device, dt)
        B = torch.cat([torch.zeros_like(Bc), Bc], dim=0)          # unconditional element untouched
        st._fired_this_step = True
        st.layer_calls_edited += 1
        st.mass += float(Bc.abs().sum())
        return B


def mass_ratio_ok(mass_a, mass_b, tol=0.05):
    """RB 13.8: arms are comparable only if their logged total bias mass differs by at most 5% (relative to the larger)."""
    hi = max(mass_a, mass_b)
    return hi == 0 or abs(mass_a - mass_b) / hi <= tol


def install_edit(unet, state):
    """Replace every text cross-attention (attn2) processor by an EditProcessor; self-attention keeps its processor.
    Side length per layer is inferred from the number of spatial tokens at call time via the layer's block (64/32/16/8 for SD 1.5)."""
    procs = {}
    side_by_block = {"down_blocks.0": 64, "down_blocks.1": 32, "down_blocks.2": 16, "up_blocks.1": 16, "up_blocks.2": 32, "up_blocks.3": 64, "mid_block": 8}
    for name, p in unet.attn_processors.items():
        if name.endswith("attn2.processor"):
            side = next((s for pre, s in side_by_block.items() if name.startswith(pre)), 16)
            procs[name] = EditProcessor(state, name, side)
        else:
            procs[name] = p
    unet.set_attn_processor(procs)
    return state
