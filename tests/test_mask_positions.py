"""positions='all' edits every position, positions='last' only the last one; decode_only hooks skip the teacher-forced
forward and act on KV-cache decode steps (DESIGN 3.1 hook semantics)."""
import pytest
import torch

from valueheads_pythia.commands.selftest import TEXT, check_mask_positions
from valueheads_pythia.hooks import HeadHooks, HeadMask, HeadSteer


def test_mask_and_steer_positions(model_tok):
    model, tok = model_tok
    res = check_mask_positions(model, tok)
    S = res["seq_len"]
    assert res["mask_all_changed"] == S
    assert res["mask_last_changed"] == 1
    assert res["mask_decode_only_changed"] == 0
    assert res["steer_all_changed"] == S
    assert res["steer_last_changed"] == 1
    assert res["steer_decode_only_changed"] == 0
    assert res["gen_step0_scores_maxdiff"] == 0.0
    assert res["gen_step1_scores_maxdiff"] > 1e-5
    assert res["ok"], res


def test_mask_scales_exactly_the_head_slice(model_tok):
    """Inside the hook, masked slices are scaled by eps at every position and other slices are untouched."""
    model, tok = model_tok
    ids = tok(TEXT, return_tensors="pt").input_ids
    seen = {}

    class Capture(HeadHooks):
        def edit(self, layer, x):
            seen[layer] = x.detach().clone()
            return x

    with torch.no_grad():
        with Capture(model, positions="all", decode_only=False):
            model(input_ids=ids)
        base = dict(seen)
        seen.clear()
        mask = HeadMask(model, {4: [1, 6]}, eps=1e-4, positions="all")
        assert mask.decode_only is False and mask.layers == [4]
        # the capture hook is registered after the mask hook, so it sees the edited input
        with mask, Capture(model, positions="all", decode_only=False):
            model(input_ids=ids)
    x0, x1 = base[4][0], seen[4][0]
    for h in range(12):
        sl = slice(h * 64, (h + 1) * 64)
        if h in (1, 6):
            assert torch.allclose(x1[:, sl], x0[:, sl] * 1e-4, atol=1e-7)
        else:
            assert torch.equal(x1[:, sl], x0[:, sl])
    # layer 5 input differs because layer 4's masked output feeds the residual stream (parallel residual: not layer 4's MLP)
    assert (seen[5][0] - base[5][0]).abs().max() > 1e-4


def test_steer_adds_vector_at_last_position_only(model_tok):
    model, tok = model_tok
    ids = tok(TEXT, return_tensors="pt").input_ids
    vec = torch.zeros(768)
    vec[2 * 64:3 * 64] = 0.5
    seen = {}

    class Capture(HeadHooks):
        def edit(self, layer, x):
            seen[layer] = x.detach().clone()
            return x

    with torch.no_grad():
        with Capture(model, positions="all", decode_only=False):
            model(input_ids=ids)
        base = seen[2][0].clone()
        steer = HeadSteer(model, {2: vec}, alpha=2.0, positions="last", decode_only=False)
        assert steer.decode_only is False
        with steer, Capture(model, positions="all", decode_only=False):
            model(input_ids=ids)
    diff = seen[2][0] - base
    assert torch.allclose(diff[-1], 2.0 * vec, atol=1e-6)
    assert torch.equal(diff[:-1], torch.zeros_like(diff[:-1]))
    # default decode_only for positions="last" is True (paper semantics)
    assert HeadSteer(model, {2: vec}).decode_only is True and HeadMask(model, {2: [0]}).decode_only is True


def test_last_index_makes_positions_last_padding_safe(model_tok):
    """Without last_index, positions='last' edits column S - 1 of every row (right for left / no padding); with
    last_index = attention_mask.sum(1) - 1 it edits the last valid token of every right-padded row."""
    model, tok = model_tok
    seqs = [tok(TEXT)["input_ids"], tok("Hi there")["input_ids"]]
    width = max(map(len, seqs))
    ids = torch.full((2, width), tok.pad_token_id)
    mask = torch.zeros((2, width), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    seen = {}

    class Capture(HeadHooks):
        def edit(self, layer, x):
            seen[layer] = x.detach().clone()
            return x

    def edited_columns(hook):
        with torch.no_grad():
            with Capture(model, positions="all", decode_only=False):
                model(input_ids=ids, attention_mask=mask)
            ref = seen[5].clone()
            with hook, Capture(model, positions="all", decode_only=False):
                model(input_ids=ids, attention_mask=mask)
        diff = (seen[5] - ref).abs().amax(-1)
        return [torch.nonzero(diff[r] > 0).flatten().tolist() for r in range(2)]

    plain = HeadMask(model, {5: [3]}, eps=1e-4, positions="last", decode_only=False)
    assert edited_columns(plain) == [[width - 1], [width - 1]]                 # row 1's real last token is column 1
    safe = HeadMask(model, {5: [3]}, eps=1e-4, positions="last", decode_only=False)
    safe.last_index = mask.sum(1) - 1
    assert edited_columns(safe) == [[len(seqs[0]) - 1], [len(seqs[1]) - 1]]
    vec = torch.zeros(768)
    vec[3 * 64:4 * 64] = 1.0
    steer = HeadSteer(model, {5: vec}, alpha=1.0, positions="last", decode_only=False)
    steer.last_index = mask.sum(1) - 1
    assert edited_columns(steer) == [[len(seqs[0]) - 1], [len(seqs[1]) - 1]]
    bad = HeadMask(model, {5: [3]}, eps=1e-4, positions="last", decode_only=False)
    bad.last_index = torch.tensor([width, 0])
    with pytest.raises(ValueError):
        edited_columns(bad)
