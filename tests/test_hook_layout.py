"""Hook site and per-head layout (DESIGN 3.1): zeroing the V rows of head 7 in layer 5 zeroes exactly slice 7 of the
o_proj input, other slices stay bit-identical; batched right-padded collection matches single-text collection;
the resolver works for GPT-NeoX and Llama-style blocks."""
import numpy as np
import pytest
import torch

from valueheads_pythia import config
from valueheads_pythia.commands.selftest import check_hook_layout
from valueheads_pythia.hooks import ActivationCollector, HeadHooks
from valueheads_pythia.models import model_geometry, o_proj


def test_geometry_matches_registry(model_tok):
    model, _ = model_tok
    L, H, D = model_geometry(model.config)
    info = config.MODELS["pythia-160m"]
    assert (L, H, D) == (info["layers"], info["heads"], info["head_dim"]) == (12, 12, 64)
    assert info["k15"] == int(0.15 * L * H) == 21
    assert model.vh_info["dtype"] == "float32" and model.vh_info["attn_impl"] == "eager"


def test_v_rows_zero_exactly_one_head_slice(model_tok):
    model, tok = model_tok
    res = check_hook_layout(model, tok, layer=5, head=7)
    assert res["hook_input_width"] == 12 * 64
    assert res["zeroed_head_absmax"] == 0.0
    assert res["other_heads_same_layer_maxdiff"] == 0.0
    assert res["earlier_layers_maxdiff"] == 0.0
    assert res["later_layer_maxdiff"] > 0
    # fp32 batched-vs-single differences are kernel numerics amplified by the massive activations (up to ~3e-3 at 300
    # tokens, even for a batch of identical rows); the policy tolerance 1e-2 applies (DESIGN 3.1), the layout is exact
    tol = config.DTYPE_POLICY["tolerance_batched_vs_single"]
    assert tol == 1e-2 and res["tolerance_batched_vs_single"] == tol
    assert res["batch_row0_vs_single_maxdiff"] < tol
    assert res["batch_row1_padded_vs_single_maxdiff"] < tol
    # the gate is scale-relative (fp32: 1e-2 of the per-layer activation maximum)
    assert res["dtype"] == "float32" and res["tolerance_rel_batched_vs_single"] == 1e-2
    assert res["batch_row0_vs_single_maxrel"] < 1e-2 and res["batch_row1_padded_vs_single_maxrel"] < 1e-2
    assert res["batch_consistent"] and res["activation_scale_max"] > 0
    assert res["pad_count_row1"] > 0
    assert res["ok"], res
    # a ~300-token text used to exceed the old 1e-3 criterion although the layout is correct
    import valueheads_pythia.commands.selftest as st
    short = st.TEXT
    st.TEXT = " ".join([short] * 15)
    try:
        long_res = check_hook_layout(model, tok, layer=5, head=7)
    finally:
        st.TEXT = short
    assert long_res["zeroed_head_absmax"] == 0.0 and long_res["other_heads_same_layer_maxdiff"] == 0.0
    assert long_res["batch_row0_vs_single_maxdiff"] < tol and long_res["ok"], long_res


def test_o_proj_resolver_gpt_neox(model_tok):
    model, _ = model_tok
    assert o_proj(model, 3) is model.gpt_neox.layers[3].attention.dense
    assert isinstance(o_proj(model, 0), torch.nn.Linear) and o_proj(model, 0).in_features == 768


def test_o_proj_resolver_llama_style():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    assert o_proj(model, 1) is model.model.layers[1].self_attn.o_proj
    L, H, D = model_geometry(cfg)
    assert (L, H, D) == (2, 4, 8)
    ids = torch.randint(0, 128, (2, 7))
    mask = torch.ones_like(ids)
    mask[1, 5:] = 0
    with ActivationCollector(model) as coll:
        acts = coll.collect(ids, mask)
    assert acts.shape == (2, L, H * D)
    with ActivationCollector(model) as coll:
        single = coll.collect(ids[1:, :5])[0]
    assert torch.allclose(acts[1], single, atol=1e-5)


def test_collector_last_token_gather(model_tok):
    """The collected vector is the o_proj input at the last valid token, also for a padded row."""
    model, tok = model_tok
    texts = ["The quick brown fox jumps over the lazy dog because it", "Hello"]
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True)
    captured = {}

    class Capture(HeadHooks):
        def edit(self, layer, x):
            captured[layer] = x.detach().clone()
            return x

    with Capture(model, positions="all", decode_only=False):
        model(**enc)
    with ActivationCollector(model) as coll:
        acts = coll.collect(enc.input_ids, enc.attention_mask)
    last = enc.attention_mask.sum(1) - 1
    for l in (0, 5, 11):
        ref = captured[l][torch.arange(2), last]
        assert torch.equal(acts[:, l], ref)
    assert acts.dtype == torch.float32 and acts.shape == (2, 12, 768)
    assert not np.isnan(acts.numpy()).any()


def test_kv_cache_decode_matches_full_recompute_within_policy(model_tok):
    """DESIGN 3.1: the o_proj input of the last token under KV-cache decoding (one-token step with past_key_values) vs. the
    full recompute differs only by kernel numerics, within DTYPE_POLICY["tolerance_kv_cache_vs_full"] (1e-2; the design
    track measured 2e-6 at L0 up to 4.9e-3 at L11)."""
    model, tok = model_tok
    ids = tok("The quick brown fox jumps over the lazy dog because it", return_tensors="pt").input_ids
    captured = {}

    class Capture(HeadHooks):
        def edit(self, layer, x):
            captured[layer] = x[:, -1].detach().clone()
            return x

    with torch.no_grad(), Capture(model, positions="all", decode_only=False):
        model(input_ids=ids, use_cache=False)
        full = dict(captured)
        captured.clear()
        prefix = model(input_ids=ids[:, :-1], use_cache=True)
        captured.clear()
        model(input_ids=ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True)
        decode = dict(captured)
    tol = config.DTYPE_POLICY["tolerance_kv_cache_vs_full"]
    diffs = {l: float((full[l] - decode[l]).abs().max()) for l in range(12)}
    assert tol == 1e-2 and max(diffs.values()) < tol, diffs
    assert all(torch.isfinite(decode[l]).all() for l in range(12)) and decode[5].shape == (1, 768)


@pytest.mark.parametrize("bad", [("bfloat16", "eager"), ("float16", "eager")])
def test_forbidden_numerics(bad):
    from valueheads_pythia.models import check_numerics
    with pytest.raises(ValueError):
        check_numerics(bad[0], bad[1])
    check_numerics(bad[0], bad[1], allow_unsafe=True)
    check_numerics("float32", "eager")
    assert config.dtype_policy("pythia-160m", "cuda") == ("float32", "eager")
    assert config.dtype_policy("pythia-1b", "cuda:0") == ("float16", "sdpa")
    assert config.dtype_policy("pythia-2.8b", "cpu") == ("float32", "eager")


def test_batch_consistency_is_scale_relative():
    """fp16 on the large models: absolute differences of ~0.3 at an activation scale of ~100 are kernel numerics
    (rel 3e-3, passes); the same absolute difference at scale 1 is a layout-sized error (rel 0.3, fails)."""
    import torch
    from valueheads_pythia.commands.selftest import batch_consistency
    g = torch.Generator().manual_seed(0)
    base = torch.randn(4, 8, generator=g) * 100.0
    short = torch.randn(4, 8, generator=g) * 100.0
    noise = torch.randn(4, 8, generator=g) * 0.1          # |noise| <= ~0.3
    res = batch_consistency(base, torch.stack([base + noise, short + noise]), short, "float16")
    assert res["tolerance_rel_batched_vs_single"] == 5e-2 and res["batch_consistent"], res
    assert res["batch_row0_vs_single_maxrel"] < 1e-2 and res["activation_scale_max"] > 100
    small_base, small_short = base / 100.0, short / 100.0
    res = batch_consistency(small_base, torch.stack([small_base + noise, small_short + noise]), small_short, "float16")
    assert not res["batch_consistent"] and res["batch_row0_vs_single_maxrel"] > 5e-2
    res = batch_consistency(base, torch.stack([base, short]), short, "float32")
    assert res["batch_consistent"] and res["batch_row0_vs_single_maxdiff"] == 0.0
