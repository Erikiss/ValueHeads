"""Qwen3 port (Llama-style blocks, GQA, head_dim != hidden / heads): geometry, hook layout, GQA self-test, registry and
numerics policy. The tiny randomly initialised Qwen3 models need no download; the last tests use Qwen/Qwen3-0.6B-Base
and are skipped unless it is in the HF cache. The Pythia policy / manifest assertions guard the unchanged Pythia path."""
import math

import numpy as np
import pytest
import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from valueheads_pythia import config
from valueheads_pythia import models as vh_models
from valueheads_pythia.commands.selftest import TEXT, check_hook_layout, check_mask_positions
from valueheads_pythia.extract import extract_features, load_features, tokenize_all
from valueheads_pythia.hooks import ActivationCollector, HeadHooks, HeadMask
from valueheads_pythia.models import kv_group_heads, model_geometry, num_kv_heads, o_proj, v_proj_rows, value_rows

VOCAB = 128


class CharTokenizer:
    """Minimal HF-like tokenizer (one id per character) for the tiny random models: __call__ with return_tensors /
    padding / add_special_tokens / truncation, padding_side, truncation_side, pad_token_id."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 0
    bos_token_id = None

    def __init__(self):
        self.padding_side = "right"
        self.truncation_side = "right"

    def encode(self, text):
        return [1 + (ord(c) % (VOCAB - 1)) for c in text]

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, add_special_tokens=True):
        texts = [text] if isinstance(text, str) else list(text)
        ids = [self.encode(t) for t in texts]
        if return_tensors is None:
            return {"input_ids": ids if not isinstance(text, str) else ids[0]}
        width = max(len(s) for s in ids)
        if not padding and any(len(s) != width for s in ids):
            raise ValueError("unequal lengths need padding=True")
        input_ids = torch.full((len(ids), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for r, s in enumerate(ids):
            sl = slice(0, len(s)) if self.padding_side == "right" else slice(width - len(s), width)
            input_ids[r, sl] = torch.tensor(s)
            mask[r, sl] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": mask})


def tiny_qwen3(heads=4, kv_heads=2, head_dim=16, hidden=48, layers=3, bias=False, attn="eager", seed=0):
    """Random Qwen3 whose head_dim is decoupled from hidden (H * D = 64 != 48) and with GQA (2 query heads per KV head)."""
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=hidden, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim, max_position_embeddings=256,
                      attention_bias=bias, tie_word_embeddings=True, pad_token_id=0, eos_token_id=0, bos_token_id=None)
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():                    # random init gives near-zero attention outputs; make them clearly non-zero
        for block in model.model.layers:
            for lin in (block.self_attn.v_proj, block.self_attn.q_proj, block.self_attn.k_proj):
                lin.weight.normal_(0, 0.3)
                if lin.bias is not None:
                    lin.bias.normal_(0, 0.3)
    model.generation_config.pad_token_id = 0
    return model


@pytest.fixture(scope="module")
def tiny():
    return tiny_qwen3(), CharTokenizer()


# ----------------------------------------------------------------------------- geometry / resolver
def test_geometry_gqa_and_decoupled_head_dim(tiny):
    model, _ = tiny
    L, H, D = model_geometry(model.config)
    assert (L, H, D) == (3, 4, 16) and H * D == 64 != model.config.hidden_size == 48
    assert num_kv_heads(model.config) == 2
    assert kv_group_heads(model.config, 0) == [0, 1] and kv_group_heads(model.config, 1) == [2, 3]
    with pytest.raises(ValueError):
        kv_group_heads(model.config, 2)
    for layer in range(L):
        proj = o_proj(model, layer)
        assert proj is model.model.layers[layer].self_attn.o_proj
        assert (proj.in_features, proj.out_features) == (H * D, 48)
    lin, rows = v_proj_rows(model, 1, 1)
    assert lin is model.model.layers[1].self_attn.v_proj and rows == slice(16, 32) and lin.out_features == 2 * 16
    assert value_rows(model, 1, 3)[2:] == ([2, 3], 1)
    assert value_rows(model, 1, 0)[2:] == ([0, 1], 0)


def test_value_rows_gpt_neox_and_mha_llama(model_tok):
    neox, _ = model_tok
    lin, rows, heads, group = value_rows(neox, 5, 7)
    assert lin is neox.gpt_neox.layers[5].attention.query_key_value and heads == [7] and group is None
    assert rows == vh_models.qkv_value_rows(neox, 5, 7)[1] == slice(7 * 192 + 128, 7 * 192 + 192)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    llama = LlamaForCausalLM(cfg).eval()
    assert num_kv_heads(cfg) == 4 and value_rows(llama, 1, 3)[2:] == ([3], 3)


def test_hook_input_is_per_query_head_attention_output(tiny):
    """The o_proj-input slice of query head h at the last token equals softmax(q_h k_g^T / sqrt(D)) v_g recomputed by hand
    from q_proj / k_proj / v_proj (+ q/k RMSNorm and rotary) with g = h // (H / KV): layout and GQA mapping."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    model, tok = tiny
    L, H, D = model_geometry(model.config)
    n_rep = H // num_kv_heads(model.config)
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :24]
    seen = {}

    def grab(layer):
        def hook(module, args, kwargs):
            seen[layer] = (kwargs["hidden_states"].detach(), kwargs["position_embeddings"])
        return hook

    handles = [model.model.layers[l].self_attn.register_forward_pre_hook(grab(l), with_kwargs=True) for l in range(L)]
    try:
        with ActivationCollector(model) as coll:
            acts = coll.collect(ids)[0]
    finally:
        for h in handles:
            h.remove()
    for layer in range(L):
        attn = model.model.layers[layer].self_attn
        x, (cos, sin) = seen[layer]
        shape = (*x.shape[:-1], -1, D)
        with torch.no_grad():
            q = attn.q_norm(attn.q_proj(x).view(shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
            v = attn.v_proj(x).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        for h in range(H):
            g = h // n_rep
            scores = (q[0, h, -1] @ k[0, g].T) / math.sqrt(D)          # last query attends to every (causal) key
            ref = torch.softmax(scores, -1) @ v[0, g]
            assert torch.allclose(acts[layer, h * D:(h + 1) * D], ref, atol=1e-5), (layer, h)
            wrong = torch.softmax((q[0, h, -1] @ k[0, h % 2].T) / math.sqrt(D), -1) @ v[0, h % 2]   # interleaved mapping
            if h % 2 != g:
                assert not torch.allclose(acts[layer, h * D:(h + 1) * D], wrong, atol=1e-3)


def test_collector_batched_right_padding_matches_single(tiny):
    model, tok = tiny
    L, H, D = model_geometry(model.config)
    batch = tok([TEXT, "Hello world"], return_tensors="pt", padding=True)
    with ActivationCollector(model) as coll:
        both = coll.collect(batch.input_ids, batch.attention_mask)
        single = [coll.collect(tok(t, return_tensors="pt").input_ids)[0] for t in (TEXT, "Hello world")]
    assert both.shape == (2, L, H * D) == (2, 3, 64)
    for r in range(2):
        assert torch.allclose(both[r], single[r], atol=1e-5)


def test_head_hooks_refuse_a_wrong_width(tiny):
    """A geometry that does not match the o_proj input (e.g. head_dim = hidden / heads = 12) fails loudly."""
    model, tok = tiny
    ids = tok("abc", return_tensors="pt").input_ids
    mask = HeadMask(model, {1: [0]}, positions="all")
    mask.head_dim = 12
    with pytest.raises(ValueError, match="width 64"):
        with mask, torch.no_grad():
            model(input_ids=ids)
    assert not mask.handles


def test_mask_scales_only_the_query_head_slice(tiny):
    """HeadMask acts per QUERY head: masking head 1 leaves head 0 (same KV group) untouched."""
    model, tok = tiny
    ids = tok(TEXT, return_tensors="pt").input_ids
    seen = {}

    class Capture(HeadHooks):
        def edit(self, layer, x):
            seen[layer] = x.detach().clone()
            return x

    with torch.no_grad():
        with Capture(model, positions="all", decode_only=False):
            model(input_ids=ids)
        base = seen[1][0]
        with HeadMask(model, {1: [1]}, eps=1e-4, positions="all"), Capture(model, positions="all", decode_only=False):
            model(input_ids=ids)
    x = seen[1][0]
    assert torch.equal(x[:, 0:16], base[:, 0:16]) and torch.equal(x[:, 32:], base[:, 32:])
    assert torch.allclose(x[:, 16:32], base[:, 16:32] * 1e-4, atol=1e-8)


# ----------------------------------------------------------------------------- GQA self-test
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("head", [0, 3])
def test_gqa_selftest_zeroes_exactly_the_kv_group(head, bias):
    model, tok = tiny_qwen3(bias=bias), CharTokenizer()
    w = model.model.layers[1].self_attn.v_proj.weight.clone()
    res = check_hook_layout(model, tok, layer=1, head=head)
    assert res["ok"], res
    assert res["kv_heads"] == 2 and res["kv_group"] == head // 2 and res["zeroed_heads"] == kv_group_heads(model.config, head // 2)
    assert res["group_size_ok"] and res["value_rows_bias"] is bias
    assert res["hook_input_width"] == 64 and res["zeroed_head_absmax"] == 0.0 and res["other_heads_same_layer_maxdiff"] == 0.0
    assert res["earlier_layers_maxdiff"] == 0.0 and res["later_layer_maxdiff"] > 0 and res["base_head_absmax"] > 0
    assert res["batch_layout_ok"] and res["pad_count_row1"] > 0
    assert torch.equal(model.model.layers[1].self_attn.v_proj.weight, w)          # weights restored


def test_gqa_selftest_detects_a_wrong_group_mapping(tiny, monkeypatch):
    """Negative control: with an interleaved head -> KV-group map (h % KV) the zeroed slices are not the predicted ones."""
    model, tok = tiny

    def interleaved(cfg, group):
        kv = num_kv_heads(cfg)
        return [h for h in range(model_geometry(cfg)[1]) if h % kv == group]

    monkeypatch.setattr(vh_models, "kv_group_heads", interleaved)
    res = check_hook_layout(model, tok, layer=1, head=0)
    assert res["zeroed_heads"] == [0, 2] and not res["ok"]
    assert res["zeroed_head_absmax"] > 0 and res["other_heads_same_layer_maxdiff"] > 0


def test_gqa_selftest_half_precision_restores_model():
    """bf16 model (sdpa): layout gate on the lossless fp32 up-cast, model returns to bf16 bit-identically."""
    model, tok = tiny_qwen3(attn="sdpa").to(torch.bfloat16), CharTokenizer()
    ids = tok(TEXT, return_tensors="pt").input_ids
    with torch.no_grad():
        before = model(input_ids=ids).logits
    res = check_hook_layout(model, tok, layer=1, head=2)
    assert model.dtype == torch.bfloat16 and res["dtype"] == "bfloat16"
    with torch.no_grad():
        assert torch.equal(before, model(input_ids=ids).logits)
    assert res["ok"] and res["layout_fp32_normrel_row1"] < 1e-2, res


def test_mask_positions_on_tiny_qwen3():
    model, tok = tiny_qwen3(layers=10), CharTokenizer()          # the check masks every head of layers 3-8
    res = check_mask_positions(model, tok)
    assert res["ok"], res


def test_extract_and_probe_tiny_qwen3(tiny, tmp_path):
    """Extraction manifest records the GQA geometry; shards hold H * D columns; head slices load as (N, D)."""
    from valueheads_pythia.probing import fit_head_probes
    model, tok = tiny
    texts = [f"Question: q{i} Answer: {'yes' * (1 + i % 3)}" for i in range(24)]
    labels = np.array([i % 2 for i in range(24)])
    out = {}
    for split, sl in (("train", slice(0, 16)), ("val", slice(16, 24))):
        out[split] = str(tmp_path / f"honesty_{split}")
        man = extract_features(model, tok, texts[sl], out[split], batch_size=4, labels=labels[sl],
                               groups=np.arange(len(texts))[sl], show_progress=False)
    assert (man["layers"], man["heads"], man["head_dim"], man["width"]) == (3, 4, 16, 64)
    assert man["kv_heads"] == 2 and man["hidden"] == 48
    acts, y, groups, _ = load_features(out["train"])
    assert acts[0].shape == (16, 64) and acts.head(2, 3).shape == (16, 16)
    df = fit_head_probes(out["train"], out["val"], feature_mode="head+mean", score="acc", class_weight=None, n_jobs=1, verbose=False)
    assert len(df) == 3 * 4 and set(df["layer"]) == {0, 1, 2} and set(df["head"]) == {0, 1, 2, 3}


def test_pythia_manifest_has_no_gqa_keys(model_tok, tmp_path):
    model, tok = model_tok
    man = extract_features(model, tok, ["Hello world", "The quick brown fox"], str(tmp_path / "p"), show_progress=False)
    assert "kv_heads" not in man and "hidden" not in man and man["width"] == 768


# ----------------------------------------------------------------------------- registry / policy
QWEN_KEYS = ["qwen3-0.6b", "qwen3-0.6b-base", "qwen3-1.7b", "qwen3-1.7b-base", "qwen3-4b", "qwen3-4b-base", "qwen3-8b", "qwen3-8b-base"]


def test_registry_entries():
    for key in QWEN_KEYS:
        info = config.MODELS[key]
        assert info["k15"] == int(info["layers"] * info["heads"] * 0.15)
        assert info["family"] == "qwen3" and info["kv_heads"] == 8 and info["head_dim"] == 128
        assert info["hf_id"].startswith("Qwen/Qwen3-") and info["hf_id"].endswith("-Base") == key.endswith("-base")
        assert config.model_short_name(info["hf_id"]) == key
    assert config.MODELS["qwen3-4b"]["k15"] == 172 and config.MODELS["qwen3-0.6b-base"]["k15"] == 67
    assert config.model_short_name("qwen3-4B") == "qwen3-4b" and config.model_short_name("qwen3-8B") == "qwen3-8b"
    assert config.resolve_model("qwen3-4B")[1] == "Qwen/Qwen3-4B"


def test_registry_matches_cached_configs():
    from transformers import AutoConfig
    checked = 0
    for key in QWEN_KEYS:
        info = config.MODELS[key]
        try:
            cfg = AutoConfig.from_pretrained(info["hf_id"], local_files_only=True)
        except Exception:
            continue
        assert (cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size) == \
            (info["layers"], info["heads"], info["kv_heads"], info["head_dim"], info["hidden"]), key
        assert model_geometry(cfg) == (info["layers"], info["heads"], info["head_dim"])
        checked += 1
    if not checked:
        pytest.skip("no Qwen3 config.json in the HF cache")


def test_dtype_policy_and_size_class():
    for key in QWEN_KEYS + ["Qwen/Qwen3-4B", "Qwen/Qwen3-14B"]:
        assert config.dtype_policy(key, "cuda") == ("bfloat16", "sdpa")
        assert config.dtype_policy(key, "cpu") == ("float32", "eager")
        assert config.model_family(key) == "qwen3"
    assert config.size_class("qwen3-0.6b-base") == "mid" and config.size_class("Qwen/Qwen3-4B") == "large"
    assert config.max_samples("t4", "harmless", "train", "qwen3-4b") == 6000
    assert config.max_samples("t4", "harmless", "train", "qwen3-0.6b") == 10000
    # Pythia (and the paper Llama / Yi entries) keep the size-class policy unchanged
    old = {"small": {"cpu": ("float32", "eager"), "cuda": ("float32", "eager")},
           "mid": {"cpu": ("float32", "eager"), "cuda": ("float16", "sdpa")},
           "large": {"cpu": ("float32", "eager"), "cuda": ("float16", "sdpa")}}
    expected_class = {"pythia-160m": "small", "pythia-410m": "small", "pythia-1b": "mid", "pythia-1.4b": "mid", "pythia-2.8b": "large",
                      "llama3.1_8B_instruct": "large", "llama3.2_3B_instruct": "large", "yi-1.5-6B": "large"}
    for name, cls in expected_class.items():
        for alias in (name, config.MODELS[name]["hf_id"]):
            assert config.size_class(alias) == cls
            for dev in ("cpu", "cuda"):
                assert config.dtype_policy(alias, dev) == old[cls][dev]


# ----------------------------------------------------------------------------- real model (cached only)
def _cached(repo, filename="model.safetensors"):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo, filename), str)
    except Exception:
        return False


@pytest.mark.skipif(not _cached("Qwen/Qwen3-0.6B-Base"), reason="Qwen/Qwen3-0.6B-Base not in the HF cache")
def test_qwen3_0_6b_base_selftest_and_tokenization():
    model, tok = vh_models.load_model("qwen3-0.6b-base", device="cpu")
    assert model.vh_info["dtype"] == "float32" and model.vh_info["attn_impl"] == "eager"
    assert model.vh_info["kv_heads"] == 8 and model.vh_info["hidden"] == 1024
    assert model_geometry(model.config) == (28, 16, 128)
    assert tok.pad_token_id == 151643 and tok.bos_token is None
    # raw text, no chat template, no BOS: what the reference repo's tokenizer(text) gives for Qwen3
    text = "Question: What happens if you crack your knuckles a lot? Answer: Nothing in particular happens"
    ids, n_trunc = tokenize_all(tok, [text], max_length=0)
    assert ids[0] == tok(text)["input_ids"] and n_trunc == 0
    assert not set(ids[0]) & set(tok.all_special_ids)
    res = check_hook_layout(model, tok, layer=5, head=7)
    assert res["ok"], res
    assert res["zeroed_heads"] == [6, 7] and res["kv_group"] == 3 and res["hook_input_width"] == 2048
    from valueheads_pythia.commands.selftest import check_query_head_layout
    q = check_query_head_layout(model, tok, layer=5, head=7)
    assert q["ok"] and q["changed_heads"] == [7], q                             # head 6 of the same KV group unchanged
    res = check_mask_positions(model, tok)
    assert res["ok"], res


# ----------------------------------------------------------------------------- per-query-head gate
@pytest.mark.parametrize("head", [0, 1, 3])
def test_query_head_gate_changes_exactly_one_head(head):
    from valueheads_pythia.commands.selftest import check_query_head_layout
    model, tok = tiny_qwen3(), CharTokenizer()
    w = model.model.layers[1].self_attn.q_proj.weight.clone()
    res = check_query_head_layout(model, tok, layer=1, head=head)
    assert res["ok"], res
    assert res["changed_heads"] == [head] and res["head_maxdiff"] > 0 and res["other_heads_same_layer_maxdiff"] == 0.0
    assert torch.equal(model.model.layers[1].self_attn.q_proj.weight, w)          # weights restored


def test_query_head_gate_detects_a_permutation_inside_a_kv_group(tiny, monkeypatch):
    """Negative control: if the hook layout swapped the two query heads of a KV group (which the v_proj group check
    cannot see), the per-query-head gate fails."""
    from valueheads_pythia.commands import selftest as st
    model, tok = tiny
    real = vh_models.q_proj_rows

    def swapped(m, layer, head):                     # head 0 <-> head 1 (same KV group): rows of the other head
        return real(m, layer, {0: 1, 1: 0}.get(head, head))

    monkeypatch.setattr(st, "q_proj_rows", swapped)
    res = st.check_query_head_layout(model, tok, layer=1, head=0)
    assert res["changed_heads"] == [1] and not res["ok"]
    assert st.check_hook_layout(model, tok, layer=1, head=0)["ok"]               # the group-level check passes


def test_query_head_gate_is_skipped_for_gpt_neox(model_tok):
    from valueheads_pythia.commands.selftest import check_query_head_layout
    model, tok = model_tok
    assert check_query_head_layout(model, tok, layer=1, head=0) is None
