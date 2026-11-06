"""Model loading, the attention-output-projection resolver and geometry helpers.

Adapted from sihuo-design/valueheads `models.py` (MIT). Changes for Pythia / transformers 5:
`dtype=` loader kwarg, `revision=` for checkpoints, numerics policy (DESIGN 3.1), and an
architecture-agnostic `o_proj` resolver (GPT-NeoX `attention.dense` and Llama-style `self_attn.o_proj`).
Qwen3 / Llama-style GQA: geometry from config.head_dim, KV-group helpers (num_kv_heads, kv_group_heads, v_proj_rows)
for the self-test, per-family numerics policy (config.FAMILY_DTYPE_POLICY).
"""
import os
import random
import re

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config

DTYPES = {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16,
          "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def set_seed(seed):
    """Seeds python, numpy and torch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_geometry(cfg):
    """(num_layers, num_heads, head_dim) of the attention stack of a HF config.

    num_heads counts QUERY heads; the o_proj input (hook site) has width num_heads * head_dim. GPTNeoXConfig has no
    `head_dim`, so hidden_size // num_heads is used (64 for pythia-160m). Qwen3 sets head_dim = 128 explicitly, which is
    not hidden_size // num_heads (Qwen3-0.6B: 16 x 128 = 2048 vs hidden 1024; Qwen3-4B: 32 x 128 = 4096 vs 2560).
    """
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads
    return num_layers, num_heads, head_dim


def num_kv_heads(cfg):
    """Number of key/value heads (GQA: Qwen3 / Llama-3 have 8; GPT-NeoX has one per query head)."""
    _, num_heads, _ = model_geometry(cfg)
    kv = getattr(cfg, "num_key_value_heads", None) or num_heads
    if num_heads % kv:
        raise ValueError(f"num_attention_heads {num_heads} is not a multiple of num_key_value_heads {kv}")
    return kv


def kv_group_heads(cfg, group):
    """Query heads that read KV head `group`: transformers' repeat_kv expands KV head g to the query heads
    g * n_rep ... (g + 1) * n_rep - 1 with n_rep = num_heads // num_kv_heads (query head h reads KV head h // n_rep)."""
    _, num_heads, _ = model_geometry(cfg)
    kv = num_kv_heads(cfg)
    if not 0 <= group < kv:
        raise ValueError(f"KV group {group} out of range for {kv} KV heads")
    n_rep = num_heads // kv
    return list(range(group * n_rep, (group + 1) * n_rep))


def resolve_device(device="auto"):
    """'auto' -> 'cuda' if available else 'cpu'."""
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def to_dtype(dtype):
    """torch dtype from a name ('float32', 'fp16', ...), a torch dtype, or None."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if str(dtype) not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(DTYPES)}")
    return DTYPES[str(dtype)]


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def check_numerics(dtype, attn_impl, allow_unsafe=False):
    """Raises for the forbidden combination eager + half precision on Pythia (DESIGN 3.1)."""
    pair = (dtype_name(dtype), attn_impl)
    if pair in [tuple(x) for x in config.DTYPE_POLICY["forbidden"]] and not allow_unsafe:
        raise ValueError(f"{pair} is forbidden by the numerics policy (eager attention overflows in half precision "
                         f"on Pythia); use float32 or attn_impl='sdpa', or pass allow_unsafe=True")


def resolve_commit_sha(hf_id, revision=None):
    """Commit SHA of `revision` from the HF cache snapshot path (None if it cannot be resolved)."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(hf_id, "config.json", revision=revision)
        match = re.search(r"snapshots/([0-9a-f]{40})", path.replace(os.sep, "/"))
        return match.group(1) if match else None
    except Exception:
        return None


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".safetensors.index.json")


def hub_weight_files(hf_id, revision=None):
    """Sorted [(path, lfs_sha256 | None)] of the weight files on the Hub branch `revision` of `hf_id` (safetensors, bin,
    index). None when the Hub cannot be reached (offline) or `hf_id` is a local directory."""
    if os.path.isdir(hf_id):
        return None
    try:
        from huggingface_hub import HfApi
        out = []
        for e in HfApi().list_repo_tree(hf_id, revision=revision):
            if e.path.endswith(WEIGHT_SUFFIXES):
                lfs = getattr(e, "lfs", None)
                out.append((e.path, getattr(lfs, "sha256", None) if lfs is not None else None))
        return sorted(out)
    except Exception:
        return None


def choose_weight_files(files):
    """Which weight files a checkpoint load must use, given hub_weight_files(). transformers prefers model.safetensors
    over a sharded index. EleutherAI/pythia-2.8b's step0 / step2000 / step16000 branches carry main's model.safetensors
    (and main's pytorch_model.bin) NEXT TO the real per-step shards, so the preferred single file is the wrong
    checkpoint. Rule: when a branch holds both a single model.safetensors and a model.safetensors.index.json, the
    shards are trusted and the single file is ignored (ambiguous = True). Returns (used_files, ambiguous)."""
    if not files:
        return None, False
    names = {p for p, _ in files}
    shards = [(p, s) for p, s in files if p.startswith("model-") and p.endswith(".safetensors")]
    if "model.safetensors.index.json" in names and shards:
        ambiguous = "model.safetensors" in names
        return shards, ambiguous
    if "model.safetensors" in names:
        return [(p, s) for p, s in files if p == "model.safetensors"], False
    return [(p, s) for p, s in files if p.endswith(".bin")], False


def weight_fingerprint(used):
    """sha256 over the LFS object ids of the weight files a load uses; None without Hub information."""
    if not used or any(s is None for _, s in used):
        return None
    import hashlib
    return hashlib.sha256("|".join(f"{p}:{s}" for p, s in sorted(used)).encode()).hexdigest()


def weight_source(hf_id, revision=None):
    """(load_from, used_files, ambiguous, fingerprint): the path to load from (hf_id, or a local snapshot holding only
    the shards + index + config/tokenizer when the branch is ambiguous, see choose_weight_files), the weight files that
    load uses, whether the branch was ambiguous, and the fingerprint of the used files."""
    files = hub_weight_files(hf_id, revision)
    used, ambiguous = choose_weight_files(files)
    if not ambiguous:
        return hf_id, used, False, weight_fingerprint(used)
    from huggingface_hub import snapshot_download
    local = snapshot_download(hf_id, revision=revision, ignore_patterns=["model.safetensors", "pytorch_model.bin", "*.msgpack", "*.h5", "*.ckpt"])
    return local, used, True, weight_fingerprint(used)


def load_tokenizer(hf_id, revision=None, trust_remote_code=False):
    """(tokenizer, revision it was loaded from). Pythia models use config.PYTHIA_TOKENIZER_REVISION for every checkpoint
    (identical encodings; early pythia-2.8b revisions lack the special-token attributes). A missing pad token falls back
    to eos (Llama-3); without either the call fails here instead of deep inside the batching code. Qwen3 tokenizers define
    pad = <|endoftext|> (id 151643; = eos for the base models, eos = <|im_end|> for the post-trained ones) and add no BOS."""
    tok_revision = config.PYTHIA_TOKENIZER_REVISION if hf_id.startswith("EleutherAI/pythia") else revision
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=tok_revision, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:          # Pythia has <|padding|> (id 1), Qwen3 <|endoftext|>; Llama-3 has none
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError(f"tokenizer of {hf_id}@{tok_revision or 'main'} defines neither a pad nor an eos token")
    return tokenizer, tok_revision


def load_model(name_or_hf_id, revision=None, device="auto", dtype=None, attn_impl=None, allow_unsafe=False,
               trust_remote_code=False):
    """Loads (model, tokenizer) by registry short name (config.MODELS) or HF id, at `revision`.

    dtype / attn_impl default to the numerics policy for the model size and device
    (float32 + eager for 160m/410m everywhere; float16 + sdpa from 1b on GPU; Qwen3: bfloat16 + sdpa on GPU, float32 +
    eager on the CPU). The model is in eval mode, with `model.vh_info` holding what the extraction manifest records.
    """
    short, hf_id, _ = config.resolve_model(name_or_hf_id)
    device = resolve_device(device)
    policy_dtype, policy_impl = config.dtype_policy(short, device)
    dtype = to_dtype(dtype) or to_dtype(policy_dtype)
    if attn_impl is None:
        attn_impl = policy_impl if dtype_name(dtype) == policy_dtype else config.DTYPE_POLICY["attn_impl_for_dtype"][dtype_name(dtype)]
    check_numerics(dtype, attn_impl, allow_unsafe)

    tokenizer, tok_revision = load_tokenizer(hf_id, revision, trust_remote_code)
    load_from, used_files, ambiguous, fingerprint = weight_source(hf_id, revision)
    model = AutoModelForCausalLM.from_pretrained(
        load_from, revision=None if load_from != hf_id else revision, dtype=dtype, attn_implementation=attn_impl,
        trust_remote_code=trust_remote_code,
    ).eval()
    model.to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.vh_info = {
        "model": short, "hf_id": hf_id, "revision": revision or "main", "tokenizer_revision": tok_revision or "main",
        "commit_sha": getattr(model.config, "_commit_hash", None) or resolve_commit_sha(hf_id, revision),
        "dtype": dtype_name(model.dtype), "attn_impl": model.config._attn_implementation,
        "device": device, "transformers": transformers.__version__, "torch": torch.__version__,
        "geometry": dict(zip(("layers", "heads", "head_dim"), model_geometry(model.config))),
        "kv_heads": num_kv_heads(model.config), "hidden": model.config.hidden_size,
        "weight_files": used_files, "weight_source": "shards (ambiguous branch)" if ambiguous else "hub",
        "weight_fingerprint": fingerprint,
    }
    return model, tokenizer


def decoder_blocks(model):
    """The list of transformer blocks of a causal LM (GPT-NeoX, Llama, Qwen, Yi, ...)."""
    base = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, model.base_model_prefix)
    return base.layers


def o_proj(model, layer):
    """The attention output projection of `layer`; its input is the concatenation of all head outputs.

    GPT-NeoX / Pythia: `gpt_neox.layers[l].attention.dense`; Llama / Qwen / Yi: `model.layers[l].self_attn.o_proj`.
    """
    block = decoder_blocks(model)[layer]
    if hasattr(block, "self_attn"):
        return block.self_attn.o_proj
    if hasattr(block, "attention"):
        return block.attention.dense
    raise AttributeError(f"no attention output projection found in block type {type(block).__name__}")


def qkv_value_rows(model, layer, head):
    """Row slice of head `head`'s V weights in the fused GPT-NeoX `query_key_value` Linear.

    GPT-NeoX stores [q_h | k_h | v_h] blocks of width 3*head_dim per head (interleaved layout,
    verified numerically in DESIGN 3.1). Used by the self-test that zeroes a head's values.
    """
    block = decoder_blocks(model)[layer]
    if not hasattr(block, "attention") or not hasattr(block.attention, "query_key_value"):
        raise AttributeError("qkv_value_rows is only defined for GPT-NeoX-style fused query_key_value")
    _, _, d = model_geometry(model.config)
    start = head * 3 * d + 2 * d
    return block.attention.query_key_value, slice(start, start + d)


def v_proj_rows(model, layer, group):
    """(v_proj Linear, row slice) of KV head `group` in a Llama-style separate `self_attn.v_proj` (Llama, Qwen, Yi).

    v_proj maps hidden -> num_kv_heads * head_dim and its output is viewed as (..., num_kv_heads, head_dim), so KV head g
    owns the rows [g * head_dim, (g + 1) * head_dim) (and the same bias entries, if the projection has a bias)."""
    block = decoder_blocks(model)[layer]
    attn = getattr(block, "self_attn", None)
    if attn is None or not hasattr(attn, "v_proj"):
        raise AttributeError(f"v_proj_rows needs a Llama-style self_attn.v_proj (block type {type(block).__name__})")
    _, _, d = model_geometry(model.config)
    if not 0 <= group < num_kv_heads(model.config):
        raise ValueError(f"KV group {group} out of range for {num_kv_heads(model.config)} KV heads")
    return attn.v_proj, slice(group * d, (group + 1) * d)


def q_proj_rows(model, layer, head):
    """(q_proj Linear, row slice) of query head `head` in a Llama-style separate `self_attn.q_proj`, or None for other
    blocks (GPT-NeoX). q_proj maps hidden -> num_heads * head_dim, viewed as (..., num_heads, head_dim)."""
    block = decoder_blocks(model)[layer]
    attn = getattr(block, "self_attn", None)
    if attn is None or not hasattr(attn, "q_proj"):
        return None
    _, num_heads, d = model_geometry(model.config)
    if not 0 <= head < num_heads:
        raise ValueError(f"head {head} out of range for {num_heads} heads")
    return attn.q_proj, slice(head * d, (head + 1) * d)


def value_rows(model, layer, head):
    """(Linear, row slice, query heads, KV group) for the self-test that zeroes the values read by query head `head`.

    GPT-NeoX (fused query_key_value, one KV head per query head): the rows of qkv_value_rows, affecting [head] only,
    KV group None. Llama-style (GQA or not): the v_proj rows of KV group g = head // n_rep, which every query head of
    that group reads, so their o_proj-input slices (kv_group_heads) must all become zero while the other heads of the
    layer stay bit-identical.
    """
    block = decoder_blocks(model)[layer]
    if hasattr(block, "attention") and hasattr(block.attention, "query_key_value"):
        linear, rows = qkv_value_rows(model, layer, head)
        return linear, rows, [head], None
    _, num_heads, _ = model_geometry(model.config)
    if not 0 <= head < num_heads:
        raise ValueError(f"head {head} out of range for {num_heads} heads")
    group = head // (num_heads // num_kv_heads(model.config))
    linear, rows = v_proj_rows(model, layer, group)
    return linear, rows, kv_group_heads(model.config, group), group
