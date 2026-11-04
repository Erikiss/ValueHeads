"""Model loading, the attention-output-projection resolver and geometry helpers.

Adapted from sihuo-design/valueheads `models.py` (MIT). Changes for Pythia / transformers 5:
`dtype=` loader kwarg, `revision=` for checkpoints, numerics policy (DESIGN 3.1), and an
architecture-agnostic `o_proj` resolver (GPT-NeoX `attention.dense` and Llama-style `self_attn.o_proj`).
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

    GPTNeoXConfig has no `head_dim`, so hidden_size // num_heads is used (64 for pythia-160m).
    """
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads
    return num_layers, num_heads, head_dim


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


def load_model(name_or_hf_id, revision=None, device="auto", dtype=None, attn_impl=None, allow_unsafe=False,
               trust_remote_code=False):
    """Loads (model, tokenizer) by registry short name (config.MODELS) or HF id, at `revision`.

    dtype / attn_impl default to the numerics policy for the model size and device
    (float32 + eager for 160m/410m everywhere; float16 + sdpa from 1b on GPU). The model is in eval
    mode, with `model.vh_info` holding what the extraction manifest records.
    """
    short, hf_id, _ = config.resolve_model(name_or_hf_id)
    device = resolve_device(device)
    policy_dtype, policy_impl = config.dtype_policy(short, device)
    dtype = to_dtype(dtype) or to_dtype(policy_dtype)
    if attn_impl is None:
        attn_impl = policy_impl if dtype_name(dtype) == policy_dtype else config.DTYPE_POLICY["attn_impl_for_dtype"][dtype_name(dtype)]
    check_numerics(dtype, attn_impl, allow_unsafe)

    tok_kwargs = dict(revision=revision, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, revision=revision, dtype=dtype, attn_implementation=attn_impl, trust_remote_code=trust_remote_code,
    ).eval()
    model.to(device)
    if tokenizer.pad_token is None:          # Pythia has <|padding|> (id 1); Llama/Qwen do not
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.vh_info = {
        "model": short, "hf_id": hf_id, "revision": revision or "main",
        "commit_sha": getattr(model.config, "_commit_hash", None) or resolve_commit_sha(hf_id, revision),
        "dtype": dtype_name(model.dtype), "attn_impl": model.config._attn_implementation,
        "device": device, "transformers": transformers.__version__, "torch": torch.__version__,
        "geometry": dict(zip(("layers", "heads", "head_dim"), model_geometry(model.config))),
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
