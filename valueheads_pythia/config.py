"""Model registry, tier defaults, numerics policy and result paths (DESIGN sections 3.1, 3.4, 3.5, 6.3, 6.4).

Everything here is torch-free so that it can be imported by analysis code without loading a model.
"""
import os
import re

# ----------------------------------------------------------------------------- models
# short name -> geometry. k15 = int(0.15 * layers * heads) (paper's top-15 % rule).
MODELS = {
    # Pythia suite (EleutherAI); all: vocab 50304, context 2048, parallel residual, attention bias,
    # partial rotary 0.25. Revisions step0 ... step143000 (154 checkpoints).
    "pythia-160m": {"hf_id": "EleutherAI/pythia-160m", "layers": 12, "heads": 12, "head_dim": 64, "hidden": 768, "k15": 21},
    "pythia-410m": {"hf_id": "EleutherAI/pythia-410m", "layers": 24, "heads": 16, "head_dim": 64, "hidden": 1024, "k15": 57},
    "pythia-1b": {"hf_id": "EleutherAI/pythia-1b", "layers": 16, "heads": 8, "head_dim": 256, "hidden": 2048, "k15": 19,
                  "note": "geometry outlier (8 x 256): equalize probe capacity or report separately"},
    "pythia-1.4b": {"hf_id": "EleutherAI/pythia-1.4b", "layers": 24, "heads": 16, "head_dim": 128, "hidden": 2048, "k15": 57},
    "pythia-2.8b": {"hf_id": "EleutherAI/pythia-2.8b", "layers": 32, "heads": 32, "head_dim": 80, "hidden": 2560, "k15": 153},
    # Paper models (for `--mode paper` comparability; gated Llama models need HF_TOKEN).
    "llama3.1_8B_instruct": {"hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "layers": 32, "heads": 32, "head_dim": 128, "hidden": 4096, "k15": 153},
    "llama3.2_3B_instruct": {"hf_id": "meta-llama/Llama-3.2-3B-Instruct", "layers": 28, "heads": 24, "head_dim": 128, "hidden": 3072, "k15": 100},
    "qwen3-8B": {"hf_id": "Qwen/Qwen3-8B", "layers": 36, "heads": 32, "head_dim": 128, "hidden": 4096, "k15": 172},
    "qwen3-4B": {"hf_id": "Qwen/Qwen3-4B", "layers": 36, "heads": 32, "head_dim": 128, "hidden": 2560, "k15": 172},
    "yi-1.5-9B": {"hf_id": "01-ai/Yi-1.5-9B-Chat", "layers": 48, "heads": 32, "head_dim": 128, "hidden": 4096, "k15": 230},
    "yi-1.5-6B": {"hf_id": "01-ai/Yi-1.5-6B-Chat", "layers": 32, "heads": 32, "head_dim": 128, "hidden": 4096, "k15": 153},
}

PYTHIA_SIZES = ["pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b"]

# Default checkpoint sweep (DESIGN 3.4): log-dense early, coarse late.
CHECKPOINTS_DEFAULT = ["step0", "step128", "step512", "step1000", "step2000", "step4000", "step8000",
                       "step16000", "step32000", "step64000", "step100000", "step143000"]
CHECKPOINTS_CACHED_LOCALLY_160M = ["step0", "step2000", "step98000", "step99000", "step143000"]
CHECKPOINTS_SMOKE = ["step0", "step2000", "step143000"]
CAUSAL_CHECKPOINTS_DEFAULT = ["step2000", "step16000", "step143000"]
# All Pythia checkpoints share one tokenizer (GPT-NeoX-20B BPE). The tokenizer files of pythia-2.8b step0..step16000 lack
# the special-token attributes under transformers 5 (pad / eos / bos = None) although their encodings are identical to
# step143000 (checked for all 5 models x 12 default checkpoints, with and without special tokens); the tokenizer is
# therefore always loaded from this revision for EleutherAI/pythia-* models (models.load_tokenizer).
PYTHIA_TOKENIZER_REVISION = "step143000"

# ----------------------------------------------------------------------------- values
# Paper order (data.py:16 of the reference repo); `harmless` there is the pooled HH-RLHF set.
PAPER_VALUES = ["harmless", "helpfulness", "honesty", "complexity", "verbosity", "coherence"]
HELPSTEER_VALUES = ["helpfulness", "complexity", "verbosity", "coherence"]
HELPSTEER_VALUES_EXT = HELPSTEER_VALUES + ["coherence_strict"]
HH_VALUES = ["harmless", "helpful", "harmless_pooled"]
# Values defined in `--mode pythia` (DESIGN 3.3).
PYTHIA_VALUES = ["harmless", "helpful", "honesty"] + HELPSTEER_VALUES_EXT
ALL_VALUES = ["harmless", "helpful", "harmless_pooled", "honesty"] + HELPSTEER_VALUES_EXT
# Unlabeled prompt sets used for sigma estimation in `--mode paper` (STD_SETS of steer.py).
PROMPT_SETS = ["jailbreak", "malicious", "beavertails"]
# Datasets accepted by `extract`: every value plus the HelpSteer2 extract-once set and the prompt sets.
EXTRACT_DATASETS = ALL_VALUES + ["helpsteer2"] + PROMPT_SETS

MODES = ["pythia", "paper"]
TOP_RATIO = 0.15
EXTRA_RATIOS = [0.05, 0.10]
MASK_EPS = 1e-4
MAX_LENGTH = 2048          # Pythia context; truncation keeps the tail (truncation_side="left")
STEER_ALPHAS = [0.3, 0.5, 1.0]


def probing_dataset_of(value, mode="pythia"):
    """Feature-directory dataset name holding the probing features of `value` (DESIGN 3.3).

    In mode `pythia` the HelpSteer2 values share the single extract-once set `helpsteer2`.
    """
    if mode == "pythia" and value in HELPSTEER_VALUES_EXT:
        return "helpsteer2"
    return value


# ----------------------------------------------------------------------------- tiers (DESIGN 6.3)
# max_samples per (tier, dataset, split); None = all. Size-dependent entries are dicts keyed by a
# size class ("small" = 160m/410m, "mid" = 1b/1.4b, "large" = 2.8b); see max_samples().
_ALL = {"train": None, "val": None}
TIERS = {
    "smoke": {
        "harmless": {"train": 1000, "val": 300},
        "helpful": {"train": 1000, "val": 300},
        "harmless_pooled": {"train": 1000, "val": 300},
        "honesty": {"train": 1000, "val": 300},
        "helpsteer2": {"train": 500, "val": 200},
        "prompt_sets": {"train": 100, "val": 100},
    },
    "t4": {
        "harmless": {"small": _ALL, "mid": {"train": 10000, "val": None}, "large": {"train": 6000, "val": None}},
        "helpful": {"small": {"train": 20000, "val": None}, "mid": {"train": 10000, "val": None}, "large": {"train": 6000, "val": None}},
        "harmless_pooled": {"small": {"train": 20000, "val": None}, "mid": {"train": 10000, "val": None}, "large": {"train": 6000, "val": None}},
        "honesty": _ALL,
        "helpsteer2": {"small": _ALL, "mid": {"train": 8000, "val": None}, "large": {"train": 4000, "val": None}},
        "prompt_sets": _ALL,
    },
    "l4": {
        "harmless": {"small": _ALL, "mid": _ALL, "large": {"train": 10000, "val": None}},
        "helpful": {"small": _ALL, "mid": _ALL, "large": {"train": 10000, "val": None}},
        "harmless_pooled": {"small": {"train": 20000, "val": None}, "mid": {"train": 20000, "val": None}, "large": {"train": 10000, "val": None}},
        "honesty": _ALL,
        "helpsteer2": {"small": _ALL, "mid": _ALL, "large": {"train": 8000, "val": None}},
        "prompt_sets": _ALL,
    },
    "a100": {
        "harmless": {"small": _ALL, "mid": _ALL, "large": {"train": 20000, "val": None}},
        "helpful": {"small": _ALL, "mid": _ALL, "large": {"train": 20000, "val": None}},
        "harmless_pooled": {"small": _ALL, "mid": _ALL, "large": {"train": 20000, "val": None}},
        "honesty": _ALL,
        "helpsteer2": _ALL,
        "prompt_sets": _ALL,
    },
    "cluster": {k: _ALL for k in ["harmless", "helpful", "harmless_pooled", "honesty", "helpsteer2", "prompt_sets"]},
}
TIER_NAMES = list(TIERS)


def size_class(model_name):
    """'small' (160m/410m), 'mid' (1b/1.4b) or 'large' (2.8b and paper models)."""
    name = model_name.lower()
    if any(s in name for s in ("160m", "410m")):
        return "small"
    if any(s in name for s in ("pythia-1b", "1.4b")):
        return "mid"
    return "large"


def tier_dataset_key(dataset):
    """Maps an `extract` dataset name onto the TIERS table row."""
    if dataset in HELPSTEER_VALUES_EXT or dataset == "helpsteer2":
        return "helpsteer2"
    if dataset in PROMPT_SETS:
        return "prompt_sets"
    return dataset


def max_samples(tier, dataset, split, model_name="pythia-160m"):
    """Tier default for max_samples (None = all texts) of `dataset`/`split` (DESIGN 6.3)."""
    if tier is None:
        return None
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {TIER_NAMES}")
    entry = TIERS[tier][tier_dataset_key(dataset)]
    if "train" not in entry:
        entry = entry[size_class(model_name)]
    return entry["train" if split == "train" else "val"]


# ----------------------------------------------------------------------------- numerics (DESIGN 3.1)
# Pythia's attention logits reach ~8e3 (massive activations); eager attention stores QK^T in the model
# dtype, which is wrong in bf16 (rounding +-32) and NaN in fp16 from layer 5 on. Hence float32 for
# 160m/410m on every device, float16 + sdpa from 1b on GPU, and never eager in half precision.
DTYPE_POLICY = {
    "small": {"cpu": ("float32", "eager"), "cuda": ("float32", "eager")},
    "mid": {"cpu": ("float32", "eager"), "cuda": ("float16", "sdpa")},
    "large": {"cpu": ("float32", "eager"), "cuda": ("float16", "sdpa")},
    "forbidden": [("bfloat16", "eager"), ("float16", "eager")],
    "attn_impl_for_dtype": {"float32": "eager", "float16": "sdpa", "bfloat16": "sdpa"},
    "tolerance_kv_cache_vs_full": 1e-2,   # fp32 KV-cache decode vs full recompute differs up to 4.9e-3 (L11)
    # batched (padded, batch > 1) vs single-text hook inputs in fp32: pure GEMM-kernel numerics amplified by the massive
    # activations, length dependent (2.8e-3 at ~300 tokens even for a batch of identical rows); the selftest / layout
    # test use this absolute tolerance, not 1e-3 (DESIGN 3.1: "Tests mit Toleranz 1e-2, nicht 1e-5")
    "tolerance_batched_vs_single": 1e-2,
    # selftest gate (DESIGN 3.1 [korr. 2026-11-04]): per-layer relative L2 error ||batch - single|| / ||single|| of the
    # hook inputs. The layout / padding check always runs in fp32 (half-precision models are cast up losslessly and back):
    # measured on pythia-2.8b@step100000 a wrong token position gives >= 0.82, fp32 kernel numerics << 1e-2. In fp16 the
    # same comparison is dominated by rounding noise (fp16 vs fp32 of ONE text: up to 0.16 in layers 24-31 of 2.8b on CPU,
    # GPU batched-vs-single ~3x the CPU value), so the native half-precision run is only a sanity gate against overflow / NaN.
    "tolerance_layout_normrel_fp32": 1e-2,
    "sanity_normrel_half_precision": 0.5,
}


def dtype_policy(model_name, device="cpu"):
    """(dtype name, attn_implementation) recommended for `model_name` on `device` ('cpu'|'cuda'|'mps')."""
    dev = "cuda" if str(device).startswith(("cuda", "mps")) else "cpu"
    return DTYPE_POLICY[size_class(model_name)][dev]


# ----------------------------------------------------------------------------- paths (DESIGN 6.4)
DEFAULT_ROOT = "./vh_runs"


def paths(root=None):
    """Result / feature / figure directories below `root` (env VH_ROOT, default ./vh_runs).

    Features default to {root}/features but can be redirected to fast local disk with env VH_FEATURES
    (Colab: VH_ROOT on Drive, VH_FEATURES=/content/features). Directories are created on demand by the
    commands, not here.
    """
    root = os.path.abspath(root or os.environ.get("VH_ROOT") or DEFAULT_ROOT)
    features = os.path.abspath(os.environ.get("VH_FEATURES") or os.path.join(root, "features"))
    results = os.path.join(root, "results")
    figures = os.path.join(root, "figures")
    return {
        "root": root,
        "features": features,
        "results": results,
        "heads": os.path.join(results, "heads"),
        "directions": os.path.join(results, "directions"),
        "eval": os.path.join(results, "eval"),
        "judge": os.path.join(results, "judge"),
        "sweep": os.path.join(results, "sweep"),
        "manifest": os.path.join(results, "manifest.json"),
        "figures": figures,
        "heatmaps": os.path.join(figures, "heatmaps"),
        "figures_sweep": os.path.join(figures, "sweep"),
        "figures_causal": os.path.join(figures, "causal"),
    }


def model_short_name(name_or_hf_id):
    """Short registry name for a model (unknown HF ids are sanitized: 'org/model' -> 'org__model')."""
    if name_or_hf_id in MODELS:
        return name_or_hf_id
    for short, info in MODELS.items():
        if info["hf_id"] == name_or_hf_id:
            return short
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name_or_hf_id)


def resolve_model(name_or_hf_id):
    """(short name, hf_id, geometry dict or None)."""
    short = model_short_name(name_or_hf_id)
    info = MODELS.get(short)
    hf_id = info["hf_id"] if info else name_or_hf_id
    return short, hf_id, info


def feature_dir(p, model, revision, dataset, split):
    """{features}/{model}/{revision}/{dataset}_{split} (DESIGN 6.4)."""
    return os.path.join(p["features"], model_short_name(model), revision or "main", f"{dataset}_{split}")


def heads_path(p, model, revision, value, mode, feature_tag=None):
    """results/heads/{model}_{revision}_{value}_{mode}[_{feature_tag}].json"""
    tag = f"_{feature_tag}" if feature_tag else ""
    return os.path.join(p["heads"], f"{model_short_name(model)}_{revision or 'main'}_{value}_{mode}{tag}.json")


def directions_path(p, model, revision, value, mode="pythia"):
    """results/directions/{model}_{revision}_{value}.npz (paper mode: ..._{value}_paper.npz)."""
    tag = "" if mode == "pythia" else f"_{mode}"
    return os.path.join(p["directions"], f"{model_short_name(model)}_{revision or 'main'}_{value}{tag}.npz")


def eval_dir(p, model, revision):
    """results/eval/{model}_{revision}/"""
    return os.path.join(p["eval"], f"{model_short_name(model)}_{revision or 'main'}")
