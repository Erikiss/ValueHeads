"""Judge stacks per Colab tier (DESIGN 4.2 table) and the judge factory.

  cpu      no judge; the refusal heuristic for harmless as a smoke test only
  t4       ArmoRM nf4 (HelpSteer values, honesty proxy; loaded without remote code, see armorm_model.py), Qwen3Guard-Gen-0.6B
           (harmless), TruthfulQA truth x info 4-bit (honesty generations); loaded one after the other (<= 6 GB VRAM;
           downloads 15 + 1.5 + 2 x 13.5 GB into HF_HOME, see `judge --cleanup_judges`)
  t4_api   the exact paper judge on OpenRouter (qwen/qwen3-30b-a3b-instruct-2507, paper prompts) for every value
  l4       as t4, plus the paper prompts on a local vLLM Qwen3-4B-Instruct-2507 (judge "openai", port 8000)
  a100     the paper judge locally (vllm serve cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit --port 30000, or the
           llama-server GGUF), harmless cross-checked with Qwen3Guard
  cluster  the repo's serve_judge.sh (Qwen/Qwen3-30B-A3B-Instruct-2507 bf16) plus the kappa cross-checks
           (Qwen3Guard for harmless, TruthfulQA truth judge for honesty)
"""
from .. import config

JUDGE_NAMES = ["heuristic", "openai", "armorm", "tqa", "qwen3guard"]
TIERS = ["cpu", "t4", "t4_api", "l4", "a100", "cluster"]
HELPSTEER = list(config.HELPSTEER_VALUES_EXT)
ALL_VALUES = ["harmless", "helpful", "honesty"] + HELPSTEER

_T4 = {**{v: ["armorm"] for v in HELPSTEER}, "helpful": ["armorm"], "honesty": ["tqa", "armorm"], "harmless": ["qwen3guard"]}
STACKS = {
    "cpu": {**{v: [] for v in ALL_VALUES}, "harmless": ["heuristic"]},
    "t4": _T4,
    "t4_api": {v: ["openai"] for v in ALL_VALUES},
    "l4": {v: _T4[v] + ["openai"] for v in ALL_VALUES},
    "a100": {**{v: ["openai"] for v in ALL_VALUES}, "harmless": ["openai", "qwen3guard"]},
    "cluster": {**{v: ["openai"] for v in ALL_VALUES}, "harmless": ["openai", "qwen3guard"], "honesty": ["openai", "tqa"]},
}
# Defaults of the "openai" judge per tier (model id must be explicit, DESIGN 4.2).
OPENAI_DEFAULTS = {
    "t4_api": {"model": "qwen/qwen3-30b-a3b-instruct-2507", "base_url": "https://openrouter.ai/api/v1"},
    "l4": {"model": "Qwen/Qwen3-4B-Instruct-2507", "base_url": "http://localhost:8000/v1"},
    "a100": {"model": "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit", "base_url": "http://localhost:30000/v1"},
    "cluster": {"model": "Qwen/Qwen3-30B-A3B-Instruct-2507", "base_url": "http://localhost:30000/v1"},
}
# Values each judge can score.
JUDGE_VALUES = {
    "heuristic": ["harmless", "harmless_pooled"],
    "openai": ["harmless", "harmless_pooled", "helpful", "honesty"] + HELPSTEER,
    "armorm": ["harmless", "harmless_pooled", "helpful", "honesty"] + HELPSTEER,
    "tqa": ["honesty"],
    "qwen3guard": ["harmless", "harmless_pooled"],
}


def stack_for(tier, value=None):
    """{value: [judge names]} of a tier, or the list for one value (aliases: harmless_pooled -> harmless)."""
    if tier not in STACKS:
        raise ValueError(f"unknown tier {tier!r}; choose from {TIERS}")
    stack = STACKS[tier]
    if value is None:
        return {v: list(j) for v, j in stack.items()}
    key = "harmless" if value == "harmless_pooled" else value
    if key not in stack:
        raise ValueError(f"no judge stack for value {value!r} in tier {tier!r}")
    return list(stack[key])


def judges_for(tier, values):
    """Ordered, de-duplicated judge names covering `values` in `tier`."""
    out = []
    for v in values:
        for j in stack_for(tier, v):
            if j not in out:
                out.append(j)
    return out


def can_judge(judge, value):
    return value in JUDGE_VALUES[judge]


def make_judge(name, tier=None, model=None, base_url=None, api_key=None, cache_path=None, device="auto", load_in_4bit=None,
               with_info=True, **kwargs):
    """Instantiates a judge by name (heavy judges import torch / transformers lazily; models load on first use)."""
    if name == "heuristic":
        from .heuristic import RefusalHeuristic
        return RefusalHeuristic()
    if name == "openai":
        from .openai_compat import OpenAICompatJudge
        defaults = OPENAI_DEFAULTS.get(tier or "", {})
        model = model or defaults.get("model")
        base_url = base_url or defaults.get("base_url") or "http://localhost:30000/v1"
        if not model:
            raise ValueError("judge 'openai' needs an explicit model id (--judge_model), or a tier with a default (t4_api, l4, a100, cluster)")
        return OpenAICompatJudge(model, base_url=base_url, api_key=api_key, cache_path=cache_path, **kwargs)
    four_bit = (tier in ("t4", "l4")) if load_in_4bit is None else bool(load_in_4bit)
    if name == "armorm":
        from .armorm import MODEL_ID, ArmoRMJudge
        return ArmoRMJudge(model_id=model or MODEL_ID, device=device, load_in_4bit=four_bit, **kwargs)
    if name == "tqa":
        from .tqa_judge import TQATruthInfoJudge
        return TQATruthInfoJudge(device=device, load_in_4bit=four_bit, with_info=with_info)
    if name == "qwen3guard":
        from .qwen3guard import Qwen3GuardJudge
        return Qwen3GuardJudge(model_id=model or "Qwen/Qwen3Guard-Gen-0.6B", device=device, **kwargs)
    raise ValueError(f"unknown judge {name!r}; choose from {JUDGE_NAMES}")
