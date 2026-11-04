"""Judges for generated answers (DESIGN 4.2): the paper's LLM judge behind an OpenAI-compatible API (with the
fixes of DESIGN 4.2), ArmoRM multi-objective rewards, the TruthfulQA truth / info judges, Qwen3Guard-Gen and a
keyword refusal heuristic. `composite.stack_for(tier)` gives the default judge set per value and Colab tier.

Every judge exposes `evaluate(question, answer, value=None) -> {"label": 0 | 1 | None, "score": float | None, ...}`.
The heavy judges are imported lazily (judge.composite.make_judge) so that this package stays importable without
torch / transformers.
"""
from .composite import JUDGE_NAMES, STACKS, TIERS, make_judge, stack_for  # noqa: F401
from .heuristic import RefusalHeuristic  # noqa: F401
from .openai_compat import SYSTEM_PROMPTS, OpenAICompatJudge, parse_verdict  # noqa: F401
