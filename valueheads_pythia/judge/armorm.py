"""ArmoRM-Llama3-8B-v0.1 (RLHFlow) multi-objective reward model as a continuous judge (DESIGN 4.2, tier t4+).

Objective names and the HelpSteer mapping are taken from the model card (README.md of
RLHFlow/ArmoRM-Llama3-8B-v0.1): `output.rewards` holds 19 objectives in the order ATTRIBUTES, the first five are
the HelpSteer attributes whose predictions map back to the original 0-4 rating scale as `reward * 5 - 0.5`;
`output.score` is the gated preference score, `output.gating_output @ model.reward_transform_matrix.T` the
per-objective coefficients. The checkpoint's remote code (modeling_custom.py, written for transformers 4.40) imports
`LLAMA_INPUTS_DOCSTRING`, which transformers >= 4.56 no longer has (review finding F1), so the default loader is the
re-implementation `armorm_model.ArmoRMForRewardWithGating` (same parameter names, no remote code; `loader="remote"` keeps
the trust_remote_code path for transformers < 4.52). On a T4 use 4-bit nf4 (~5.5 GB) with float16 compute (no bf16 on
T4). Llama-3 license. Download: 15.0 GB (4 safetensors shards).

Value mapping of this project: helpfulness / coherence / complexity / verbosity (and coherence_strict) -> the
helpsteer-* objectives on the 0-4 scale, binarized with the paper's rating rule (helpfulness / coherence: 4,
complexity / verbosity: 3-4); honesty -> ultrafeedback-honesty; harmless -> beavertails-is_safe; helpful ->
ultrafeedback-helpfulness (raw 0-1 rewards, binarized at 0.5).
"""
import torch

MODEL_ID = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
ATTRIBUTES = ["helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence", "helpsteer-complexity", "helpsteer-verbosity",
              "ultrafeedback-overall_score", "ultrafeedback-instruction_following", "ultrafeedback-truthfulness", "ultrafeedback-honesty",
              "ultrafeedback-helpfulness", "beavertails-is_safe", "prometheus-score", "argilla-overall_quality", "argilla-judge_lm",
              "code-complexity", "code-style", "code-explanation", "code-instruction-following", "code-readability"]
HELPSTEER_ATTRIBUTES = ATTRIBUTES[:5]
VALUE_OBJECTIVE = {"helpfulness": "helpsteer-helpfulness", "coherence": "helpsteer-coherence", "coherence_strict": "helpsteer-coherence",
                   "complexity": "helpsteer-complexity", "verbosity": "helpsteer-verbosity", "honesty": "ultrafeedback-honesty",
                   "harmless": "beavertails-is_safe", "harmless_pooled": "beavertails-is_safe", "helpful": "ultrafeedback-helpfulness"}
# Paper binarization of the HelpSteer2 ratings (data.helpsteer_label): positive at rating 4 (helpfulness / coherence)
# or 3-4 (complexity / verbosity); predictions are continuous, so the thresholds sit halfway between the ratings.
HELPSTEER_THRESHOLD = {"helpfulness": 3.5, "coherence": 3.5, "coherence_strict": 3.5, "complexity": 2.5, "verbosity": 2.5}


def helpsteer_scale(reward):
    """ArmoRM helpsteer-* reward (about 0-1) -> the original 0-4 rating scale (model card: reward * 5 - 0.5)."""
    return float(reward) * 5.0 - 0.5


def binarize(value, score):
    """0/1 label of a mapped value score (HelpSteer values on the 0-4 scale, other objectives at 0.5)."""
    thr = HELPSTEER_THRESHOLD.get(value, 0.5)
    return int(float(score) >= thr)


def map_rewards(rewards, preference=None):
    """{objective: reward} of a 19-vector plus the HelpSteer attributes on the 0-4 scale and the per-value scores."""
    rewards = [float(r) for r in rewards]
    if len(rewards) != len(ATTRIBUTES):
        raise ValueError(f"expected {len(ATTRIBUTES)} objectives, got {len(rewards)}")
    by_name = dict(zip(ATTRIBUTES, rewards))
    helpsteer = {a.split("-", 1)[1]: helpsteer_scale(by_name[a]) for a in HELPSTEER_ATTRIBUTES}
    value_scores = {}
    for value, obj in VALUE_OBJECTIVE.items():
        value_scores[value] = helpsteer_scale(by_name[obj]) if obj in HELPSTEER_ATTRIBUTES else by_name[obj]
    return {"preference": None if preference is None else float(preference), "rewards": by_name, "helpsteer": helpsteer,
            "value_scores": value_scores}


LOADERS = ["native", "remote"]


def load_armorm(model_id=MODEL_ID, device="auto", dtype=None, load_in_4bit=False, loader="native"):
    """(model, tokenizer); 4-bit nf4 + float16 compute for T4, else float16 on GPU / float32 on CPU.

    loader "native" (default): `armorm_model.ArmoRMForRewardWithGating.from_pretrained` (works on transformers >= 4.56);
    loader "remote": the checkpoint's own modeling_custom.py via trust_remote_code (needs transformers < 4.52).
    """
    from transformers import AutoTokenizer
    if loader not in LOADERS:
        raise ValueError(f"loader must be one of {LOADERS}")
    kwargs = {}
    cuda = torch.cuda.is_available() and str(device) != "cpu"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = dtype or (torch.float16 if cuda else torch.float32)
        kwargs["device_map"] = ("auto" if cuda else "cpu") if device in (None, "auto") else device
    if loader == "native":
        from .armorm_model import ArmoRMForRewardWithGating
        model = ArmoRMForRewardWithGating.from_pretrained(model_id, **kwargs).eval()
    else:
        from transformers import AutoModelForSequenceClassification
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True, **kwargs).eval()
        except ImportError as e:
            raise ImportError(f"ArmoRM remote code cannot be imported on this transformers version ({e}); it needs transformers < 4.52 "
                              "(DESIGN 4.2 / 8). Use loader='native' (judge --armorm_loader native, the default), which loads the same "
                              "weights without remote code.") from e
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tokenizer


def chat_input_ids(tokenizer, messages, max_length=4096):
    """Chat-template token ids (1, S) of `messages`, truncated to `max_length`, for any transformers version."""
    out = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", truncation=True, max_length=max_length)
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    out = torch.as_tensor(out)
    return out.reshape(1, -1) if out.dim() == 1 else out


class ArmoRMJudge:
    """score(question, answer) -> dict of objectives; evaluate(question, answer, value) -> uniform record.

    `model` / `tokenizer` may be given (e.g. mocks); otherwise the model is loaded lazily on first use.
    """

    name = "armorm"

    def __init__(self, model=None, tokenizer=None, model_id=MODEL_ID, device="auto", dtype=None, load_in_4bit=False,
                 max_length=4096, max_question_chars=8000, max_answer_chars=4000, loader="native"):
        self.model, self.tokenizer = model, tokenizer
        self.model_id, self.device, self.dtype, self.load_in_4bit = model_id, device, dtype, load_in_4bit
        self.loader = loader
        self.max_length, self.max_question_chars, self.max_answer_chars = max_length, max_question_chars, max_answer_chars

    def _ensure(self):
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_armorm(self.model_id, self.device, self.dtype, self.load_in_4bit, self.loader)

    @torch.no_grad()
    def score(self, question, answer):
        """{"preference", "rewards" (19 objectives), "helpsteer" (0-4 scale), "value_scores", "coefficients", "truncated"}."""
        from .openai_compat import truncate_text
        self._ensure()
        q, tq = truncate_text(question, self.max_question_chars)
        a, ta = truncate_text(answer, self.max_answer_chars)
        messages = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        input_ids = chat_input_ids(self.tokenizer, messages, self.max_length)
        device = getattr(self.model, "device", None)
        if device is not None:
            input_ids = input_ids.to(device)
        output = self.model(input_ids)
        rewards = output.rewards[0].detach().float().cpu()
        preference = output.score.reshape(-1)[0].detach().float().cpu()
        out = map_rewards(rewards.tolist(), preference.item())
        gating = getattr(output, "gating_output", None)
        transform = getattr(self.model, "reward_transform_matrix", None)
        if gating is not None and transform is not None:
            coeffs = (gating[0].detach().float().cpu() @ transform.data.detach().float().cpu().T)
            out["coefficients"] = dict(zip(ATTRIBUTES, [float(c) for c in coeffs]))
        out["truncated"] = bool(tq or ta)
        return out

    def free(self):
        """Drops the model (two-phase judging; the HF cache entry is removed by `judge --cleanup_judges`)."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate(self, question, answer, value):
        if value not in VALUE_OBJECTIVE:
            raise ValueError(f"ArmoRM has no objective for value {value!r}; choose from {list(VALUE_OBJECTIVE)}")
        s = self.score(question, answer)
        score = s["value_scores"][value]
        return {"judge": self.name, "value": value, "label": binarize(value, score), "score": float(score),
                "objective": VALUE_OBJECTIVE[value], "preference": s["preference"], "helpsteer": s["helpsteer"],
                "rewards": s["rewards"], "truncated": s["truncated"]}
