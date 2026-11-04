"""Free-form answer generation under Base / TopK / Random / Steer conditions (DESIGN 3.2 and 7; paper Sec. 4.2-5).

Adapted from sihuo-design/valueheads `generation.py` (MIT): the `paper` prompt style reproduces it exactly
(template "\\nAnswer question:\\n{question}\\n", no chat template, greedy, 128 new tokens, stop as soon as the last
token is "]", decode of the new tokens with skip_special_tokens). Additions for base models (DESIGN 3.2):

  fewshot_qa   the lm-eval-harness TruthfulQA 6-shot primer followed by "Q: {question}\\nA:", greedy, 48 new
               tokens, stop at "\\n\\n" or "\\nQ:" (default for honesty / hh-rlhf / beavertails / jailbreak / malicious)
  qa_plain     "Question: {prompt}\\nAnswer:" (= the probing format), 64 new tokens, stop at "\\n\\nQuestion:"
               (default for the HelpSteer2 prompt sets)

Generation is batched with left padding (prompts longer than the context minus the new tokens are truncated from
the left, i.e. the tail is kept). Interventions are the core hooks (HeadMask / HeadSteer); the paper semantics are
positions="last" + decode_only=True (prompt forward untouched, every KV-cache decoding step edited), positions="all"
additionally edits every position of the prompt forward. Each condition is written to one jsonl file: a header line
with the condition metadata, then one record per question. Answers of the harmful prompt sets (HARMFUL_SETS) are
written to disk for the judges but never printed.
"""
import contextlib
import json
import os
import time
from collections import Counter

import torch
from tqdm.auto import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from . import config
from .data import EVAL_SETS
from .eval_likelihood import FEWSHOT_PRIMER
from .hooks import HeadMask, HeadSteer

PAPER_TEMPLATE = "\nAnswer question:\n{question}\n"            # generation.py:5 of the reference repo
FEWSHOT_TEMPLATE = FEWSHOT_PRIMER + "Q: {question}\nA:"          # DESIGN 3.2 (lm-eval-harness primer)
QA_PLAIN_TEMPLATE = "Question: {question}\nAnswer:"              # = data.helpsteer_text without the response

PROMPT_STYLES = {
    "paper": {"template": PAPER_TEMPLATE, "max_new_tokens": 128, "stop_strings": [], "stop_token": "]"},
    "fewshot_qa": {"template": FEWSHOT_TEMPLATE, "max_new_tokens": 48, "stop_strings": ["\n\n", "\nQ:"], "stop_token": None},
    "qa_plain": {"template": QA_PLAIN_TEMPLATE, "max_new_tokens": 64, "stop_strings": ["\n\nQuestion:"], "stop_token": None},
}
PROMPT_STYLE_NAMES = list(PROMPT_STYLES)

# Prompt sets whose contents (and the generated answers) must never be printed or shown in notebook outputs.
HARMFUL_SETS = {"jailbreak", "malicious", "beavertails", "hh-rlhf"}
STOP_REASONS = ["stop_string", "stop_token", "eos", "max_new_tokens"]


def default_prompt_style(eval_set, mode="pythia"):
    """paper -> repo template; HelpSteer2 prompt sets -> qa_plain; everything else -> fewshot_qa (DESIGN 3.2)."""
    if mode == "paper":
        return "paper"
    if eval_set in config.HELPSTEER_VALUES_EXT:
        return "qa_plain"
    return "fewshot_qa"


def build_prompt(prompt_style, question):
    """The prompt text of `question` under `prompt_style` (no chat template)."""
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {prompt_style!r}; choose from {PROMPT_STYLE_NAMES}")
    return PROMPT_STYLES[prompt_style]["template"].format(question=question)


def judge_values_of(eval_set):
    """Values the judges score on `eval_set` (single-value sets: one value; hh-rlhf / beavertails: all six)."""
    if eval_set not in EVAL_SETS:
        raise ValueError(f"unknown eval set {eval_set!r}; choose from {list(EVAL_SETS)}")
    v = EVAL_SETS[eval_set]
    return [v] if v else list(config.PAPER_VALUES)


# ----------------------------------------------------------------------------- stopping criteria
class StopOnToken(StoppingCriteria):
    """Per-row stop as soon as the last generated token equals `token_id` (repo: "]")."""

    def __init__(self, token_id):
        self.token_id = int(token_id)

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[:, -1] == self.token_id


class StopOnStrings(StoppingCriteria):
    """Per-row stop as soon as one of `stop_strings` occurs in the text generated after the first `prompt_len` tokens."""

    def __init__(self, tokenizer, stop_strings, prompt_len):
        self.tokenizer = tokenizer
        self.stop_strings = list(stop_strings)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids, scores, **kwargs):
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if not self.stop_strings or input_ids.shape[1] <= self.prompt_len:
            return done
        texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True)
        for i, text in enumerate(texts):
            if any(s in text for s in self.stop_strings):
                done[i] = True
        return done


def cut_at_stop(text, stop_strings):
    """(text before the first occurrence of any stop string, whether one was found)."""
    cut, found = len(text), False
    for s in stop_strings:
        i = text.find(s)
        if 0 <= i < cut:
            cut, found = i, True
    return text[:cut], found


# ----------------------------------------------------------------------------- encoding
def encode_prompts(tokenizer, prompts, max_prompt_tokens=None):
    """Left-padded batch encoding (left truncation to `max_prompt_tokens`, i.e. the tail of a long prompt is kept).

    Returns (BatchEncoding with input_ids / attention_mask, [untruncated token count per prompt], [truncated flags]).
    """
    lengths = [len(ids) for ids in tokenizer(prompts)["input_ids"]]
    old_pad, old_trunc = tokenizer.padding_side, tokenizer.truncation_side
    tokenizer.padding_side, tokenizer.truncation_side = "left", "left"
    try:
        kwargs = dict(return_tensors="pt", padding=True)
        if max_prompt_tokens is not None:
            kwargs.update(truncation=True, max_length=int(max_prompt_tokens))
        enc = tokenizer(prompts, **kwargs)
    finally:
        tokenizer.padding_side, tokenizer.truncation_side = old_pad, old_trunc
    truncated = [max_prompt_tokens is not None and n > max_prompt_tokens for n in lengths]
    return enc, lengths, truncated


# ----------------------------------------------------------------------------- generation
def _row_tokens(new_ids, eos_id, pad_id):
    """Generated tokens of one row up to (excluding) the first EOS / padding token, and whether an EOS was hit."""
    toks = new_ids.tolist()
    n_eos = next((i for i, t in enumerate(toks) if t == eos_id), None)
    n_pad = next((i for i, t in enumerate(toks) if t == pad_id), None) if pad_id != eos_id else None
    cut = min(x for x in (n_eos, n_pad, len(toks)) if x is not None)
    return toks[:cut], n_eos is not None and n_eos == cut


@torch.no_grad()
def generate_answers(model, tokenizer, questions, prompt_style="fewshot_qa", hook=None, batch_size=8, max_new_tokens=None,
                     temperature=None, top_p=0.9, max_length=config.MAX_LENGTH, progress=False, desc=None):
    """Answers `questions` under `prompt_style`; `hook` (HeadMask / HeadSteer or None) is active for every batch.

    Greedy by default; with `temperature`, samples with top-p (repo robustness variant). Batches are formed in
    descending prompt length (left padding), prompts are truncated from the left to max_length - max_new_tokens.
    Returns one record per question (input order): {idx, question, answer, raw_answer, n_new_tokens, stop_reason,
    prompt_tokens, prompt_truncated}. `answer` is the decoded text cut at the first stop string and stripped;
    `raw_answer` the uncut decode of the generated tokens (paper style: identical, the "]" is kept as in the repo).
    """
    spec = PROMPT_STYLES[prompt_style]
    max_new = int(max_new_tokens or spec["max_new_tokens"])
    stop_strings = spec["stop_strings"]
    stop_token_id = tokenizer.convert_tokens_to_ids(spec["stop_token"]) if spec["stop_token"] else None
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
    prompts = [build_prompt(prompt_style, q) for q in questions]
    lengths = [len(ids) for ids in tokenizer(prompts)["input_ids"]] if prompts else []
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    max_prompt = max(8, int(max_length) - max_new)
    sampling = dict(do_sample=True, temperature=float(temperature), top_p=top_p) if temperature else dict(do_sample=False)
    records = [None] * len(prompts)
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    ctx = hook if hook is not None else contextlib.nullcontext()
    with ctx:
        for idx in tqdm(batches, desc=desc or "generate", disable=not progress):
            enc, _, truncated = encode_prompts(tokenizer, [prompts[i] for i in idx], max_prompt)
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
            prompt_len = input_ids.shape[1]
            if hook is not None and getattr(hook, "decode_only", False) and prompt_len < 2:
                # a 1-token prompt forward has the shape of a decoding step, so a decode-only hook would edit the prefill
                raise ValueError("decode-only hooks cannot distinguish a 1-token prompt from a decoding step (the prompt forward "
                                 "would be edited); use prompts of >= 2 tokens or decode_only=False")
            criteria = StoppingCriteriaList([StopOnStrings(tokenizer, stop_strings, prompt_len)] if stop_strings else [])
            if stop_token_id is not None:
                criteria.append(StopOnToken(stop_token_id))
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new, **sampling,
                                 stopping_criteria=criteria, pad_token_id=pad_id)
            for row, i in enumerate(idx):
                toks, hit_eos = _row_tokens(out[row, prompt_len:], eos_id, pad_id)
                raw = tokenizer.decode(toks, skip_special_tokens=True)
                answer, hit_stop = cut_at_stop(raw, stop_strings)
                if hit_stop:
                    reason = "stop_string"
                elif stop_token_id is not None and toks and toks[-1] == stop_token_id:
                    reason = "stop_token"
                elif hit_eos:
                    reason = "eos"
                elif len(toks) >= max_new:
                    reason = "max_new_tokens"
                else:
                    reason = "eos"       # pad == eos: the row finished with an EOS that also serves as padding
                records[i] = {"idx": i, "question": questions[i], "answer": answer.strip(), "raw_answer": raw,
                              "n_new_tokens": len(toks), "stop_reason": reason, "prompt_tokens": int(lengths[i]),
                              "prompt_truncated": bool(truncated[row])}
    return records


def generation_hook(model, intervention, heads_by_layer=None, vectors=None, eps=config.MASK_EPS, alpha=1.0, positions="last",
                    decode_only=None):
    """HeadMask / HeadSteer for the generation path (None for intervention "none").

    positions="last" with decode_only=None -> decode_only=True (paper semantics); positions="all" -> decode_only=False
    unless given explicitly, so that the prompt forward is edited at every position too.
    """
    if intervention == "none":
        return None
    if decode_only is None:
        decode_only = positions == "last"
    if intervention == "mask":
        return HeadMask(model, heads_by_layer, eps=eps, positions=positions, decode_only=decode_only)
    if intervention == "steer":
        return HeadSteer(model, vectors, alpha=alpha, positions=positions, decode_only=decode_only)
    raise ValueError(f"unknown intervention {intervention!r}")


# ----------------------------------------------------------------------------- persistence
def gen_tag(eval_set, prompt_style):
    return f"gen-{eval_set}-{prompt_style}"


def condition_tag(intervention, head_values=None, select="topk", k=None, random_seed=None, alpha=None, positions="last", decode_only=None):
    """File tag of a condition: none | mask-{values}_{select}_k{k}[_a{alpha}][_seed{s}][_all][_decodeonly|_prefill].

    Paper semantics (positions last, decode_only) carry no suffix; positions="all" adds "_all", and a decode_only
    that deviates from the default of its positions adds "_decodeonly" / "_prefill".
    """
    if intervention == "none":
        return "none"
    values = "+".join(head_values) if isinstance(head_values, (list, tuple)) else str(head_values)
    tag = f"{intervention}-{values}_{select}_k{k}"
    if alpha is not None:
        tag += f"_a{alpha:g}"
    if random_seed is not None:
        tag += f"_seed{random_seed}"
    if positions == "all":
        tag += "_all"
    if decode_only is not None and bool(decode_only) != (positions == "last"):
        tag += "_decodeonly" if decode_only else "_prefill"
    return tag


# header fields that identify a stored condition (a cached file is reused only when all of them match)
CONDITION_KEYS = ["intervention", "select", "head_values", "k", "eps", "alpha", "positions", "decode_only", "random_seed", "weights",
                  "prompt_style", "n", "max_new_tokens", "temperature", "model", "revision", "mode"]


def same_condition(header, wanted):
    """True when every CONDITION_KEYS entry of `wanted` equals the stored header's (JSON round-trip tolerant)."""
    return all(json.loads(json.dumps(header.get(k), default=str)) == json.loads(json.dumps(wanted.get(k), default=str)) for k in CONDITION_KEYS)


def generations_path(eval_dir, eval_set, prompt_style, cond_tag):
    """results/eval/{model}_{revision}/gen-{eval_set}-{prompt_style}_{condition}.jsonl"""
    return os.path.join(eval_dir, f"{gen_tag(eval_set, prompt_style)}_{cond_tag}.jsonl")


def write_generations(path, header, records):
    """Header line ({"type": "header", ...}) followed by one {"type": "record", ...} line per question."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"type": "header", **header}, default=str) + "\n")
        for rec in records:
            f.write(json.dumps({"type": "record", **rec}, default=str) + "\n")
    return path


def read_generations(path):
    """(header dict, [record dict, ...]) of a generations jsonl written by write_generations."""
    header, records = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "header":
                header = obj
            else:
                records.append(obj)
    if header is None:
        raise ValueError(f"{path}: no header line (not an eval-generate output)")
    return header, records


# ----------------------------------------------------------------------------- summaries
def summarize_records(records):
    """Content-free statistics of a condition: counts, token / character lengths, stop reasons, heuristic flags."""
    from .judge.heuristic import RefusalHeuristic
    heur = RefusalHeuristic()
    n = len(records)
    out = {"n": n, "mean_new_tokens": 0.0, "mean_answer_chars": 0.0, "empty": 0, "truncated_prompts": 0,
           "refusal_rate": 0.0, "no_comment_rate": 0.0, "degenerate_rate": 0.0}
    for r in STOP_REASONS:
        out[f"stop_{r}"] = 0
    if n == 0:
        return out
    flags = [heur.score(r["question"], r["answer"]) for r in records]
    out["mean_new_tokens"] = sum(r["n_new_tokens"] for r in records) / n
    out["mean_answer_chars"] = sum(len(r["answer"]) for r in records) / n
    out["empty"] = sum(1 for r in records if not r["answer"])
    out["truncated_prompts"] = sum(1 for r in records if r.get("prompt_truncated"))
    out["refusal_rate"] = sum(f["refusal"] for f in flags) / n
    out["no_comment_rate"] = sum(f["no_comment"] for f in flags) / n
    out["degenerate_rate"] = sum(f["degenerate"] for f in flags) / n
    for reason, c in Counter(r["stop_reason"] for r in records).items():
        out[f"stop_{reason}"] = c
    return out


def format_examples(conditions, n=3, harmful=False, width=160):
    """Side-by-side examples (question, then the answer of every condition). Empty string for harmful sets."""
    if harmful or not conditions:
        return ""
    names = list(conditions)
    base = conditions[names[0]]
    lines = []
    for rec in base[:n]:
        lines.append(f"Q: {rec['question'][:width]}")
        for name in names:
            match = next((r for r in conditions[name] if r["idx"] == rec["idx"]), None)
            ans = (match["answer"] if match else "<missing>").replace("\n", "\\n")
            lines.append(f"  [{name}] {ans[:width]}")
    return "\n".join(lines)
