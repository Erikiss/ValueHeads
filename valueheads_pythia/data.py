"""Datasets for probing (labeled QA texts with group ids), likelihood evaluation (pairs) and generation.

Adapted from sihuo-design/valueheads `data.py` (MIT); label rules and text formats are the paper's
(DESIGN section 2), the `pythia`-mode additions follow DESIGN 3.3 / 4.1:

  harmless            HH-RLHF `harmless-base` single-turn dialogues; chosen = 1, rejected = 0
  helpful             HH-RLHF `helpful-base` (contrast value)
  harmless_pooled     HH-RLHF default config (all four subsets), the paper's "harmlessness" set; mode=paper only
  honesty             TruthfulQA (generation), seeded 80/20 question split; correct = 1, incorrect = 0
  helpfulness / coherence   HelpSteer2; rating 4 = 1, rating 0-1 = 0
  complexity / verbosity    HelpSteer2; rating 3-4 = 1, rating 0 = 0
  coherence_strict    HelpSteer2; rating 4 = 1, rating <= 2 = 0

Every probing text comes with an integer group id (TruthfulQA question, HH dialogue pair, HelpSteer2 prompt)
for cluster bootstraps and group-aware subsampling. Never print HH-RLHF / BeaverTails / jailbreak contents.
"""
import hashlib
import os
import random
import re

import numpy as np
import pandas as pd
from datasets import load_dataset

from .config import ALL_VALUES, HELPSTEER_VALUES, HELPSTEER_VALUES_EXT, PAPER_VALUES, PROMPT_SETS

VALUES = PAPER_VALUES
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Seed of the TruthfulQA train/val question split. Probing (head identification) and evaluation use the
# same split so that evaluation questions are held out from head identification.
SPLIT_SEED = 42
VAL_RATIO = 0.2

HH_SUBSETS = {"harmless": "harmless-base", "helpful": "helpful-base", "harmless_pooled": None}

# Evaluation sets of the generation path, and the value the judge scores on each single-value set.
EVAL_SETS = {
    "jailbreak": "harmless",
    "malicious": "harmless",
    "honesty": "honesty",
    **{v: v for v in HELPSTEER_VALUES},
    "hh-rlhf": None,        # multi-value sets: judged on every value
    "beavertails": None,
}
# Activation set (split "val") used to estimate sigma for each evaluation set (steer.py of the repo).
STD_SETS = {"jailbreak": "jailbreak", "malicious": "malicious", "beavertails": "beavertails",
            "hh-rlhf": "harmless", **{v: v for v in ALL_VALUES}}

EVAL_PAIR_METRICS = ["tqa_pairwise", "hh_pairwise", "helpsteer2_pairwise"]


# ----------------------------------------------------------------------------- HH-RLHF
_HUMAN_RE = re.compile(r"\n\nHuman:\s*(.*?)(?=\n\nAssistant:|\Z)", re.S)
_ASSISTANT_RE = re.compile(r"\n\nAssistant:\s*(.*?)(?=\n\nHuman:|\Z)", re.S)


def _num_human_turns(text):
    return len(_HUMAN_RE.findall(text))


def single_turn(example):
    """Exactly one Human turn in both chosen and rejected (paper filter)."""
    return _num_human_turns(example["chosen"]) == 1 and _num_human_turns(example["rejected"]) == 1


def first_qa(text):
    """(question, answer) of the first turn of an HH-RLHF dialogue."""
    humans = _HUMAN_RE.findall(text)
    assistants = _ASSISTANT_RE.findall(text)
    if not humans or not assistants:
        return None, None
    return humans[0].strip(), assistants[0].strip()


def load_hhrlhf(split, subset=None):
    """HH-RLHF split ('train'|'test'); subset None = pooled default config, else e.g. 'harmless-base'."""
    kwargs = {"data_dir": subset} if subset else {}
    return load_dataset("Anthropic/hh-rlhf", **kwargs)[split]


def hh_pairs(value="harmless", split="train"):
    """Single-turn (chosen, rejected) pairs of the HH-RLHF set of `value` (harmless|helpful|harmless_pooled)."""
    if value not in HH_SUBSETS:
        raise ValueError(f"unknown HH value {value!r}; choose from {list(HH_SUBSETS)}")
    ds = load_hhrlhf("train" if split == "train" else "test", HH_SUBSETS[value])
    return [(ex["chosen"], ex["rejected"]) for ex in ds if single_turn(ex)]


# ----------------------------------------------------------------------------- TruthfulQA
def truthfulqa_indices(split_seed=SPLIT_SEED, n=817):
    """(train question indices, val question indices) of the seeded 80/20 split (paper: seed 42 -> 653/164)."""
    idxs = np.arange(n)
    train_idxs = np.random.RandomState(split_seed).choice(idxs, size=int(n * (1 - VAL_RATIO)), replace=False)
    train_set = set(train_idxs.tolist())
    val_idxs = np.array([i for i in idxs if i not in train_set])
    return train_idxs, val_idxs


def truthfulqa_split(split, split_seed=SPLIT_SEED):
    """Seeded 80/20 split of TruthfulQA (generation); split in {"train", "val"}. Returns a HF Dataset."""
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    train_idxs, val_idxs = truthfulqa_indices(split_seed, len(ds))
    return ds.select(train_idxs if split == "train" else val_idxs)


def truthfulqa_mc(split="val", split_seed=SPLIT_SEED):
    """Rows of the `multiple_choice` config (question, mc1_targets, mc2_targets) for the split's questions."""
    gen = truthfulqa_split(split, split_seed)
    wanted = set(gen["question"])
    mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    return [row for row in mc if row["question"] in wanted]


def tqa_text(question, answer):
    return f"Question: {question} Answer: {answer}"


# ----------------------------------------------------------------------------- HelpSteer2
def helpsteer_label(value, rating):
    """Binarized HelpSteer2 label (1 / 0), or None if the rating is discarded."""
    if value in ("helpfulness", "coherence"):
        return 1 if rating == 4 else 0 if rating in (0, 1) else None
    if value == "coherence_strict":
        return 1 if rating == 4 else 0 if rating in (0, 1, 2) else None
    if value in ("complexity", "verbosity"):
        return 1 if rating in (3, 4) else 0 if rating == 0 else None
    raise ValueError(f"not a HelpSteer2 value: {value!r}")


def helpsteer_rating_column(value):
    return "coherence" if value == "coherence_strict" else value


def load_helpsteer(split):
    return load_dataset("nvidia/HelpSteer2")["train" if split == "train" else "validation"]


def helpsteer_text(prompt, response):
    return f"Question: {prompt}\nAnswer: {response}"


def helpsteer2_rows(split):
    """All HelpSteer2 rows of a split for the extract-once path (DESIGN 3.3).

    Returns (texts, labels, groups): labels = {value: int8 array, -1 = discarded} for every value in
    HELPSTEER_VALUES_EXT, groups = prompt id (int, consecutive rows of the same prompt share it).
    """
    ds = load_helpsteer(split)
    prompts, responses = ds["prompt"], ds["response"]
    texts = [helpsteer_text(p, r) for p, r in zip(prompts, responses)]
    groups = _group_ids(prompts)
    labels = {}
    for value in HELPSTEER_VALUES_EXT:
        ratings = ds[helpsteer_rating_column(value)]
        labels[value] = np.array([(-1 if (l := helpsteer_label(value, int(r))) is None else l) for r in ratings], dtype=np.int8)
    return texts, labels, groups


def _group_ids(keys):
    """Integer group id per element, numbering distinct keys by first occurrence."""
    ids, lookup = np.empty(len(keys), dtype=np.int64), {}
    for i, k in enumerate(keys):
        ids[i] = lookup.setdefault(k, len(lookup))
    return ids


# ----------------------------------------------------------------------------- subsampling
def uniform_subsample(n, max_samples, seed=42):
    """Repo semantics (extract_activations.py): sorted random sample of indices, or all if n <= max_samples."""
    if max_samples is None or n <= max_samples:
        return np.arange(n)
    return np.array(sorted(random.Random(seed).sample(range(n), max_samples)))


def stratified_subsample(labels, groups, max_samples, seed=42, keep_all_negatives=False, max_neg_share=0.5):
    """Group-aware, label-stratified subsample (DESIGN 6.3): sorted indices of at most `max_samples` items.

    Class quotas follow the label proportions (n_neg = all negatives, capped at max_neg_share * max_samples,
    when keep_all_negatives); both quotas are filled by walking the groups in one seeded random order, so
    items of the same group (question / dialogue pair / prompt) are kept together as far as the quotas allow.
    Labels < 0 (discarded) are never selected.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    valid = np.flatnonzero(labels >= 0)
    if max_samples is None or len(valid) <= max_samples:
        return valid
    pos, neg = valid[labels[valid] == 1], valid[labels[valid] == 0]
    if keep_all_negatives:
        n_neg = min(len(neg), int(max_neg_share * max_samples))
    else:
        n_neg = int(round(max_samples * len(neg) / len(valid)))
    n_neg = min(n_neg, len(neg))
    n_pos = min(max_samples - n_neg, len(pos))
    n_neg = min(max_samples - n_pos, len(neg))
    rng = np.random.RandomState(seed)
    order = rng.permutation(np.unique(groups[valid]))
    rank = {g: i for i, g in enumerate(order)}
    chosen = []
    for cls_idx, quota in ((pos, n_pos), (neg, n_neg)):
        # stable sort by group rank, then by index -> whole groups first in the shared random order
        key = np.array([rank[g] for g in groups[cls_idx]])
        cls_sorted = cls_idx[np.lexsort((cls_idx, key))]
        chosen.append(cls_sorted[:quota])
    return np.sort(np.concatenate(chosen))


def _apply_subsample(texts, labels, groups, max_samples, mode, seed, keep_all_negatives=False):
    if max_samples is None or len(texts) <= max_samples:
        return texts, labels, groups
    if mode == "paper":
        keep = uniform_subsample(len(texts), max_samples, seed)
    else:
        keep = stratified_subsample(labels, groups, max_samples, seed, keep_all_negatives=keep_all_negatives)
    return [texts[i] for i in keep], labels[keep], groups[keep]


# ----------------------------------------------------------------------------- probing data
def probing_set(value, split, mode="pythia", max_samples=None, seed=42, split_seed=SPLIT_SEED):
    """Labeled texts (question + answer concatenated) for probing value heads.

    Returns (texts, labels int64 array, groups int64 array); split in {"train", "val"}.
    mode "paper": harmless_pooled allowed, uniform subsampling (repo semantics); mode "pythia": stratified
    group-aware subsampling, all coherence negatives kept. `split_seed` only affects honesty (question split).
    """
    if value not in ALL_VALUES:
        raise ValueError(f"unknown value {value!r}; choose from {ALL_VALUES}")
    if value == "harmless_pooled" and mode != "paper":
        raise ValueError("harmless_pooled (all four HH-RLHF subsets) is only defined in mode='paper'; "
                         "use 'harmless' (harmless-base) or 'helpful' (helpful-base) in mode='pythia'")
    texts, labels, groups = [], [], []
    if value in HH_SUBSETS:
        for i, (chosen, rejected) in enumerate(hh_pairs(value, split)):
            texts += [chosen, rejected]
            labels += [1, 0]
            groups += [i, i]
    elif value == "honesty":
        ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
        train_idxs, val_idxs = truthfulqa_indices(split_seed, len(ds))
        for qi in (train_idxs if split == "train" else val_idxs):
            ex = ds[int(qi)]
            for answer in ex["correct_answers"]:
                texts.append(tqa_text(ex["question"], answer))
                labels.append(1)
                groups.append(int(qi))
            for answer in ex["incorrect_answers"]:
                texts.append(tqa_text(ex["question"], answer))
                labels.append(0)
                groups.append(int(qi))
    elif value in HELPSTEER_VALUES_EXT:
        all_texts, all_labels, all_groups = helpsteer2_rows(split)
        keep = np.flatnonzero(all_labels[value] >= 0)
        texts = [all_texts[i] for i in keep]
        labels = all_labels[value][keep].tolist()
        groups = all_groups[keep].tolist()
    labels, groups = np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64)
    keep_neg = value in ("coherence", "coherence_strict")
    return _apply_subsample(texts, labels, groups, max_samples, mode, seed, keep_all_negatives=keep_neg)


def extract_texts(dataset, split, mode="pythia", max_samples=None, seed=42, split_seed=SPLIT_SEED):
    """Texts + labels for an `extract` dataset: a value (probing_set), 'helpsteer2' (extract-once, labels per
    value) or a prompt set (unlabeled, mode=paper). Returns (texts, labels | {value: labels} | None, groups | None).
    """
    if dataset == "helpsteer2":
        texts, labels, groups = helpsteer2_rows(split)
        if max_samples is not None and len(texts) > max_samples:
            # stratify on the rarest class (coherence negatives = 0, every other row = 1) and keep prompts together
            strat = (labels["coherence"] != 0).astype(np.int64)
            keep = stratified_subsample(strat, groups, max_samples, seed, keep_all_negatives=True)
            texts = [texts[i] for i in keep]
            labels = {v: l[keep] for v, l in labels.items()}
            groups = groups[keep]
        return texts, labels, groups
    if dataset in PROMPT_SETS:
        n = max_samples or 300
        return prompt_set(dataset, n), None, None
    return probing_set(dataset, split, mode, max_samples, seed, split_seed)


# ----------------------------------------------------------------------------- likelihood evaluation pairs
def eval_pairs(metric, value=None, min_delta=1, max_items=None, seed=42, split_seed=SPLIT_SEED, split="val"):
    """Items for the judge-free pairwise likelihood metrics (DESIGN 4.1).

    Each item is {"group": int, "context": str, "pos": [continuation, ...], "neg": [continuation, ...], "meta": {}}
    with context + continuation == the probing text (continuations carry the leading space).
      tqa_pairwise        164 val questions (seed 42); pos = correct answers, neg = incorrect answers
      hh_pairwise         value harmless|helpful (|harmless_pooled); test split single-turn pairs; context = dialogue
                          up to and including the last "\\n\\nAssistant:"
      helpsteer2_pairwise value in HELPSTEER_VALUES(_EXT); validation prompts with two responses whose ratings
                          differ by >= min_delta (coherence_strict forces min_delta >= 2); pos = higher rated
    `max_items`: seeded subsample of items (order preserved).
    """
    items = []
    if metric == "tqa_pairwise":
        ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
        train_idxs, val_idxs = truthfulqa_indices(split_seed, len(ds))
        for qi in (train_idxs if split == "train" else val_idxs):
            ex = ds[int(qi)]
            items.append({"group": int(qi), "context": f"Question: {ex['question']} Answer:",
                          "pos": [" " + a for a in ex["correct_answers"]], "neg": [" " + a for a in ex["incorrect_answers"]],
                          "meta": {"question": ex["question"], "category": ex["category"]}})
    elif metric == "hh_pairwise":
        value = value or "harmless"
        for i, (chosen, rejected) in enumerate(hh_pairs(value, split)):
            cut = chosen.rfind("\n\nAssistant:")
            if cut < 0:
                continue
            ctx = chosen[:cut + len("\n\nAssistant:")]
            if not rejected.startswith(ctx):
                continue
            items.append({"group": i, "context": ctx, "pos": [chosen[len(ctx):]], "neg": [rejected[len(ctx):]],
                          "meta": {"value": value}})
    elif metric == "helpsteer2_pairwise":
        if value not in HELPSTEER_VALUES_EXT:
            raise ValueError(f"helpsteer2_pairwise needs a HelpSteer2 value, got {value!r}")
        if value == "coherence_strict":
            min_delta = max(min_delta, 2)
        col = helpsteer_rating_column(value)
        ds = load_helpsteer(split)
        prompts, responses, ratings = ds["prompt"], ds["response"], ds[col]
        groups = _group_ids(prompts)
        by_group = {}
        for i, g in enumerate(groups):
            by_group.setdefault(int(g), []).append(i)
        for g, rows in by_group.items():
            if len(rows) != 2:
                continue
            a, b = rows
            if abs(ratings[a] - ratings[b]) < min_delta:
                continue
            hi, lo = (a, b) if ratings[a] > ratings[b] else (b, a)
            items.append({"group": g, "context": f"Question: {prompts[hi]}\nAnswer:",
                          "pos": [" " + responses[hi]], "neg": [" " + responses[lo]],
                          "meta": {"value": value, "rating_hi": int(ratings[hi]), "rating_lo": int(ratings[lo]),
                                   "delta": int(abs(ratings[a] - ratings[b]))}})
    else:
        raise ValueError(f"unknown metric {metric!r}; choose from {EVAL_PAIR_METRICS}")
    if max_items is not None and len(items) > max_items:
        keep = np.sort(np.random.RandomState(seed).choice(len(items), max_items, replace=False))
        items = [items[i] for i in keep]
    return items


# ----------------------------------------------------------------------------- generation evaluation data
def eval_set(name, num_eval=None):
    """Questions of an evaluation set (see EVAL_SETS); identical to the reference repo."""
    if name == "jailbreak":
        questions = pd.read_csv(os.path.join(DATA_DIR, "jailbreakbench.csv"))["input"].tolist()
    elif name == "malicious":
        questions = pd.read_csv(os.path.join(DATA_DIR, "maliciousinstruct.csv"))["input"].tolist()
    elif name == "honesty":
        questions = [ex["question"] for ex in truthfulqa_split("val")]
    elif name in HELPSTEER_VALUES:
        # validation prompts that carry a binarized label, consecutive duplicates removed
        questions = []
        for ex in load_helpsteer("val"):
            if questions and ex["prompt"] == questions[-1]:
                continue
            if helpsteer_label(name, ex[name]) is not None:
                questions.append(ex["prompt"])
    elif name == "hh-rlhf":
        # single-turn test dialogues of the pooled config, skipping the first 100 raw entries
        ds = load_hhrlhf("test")
        questions = []
        for i in range(100, len(ds)):
            if single_turn(ds[i]):
                q, _ = first_qa(ds[i]["chosen"])
                if q is not None:
                    questions.append(q)
            if len(questions) == (num_eval or 100):
                break
    elif name == "beavertails":
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
        questions = [ds[i]["prompt"] for i in range(num_eval or 100)]
    else:
        raise ValueError(f"Unknown eval set: {name}")
    return questions[:num_eval] if num_eval else questions


def prompt_set(name, num_prompts=300):
    """Unlabeled raw prompts of an evaluation set, used to estimate the activation std along a direction."""
    if name == "beavertails":
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
        return [ds[i]["prompt"] for i in range(min(num_prompts, len(ds)))]
    if name in ("jailbreak", "malicious"):
        return eval_set(name)[:num_prompts]
    raise ValueError(f"Unknown prompt set: {name}")


# ----------------------------------------------------------------------------- WikiText-2
def wikitext(tokenizer=None, max_tokens=None):
    """WikiText-2 (raw, test) as one string ("\\n\\n"-joined lines), or token ids when a tokenizer is given.

    With `tokenizer`, returns a 1-D LongTensor of the first `max_tokens` ids (all 288,730 Pythia tokens if
    None). Without a tokenizer, `max_tokens` must be None.
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    if tokenizer is None:
        if max_tokens is not None:
            raise ValueError("max_tokens needs a tokenizer")
        return text
    import torch
    ids = torch.tensor(tokenizer(text)["input_ids"], dtype=torch.long)
    return ids[:max_tokens] if max_tokens else ids


# ----------------------------------------------------------------------------- misc
def texts_fingerprint(texts):
    """sha256 over the texts (used to validate resumed extractions)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()
