"""Portions adapted from andrewgcodes/in-context-learning-circuit @ 26c44caf8f9b (MIT License, Copyright (c) 2026 Andrew
Gao, William Gao; see LICENSE).

ICL label-retrieval battery and query-position knockout: the positive control and reverse test M3 of the causal ablation
test (docs/ABLATION_DESIGN.md, part C, section 7). The battery and the protocol are those of the D188 head census of
Gao & Gao, "A Circuit for In-Context Learning" (arXiv 2611.05344): concept12n prompts, every head of a set mean-ablated
at the query arrow only, with the per-kind query mean as the reference, and the label-retrieval margin
log p(target) - log p(alt).

Contents
--------
* Battery: `load_concepts`, `concept12n`, `battery_fingerprint`, `ICLItem`.
* Knockout and scoring: `QueryKnock` (one position per row, per-row reference), `query_means` (per-kind query means from
  a clean prompt-only pass), `score_items` (batched, right padded, logits only at the scored positions).
* Statistics helpers: `summarize` (intact margins per label kind, R / P on nat-flip pairs), `effects` (per-item dret and
  per-pair dR / dP, clusters = (concept, seq)), `census_ranked` (CR20 / CB24 from a census), `calibration_check`
  (against data/D188_summary.json), `CALIBRATION`, `POSITIVE_CONTROL_TARGETS`.

Vendored from the ICL repository (logic verbatim; ABLATION_DESIGN 7.1)
--------------------------------------------------------------------
* icl/tokspans.py:2-20 `is_bare_ws`, `label_spans`, `target_ids`. The `assert`s became explicit `ValueError`s (same
  conditions; they must also hold under `python -O`).
* icl/tasks.py:24-31 `build`, `mk`. `build` returns the prompt ids as a list instead of a tensor; `mk` returns an
  `ICLItem` and computes target / alt ids once (icl/run.py computes them again at scoring time with the same function).
* icl/tasks.py:279 `NATLAB`.
* The concept-battery loop of icl/tasks.py:294-322 (`fam_concept`) with the `fam_concept12n` settings of :280-287
  (CONCEPT_PER = 6, kind suffix "_12n", concepts {topic, lang, question, implied}, label sets sym, rare, nat, flip).
  `random.seed(13 + BATTERY_SEED)` became the local `random.Random(13 + seed)` with the identical call order (per seq up
  to 50 x (sample A, sample B, token-length check), then `random()`, then `shuffle(demos)`); concepts are iterated in
  JSON order and skipped concepts consume no randomness. Dropped: `add_twins` (no RNG calls; the twins are not used
  here), `SENTPOOL`, the `CONCEPT_ORDER` / `CONCEPT_PERM` environment hooks (unset in the census), module globals and the
  import-time JSON lookup.

Re-implemented (not copied): the query-position knockout of icl/hooks.py (`KNOCK`, `KNOCK_POS = [T - 1]`, `MEANO` per kind
from icl/run.py:513-533) as `QueryKnock` on top of `hooks.HeadHooks`, batched; the margin scoring of icl/score.py
`seq_logprob` and icl/run.py:657-660 (single-token alt read at T - 1 of the target row, multi-token alt from its own
teacher-forced row) as `score_items`.

Protocol (ABLATION_DESIGN 7.3): the knock replaces the o_proj-input slice of every set head at T - 1 = the query arrow
(the last prompt token) only; label slots and all other positions stay intact. The reference is the mean over the
prompts of the same kind (same concept and label set, the item itself included) of that slice at T - 1 in a clean
prompt-only pass. Because attention is causal, position T - 1 does not depend on the appended target / alt tokens, so
the prompt-only mean is exact for every scored row.

Validation (recorded 2026-11-05, CPU fp32 + eager, Qwen/Qwen3-0.6B-Base, batch_tokens 2048 / max_batch 16)
-----------------------------------------------------------------------------------------------------
* The vendored builder reproduces the original `icl/tasks.py build_battery(tok, ("concept12n",), smoke)` byte for byte
  (fingerprints for seed 1, seed 2 and the seed-2 smoke battery; pinned in tests/test_icl_battery.py).
* Calibration on the census draw (seed 1): symrare 0.4601, nat 2.1563, flip -0.0241, R 1.0661, P 1.0902; differences
  to data/D188_summary.json -0.0107 / -0.0195 / +0.0098 / -0.0049 / -0.0146 nats (all within 0.02; the census ran bf16).
* Fresh draw (seed 2), 0.6B CR20 knocked jointly at the query with the per-kind mean: intact sym+rare margin 0.161,
  dret -0.130, dR -0.632, dP -0.747 (no random-set comparison; that is the gate of the `ablate` command).

Deviations from the spec (docs/ABLATION_DESIGN.md section 7.2), reported to the orchestrator
------------------------------------------------------------------------------------------
* `concept12n` takes an extra keyword `n_seq` (draws per concept; default 24, smoke 1). With smoke=True every kind holds a
  single prompt, so the per-kind mean equals the item's own query value and every knock is an exact identity; a smoke run
  that should exercise the knock needs n_seq >= 2 (32 prompts, 2 per kind). n_seq != 24 is not a prefix of the full
  battery (the RNG stream of a concept starts after the previous concept's draws).
* `score_items` takes an extra keyword `logit_chunk` (2048) and returns, besides the specified keys, "tokens_padded" and
  "n_rows"; `summarize` also returns "n_pairs".
* `calibration_check` returns ok=None (with a "skipped" reason) for a model without a reference in `CALIBRATION`.
* `census_ranked` also accepts role "p_brakes" / "p_carriers" (ranked by dP; p_brakes largest first).
* Measured battery facts (Qwen3 tokenizer; identical to the original builder, see tests/test_icl_battery.py): prompt
  lengths 118-181 tokens and 337 (seed 1) / 340 (seed 2) single-token alts, not the scouting values 115-183 / 341 of
  section 7.5. The difference is in the scouting numbers, not in the builder.
* LICENSE: part C appended the attribution as its own block; part D merged it into the icl_sets block as section 8.10
  describes ("Attribution: in-context-learning circuit (head sets, battery)").
"""
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import icl_sets
from .extract import plan_batches
from .hooks import ActivationCollector, HeadHooks
from .models import model_geometry

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONCEPT_TASKS_PATH = os.path.join(DATA_DIR, "icl", "concept_tasks.json")
CONCEPT_TASKS_SHA256 = "bf242720283bdc9967ffdb22da277f2fc0237d374de2b8214c45790187e60df0"
CONCEPT_TASKS_REPO_PATH = os.path.join("batteries", "concept_tasks.json")    # inside an ICL repo checkout
CONCEPTS_12N = ("topic", "lang", "question", "implied")
NATLAB = {"topic": ("animal", "food"), "lang": ("English", "Spanish"), "question": ("question", "statement"),
          "implied": ("good", "bad")}                     # icl/tasks.py:279 (D185 natural-word labels: class A, class B)
LABELSETS = ("sym", "rare", "nat", "flip")
SYMBOL_LABELS = {"sym": ("foo", "bar"), "rare": ("zorp", "quil")}
CONCEPT_PER = 6           # demos per class (fam_concept12n)
KIND_SUFFIX = "_12n"
N_SEQ = 24                # draws per concept (fam_concept: range(1 if smoke else 24))
MAX_TRIES = 50            # fam_concept: token-length matching attempts per draw
RNG_OFFSET = 13           # fam_concept: random.seed(13 + BATTERY_SEED)
FRESH_SEED = 2            # ABLATION_DESIGN F21: the fresh draw (random.Random(15))
CENSUS_SEED = 1           # the census draw (random.Random(14)); calibration only

CALIBRATION = {           # data/D188_summary.json @ ICL_COMMIT (intact margins of the census battery, nats)
    "qwen3-4b-base": {"symrare": 1.5991635416666667, "nat": 5.398438541666667, "flip": 0.3502604166666667,
                      "R": 2.8743494791666664, "P": 2.5240890625},
    "qwen3-0.6b-base": {"symrare": 0.4707578125, "nat": 2.17578125, "flip": -0.033854166666666664,
                        "R": 1.0709635416666667, "P": 1.1048177083333333},
}
CALIBRATION_KEYS = ("symrare", "nat", "flip", "R", "P")
CALIBRATION_REFERENCE = "data/D188_summary.json"

# ABLATION_DESIGN Anhang B as data (for the report). value / ci in nats; ci None where the paper gives none; "random" is
# the paper's random-set comparison of the same protocol (value, ci); "id" is the key the report selects by. Checked
# against the paper at ICL_COMMIT 26c44caf (main.tex, figures/fig6_populations.py, data/D188_summary.json).
POSITIVE_CONTROL_TARGETS = [
    {"id": "reader20_4b", "quantity": "reader-20 joint knock, fresh draw", "model": "Qwen3-4B-Base",
     "battery": "concept12 (12 demos, 144 prompts, 6 kinds)", "value": -0.74, "ci": [-0.95, -0.55],
     "random": {"value": 0.02, "ci": [-0.03, 0.06], "what": "20 random heads, same layers"},
     "source": "main.tex:263, :316; tab:claims main.tex:592; figures/fig6_populations.py:4-6"},
    {"id": "random20_4b", "quantity": "20 random heads, same layers, fresh draw", "model": "Qwen3-4B-Base",
     "battery": "concept12", "value": 0.02, "ci": [-0.03, 0.06], "source": "main.tex:263; figures/fig6_populations.py:6"},
    {"id": "intact_4b", "quantity": "intact margin of that battery (sym +0.55 / rare +1.13)", "model": "Qwen3-4B-Base",
     "battery": "concept12", "value": 0.84, "ci": None,
     "source": "figures/fig6_populations.py:4; data/D173Q2b_effects.txt"},
    {"id": "brake24_4b", "quantity": "brake-24 joint knock, fresh draw (24 random: +0.22 [+0.14, +0.31])",
     "model": "Qwen3-4B-Base", "battery": "concept12", "value": 1.39, "ci": [0.88, 1.88],
     "random": {"value": 0.22, "ci": [0.14, 0.31], "what": "24 random heads"},
     "source": "main.tex:263, :593; figures/fig6_populations.py:6"},
    {"id": "selection_readers_4b", "quantity": "selection draw (intact +2.46): readers", "model": "Qwen3-4B-Base",
     "battery": "concept12", "value": -1.96, "ci": [-2.39, -1.53], "source": "figures/fig6_populations.py:4-6"},
    {"id": "selection_brakes_4b", "quantity": "selection draw (intact +2.46): brakes", "model": "Qwen3-4B-Base",
     "battery": "concept12", "value": 4.88, "ci": [3.52, 6.24], "source": "figures/fig6_populations.py:4-6"},
    {"id": "cr20_llama", "quantity": "census-ranked reader-20 (precedent for CR20; random 0.00)", "model": "Llama-3.2-3B",
     "battery": "12 demos, 144 prompts", "value": -0.56, "ci": [-1.01, -0.12],
     "random": {"value": 0.0, "ci": None, "what": "random heads"}, "source": "main.tex:394, :601"},
    {"id": "cb24_llama", "quantity": "census-ranked brake-24 (random +0.12)", "model": "Llama-3.2-3B",
     "battery": "12 demos, 144 prompts", "value": 1.90, "ci": [1.12, 2.67],
     "random": {"value": 0.12, "ci": None, "what": "random heads"}, "source": "main.tex:394, :601"},
    {"id": "cb24_llama_6demos", "quantity": "census-ranked brake-24, 6 demos (random +0.16)", "model": "Llama-3.2-3B",
     "battery": "6 demos, 432 prompts", "value": 0.57, "ci": [0.19, 0.93],
     "random": {"value": 0.16, "ci": None, "what": "random heads"}, "source": "main.tex:602"},
    {"id": "cr20_llama_6demos", "quantity": "census-ranked reader-20, 6 demos", "model": "Llama-3.2-3B",
     "battery": "6 demos, 432 prompts", "value": -0.06, "ci": [-0.32, 0.21], "source": "main.tex:602"},
    {"id": "census_readers_4b", "quantity": "census 4B: 89 readers, summed single-head dret", "model": "Qwen3-4B-Base",
     "battery": "concept12n (384 prompts), single heads at the query, per-kind mean", "value": -4.69, "ci": None,
     "source": "main.tex:460, :475, :480-485; data/D188_summary.json"},
    {"id": "census_brakes_4b", "quantity": "census 4B: 56 brakes, summed single-head dret", "model": "Qwen3-4B-Base",
     "battery": "concept12n", "value": 3.76, "ci": None, "source": "data/D188_summary.json"},
    {"id": "census_readers_06b", "quantity": "census 0.6B: 70 readers, summed single-head dret", "model": "Qwen3-0.6B-Base",
     "battery": "concept12n", "value": -2.08, "ci": None, "source": "data/D188_summary.json; main.tex:480-485"},
    {"id": "cr20_sum_4b", "quantity": "CR20, summed single-head dret (selection draw, seed 1)", "model": "Qwen3-4B-Base",
     "battery": "concept12n", "value": -2.547, "ci": None,
     "source": "data/D188_heads_Qwen3-4B-Base_lbD188SC4.json via icl_sets.census_sets"},
    {"id": "instruct_removal", "quantity": "instruct transfer, 189-head removal, symbol / rare labels (all heads removed: "
     "-0.60 / -1.00)", "model": "Qwen3-4B", "battery": "concept12 symbol / rare battery (12 demos, 144 prompts)",
     "value": -0.61, "values": {"symbol": -0.61, "rare": -1.00}, "ci": None,
     "random": None, "all_heads": {"symbol": -0.60, "rare": -1.00}, "source": "main.tex:470, :605"},
]


# ----------------------------------------------------------------------------------------------------------------
# vendored helpers (icl/tokspans.py:2-20)
# ----------------------------------------------------------------------------------------------------------------

def is_bare_ws(tok, tid):
    """True if token `tid` decodes to non-empty whitespace only (icl/tokspans.py:2)."""
    d = tok.decode([tid])
    return d != "" and d.strip() == ""


def label_spans(tok, ids, pairs, labpos):
    """(s0, s1, label) per demo from the FULL prompt tokenization `ids`; skips a leading bare-whitespace token (Qwen:
    ' 7' -> [' ', '7']) so the span starts on content; checks that the decoded span equals the label (icl/tokspans.py:3-10)."""
    out = []
    for (x, y), lp_ in zip(pairs, labpos):
        n = len(tok(" " + y)["input_ids"])
        s0, s1 = lp_, lp_ + n
        while s1 - s0 > 1 and is_bare_ws(tok, ids[s0]):
            s0 += 1
        dec = tok.decode(ids[s0:s1])
        if not (dec.strip() == y.strip() and dec.strip() != ""):
            raise ValueError(f"span mismatch: label span {s0}:{s1} does not decode to the label (ids {ids[s0:s1]})")
        out.append((s0, s1, y))
    return out


def target_ids(tok, ids, target):
    """Token ids of ' ' + target as the model sees them AFTER the prompt `ids` (full-text tokenization, suffix), falling
    back to standalone tokenization when the prompt does not re-tokenize to itself. Returns (ids, used_fallback)
    (icl/tokspans.py:11-20)."""
    ptxt = tok.decode(ids)
    full = tok(ptxt + " " + target, add_special_tokens=False)["input_ids"]
    ids = list(ids)
    bos = tok.bos_token_id
    if bos is not None and ids and ids[0] == bos and (not full or full[0] != bos):
        ids = ids[1:]     # Llama-style tokenizers prepend BOS to the prompt; decode() may drop it
    if full[:len(ids)] == ids:
        tgt, fb = full[len(ids):], False
    else:
        tgt, fb = tok(" " + target, add_special_tokens=False)["input_ids"], True
    dec = tok.decode(tgt)
    if not (dec.strip() == target.strip() and dec.strip() != "" and len(tgt) >= 1):
        raise ValueError(f"target mismatch: {len(tgt)} target tokens do not decode to the target")
    return list(tgt), fb


# ----------------------------------------------------------------------------------------------------------------
# battery (icl/tasks.py:24-31, :279-322)
# ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ICLItem:
    """One concept12n prompt. `ids` ends with the query arrow ' ->' (query position T - 1 = len(ids) - 1)."""
    index: int              # position in the battery
    kind: str               # icl_concept_{concept}_{labelset}_12n
    concept: str
    labelset: str
    seq: int                # draw index within the concept
    qpos: bool              # query belongs to class A ("P")
    ids: tuple              # prompt ids
    target: str
    alt: str
    target_ids: tuple       # target_ids(tok, ids, target)[0]
    alt_ids: tuple          # target_ids(tok, ids, alt)[0]
    cluster: int            # concept_index * 100 + seq (the four label sets of one draw share a cluster)


def build(tok, pairs, query):
    """Prompt ids of `x -> y\\n` demo lines plus the query line `query ->`; returns (ids, label positions, T - 1)
    (icl/tasks.py:24-28; ids as a list instead of a tensor)."""
    text = ""
    labpos = []
    for x, y in pairs:
        pre = tok(text + x + " ->")["input_ids"]
        text = text + x + " -> " + y + "\n"
        labpos.append(len(pre))
    q = tok(text + query + " ->")["input_ids"]
    return list(q), labpos, len(q) - 1


def mk(tok, index, kind, pairs, query, target, alt, *, concept, labelset, seq, qpos, cluster):
    """Builds one item (icl/tasks.py:30-32): prompt, label-span check, target / alt ids after the prompt."""
    ids, labpos, _ = build(tok, pairs, query)
    label_spans(tok, ids, pairs, labpos)          # the labels must be recoverable from the full tokenization
    tgt, _ = target_ids(tok, ids, target)         # assert the answer is scorable now, not at run time
    alt_t, _ = target_ids(tok, ids, alt)
    return ICLItem(int(index), kind, concept, labelset, int(seq), bool(qpos), tuple(int(i) for i in ids), target, alt,
                   tuple(int(i) for i in tgt), tuple(int(i) for i in alt_t), int(cluster))


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _validate_concepts(concepts):
    if not isinstance(concepts, dict):
        raise ValueError("concept tasks must be a JSON object {concept: [class-A sentences, class-B sentences]}")
    out = {}
    for name, value in concepts.items():
        if not isinstance(name, str) or not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"concept {name!r}: expected a pair [class-A sentences, class-B sentences]")
        a, b = value
        for cls in (a, b):
            if not isinstance(cls, (list, tuple)) or not all(isinstance(s, str) and s for s in cls):
                raise ValueError(f"concept {name!r}: every class must be a list of non-empty strings")
        out[name] = (list(a), list(b))
    for name in CONCEPTS_12N:
        if name not in out:
            raise ValueError(f"concept tasks lack the concept12n concept {name!r}")
        if min(len(out[name][0]), len(out[name][1])) < CONCEPT_PER + 1:
            raise ValueError(f"concept {name!r}: every class needs at least {CONCEPT_PER + 1} sentences")
    return out


def load_concepts(path=None, verify=True):
    """{concept: (class-A sentences, class-B sentences)} in file order.

    path: None (data/icl/concept_tasks.json), the JSON file, or a checkout of the ICL repo (batteries/concept_tasks.json).
    verify: the bytes must match CONCEPT_TASKS_SHA256 (else ValueError)."""
    path = CONCEPT_TASKS_PATH if path is None else str(path)
    if os.path.isdir(path):
        path = os.path.join(path, CONCEPT_TASKS_REPO_PATH)
    with open(path, "rb") as f:
        data = f.read()
    digest = _sha256(data)
    if verify and digest != CONCEPT_TASKS_SHA256:
        raise ValueError(f"{path}: sha256 {digest} != pinned {CONCEPT_TASKS_SHA256} (batteries/concept_tasks.json @ "
                         f"{icl_sets.ICL_COMMIT[:12]})")
    return _validate_concepts(json.loads(data.decode("utf-8")))


def concept12n(tok, seed=FRESH_SEED, smoke=False, concepts=None, *, n_seq=None):
    """The concept12n battery of the D188 census (icl/tasks.py fam_concept12n): for each of topic, lang, question,
    implied and each of `n_seq` draws (24; smoke 1), 6 + 6 demos and a query, under the four label sets sym (foo/bar),
    rare (zorp/quil), nat (the concept's own words) and flip (the same words with exchanged roles). The four label sets of
    one draw share sentences, demo order and query. Full battery: 4 x 24 x 4 = 384 items, 16 kinds of 24.

    seed: the BATTERY_SEED of the original (RNG = random.Random(13 + seed)); FRESH_SEED = 2 is the fresh draw,
    CENSUS_SEED = 1 the census draw. concepts: None (load_concepts()), a path, or an already loaded dict.
    n_seq: draws per concept (extension, see the module docstring)."""
    if concepts is None:
        concepts = load_concepts()
    elif isinstance(concepts, (str, os.PathLike)):
        concepts = load_concepts(concepts)
    else:
        concepts = _validate_concepts(concepts)
    n_seq = (1 if smoke else N_SEQ) if n_seq is None else int(n_seq)
    if not 1 <= n_seq < 100:
        raise ValueError(f"n_seq must be in [1, 99] (cluster = concept_index * 100 + seq), got {n_seq}")
    rng = random.Random(RNG_OFFSET + int(seed))
    out = []
    for cname, (A, Bs) in concepts.items():
        if cname not in CONCEPTS_12N:
            continue                                   # skipped concepts consume no randomness
        cidx = CONCEPTS_12N.index(cname)
        LS = {"sym": SYMBOL_LABELS["sym"], "rare": SYMBOL_LABELS["rare"], "nat": NATLAB[cname],
              "flip": NATLAB[cname][::-1]}
        for s_ in range(n_seq):
            for _try in range(MAX_TRIES):
                ps = rng.sample(A, CONCEPT_PER + 1)
                ns = rng.sample(Bs, CONCEPT_PER + 1)
                la = {len(tok(x)["input_ids"]) for x in ps[:3]}
                lb = {len(tok(x)["input_ids"]) for x in ns[:3]}
                if la & lb:
                    break
            qpos = rng.random() < 0.5
            demos = [(x, "P") for x in ps[:CONCEPT_PER]] + [(x, "N") for x in ns[:CONCEPT_PER]]
            rng.shuffle(demos)
            q = ps[CONCEPT_PER] if qpos else ns[CONCEPT_PER]
            for name in LABELSETS:
                lp, ln = LS[name]
                dd = [(x, lp if y == "P" else ln) for x, y in demos]
                tgt = lp if qpos else ln
                out.append(mk(tok, len(out), f"icl_concept_{cname}_{name}" + KIND_SUFFIX, dd, q, tgt,
                              ln if qpos else lp, concept=cname, labelset=name, seq=s_, qpos=qpos,
                              cluster=cidx * 100 + s_))
    if not out:
        raise ValueError("concept12n produced no items")
    return out


def battery_fingerprint(items):
    """sha256 over (kind, ids, target_ids, alt_ids) of all items, in order (canonical compact JSON)."""
    rows = [[it.kind, [int(i) for i in it.ids], [int(i) for i in it.target_ids], [int(i) for i in it.alt_ids]]
            for it in items]
    return _sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8"))


def kinds_of(items):
    """Kinds in first-appearance order."""
    return list(dict.fromkeys(it.kind for it in items))


# ----------------------------------------------------------------------------------------------------------------
# knockout and scoring
# ----------------------------------------------------------------------------------------------------------------

def heads_by_layer(heads, n_layers, n_heads):
    """{layer: sorted unique heads} (non-empty layers only) from an iterable of (layer, head) or a {layer: [heads]}."""
    if heads is None:
        return {}
    pairs = ([(l, h) for l, hs in heads.items() for h in hs] if isinstance(heads, dict)
             else [tuple(lh) for lh in heads])
    out = {}
    for lh in pairs:
        if len(lh) != 2:
            raise ValueError(f"head {lh!r} is not a (layer, head) pair")
        l, h = int(lh[0]), int(lh[1])
        if not (0 <= l < n_layers and 0 <= h < n_heads):
            raise ValueError(f"head {l}:{h} outside {n_layers} layers x {n_heads} heads")
        out.setdefault(l, set()).add(h)
    return {l: sorted(out[l]) for l in sorted(out)}


class QueryKnock(HeadHooks):
    """Replaces the set heads' o_proj-input slices at one position per row (the query arrow) with a per-row reference.

    heads: iterable of (layer, head) or {layer: [heads]}; an empty set hooks no layer (identity). Before every forward call
    `set_rows(query_index, refs)`: query_index (B,) long = the edited position of each row; refs {layer: (B, n_l, D)} with
    n_l = len(self.heads[layer]) in ascending head order. With grouped-query attention every query head owns its slice:
    knocking head h never touches the other heads of its KV group."""

    def __init__(self, model, heads):
        n_layers, n_heads, _ = model_geometry(model.config)
        self.heads = heads_by_layer(heads, n_layers, n_heads)
        super().__init__(model, positions="all", decode_only=False, layers=list(self.heads))
        self._index = {l: torch.tensor(hs, dtype=torch.long) for l, hs in self.heads.items()}
        self.query_index = None
        self.refs = None
        self._qmax = None

    def set_rows(self, query_index, refs):
        q = torch.as_tensor(query_index, dtype=torch.long).reshape(-1)
        B = q.numel()
        if B and int(q.min()) < 0:
            raise ValueError("query_index must be >= 0")
        out = {}
        for layer, hs in self.heads.items():
            if layer not in refs:
                raise ValueError(f"refs lack layer {layer}")
            r = torch.as_tensor(refs[layer])
            if tuple(r.shape) != (B, len(hs), self.head_dim):
                raise ValueError(f"refs[{layer}] has shape {tuple(r.shape)}, expected {(B, len(hs), self.head_dim)}")
            out[layer] = r
        self.query_index, self.refs = q, out
        self._qmax = int(q.max()) if B else -1
        return self

    def clear_rows(self):
        self.query_index = self.refs = self._qmax = None

    def edit(self, layer, x):
        idx = self._index.get(layer)
        if idx is None:
            return x
        if self.query_index is None:
            raise RuntimeError("QueryKnock: call set_rows(query_index, refs) before the forward")
        B, S, _ = x.shape
        if self.query_index.numel() != B or self._qmax >= S:
            raise ValueError(f"query_index (n={self.query_index.numel()}, max {self._qmax}) does not fit a batch of "
                             f"shape {tuple(x.shape)}")
        x = x.clone()
        view = x.view(B, S, self.num_heads, self.head_dim)
        rows = torch.arange(B, device=x.device)[:, None]
        cols = self.query_index.to(x.device)[:, None]
        view[rows, cols, idx.to(x.device)[None, :]] = self.refs[layer].to(x.device, x.dtype)
        return x


def _decoder(model):
    return model.get_decoder() if hasattr(model, "get_decoder") else model


def _pad(seqs, idx, pad_id):
    width = max(len(seqs[i]) for i in idx)
    input_ids = torch.full((len(idx), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(idx), width), dtype=torch.long)
    for r, i in enumerate(idx):
        input_ids[r, :len(seqs[i])] = torch.tensor(seqs[i], dtype=torch.long)
        mask[r, :len(seqs[i])] = 1
    return input_ids, mask


@torch.no_grad()
def query_means(model, items, pad_id, *, batch_tokens=8192, max_batch=64):
    """{kind: float32 (L, H*D)}: mean over the items of that kind of the o_proj input at T - 1 from a clean prompt-only
    pass (hooks.ActivationCollector on the decoder, right padding; the item itself is included, icl/run.py:513-521).
    The caller must not have intervention hooks installed."""
    if not items:
        raise ValueError("query_means needs at least one item")
    n_layers, n_heads, head_dim = model_geometry(model.config)
    seqs = [list(it.ids) for it in items]
    lengths = [len(s) for s in seqs]
    sums = {k: np.zeros((n_layers, n_heads * head_dim), np.float64) for k in kinds_of(items)}
    counts = dict.fromkeys(sums, 0)
    was_training = model.training
    model.eval()
    try:
        collector = ActivationCollector(_decoder(model))
        with collector:
            for idx in plan_batches(lengths, max_batch, batch_tokens):
                input_ids, mask = _pad(seqs, idx, pad_id)
                states = collector.collect(input_ids, mask).numpy()       # (B, L, H*D) float32
                for r, i in enumerate(idx):
                    sums[items[i].kind] += states[r]
                    counts[items[i].kind] += 1
    finally:
        model.train(was_training)
    return {k: (sums[k] / counts[k]).astype(np.float32) for k in sums}


@torch.no_grad()
def score_items(model, items, pad_id, *, heads=None, kind_means=None, batch_tokens=8192, max_batch=64,
                logit_chunk=2048):
    """Label-retrieval margins of the battery, optionally under a query knock.

    Rows: prompt + target_ids for every item; prompt + alt_ids only when len(alt_ids) > 1 (a single-token alt is read
    as log p(alt) at T - 1 of the target row, icl/run.py:657-660). With `heads` (non-empty) the knock is active in every
    row at T - 1 with the row's kind mean from `kind_means` ({kind: (L, H*D)}, see `query_means`). Logits are computed
    only at the scored positions (lm_head in chunks of `logit_chunk`, fp32 log-softmax). Batches follow
    extract.plan_batches over the row lengths (right padding).

    Returns {"tlp", "alt", "margin": float64 (N,), "tokens_processed", "tokens_padded", "n_rows", "seconds"} where
    tlp = sum log p(target tokens), alt = sum log p(alt tokens), margin = tlp - alt."""
    t0 = time.time()
    n_layers, n_heads, head_dim = model_geometry(model.config)
    N = len(items)
    rows = []                 # (item position, kind, ids, T, continuation, single-token alt id or -1, is_alt_row)
    for n, it in enumerate(items):
        T = len(it.ids)
        if T < 1 or not it.target_ids or not it.alt_ids:
            raise ValueError(f"item {it.index}: empty prompt, target or alt")
        a1 = int(it.alt_ids[0]) if len(it.alt_ids) == 1 else -1
        rows.append((n, it.kind, list(it.ids) + list(it.target_ids), T, list(it.target_ids), a1, False))
        if len(it.alt_ids) > 1:
            rows.append((n, it.kind, list(it.ids) + list(it.alt_ids), T, list(it.alt_ids), -1, True))
    seqs = [r[2] for r in rows]
    lengths = [len(s) for s in seqs]

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hb = heads_by_layer(heads, n_layers, n_heads)
    knock, tables, kind_index = None, {}, {}
    if hb:
        if kind_means is None:
            raise ValueError("a knock needs kind_means (query_means of the battery)")
        kinds = kinds_of(items)
        missing = [k for k in kinds if k not in kind_means]
        if missing:
            raise ValueError(f"kind_means lack the kinds {missing}")
        kind_index = {k: i for i, k in enumerate(kinds)}
        arrs = {k: np.asarray(kind_means[k], np.float32).reshape(n_layers, n_heads, head_dim) for k in kinds}
        for layer, hs in hb.items():
            tables[layer] = torch.from_numpy(np.stack([arrs[k][layer][hs] for k in kinds])).to(device, dtype)
        knock = QueryKnock(model, hb)

    tlp = np.zeros(N, np.float64)
    alt = np.zeros(N, np.float64)
    tokens_processed = tokens_padded = 0
    decoder = _decoder(model)
    lm_head = model.get_output_embeddings()
    was_training = model.training
    model.eval()
    try:
        if knock is not None:
            knock.install()
        for idx in plan_batches(lengths, max_batch, batch_tokens):
            input_ids, mask = _pad(seqs, idx, pad_id)
            tokens_processed += int(mask.sum())
            tokens_padded += int(mask.numel())
            if knock is not None:
                q = torch.tensor([rows[i][3] - 1 for i in idx], dtype=torch.long)
                kidx = torch.tensor([kind_index[rows[i][1]] for i in idx], dtype=torch.long, device=device)
                knock.set_rows(q, {layer: tab[kidx] for layer, tab in tables.items()})
            hidden = decoder(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                             use_cache=False).last_hidden_state
            r_idx, p_idx, toks, owner, is_alt, alt_tok = [], [], [], [], [], []
            for r, i in enumerate(idx):
                n, _, _, T, cont, a1, alt_row = rows[i]
                for j, t in enumerate(cont):
                    r_idx.append(r)
                    p_idx.append(T - 1 + j)
                    toks.append(t)
                    owner.append(n)
                    is_alt.append(alt_row)
                    alt_tok.append(a1 if j == 0 else -1)
            h = hidden[torch.tensor(r_idx, device=device), torch.tensor(p_idx, device=device)]
            toks_t = torch.tensor(toks, dtype=torch.long, device=device)
            alt_tok_t = torch.tensor(alt_tok, dtype=torch.long, device=device)
            vals = np.empty(len(toks), np.float64)
            alt_vals = np.full(len(toks), np.nan, np.float64)
            for s in range(0, len(toks), logit_chunk):
                e = min(s + logit_chunk, len(toks))
                lsm = torch.log_softmax(lm_head(h[s:e]).float(), dim=-1)
                vals[s:e] = lsm.gather(1, toks_t[s:e, None])[:, 0].double().cpu().numpy()
                sel = alt_tok_t[s:e] >= 0
                if bool(sel.any()):
                    pos = torch.nonzero(sel)[:, 0]
                    got = lsm[pos, alt_tok_t[s:e][pos]].double().cpu().numpy()
                    alt_vals[s + pos.cpu().numpy()] = got
            owner = np.asarray(owner, np.int64)
            is_alt = np.asarray(is_alt, bool)
            np.add.at(tlp, owner[~is_alt], vals[~is_alt])
            np.add.at(alt, owner[is_alt], vals[is_alt])
            single = ~np.isnan(alt_vals)
            alt[owner[single]] = alt_vals[single]          # exactly one T - 1 position per single-token-alt item
            if knock is not None:
                knock.clear_rows()
    finally:
        if knock is not None:
            knock.remove()
        model.train(was_training)
    return {"tlp": tlp, "alt": alt, "margin": tlp - alt, "tokens_processed": tokens_processed,
            "tokens_padded": tokens_padded, "n_rows": len(rows), "seconds": time.time() - t0}


# ----------------------------------------------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------------------------------------------

def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def _pairs(items):
    """[(cluster, nat position, flip position)] per (concept, seq), matched by meta (not by order), sorted by cluster."""
    by = {}
    for n, it in enumerate(items):
        key = (it.concept, it.seq, it.labelset)
        if key in by:
            raise ValueError(f"duplicate item for (concept, seq, labelset) = {key}")
        by[key] = n
    out = []
    for (concept, seq, labelset), n in by.items():
        if labelset == "nat" and (concept, seq, "flip") in by:
            out.append((items[n].cluster, n, by[(concept, seq, "flip")]))
    return sorted(out)


def summarize(items, margin):
    """Mean margins: per label set, "symrare" (sym + rare items), "R" = mean (nat + flip) / 2 and "P" = mean
    (nat - flip) / 2 over the nat/flip pairs of one draw (analyze_D188_scale.py), and "n" = number of items."""
    m = np.asarray(margin, np.float64).reshape(-1)
    if m.size != len(items):
        raise ValueError(f"{m.size} margins for {len(items)} items")
    out = {ls: _mean(m[n] for n, it in enumerate(items) if it.labelset == ls) for ls in LABELSETS}
    out["symrare"] = _mean(m[n] for n, it in enumerate(items) if it.labelset in ("sym", "rare"))
    pairs = _pairs(items)
    out["R"] = _mean((m[a] + m[b]) / 2 for _, a, b in pairs)
    out["P"] = _mean((m[a] - m[b]) / 2 for _, a, b in pairs)
    out["n"] = len(items)
    out["n_pairs"] = len(pairs)
    return out


def effects(items, m_clean, m_cond):
    """Paired knock effects, condition - clean, on the same items:
    "dret": (per sym + rare item delta, clusters) in item order;
    "dR" / "dP": (per (concept, seq) pair delta of (nat + flip) / 2 / (nat - flip) / 2, clusters), sorted by cluster.
    nat and flip are paired by explicit meta (concept, seq), not by order. Clusters = concept_index * 100 + seq."""
    a = np.asarray(m_clean, np.float64).reshape(-1)
    b = np.asarray(m_cond, np.float64).reshape(-1)
    if a.size != len(items) or b.size != len(items):
        raise ValueError(f"margins ({a.size}, {b.size}) do not match {len(items)} items")
    d = b - a
    ret = [n for n, it in enumerate(items) if it.labelset in ("sym", "rare")]
    pairs = _pairs(items)
    cl = np.asarray([c for c, _, _ in pairs], np.int64)
    return {
        "dret": (d[ret], np.asarray([items[n].cluster for n in ret], np.int64)),
        "dR": (np.asarray([(d[x] + d[y]) / 2 for _, x, y in pairs], np.float64), cl),
        "dP": (np.asarray([(d[x] - d[y]) / 2 for _, x, y in pairs], np.float64), cl.copy()),
    }


def census_ranked(census, role="readers", k=20):
    """[(layer, head)] of the census role ranked by its single-head effect (ABLATION_DESIGN F22): readers by
    (dret, l, h) ascending, brakes by (-dret, l, h) (largest first); p_brakes / p_carriers by dP (p_brakes largest
    first). k=None returns all; fewer members than k return all (0.6B has 13 brakes)."""
    if role not in icl_sets.CENSUS_ROLES:
        raise KeyError(f"unknown census role {role!r} (known: {icl_sets.CENSUS_ROLES})")
    eff = census.effects("dP" if role.startswith("p_") else "dret")
    largest = role in ("brakes", "p_brakes")
    members = [(int(l), int(h)) for l, h in census.roles[role]]
    ranked = sorted(members, key=lambda lh: (-eff[lh] if largest else eff[lh], lh[0], lh[1]))
    return ranked if k is None else ranked[:int(k)]


def _calibration_key(model):
    return str(model).strip().split("/")[-1].lower()


def calibration_check(summary, model, tol=0.15):
    """Compares a seed-1 (census draw) summary with data/D188_summary.json: ok <=> |diff| <= tol for symrare, nat, flip,
    R and P (ABLATION_DESIGN F23a). A model without a reference gives ok=None and a "skipped" reason."""
    key = _calibration_key(model)
    out = {"ok": None, "diffs": {}, "tol": float(tol), "reference": CALIBRATION_REFERENCE, "model": key}
    ref = CALIBRATION.get(key)
    if ref is None:
        out["skipped"] = f"no D188 calibration reference for {key!r} (known: {sorted(CALIBRATION)})"
        return out
    diffs = {k: float(summary[k]) - ref[k] for k in CALIBRATION_KEYS}
    out["diffs"] = diffs
    out["values"] = {k: float(summary[k]) for k in CALIBRATION_KEYS}
    out["expected"] = dict(ref)
    out["ok"] = bool(all(math.isfinite(v) and abs(v) <= tol for v in diffs.values()))
    return out
