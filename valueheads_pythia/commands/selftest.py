"""`selftest`: hook layout, mask positions and data counts (DESIGN 7, notebook cell 4). Exits 1 on failure.

    python -m valueheads_pythia.cli selftest --model pythia-160m --revision step143000 [--data quick|full|none]

The check functions are also what tests/test_hook_layout.py, tests/test_mask_positions.py and
tests/test_data_counts.py assert on.
"""
import sys
import time

import numpy as np
import torch

from .. import config
from ..cli import add_common_args
from ..data import eval_pairs, eval_set, hh_pairs, probing_set, truthfulqa_indices, wikitext
from ..hooks import ActivationCollector, HeadMask, HeadSteer
from ..models import load_model, model_geometry, o_proj, qkv_value_rows

TEXT = "Question: What happens if an earthworm is cut in half? Answer: It becomes two worms."
SHORT = "Hello world"

# Verified counts (DESIGN section 2 table; data track + adversarial re-check).
EXPECTED = {
    "tqa_questions": (653, 164),
    "honesty_texts": {"train": (4712, 2075, 2637), "val": (1206, 525, 681)},
    "helpsteer2": {  # value: {split: (n, pos, neg)}
        "helpfulness": {"train": (12002, 8434, 3568), "val": (603, 432, 171)},
        "coherence": {"train": (14824, 14545, 279), "val": (769, 753, 16)},
        "complexity": {"train": (2627, 2005, 622), "val": (139, 94, 45)},
        "verbosity": {"train": (4696, 3903, 793), "val": (243, 185, 58)},
    },
    "helpsteer2_rows": {"train": 20324, "val": 1038},
    "hh_pairs_test": {"harmless": 662, "helpful": 700, "harmless_pooled": 2612},
    "hh_pairs_train": {"harmless": 12253, "helpful": 12843, "harmless_pooled": 49436},
    "helpsteer2_pairs_ge1": {"helpfulness": 373, "complexity": 143, "verbosity": 254, "coherence": 196},
    "helpsteer2_pairs_ge2": {"helpfulness": 182, "complexity": 16, "verbosity": 83, "coherence": 44},
    "eval_sets": {"jailbreak": 100, "malicious": 100, "honesty": 164, "helpfulness": 426, "complexity": 100,
                  "verbosity": 205, "coherence": 472, "hh-rlhf": 100, "beavertails": 100},
    "wikitext_tokens": 288730,
}


# ----------------------------------------------------------------------------- hook layout
@torch.no_grad()
def check_hook_layout(model, tokenizer, layer=5, head=7):
    """Zeroing the V rows of `head` in `layer` zeroes exactly that head's slice at the o_proj input (others
    unchanged); batched right-padded collection equals per-text collection. Returns a dict with 'ok'."""
    L, H, D = model_geometry(model.config)
    enc = tokenizer(TEXT, return_tensors="pt")
    coll = ActivationCollector(model)
    with coll:
        base = coll.collect(enc.input_ids, enc.attention_mask)[0]            # (L, H*D)
    qkv, rows = qkv_value_rows(model, layer, head)
    w_backup, b_backup = qkv.weight[rows].clone(), qkv.bias[rows].clone()
    qkv.weight[rows] = 0
    qkv.bias[rows] = 0
    try:
        with coll:
            zeroed = coll.collect(enc.input_ids, enc.attention_mask)[0]
    finally:
        qkv.weight[rows] = w_backup
        qkv.bias[rows] = b_backup
    sl = slice(head * D, (head + 1) * D)
    other = torch.ones(H * D, dtype=torch.bool)
    other[sl] = False
    res = {
        "geometry": (L, H, D),
        "hook_input_width": int(base.shape[-1]),
        "zeroed_head_absmax": float(zeroed[layer, sl].abs().max()),
        "other_heads_same_layer_maxdiff": float((zeroed[layer, other] - base[layer, other]).abs().max()),
        "earlier_layers_maxdiff": float((zeroed[:layer] - base[:layer]).abs().max()),
        "later_layer_maxdiff": float((zeroed[layer + 1] - base[layer + 1]).abs().max()) if layer + 1 < L else None,
        "base_head_absmax": float(base[layer, sl].abs().max()),
    }
    # batched right padding vs single texts
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        batch = tokenizer([TEXT, SHORT], return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = old_side
    with coll:
        both = coll.collect(batch.input_ids, batch.attention_mask)
        single_short = coll.collect(tokenizer(SHORT, return_tensors="pt").input_ids)[0]
    res["batch_row0_vs_single_maxdiff"] = float((both[0] - base).abs().max())
    res["batch_row1_padded_vs_single_maxdiff"] = float((both[1] - single_short).abs().max())
    res["pad_count_row1"] = int((batch.attention_mask[1] == 0).sum())
    res["o_proj_module"] = type(o_proj(model, layer)).__name__
    # batched-vs-single differences are kernel numerics (length dependent, up to ~3e-3 at 300 tokens in fp32), not a
    # layout error: the policy tolerance of DESIGN 3.1 (1e-2) applies, the layout itself is checked exactly above
    tol = config.DTYPE_POLICY["tolerance_batched_vs_single"]
    res["tolerance_batched_vs_single"] = tol
    res["ok"] = (res["hook_input_width"] == H * D and res["zeroed_head_absmax"] == 0.0
                 and res["other_heads_same_layer_maxdiff"] == 0.0 and res["earlier_layers_maxdiff"] == 0.0
                 and (res["later_layer_maxdiff"] or 0) > 0 and res["base_head_absmax"] > 0
                 and res["batch_row0_vs_single_maxdiff"] < tol and res["batch_row1_padded_vs_single_maxdiff"] < tol
                 and res["pad_count_row1"] > 0)
    return res


# ----------------------------------------------------------------------------- mask positions
@torch.no_grad()
def check_mask_positions(model, tokenizer, eps=config.MASK_EPS, tol=1e-5):
    """positions='all' changes log p at every position, 'last' only at the last one, decode_only hooks are a
    no-op in a teacher-forced forward but act on KV-cache decoding steps. Returns a dict with 'ok'."""
    L, H, D = model_geometry(model.config)
    ids = tokenizer(TEXT, return_tensors="pt").input_ids.to(model.device)      # model device (cuda on Colab)
    S = ids.shape[1]
    heads = {l: list(range(H)) for l in range(3, min(9, L))}

    def logp(hook=None):
        if hook is None:
            return torch.log_softmax(model(input_ids=ids).logits[0], -1)
        with hook:
            return torch.log_softmax(model(input_ids=ids).logits[0], -1)

    def changed(a, b):
        return int(((a - b).abs().max(-1).values > tol).sum())

    base = logp()
    res = {"seq_len": S}
    res["mask_all_changed"] = changed(base, logp(HeadMask(model, heads, eps, positions="all")))
    res["mask_last_changed"] = changed(base, logp(HeadMask(model, heads, eps, positions="last", decode_only=False)))
    res["mask_decode_only_changed"] = changed(base, logp(HeadMask(model, heads, eps, positions="last", decode_only=True)))
    gen = torch.Generator().manual_seed(0)
    vectors = {l: torch.randn(H * D, generator=gen) for l in heads}
    res["steer_all_changed"] = changed(base, logp(HeadSteer(model, vectors, alpha=1.0, positions="all")))
    res["steer_last_changed"] = changed(base, logp(HeadSteer(model, vectors, alpha=1.0, positions="last", decode_only=False)))
    res["steer_decode_only_changed"] = changed(base, logp(HeadSteer(model, vectors, alpha=1.0, positions="last", decode_only=True)))
    # generation: decode-only mask leaves the prefill (first generated step) untouched, later steps differ
    kw = dict(max_new_tokens=3, do_sample=False, output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.pad_token_id)
    g_base = model.generate(ids, **kw)
    with HeadMask(model, heads, eps):                      # defaults: positions="last", decode_only=True
        g_mask = model.generate(ids, **kw)
    res["gen_step0_scores_maxdiff"] = float((g_base.scores[0] - g_mask.scores[0]).abs().max())
    res["gen_step1_scores_maxdiff"] = float((g_base.scores[1] - g_mask.scores[1]).abs().max())
    res["ok"] = (res["mask_all_changed"] == S and res["mask_last_changed"] == 1 and res["mask_decode_only_changed"] == 0
                 and res["steer_all_changed"] == S and res["steer_last_changed"] == 1 and res["steer_decode_only_changed"] == 0
                 and res["gen_step0_scores_maxdiff"] == 0.0 and res["gen_step1_scores_maxdiff"] > tol)
    return res


# ----------------------------------------------------------------------------- data counts
def _counts(labels):
    labels = np.asarray(labels)
    return (len(labels), int((labels == 1).sum()), int((labels == 0).sum()))


def check_data_counts(level="quick", tokenizer=None):
    """Compares dataset sizes with the verified counts. level 'quick' (~1 min) skips the HH train sets, the
    pooled HH config, BeaverTails and WikiText; 'full' checks everything. Returns a dict with 'ok' and 'mismatches'."""
    exp = EXPECTED
    got, mismatches = {}, []

    def expect(key, value, expected):
        got[key] = value
        if value != expected:
            mismatches.append(f"{key}: got {value}, expected {expected}")

    tr, va = truthfulqa_indices()
    expect("tqa_questions", (len(tr), len(va)), exp["tqa_questions"])
    expect("tqa_disjoint", len(set(tr.tolist()) & set(va.tolist())), 0)
    for split in ("train", "val"):
        texts, labels, groups = probing_set("honesty", split)
        expect(f"honesty_{split}", _counts(labels), exp["honesty_texts"][split])
        expect(f"honesty_{split}_groups", len(np.unique(groups)), exp["tqa_questions"][0 if split == "train" else 1])
    for value, per_split in exp["helpsteer2"].items():
        for split, expected in per_split.items():
            texts, labels, groups = probing_set(value, split)
            expect(f"helpsteer2_{value}_{split}", _counts(labels), expected)
    for split in ("train", "val"):
        texts, labels, groups = probing_set("coherence_strict", split)
        n, pos, neg = _counts(labels)
        expect(f"coherence_strict_{split}_pos", pos, exp["helpsteer2"]["coherence"][split][1])
        got[f"coherence_strict_{split}"] = (n, pos, neg)
        if neg < exp["helpsteer2"]["coherence"][split][2]:
            mismatches.append(f"coherence_strict_{split}: {neg} negatives < coherence negatives")
    for value in ("harmless", "helpful"):
        expect(f"hh_{value}_test_pairs", len(hh_pairs(value, "val")), exp["hh_pairs_test"][value])
    for value, expected in exp["helpsteer2_pairs_ge1"].items():
        expect(f"helpsteer2_pairs_ge1_{value}", len(eval_pairs("helpsteer2_pairwise", value, min_delta=1)), expected)
    for value, expected in exp["helpsteer2_pairs_ge2"].items():
        expect(f"helpsteer2_pairs_ge2_{value}", len(eval_pairs("helpsteer2_pairwise", value, min_delta=2)), expected)
    tqa_items = eval_pairs("tqa_pairwise")
    expect("tqa_pairwise_items", len(tqa_items), exp["tqa_questions"][1])
    got["tqa_pairwise_pairs"] = sum(len(i["pos"]) * len(i["neg"]) for i in tqa_items)
    if not 2000 <= got["tqa_pairwise_pairs"] <= 2400:
        mismatches.append(f"tqa_pairwise_pairs: {got['tqa_pairwise_pairs']} outside 2000-2400")
    hh_items = eval_pairs("hh_pairwise", "harmless")
    expect("hh_pairwise_harmless_items", len(hh_items), exp["hh_pairs_test"]["harmless"])
    for name in ("jailbreak", "malicious", "honesty", "helpfulness", "complexity", "verbosity", "coherence"):
        expect(f"eval_set_{name}", len(eval_set(name)), exp["eval_sets"][name])
    # subsampling invariants (smoke tier on honesty train)
    texts, labels, groups = probing_set("honesty", "train", max_samples=1000)
    got["honesty_train_smoke"] = _counts(labels)
    if len(texts) != 1000:
        mismatches.append(f"honesty_train_smoke: {len(texts)} != 1000")
    full_labels = probing_set("honesty", "train")[1]
    if abs(labels.mean() - full_labels.mean()) > 0.05:
        mismatches.append(f"honesty_train_smoke pos rate {labels.mean():.3f} vs full {full_labels.mean():.3f}")
    texts, labels, groups = probing_set("coherence", "val", max_samples=200)
    got["coherence_val_smoke"] = _counts(labels)
    if (labels == 0).sum() != 16 or len(labels) != 200:
        mismatches.append(f"coherence_val_smoke: {_counts(labels)} (expected 200 texts with all 16 negatives)")

    if level == "full":
        for value in ("harmless", "helpful", "harmless_pooled"):
            expect(f"hh_{value}_train_pairs", len(hh_pairs(value, "train")), exp["hh_pairs_train"][value])
        expect("hh_harmless_pooled_test_pairs", len(hh_pairs("harmless_pooled", "val")), exp["hh_pairs_test"]["harmless_pooled"])
        expect("eval_set_hh-rlhf", len(eval_set("hh-rlhf")), exp["eval_sets"]["hh-rlhf"])
        expect("eval_set_beavertails", len(eval_set("beavertails")), exp["eval_sets"]["beavertails"])
        if tokenizer is not None:
            expect("wikitext_tokens", int(wikitext(tokenizer).numel()), exp["wikitext_tokens"])
    got["mismatches"] = mismatches
    got["ok"] = not mismatches
    return got


# ----------------------------------------------------------------------------- command
def add_args(parser):
    add_common_args(parser, root=False, mode=False, seed=False)
    parser.add_argument("--data", default="quick", choices=["none", "quick", "full"], help="data-count check level")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--head", type=int, default=7)


def run(args):
    t0 = time.time()
    model, tokenizer = load_model(args.model, revision=args.revision, device=args.device, dtype=args.dtype, attn_impl=args.attn_impl)
    info = model.vh_info
    print(f"[selftest] {info['hf_id']}@{info['revision']} dtype={info['dtype']} attn={info['attn_impl']} device={info['device']} "
          f"transformers={info['transformers']} torch={info['torch']}")
    results = {}
    results["hook_layout"] = check_hook_layout(model, tokenizer, args.layer, args.head)
    results["mask_positions"] = check_mask_positions(model, tokenizer)
    if args.data != "none":
        results["data_counts"] = check_data_counts(args.data, tokenizer)
    ok = True
    for name, res in results.items():
        status = "PASS" if res["ok"] else "FAIL"
        ok &= res["ok"]
        detail = {k: v for k, v in res.items() if k != "ok"}
        print(f"[selftest] {status} {name}: {detail}")
    print(f"[selftest] {'ALL PASS' if ok else 'FAILED'} in {time.time() - t0:.1f}s")
    if not ok:
        sys.exit(1)
    return 0
