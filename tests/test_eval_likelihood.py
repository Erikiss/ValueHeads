"""eval_likelihood.py: continuation boundaries of the batched log-likelihood scorer against a per-example computation,
metric definitions on synthetic scores, sliding-window perplexity, conditions with hooks, persistence and summary rows."""
import json

import numpy as np
import pytest
import torch

from valueheads_pythia.controls import random_sets_layer_matched
from valueheads_pythia.eval_likelihood import (FEWSHOT_PRIMER, METRIC_SPECS, MCMetric, PairwiseMetric, append_condition,
                                               append_summary, build_metric, compare_conditions, condition_key, encode_pair,
                                               flat_heads, format_table, jsonl_path, load_conditions, loglik_batched, mask_hook,
                                               run_condition, score_pairs, signif_heads_from_file, steer_hook,
                                               steering_vectors_from_directions, summary_row)
from valueheads_pythia.hooks import HeadMask

PAIRS = [
    ("Question: What happens if an earthworm is cut in half? Answer:", " It becomes two worms."),
    ("Question: What is the capital of France? Answer:", " Paris"),
    ("\n\nHuman: Can you recommend a book?\n\nAssistant:", " Sure! A classic choice is 'Pride and Prejudice' by Jane Austen, "
                                                           "which many readers enjoy for its wit and its memorable characters."),
    ("Q: Where were the 1992 Olympics held?\nA:", " The 1992 Olympics were held in Barcelona, Spain."),
    ("", " Hello world"),                                      # empty context -> boundary 1, first token unscored
    ("The capital of France is", " Paris."),
    ("Question: What is 2 + 2?\nAnswer:", " 4"),
]


def unbatched_reference(model, tok, ctx, cont):
    """Per-example computation: joint tokenization, log-softmax of a single unpadded forward, sum over continuation tokens."""
    ids = tok(ctx + cont, add_special_tokens=False)["input_ids"]
    n_ctx = len(tok(ctx, add_special_tokens=False)["input_ids"])
    assert ids[:n_ctx] == tok(ctx, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0].float()
    lp = torch.log_softmax(logits[:-1], -1)
    tok_lp = lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]
    start = max(n_ctx, 1)
    return float(tok_lp[start - 1:].sum()), len(ids) - start, ids[n_ctx:]


def test_batched_scoring_matches_unbatched(model_tok):
    model, tok = model_tok
    # Numerics: batched (padded) and single forwards differ by up to ~1e-3 per token on pythia-160m in fp32 (massive
    # activations; DESIGN 3.1 prescribes a 1e-2 tolerance), so the boundary / token count is checked exactly and the
    # log-likelihoods with that tolerance.
    batched = score_pairs(model, tok, PAIRS, batch_tokens=70)       # several right-padded batches of mixed lengths
    for (ctx, cont), res in zip(PAIRS, batched):
        ref_sum, ref_n, cont_ids = unbatched_reference(model, tok, ctx, cont)
        assert res["n"] == ref_n, (ctx, cont)
        assert abs(res["sum"] - ref_sum) < 1e-2, (ctx, cont, res["sum"], ref_sum)
        assert np.isclose(res["mean"], res["sum"] / res["n"])
        assert not res["fallback"] and res["cut"] == 0
        if ctx:
            assert tok.decode(cont_ids) == cont                    # continuation tokens = exactly the continuation (leading space)
            assert tok.decode(cont_ids[:1]).startswith(" ")
    # one big batch gives the same numbers as the small batches
    single = score_pairs(model, tok, PAIRS, batch_tokens=10 ** 6)
    for a, b in zip(batched, single):
        assert abs(a["sum"] - b["sum"]) < 1e-2 and a["n"] == b["n"]
    # per-token log-probs of a padded row line up position by position with the unpadded reference
    ids = [tok(c + k, add_special_tokens=False)["input_ids"] for c, k in PAIRS[:3]]
    bounds = [len(tok(c, add_special_tokens=False)["input_ids"]) for c, _ in PAIRS[:3]]
    _, counts, per_token = loglik_batched(model, ids, bounds, tok.pad_token_id, batch_tokens=10 ** 6, return_tokens=True)
    for seq, b, n, lp in zip(ids, bounds, counts, per_token):
        with torch.no_grad():
            ref = torch.log_softmax(model(input_ids=torch.tensor([seq])).logits[0, b - 1:-1].float(), -1)
        ref = ref[torch.arange(len(seq) - b), torch.tensor(seq[b:])].numpy()
        assert n == len(seq) - b == len(lp) and np.abs(lp - ref).max() < 1e-2
    # duplicates are scored once and reported in input order
    dup = score_pairs(model, tok, [PAIRS[1], PAIRS[0], PAIRS[1]], batch_tokens=10 ** 6)
    assert dup[0] == dup[2] and abs(dup[1]["sum"] - batched[0]["sum"]) < 1e-2
    # the first position of a sequence is never scored
    ids = tok("Hello world", add_special_tokens=False)["input_ids"]
    sums, counts = loglik_batched(model, [ids], [1], tok.pad_token_id)
    assert counts[0] == len(ids) - 1
    sums0, counts0 = loglik_batched(model, [ids], [len(ids)], tok.pad_token_id)
    assert counts0[0] == 0 and sums0[0] == 0.0


class MergingTokenizer:
    """Character tokenizer whose joint tokenization merges ':' + ')' into one token (BPE merge across the boundary)."""
    pad_token_id = 0
    bos_token_id = 7

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text[i:i + 2] == ":)":
                ids.append(99)
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}


def test_encode_pair_boundary_fallback_truncation_bos():
    tok = MergingTokenizer()
    ids, b, fallback, cut = encode_pair(tok, "ab", " cd")
    assert ids == [97, 98, 32, 99, 100] and b == 2 and not fallback and cut == 0
    ids, b, fallback, cut = encode_pair(tok, "hi:", ")")              # joint tokenization merges across the boundary
    assert fallback and ids == [104, 105, 58, 41] and b == 3
    ids, b, fallback, cut = encode_pair(tok, "0123456789", "abc", max_length=5)
    assert ids == [54, 55, 56, 57, 58][:0] + [ord(c) for c in "89abc"] and b == 2 and cut == 8     # tail kept, boundary moved
    ids, b, _, cut = encode_pair(tok, "0123456789", "abc", max_length=2)
    assert ids == [98, 99] and b == 1 and cut == 11                    # continuation itself truncated: first token unscored
    ids, b, _, _ = encode_pair(tok, "", "xy")
    assert b == 1 and ids == [120, 121]
    ids, b, _, _ = encode_pair(tok, "ab", "c", add_bos=True)
    assert ids == [7, 97, 98, 99] and b == 3


def test_pairwise_metric_definitions():
    # two questions: A has one pair (win), B has two pairs (one win) -> PairAcc = (1 + 0.5) / 2, pair level 2 / 3
    items = [{"group": 10, "pos": [[-2.0, 2]], "neg": [[-6.0, 2]]},
             {"group": 11, "pos": [[-4.0, 1]], "neg": [[-1.0, 1], [-5.0, 1]]}]
    m = PairwiseMetric.__new__(PairwiseMetric)
    m.name, m.value, m.spec = "tqa_pairwise", None, METRIC_SPECS["tqa_pairwise"]
    s = m.summarize(items)
    assert s["pair_acc"] == 0.75 and np.isclose(s["pair_acc_pairs"], 2 / 3) and s["n_questions"] == 2 and s["n_pairs"] == 3
    assert np.isclose(s["margin_mean"], np.mean([(-1) - (-3), -4 - (-1), -4 - (-5)]))
    vals, groups, w = m.item_values(items)
    assert groups.tolist() == [10, 11, 11] and vals["win"].tolist() == [1.0, 0.0, 1.0] and w is None
    # HH: sum vs mean accuracy can disagree; length baseline
    hh = PairwiseMetric.__new__(PairwiseMetric)
    hh.name, hh.value, hh.spec = "hh_pairwise", "harmless", METRIC_SPECS["hh_pairwise"]
    items = [{"group": 0, "pos": [[-10.0, 10]], "neg": [[-8.0, 4]]},     # sum prefers neg, mean prefers pos (-1.0 > -2.0)
             {"group": 1, "pos": [[-3.0, 3]], "neg": [[-9.0, 3]]}]
    s = hh.summarize(items)
    assert s["acc_sum"] == 0.5 and s["acc_mean"] == 1.0 and s["len_baseline"] == 0.5 and s["n_items"] == 2
    assert np.isclose(s["margin_sum"], np.mean([-2.0, 6.0])) and np.isclose(s["margin_mean"], np.mean([1.0, 2.0]))
    hs = PairwiseMetric.__new__(PairwiseMetric)
    hs.name, hs.value, hs.spec = "helpsteer2_pairwise", "verbosity", METRIC_SPECS["helpsteer2_pairwise"]
    items = [dict(it, delta_rating=d) for it, d in zip(items, (1, 2))]
    s = hs.summarize(items)
    assert s["n_ge2"] == 1 and s["acc_mean_ge2"] == 1.0 and s["acc_sum_ge2"] == 1.0 and s["acc_mean"] == 1.0


def test_tqa_pairwise_items_and_run_condition(model_tok):
    model, tok = model_tok
    metric = build_metric("tqa_pairwise", tok, max_items=3, seed=0)
    assert metric.tag == "tqa_pairwise" and metric.config["n_items"] == 3 and metric.value is None
    assert all(c.endswith(" Answer:") for c, _ in metric.requests) and all(k.startswith(" ") for _, k in metric.requests)
    base = run_condition(model, tok, None, metric, batch_tokens=2048)
    s = base["summary"]
    assert 0 <= s["pair_acc"] <= 1 and s["n_questions"] == 3 and s["n_pairs"] == sum(len(i["pos"]) * len(i["neg"]) for i in metric.items)
    assert base["stats"]["hook"] is None and base["stats"]["n_requests"] == len(metric.requests) and base["stats"]["n_fallback"] == 0
    heads = {4: [0, 2, 3], 6: [0, 4]}
    masked = run_condition(model, tok, mask_hook(model, heads), metric, batch_tokens=2048)
    assert masked["stats"]["hook"] == "HeadMask"
    assert any(abs(a["pos"][0][0] - b["pos"][0][0]) > 1e-4 for a, b in zip(masked["items"], base["items"]))
    with pytest.raises(ValueError):
        run_condition(model, tok, HeadMask(model, heads), metric)        # paper-semantics hook: decode_only -> no-op here
    # positions="last" would edit only the final (padded) column of a right-padded batch, never a scored position
    # (a silent no-op reproducing the base scores) -> refused
    with pytest.raises(ValueError, match="positions='all'"):
        run_condition(model, tok, mask_hook(model, heads, positions="last"), metric, batch_tokens=2048)
    # deltas, random controls and the summary row
    deltas = compare_conditions(metric, masked, base, B=100, seed=0)
    assert set(deltas) == {"margin", "win"} and deltas["margin"]["n"] == s["n_pairs"] and deltas["margin"]["n_groups"] == 3
    assert deltas["margin"]["lo"] <= deltas["margin"]["mean"] <= deltas["margin"]["hi"]
    sets = random_sets_layer_matched(heads, 12, 12, 2, seed=0)
    randoms = [run_condition(model, tok, mask_hook(model, r), metric, batch_tokens=2048) for r in sets]
    row = summary_row(metric, base, masked, randoms, "mask", B=50, seed=0, meta={"model": "pythia-160m"})
    assert row["R"] == 2 and row["primary"] == "pair_acc" and row["p_direction"] == "le"
    assert row["p_emp_primary"] in {1 / 3, 2 / 3, 1.0} and row["p_emp_delta"] in {1 / 3, 2 / 3, 1.0}
    assert 0 <= row["p_ttest_paper"] <= 1 or np.isnan(row["p_ttest_paper"])
    assert np.isclose(row["delta_primary"], masked["summary"]["pair_acc"] - base["summary"]["pair_acc"])
    assert row["delta_ci_lo"] <= row["delta_mean"] <= row["delta_ci_hi"] and json.loads(row["delta_other"])["win"]
    table = format_table(metric, base, masked, randoms)
    assert table.count("\n") == 5 and "random mean+-std" in table    # header, base, topk, 2 random, random stats
    row_base = summary_row(metric, base, None, [], "none", B=10)
    assert "topk_pair_acc" not in row_base and row_base["R"] == 0
    # steering with the stored directions changes the scores too
    rng = np.random.RandomState(0)
    directions = rng.randn(144, 64).astype(np.float32)
    vec = steering_vectors_from_directions(directions, np.full(144, 0.5, dtype=np.float32), flat_heads(heads, 12), 12, 64)
    assert set(vec) == {4, 6} and np.isclose(float(vec[4][0:64].norm()), 0.5)
    steered = run_condition(model, tok, steer_hook(model, vec, alpha=1.0), metric, batch_tokens=2048)
    assert abs(steered["summary"]["margin_mean"] - base["summary"]["margin_mean"]) > 0 or steered["items"] != base["items"]
    with pytest.raises(ValueError):
        steering_vectors_from_directions(directions, np.full(144, np.nan, dtype=np.float32), [48], 12, 64)
    # compare_conditions refuses mismatched item sets
    other = build_metric("tqa_pairwise", tok, max_items=2, seed=1)
    with pytest.raises(ValueError):
        compare_conditions(metric, run_condition(model, tok, None, other), base)


def test_mc_metric(model_tok):
    model, tok = model_tok
    metric = build_metric("tqa_mc1_mc2", tok, max_items=2, seed=0)
    assert metric.config["n_items"] == 2 and FEWSHOT_PRIMER.endswith("Spain.\n\n")
    n_unique = sum(len(set(r["mc1_targets"]["choices"] + r["mc2_targets"]["choices"])) for r in metric.rows)
    assert len(metric.requests) == 2 * n_unique
    assert metric.requests[0][0].startswith("Q: What is human life expectancy") and metric.requests[0][0].endswith("\nA:")
    res = run_condition(model, tok, None, metric, batch_tokens=4096)
    s = res["summary"]
    assert s["n_questions"] == 2 and s["mc1"] in {0.0, 0.5, 1.0} and 0 <= s["mc2"] <= 1 and 0 <= s["mc2_lennorm"] <= 1
    assert s["mc1_zeroshot"] in {0.0, 0.5, 1.0} and 0 < s["chance_mc1"] < 1
    rec = res["items"][0]
    assert set(MCMetric.KEYS) <= set(rec)
    vals, groups, w = metric.item_values(res["items"])
    assert groups.tolist() == [0, 1] and w is None and vals["mc2"].shape == (2,)
    # MC2 definition: softmax mass of the true options; recompute from the stored request scores
    row, slots = metric.rows[0], metric._slots[0]["fewshot"]
    scored = score_pairs(model, tok, [metric.requests[slots[c]] for c in row["mc2_targets"]["choices"]], batch_tokens=4096)
    ll = np.array([x["sum"] for x in scored])
    p = np.exp(ll - ll.max())
    p /= p.sum()
    assert np.isclose(p[np.array(row["mc2_targets"]["labels"], dtype=bool)].sum(), rec["mc2"], atol=1e-4)


def test_wikitext_ppl_windows(model_tok):
    model, tok = model_tok
    # DESIGN 4.1 item 5: every window scores only its last `stride` tokens, each with >= window - stride tokens of context;
    # the first window - stride tokens of the corpus are never scored
    metric = build_metric("wikitext_ppl", tok, max_tokens=2000, window=1024, stride=512)
    assert metric.windows == [(0, 1024, 512), (512, 1536, 1024), (1024, 2000, 1536)] and metric.min_context == 512
    assert metric.config["n_tokens_total"] == 2000 and metric.config["n_items"] == 3 and metric.config["min_context"] == 512
    assert all(t - b >= 512 for b, _, t in metric.windows)
    res = run_condition(model, tok, None, metric, batch_tokens=2048)
    s = res["summary"]
    assert s["n_tokens_scored"] == 1488 and s["n_windows"] == 3 and s["ppl"] > 1 and np.isclose(s["log_ppl"], np.log(s["ppl"]))
    vals, groups, w = metric.item_values(res["items"])
    assert w.tolist() == [512, 512, 464] and np.isclose(np.average(vals["nll"], weights=w), s["nll"])
    # min_context = 0 is the Hugging Face recipe (first window scores everything but position 0)
    hf = build_metric("wikitext_ppl", tok, max_tokens=2000, window=1024, stride=512, min_context=0)
    assert hf.windows == [(0, 1024, 1), (512, 1536, 1024), (1024, 2000, 1536)] and hf.config["min_context"] == 0
    assert condition_key(hf.config, {"intervention": "none"}) != condition_key(metric.config, {"intervention": "none"})
    # too few tokens for the minimal context, or a bad stride / min_context, is an error, not a silent empty metric
    with pytest.raises(ValueError):
        build_metric("wikitext_ppl", tok, max_tokens=300)
    with pytest.raises(ValueError):
        build_metric("wikitext_ppl", tok, max_tokens=2000, window=1024, stride=512, min_context=1024)
    with pytest.raises(ValueError):
        build_metric("wikitext_ppl", tok, max_tokens=2000, window=512, stride=1024)
    # a single window without a context requirement equals the plain full-sequence perplexity
    small = build_metric("wikitext_ppl", tok, max_tokens=300, min_context=0)
    assert small.windows == [(0, 300, 1)]
    res_small = run_condition(model, tok, None, small, batch_tokens=2048)
    with torch.no_grad():
        logits = model(input_ids=small.ids[None]).logits[0].float()
    lp = torch.log_softmax(logits[:-1], -1)[torch.arange(299), small.ids[1:]]
    assert np.isclose(res_small["summary"]["ppl"], float(torch.exp(-lp.mean())), rtol=1e-4)
    # masking raises the perplexity; the delta is token-weighted
    masked = run_condition(model, tok, mask_hook(model, {l: list(range(12)) for l in range(3, 9)}), small, batch_tokens=2048)
    assert masked["summary"]["ppl"] > res_small["summary"]["ppl"]
    d = compare_conditions(small, masked, res_small, B=20)["nll"]
    assert np.isclose(d["mean"], masked["summary"]["log_ppl"] - res_small["summary"]["log_ppl"])
    row = summary_row(small, res_small, masked, [], "mask", B=10)
    assert row["effect"] == "delta_log_ppl" and row["p_direction"] == "ge" and np.isnan(row["p_emp_primary"]) and row["min_context"] == 0
    with pytest.raises(ValueError):
        build_metric("wikitext_ppl", None)
    with pytest.raises(ValueError):
        build_metric("nope", tok)


def test_tqa_pairwise_lmeval_context(model_tok):
    """--context lmeval: the same answer pairs behind the lm-eval 6-shot primer + 'Q: {q}\\nA:' (DESIGN 3.2 second column)."""
    model, tok = model_tok
    probing = build_metric("tqa_pairwise", tok, max_items=2, seed=0)
    lmeval = build_metric("tqa_pairwise", tok, max_items=2, seed=0, context="lmeval")
    assert lmeval.tag == "tqa_pairwise-lmeval" and probing.tag == "tqa_pairwise" and lmeval.config["context"] == "lmeval"
    assert [it["group"] for it in lmeval.items] == [it["group"] for it in probing.items]
    for it_p, it_l in zip(probing.items, lmeval.items):
        assert it_l["context"] == FEWSHOT_PRIMER + f"Q: {it_p['meta']['question']}\nA:" and it_l["pos"] == it_p["pos"] and it_l["neg"] == it_p["neg"]
    assert all(c.startswith("Q: What is human life expectancy") and c.endswith("\nA:") for c, _ in lmeval.requests)
    res = run_condition(model, tok, None, lmeval, batch_tokens=4096)
    assert res["summary"]["n_questions"] == 2 and res["stats"]["n_fallback"] == 0
    row = summary_row(lmeval, res, None, [], "none", B=10)
    assert row["context"] == "lmeval" and row["metric"] == "tqa_pairwise"
    assert condition_key(lmeval.config, {"intervention": "none"}) != condition_key(probing.config, {"intervention": "none"})
    with pytest.raises(ValueError):
        build_metric("hh_pairwise", tok, max_items=2, context="lmeval")
    with pytest.raises(ValueError):
        PairwiseMetric("tqa_pairwise", max_items=2, context="chat")


def test_hh_and_helpsteer_items_have_no_text(tmp_path):
    hh = build_metric("hh_pairwise", None, value=None, max_items=4)
    assert hh.value == "harmless" and hh.tag == "hh_pairwise-harmless" and len(hh.items) == 4
    assert all(c.endswith("\n\nAssistant:") for c, _ in hh.requests)
    hs = build_metric("helpsteer2_pairwise", None, value="verbosity", max_items=5, min_delta=2)
    assert hs.tag == "helpsteer2_pairwise-verbosity" and all(it["meta"]["delta"] >= 2 for it in hs.items)
    with pytest.raises(ValueError):
        build_metric("helpsteer2_pairwise", None, value="honesty")
    # persistence: records carry scores and group ids only
    path = jsonl_path(str(tmp_path), hh.tag, "none")
    rec = {"condition": "base", "intervention": "none", "heads": None, "key": condition_key(hh.config, {"intervention": "none"}),
           "summary": {"acc_sum": 0.5}, "items": [{"group": 0, "pos": [[-1.0, 1]], "neg": [[-2.0, 1]]}], "stats": {"seconds": 1.0}}
    append_condition(path, rec)
    append_condition(path, dict(rec, condition="random0", heads={3: [1]}))
    loaded = load_conditions(path)
    assert set(loaded) == {"base", "random0"} and loaded["random0"]["heads"] == {"3": [1]}
    assert "Assistant" not in open(path).read()
    assert condition_key(hh.config, {"intervention": "none"}) != condition_key(hh.config, {"intervention": "mask"})
    assert load_conditions(str(tmp_path / "missing.jsonl")) == {}
    # summary.csv alignment across rows with different columns
    csv = str(tmp_path / "summary.csv")
    append_summary(csv, [{"metric": "a", "x": 1}])
    append_summary(csv, [{"metric": "b", "y": 2}])
    import pandas as pd
    df = pd.read_csv(csv)
    assert len(df) == 2 and set(df.columns) == {"metric", "x", "y", "run_key"}
    append_summary(csv, [{"metric": "a", "x": 3}])                   # same run key -> replaces the earlier row
    df = pd.read_csv(csv)
    assert len(df) == 2 and df.loc[df.metric == "a", "x"].item() == 3
    # significant-set selection needs the bootstrap / permutation columns
    heads_json = str(tmp_path / "heads.json")
    with open(heads_json, "w") as f:
        json.dump({"meta": {"perm_threshold": 0.6}, "heads": [{"flat_idx": 5, "rank": 1, "ci_lo": 0.7}, {"flat_idx": 9, "rank": 2, "ci_lo": 0.55}]}, f)
    assert signif_heads_from_file(heads_json) == [5]
    with open(heads_json, "w") as f:                       # probe-stats output: explicit list, rank order kept
        json.dump({"meta": {"perm_threshold": 0.6, "signif_heads": [9, 5]},
                   "heads": [{"flat_idx": 5, "rank": 1, "ci_lo": 0.7}, {"flat_idx": 9, "rank": 2, "ci_lo": 0.65}, {"flat_idx": 1, "rank": 3}]}, f)
    assert signif_heads_from_file(heads_json) == [5, 9]
    with open(heads_json, "w") as f:
        json.dump({"meta": {}, "heads": [{"flat_idx": 5, "rank": 1}]}, f)
    with pytest.raises(ValueError):
        signif_heads_from_file(heads_json)
