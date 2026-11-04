"""Dataset sizes against the verified counts (DESIGN section 2 table): 653/164 TruthfulQA questions, 4,712/1,206
honesty texts, HelpSteer2 label counts, HH-RLHF single-turn pairs, pairwise evaluation items, subsampling invariants.

Set VH_TEST_DATA=full to also scan the HH-RLHF train splits, the pooled config, BeaverTails and WikiText (~1-2 min)."""
import os

import numpy as np
import pytest

from valueheads_pythia import config
from valueheads_pythia.commands.selftest import EXPECTED, check_data_counts
from valueheads_pythia.data import (eval_pairs, extract_texts, helpsteer2_rows, probing_set, stratified_subsample,
                                    truthfulqa_mc, uniform_subsample, wikitext)

LEVEL = os.environ.get("VH_TEST_DATA", "quick")


@pytest.fixture(scope="module")
def counts():
    tokenizer = None
    if LEVEL == "full":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision="step143000")
    return check_data_counts(LEVEL, tokenizer)


def test_verified_counts(counts):
    assert counts["ok"], counts["mismatches"]
    assert counts["tqa_questions"] == (653, 164)
    assert counts["honesty_train"] == (4712, 2075, 2637) and counts["honesty_val"] == (1206, 525, 681)
    assert counts["hh_harmless_test_pairs"] == 662 and counts["hh_helpful_test_pairs"] == 700
    assert counts["helpsteer2_coherence_val"] == (769, 753, 16)
    assert counts["helpsteer2_pairs_ge1_helpfulness"] == 373 and counts["helpsteer2_pairs_ge2_complexity"] == 16
    assert 2000 <= counts["tqa_pairwise_pairs"] <= 2400


def test_helpsteer2_extract_once_labels():
    texts, labels, groups = helpsteer2_rows("val")
    assert len(texts) == EXPECTED["helpsteer2_rows"]["val"] == 1038
    assert set(labels) == set(config.HELPSTEER_VALUES_EXT)
    for value, (n, pos, neg) in ((v, EXPECTED["helpsteer2"][v]["val"]) for v in config.HELPSTEER_VALUES):
        l = labels[value]
        assert (int((l >= 0).sum()), int((l == 1).sum()), int((l == 0).sum())) == (n, pos, neg)
    strict = labels["coherence_strict"]
    assert int((strict == 1).sum()) == 753 and int((strict == 0).sum()) >= 16
    assert len(np.unique(groups)) == 519 and np.all(np.bincount(groups) == 2)
    assert texts[0].startswith("Question: ") and "\nAnswer: " in texts[0]
    # the per-value probing set is the labeled subset of the extract-once rows
    t2, l2, g2 = probing_set("helpfulness", "val")
    keep = np.flatnonzero(labels["helpfulness"] >= 0)
    assert t2 == [texts[i] for i in keep] and np.array_equal(l2, labels["helpfulness"][keep]) and np.array_equal(g2, groups[keep])


def test_extract_once_smoke_subsample_keeps_coherence_negatives():
    texts, labels, groups = extract_texts("helpsteer2", "val", max_samples=200)
    assert len(texts) == 200 and all(len(l) == 200 for l in labels.values()) and len(groups) == 200
    assert int((labels["coherence"] == 0).sum()) == 16
    # prompts are kept together as far as the quotas allow: at least 90 % of groups complete
    sizes = np.bincount(groups)
    assert (sizes[sizes > 0] == 2).mean() > 0.9


def test_stratified_subsample_invariants():
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(300), 2)
    labels = np.tile([1, 0], 300)
    keep = stratified_subsample(labels, groups, 100, seed=1)
    assert len(keep) == 100 and np.all(np.diff(keep) > 0)
    assert labels[keep].sum() == 50
    assert np.all(np.bincount(groups[keep])[np.bincount(groups[keep]) > 0] == 2)      # whole pairs
    # discarded rows (-1) never selected; all negatives kept when requested
    labels2 = rng.choice([1, 1, 1, 0, -1], size=1000)
    groups2 = np.arange(1000) // 3
    keep2 = stratified_subsample(labels2, groups2, 300, seed=3, keep_all_negatives=True)
    assert np.all(labels2[keep2] >= 0) and len(keep2) == 300
    n_neg = int((labels2 == 0).sum())
    assert int((labels2[keep2] == 0).sum()) == min(n_neg, 150)
    assert stratified_subsample(labels2, groups2, 1, seed=0).shape == (1,)
    assert len(stratified_subsample(labels2, groups2, None)) == int((labels2 >= 0).sum())
    # paper mode: repo semantics
    assert uniform_subsample(10, 20).tolist() == list(range(10))
    u = uniform_subsample(1000, 100, seed=42)
    assert len(u) == 100 and np.all(np.diff(u) > 0)


def test_probing_set_modes_and_seeds():
    with pytest.raises(ValueError):
        probing_set("harmless_pooled", "val", mode="pythia")
    t42, _, g42 = probing_set("honesty", "val")
    t1, _, g1 = probing_set("honesty", "val", split_seed=1)
    assert len(t42) == 1206 and set(g42.tolist()) != set(g1.tolist())
    texts, labels, groups = probing_set("honesty", "train", mode="paper", max_samples=1000, seed=42)
    assert len(texts) == 1000 and labels.shape == (1000,) and groups.shape == (1000,)
    assert texts[0].startswith("Question: ") and " Answer: " in texts[0] and "\n" not in texts[0].split(" Answer: ")[0]


def test_eval_pairs_formats():
    items = eval_pairs("tqa_pairwise", max_items=10, seed=0)
    assert len(items) == 10
    for it in items:
        assert it["context"].endswith(" Answer:") and all(c.startswith(" ") for c in it["pos"] + it["neg"])
    full = eval_pairs("tqa_pairwise")
    texts, labels, groups = probing_set("honesty", "val")
    probing_texts = set(texts)
    assert all(it["context"] + c in probing_texts for it in full for c in it["pos"] + it["neg"])
    assert {it["group"] for it in full} == set(groups.tolist())
    hh = eval_pairs("hh_pairwise", "harmless")
    assert len(hh) == 662
    hh_texts = set(probing_set("harmless", "val")[0])
    assert all(it["context"] + it["pos"][0] in hh_texts and it["context"] + it["neg"][0] in hh_texts for it in hh)
    assert all(it["context"].endswith("\n\nAssistant:") for it in hh)
    hs = eval_pairs("helpsteer2_pairwise", "verbosity", min_delta=1)
    assert len(hs) == 254 and all(it["meta"]["rating_hi"] > it["meta"]["rating_lo"] for it in hs)
    hs_texts = set(probing_set("verbosity", "val")[0]) | set(helpsteer2_rows("val")[0])
    assert all(it["context"] + it["pos"][0] in hs_texts for it in hs)
    strict = eval_pairs("helpsteer2_pairwise", "coherence_strict")
    assert len(strict) == 44
    with pytest.raises(ValueError):
        eval_pairs("nope")


def test_truthfulqa_mc_rows():
    rows = truthfulqa_mc("val")
    assert len(rows) == 164 and {"question", "mc1_targets", "mc2_targets"} <= set(rows[0])
    assert sum(rows[0]["mc1_targets"]["labels"]) == 1


def test_wikitext_text():
    text = wikitext()
    assert isinstance(text, str) and len(text) > 1_000_000 and "\n\n" in text
    with pytest.raises(ValueError):
        wikitext(max_tokens=10)


def test_tier_defaults():
    assert config.max_samples("smoke", "honesty", "train") == 1000 and config.max_samples("smoke", "honesty", "val") == 300
    assert config.max_samples("smoke", "helpsteer2", "val") == 200 and config.max_samples("smoke", "coherence", "train") == 500
    assert config.max_samples("t4", "harmless", "train", "pythia-160m") is None
    assert config.max_samples("t4", "harmless", "train", "pythia-1b") == 10000
    assert config.max_samples("t4", "helpsteer2", "train", "pythia-2.8b") == 4000
    assert config.max_samples("a100", "helpful", "train", "pythia-2.8b") == 20000
    assert config.max_samples(None, "honesty", "train") is None
    with pytest.raises(ValueError):
        config.max_samples("v100", "honesty", "train")
    p = config.paths("/tmp/x")
    assert p["heads"] == "/tmp/x/results/heads" and p["features"].endswith("features")
    assert config.feature_dir(p, "EleutherAI/pythia-160m", "step0", "honesty", "val").endswith("pythia-160m/step0/honesty_val")
    assert config.heads_path(p, "pythia-160m", "step0", "honesty", "pythia").endswith("pythia-160m_step0_honesty_pythia.json")
