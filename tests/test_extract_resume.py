"""Sharded extraction: layer-major float16 shards, manifest, DONE marker, deterministic batching under a token budget,
tail-keep truncation, and resuming an interrupted run yields bit-identical shards."""
import json
import os

import numpy as np
import pytest

from valueheads_pythia import extract as ex
from valueheads_pythia.hooks import ActivationCollector

TEXTS = [f"Question: Is {i} an even number? Answer: {'Yes' if i % 2 == 0 else 'No'}, {i} is {'even' if i % 2 == 0 else 'odd'}." for i in range(23)]
TEXTS += ["Short.", "A somewhat longer sentence that will sort first because it has the most tokens of all texts here, really."]


def test_plan_batches_token_budget():
    lengths = [5, 50, 7, 50, 3, 20]
    batches = ex.plan_batches(lengths, batch_size=4, max_tokens_per_batch=100)
    assert batches == [[1, 3], [5, 2, 0, 4]]          # descending length, 100 // 50 = 2 per batch, then 4
    assert sum(len(b) for b in batches) == len(lengths)
    assert all(len(b) * max(lengths[i] for i in b) <= 100 or len(b) == 1 for b in batches)
    assert ex.plan_batches([300], 8, 100) == [[0]]


def test_tokenize_all_keeps_tail(model_tok):
    _, tok = model_tok
    ids, n_trunc = ex.tokenize_all(tok, TEXTS, max_length=6)
    full = tok(TEXTS[0])["input_ids"]
    assert n_trunc == len(TEXTS) - 1 and ids[0] == full[-6:] and ids[-2] == tok("Short.")["input_ids"]
    assert tok.truncation_side == "right"           # restored


def test_extract_resume_identical(model_tok, tmp_path):
    model, tok = model_tok
    labels = np.array([i % 2 for i in range(len(TEXTS))])
    groups = np.arange(len(TEXTS))
    ref_dir = str(tmp_path / "honesty_train")
    man = ex.extract_features(model, tok, TEXTS, ref_dir, batch_size=4, resume=True, labels=labels, groups=groups,
                              flush_every=1, show_progress=False, extra_manifest={"dataset": "honesty"})
    assert ex.is_done(ref_dir) and man["n"] == len(TEXTS) and man["n_done"] == len(TEXTS) and man["dataset"] == "honesty"
    assert man["dtype"] == "float32" and man["attn_impl"] == "eager" and man["commit_sha"] and man["transformers"]
    acts, lab, grp, man2 = ex.load_features(ref_dir)
    assert acts[0].dtype == np.float16 and acts[0].shape == (len(TEXTS), 768) and len(acts.keys()) == 12
    assert np.array_equal(lab, labels) and np.array_equal(grp, groups) and man2["n_tokens"] == man["n_tokens"]
    # shards hold the last-token activations of each text (compare with a direct single-text collection)
    with ActivationCollector(model) as coll:
        single = coll.collect(tok(TEXTS[3], return_tensors="pt").input_ids)[0].numpy().astype(np.float16)
    assert np.abs(acts[7][3].astype(np.float32) - single[7].astype(np.float32)).max() < 1e-2

    # interrupted run: stop after 2 batches, then resume -> identical shards, no re-extraction of done batches
    class Stop(Exception):
        pass

    calls = {"n": 0}
    orig = ActivationCollector.collect

    def limited(self, *a, **k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise Stop()
        return orig(self, *a, **k)

    res_dir = str(tmp_path / "honesty_val")
    ActivationCollector.collect = limited
    try:
        with pytest.raises(Stop):
            ex.extract_features(model, tok, TEXTS, res_dir, batch_size=4, labels=labels, groups=groups, flush_every=1, show_progress=False)
    finally:
        ActivationCollector.collect = orig
    partial = ex.read_manifest(res_dir)
    assert not ex.is_done(res_dir) and 0 < partial["n_done"] < len(TEXTS)
    calls["n"] = 0
    ActivationCollector.collect = lambda self, *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(self, *a, **k))[1]
    try:
        man3 = ex.extract_features(model, tok, TEXTS, res_dir, batch_size=4, resume=True, labels=labels, groups=groups, flush_every=1, show_progress=False)
    finally:
        ActivationCollector.collect = orig
    assert ex.is_done(res_dir) and man3["n_done"] == len(TEXTS)
    assert calls["n"] < len(ex.plan_batches([len(s) for s in ex.tokenize_all(tok, TEXTS)[0]], 4, 16384))
    acts_res = ex.load_features(res_dir)[0]
    for l in range(12):
        assert np.array_equal(np.asarray(acts_res[l]), np.asarray(acts[l]))
    # a DONE directory is skipped on resume and rejected for different texts
    assert ex.extract_features(model, tok, TEXTS, res_dir, resume=True, show_progress=False)["n_done"] == len(TEXTS)
    with pytest.raises(RuntimeError):
        ex.extract_features(model, tok, TEXTS[:-1], res_dir, resume=True, show_progress=False)
    # per-value label sets (HelpSteer2 extract-once path)
    multi_dir = str(tmp_path / "helpsteer2_val")
    multi = {"coherence": labels[:8], "verbosity": np.where(labels == 1, -1, 0)[:8]}
    with pytest.raises(ValueError):
        ex.extract_features(model, tok, TEXTS[:8], multi_dir, labels={"coherence": labels}, show_progress=False)
    ex.extract_features(model, tok, TEXTS[:8], multi_dir, batch_size=8, labels=multi, show_progress=False)
    assert sorted(json.load(open(os.path.join(multi_dir, ex.MANIFEST)))["label_sets"]) == ["coherence", "verbosity"]
    assert np.array_equal(ex.load_features(multi_dir, value="verbosity")[1], multi["verbosity"])
    assert ex.load_features(multi_dir)[1] is None


def test_empty_text_and_missing_shard_are_refused(model_tok, tmp_path):
    """An empty text raises with its index (it would be gathered at a pad column), a left-padded batch is refused by the
    collector, and resuming a partial directory with a missing layer shard gives an actionable error."""
    model, tok = model_tok
    with pytest.raises(ValueError, match="text 1 tokenizes to zero tokens"):
        ex.extract_features(model, tok, ["Hello world", ""], str(tmp_path / "empty"), show_progress=False)
    with pytest.raises(ValueError, match="zero tokens"):
        ex.tokenize_all(tok, [""])
    enc = tok(["Hello world", "Hi"], return_tensors="pt", padding=True)
    left = enc.attention_mask.flip(1)                                 # a left-padded mask (first column 0 for the short row)
    with ActivationCollector(model) as coll:
        coll.collect(enc.input_ids, enc.attention_mask)               # right padding: fine
        with pytest.raises(ValueError, match="right padding"):
            coll.collect(enc.input_ids.flip(1), left)
        with pytest.raises(ValueError):
            coll.collect(enc.input_ids[:, :0])

    class Stop(Exception):
        pass

    calls = {"n": 0}
    orig = ActivationCollector.collect

    def limited(self, *a, **k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise Stop()
        return orig(self, *a, **k)

    part = str(tmp_path / "partial")
    ActivationCollector.collect = limited
    try:
        with pytest.raises(Stop):
            ex.extract_features(model, tok, TEXTS, part, batch_size=4, flush_every=1, show_progress=False)
    finally:
        ActivationCollector.collect = orig
    assert not ex.is_done(part) and ex.read_manifest(part)["n_done"] > 0
    os.remove(ex.shard_path(part, 7))
    with pytest.raises(RuntimeError, match=r"missing layer shards \[7\]"):
        ex.extract_features(model, tok, TEXTS, part, batch_size=4, resume=True, show_progress=False)
    man = ex.extract_features(model, tok, TEXTS, part, batch_size=4, resume=False, show_progress=False)   # --force rebuilds
    assert ex.is_done(part) and man["n_done"] == len(TEXTS)


def test_resume_refuses_changed_batching_and_max_length_zero(model_tok, tmp_path, capsys):
    """Review finding F6: a partial directory resumed with another --batch_size / --max_tokens_per_batch mixed batch
    compositions (fp16 differences up to ~4e-3) and kept the old manifest values; both keys are now compatibility keys.
    Also: max_length 0 = no truncation (paper-mode tokenization) and the throughput / ETA line after `calibrate_after` texts."""
    model, tok = model_tok

    class Stop(Exception):
        pass

    calls = {"n": 0}
    orig = ActivationCollector.collect

    def limited(self, *a, **k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise Stop()
        return orig(self, *a, **k)

    part = str(tmp_path / "partial")
    ActivationCollector.collect = limited
    try:
        with pytest.raises(Stop):
            ex.extract_features(model, tok, TEXTS, part, batch_size=4, flush_every=1, show_progress=False)
    finally:
        ActivationCollector.collect = orig
    assert ex.read_manifest(part)["batch_size"] == 4 and not ex.is_done(part)
    with pytest.raises(RuntimeError, match="batch_size=4, now 16"):
        ex.extract_features(model, tok, TEXTS, part, batch_size=16, resume=True, show_progress=False)
    with pytest.raises(RuntimeError, match="max_tokens_per_batch"):
        ex.extract_features(model, tok, TEXTS, part, batch_size=4, max_tokens_per_batch=8192, resume=True, show_progress=False)
    assert not ex.is_done(part)
    man = ex.extract_features(model, tok, TEXTS, part, batch_size=4, resume=True, show_progress=False)       # same settings: resumes
    assert ex.is_done(man and part) and man["batch_size"] == 4 and man["max_tokens_per_batch"] == 16384
    # max_length 0 / None = no truncation
    ids, n_trunc = ex.tokenize_all(tok, TEXTS, max_length=0)
    assert n_trunc == 0 and ids[0] == tok(TEXTS[0])["input_ids"] and ex.tokenize_all(tok, TEXTS, max_length=None)[0] == ids
    man0 = ex.extract_features(model, tok, TEXTS[:5], str(tmp_path / "nolimit"), batch_size=8, max_length=0, show_progress=False)
    assert man0["max_length"] == 0 and man0["n_truncated"] == 0
    # throughput / ETA calibration after the first `calibrate_after` texts (DESIGN 6 / 8), with the budget warning
    capsys.readouterr()
    man_c = ex.extract_features(model, tok, TEXTS, str(tmp_path / "calib"), batch_size=4, calibrate_after=8, budget_seconds=0.0,
                                show_progress=False)
    out = capsys.readouterr().out
    assert man_c["calibration"]["after_texts"] >= 8 and man_c["calibration"]["tokens_per_second"] > 0
    assert "[extract] after" in out and "tok/s" in out and "warning: ETA exceeds" in out
