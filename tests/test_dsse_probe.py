"""DSSE side B (docs/DSSE_SIDEB.md section 9): synthetic rows only (make_synthetic_raw), random matrices, a tiny random
GPT-NeoX with a byte tokenizer, and one real-model collector check via the session fixture `model_tok`. No test prints
or checks dataset text; the synthetic rows are pseudo-words."""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.error
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from valueheads_pythia import cli, config
from valueheads_pythia import dsse_probe as D
from valueheads_pythia.commands import dsse_probe as C
from valueheads_pythia.extract import _pad_batch
from valueheads_pythia.models import decoder_blocks



# ----------------------------------------------------------------------------- helpers

def _pseudo_text(rng, n_words):
    return " ".join("".join(rng.choice(list("bdklmnprst")) + rng.choice(list("aeiou")) for _ in range(2)) for _ in range(n_words))


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("dsse_syn")
    info = D.make_synthetic_raw(str(out), n_datasets=40, n_behaviors=4, rows_per_set=24, seed=0)
    assert info["n"] == 40 and len(info["behaviors"]) == 4
    return str(out)


# ----------------------------------------------------------------------------- 1-2: sampling, cut rule

def test_sample_rows_fixed():
    a = D.sample_rows("dsse1-00042", 150, n_rows=16, seed=0)
    b = D.sample_rows("dsse1-00042", 150, n_rows=16, seed=0)
    assert np.array_equal(a, b) and len(a) == 16 and len(set(a.tolist())) == 16
    assert np.array_equal(a, np.sort(a)) and a.min() >= 0 and a.max() < 150
    assert not np.array_equal(a, D.sample_rows("dsse1-00042", 150, n_rows=16, seed=1))
    assert not np.array_equal(a, D.sample_rows("dsse1-00043", 150, n_rows=16, seed=0))
    assert np.array_equal(D.sample_rows("synth-00001", 10, n_rows=16), np.arange(10))     # n_available < n_rows -> all


def test_encode_row_cut_rule():
    tok = D.ByteTokenizer()
    ids, start, cp, cr = D.encode_row(tok, "abc", "defg")
    assert ids == [b + 2 for b in b"abc\ndefg"] and start == 4 and not cp and not cr
    ids, start, cp, cr = D.encode_row(tok, "p" * 400, "r" * 10)             # sum 411 <= 512: nothing is cut (F4)
    assert start == 401 and len(ids) == 411 and not cp and not cr
    ids, start, cp, cr = D.encode_row(tok, "p" * 400, "r" * 200)            # long prompt, short response -> prompt tail 256
    assert start == 256 and len(ids) == 456 and cp and not cr and ids[:256] == [ord("p") + 2] * 255 + [ord("\n") + 2]
    ids, start, cp, cr = D.encode_row(tok, "p" * 9, "r" * 1000)             # short prompt, long response -> head 512 - 10
    assert start == 10 and len(ids) == 512 and not cp and cr
    ids, start, cp, cr = D.encode_row(tok, "p" * 400, "r" * 1000)           # both long -> 256 + 256
    assert start == 256 and len(ids) == 512 and cp and cr
    ids, start, cp, cr = D.encode_row(tok, "abc", "")                       # empty response
    assert start == len(ids) == 4 and not cp and not cr
    ids2, start2, _, _ = D.encode_row(tok, "p" * 200, "r" * 300)            # exactly at the limit: no cut
    assert len(ids2) == 501 and start2 == 201


# ----------------------------------------------------------------------------- 3-4: collector against hidden states

def _rows_for_model(tokenizer, n=5, seed=0):
    rng = np.random.default_rng(seed)
    texts = [_pseudo_text(rng, k) for k in (4, 9, 14, 21, 30)][:n]
    ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    lengths = [len(x) for x in ids]
    starts = np.array([2, 0, lengths[2] // 2, lengths[3], 1][:n])        # row 3: zero response tokens
    input_ids, mask = _pad_batch(ids, list(range(n)), tokenizer.pad_token_id)
    return ids, lengths, starts, input_ids, mask


def test_collector_matches_hidden_states(model_tok):
    model, tokenizer = model_tok
    ids, lengths, starts, input_ids, mask = _rows_for_model(tokenizer)
    assert len(set(lengths)) == 5
    with D.ResidualCollector(model) as col:
        last, resp, count = col.collect(input_ids, mask, starts)
    L = model.config.num_hidden_layers
    assert last.shape == (5, L, model.config.hidden_size) and resp.shape == last.shape
    assert count.tolist() == [int(l - s) for l, s in zip(lengths, starts)]
    captured = {}
    h = decoder_blocks(model)[L - 1].register_forward_hook(lambda m, a, o: captured.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    finally:
        h.remove()
    hs = [out.hidden_states[l + 1].float() for l in range(L - 1)] + [captured["x"]]   # hidden_states[-1] is post final-LN
    for l in range(L):
        for r in range(5):
            ref_last = hs[l][r, lengths[r] - 1]
            assert torch.allclose(last[r, l], ref_last, atol=1e-5), (l, r)
            if count[r] > 0:
                ref_resp = hs[l][r, starts[r]:lengths[r]].mean(0)
                assert torch.allclose(resp[r, l], ref_resp, atol=1e-5), (l, r)
            else:
                assert float(resp[r, l].abs().max()) == 0.0


def test_collector_batch_vs_single(model_tok):
    model, tokenizer = model_tok
    ids, lengths, starts, input_ids, mask = _rows_for_model(tokenizer)
    with D.ResidualCollector(model) as col:
        last_b, resp_b, _ = col.collect(input_ids, mask, starts)
        single_ids, single_mask = _pad_batch(ids, [1], tokenizer.pad_token_id)
        last_s, resp_s, _ = col.collect(single_ids, single_mask, starts[1:2])
        tol = config.DTYPE_POLICY["tolerance_batched_vs_single"]
        for l in range(model.config.num_hidden_layers):
            for a, b in ((last_b[1, l], last_s[0, l]), (resp_b[1, l], resp_s[0, l])):
                assert float((a - b).norm() / b.norm()) <= tol, l
        left_ids = torch.flip(input_ids, [1])
        left_mask = torch.flip(mask, [1])
        with pytest.raises(ValueError):
            col.collect(left_ids, left_mask, starts)
        with pytest.raises(ValueError):
            col.collect(input_ids, mask, np.array(lengths) + 1)              # resp_start > length
        hole = mask.clone()
        hole[0, 1] = 0                                                       # interior hole: 1 0 1 1 ... (not right padding)
        with pytest.raises(ValueError, match="right-padded"):
            col.collect(input_ids, hole, starts)


# ----------------------------------------------------------------------------- 5-6: layer selection, extraction

def test_select_layers():
    assert D.select_layers(12, 768, 6915) == list(range(12))
    assert D.select_layers(24, 2048, 6915) == list(range(24))                 # 1.36 GB <= 1.5 GB
    sel = D.select_layers(32, 2560, 6915)
    assert sel == sorted(set(range(0, 32, 2)) | {31}) and len(sel) == 17 and 31 in sel
    assert D.select_layers(12, 768, 6915, stride="2") == [0, 2, 4, 6, 8, 10, 11]
    assert D.select_layers(32, 2560, 6915, stride="1") == list(range(32))
    assert D.select_layers(12, 768, 6915, cap_bytes=1e6) == [0, 2, 4, 6, 8, 10, 11]
    assert D.feature_bytes(6915, 17, 2560) == 6915 * 17 * 2560 * 4


def test_extract_synthetic_tiny(synthetic_dir, tmp_path):
    rows = D.build_rows(synthetic_dir, n_rows=8)
    assert len(rows.dataset_ids) == 40 and len(rows.rows) == 320 and list(rows.dataset_ids) == sorted(rows.dataset_ids)
    model, tok = D.tiny_model(seed=0)
    out = str(tmp_path / "tiny-gptneox_step0.npz")
    man = D.extract_dsse(model, tok, rows, out, batch_size=16, max_tokens_per_batch=4096, show_progress=False, say=lambda *a, **k: None)
    assert os.path.exists(out) and os.path.exists(D.manifest_path(out))
    meta = D.load_feature_meta(out)
    assert meta["layers"] == [0, 1, 2] and list(meta["dataset_ids"]) == list(rows.dataset_ids)
    assert (meta["n_rows_used"] == 8).all() and (meta["n_resp_rows"] == 8).all()
    for key in ("token_stats", "layers", "rows_sha256", "tiny", "version", "revision", "hidden"):
        assert key in man and man[key] == meta["manifest"][key]
    assert man["tiny"] is True and man["token_stats"]["n_rows"] == 320 and man["rows_sha256"] == rows.rows_sha256()
    with np.load(out) as z:
        keys = set(z.files)
        for l in range(3):
            for pre in ("last", "resp"):
                arr = z[f"{pre}_L{l:02d}"]
                assert arr.shape == (40, 64) and arr.dtype == np.float16
        assert {"dataset_ids", "layers", "n_rows_used", "n_resp_rows", "manifest"} <= keys
        assert z["layers"].dtype == np.int16
    X = D.load_feature_layer(out, 1, "resp_mean")
    assert X.shape == (40, 64) and X.dtype == np.float32
    # one dataset recomputed row by row with the collector
    i = 7
    mine = [(p, r) for j, p, r in rows.rows if j == i]
    assert len(mine) == 8
    acc_last, acc_resp = np.zeros((3, 64)), np.zeros((3, 64))
    with D.ResidualCollector(model) as col:
        for p, r in mine:
            ids, start, _, _ = D.encode_row(tok, p, r)
            input_ids, mask = _pad_batch([ids], [0], tok.pad_token_id)
            last, resp, count = col.collect(input_ids, mask, np.array([start]))
            acc_last += last[0].numpy()
            acc_resp += resp[0].numpy()
    with np.load(out) as z:
        for l in range(3):
            assert np.allclose(z[f"last_L{l:02d}"][i], acc_last[l] / 8, atol=2e-3, rtol=2e-3)
            assert np.allclose(z[f"resp_L{l:02d}"][i], acc_resp[l] / 8, atol=2e-3, rtol=2e-3)
    expect = {"version": D.VERSION, "n_rows": 8, "row_seed": 0, "max_length": 512, "prompt_cap": 256, "layers": [0, 1, 2],
              "hf_id": None, "revision": "main", "rows_sha256": rows.rows_sha256()}
    assert D.feature_done(out, expect) is True
    assert D.feature_done(str(tmp_path / "missing.npz"), expect) is False
    with pytest.raises(RuntimeError, match="n_rows"):
        D.feature_done(out, dict(expect, n_rows=16))


# ----------------------------------------------------------------------------- 7-10: splits, probes, bootstrap

def test_splits():
    rng = np.random.default_rng(0)
    behavior = np.array([f"b{i % 4}" for i in range(100)])
    method = np.array([["DE", "PT", "PT-noterse", "DPG"][rng.integers(0, 4)] for _ in range(100)])
    method[behavior == "b3"] = "DE"                                          # b3 has no group-B dataset
    splits = D.random_splits(behavior, 5, seed=1)
    tests = [set(s.test.tolist()) for s in splits]
    assert len(splits) == 5 and all(len(a & b) == 0 for i, a in enumerate(tests) for b in tests[i + 1:])
    assert set().union(*tests) == set(range(100))
    for s in splits:
        assert set(behavior[s.test]) == set(behavior) and set(s.train.tolist()).isdisjoint(s.test.tolist())
    ms = D.method_splits(behavior, method)
    assert [s.name for s in ms] == ["A2B", "B2A"] and D.shared_behaviors(behavior, method) == ["b0", "b1", "b2"]
    a2b, b2a = ms
    assert set(method[a2b.train]) <= {"DE", "PT"} and set(method[a2b.test]) <= {"PT-noterse", "DPG", "LLS"}
    assert "b3" not in set(behavior[a2b.train]) | set(behavior[a2b.test])
    shared_idx = set(np.flatnonzero(behavior != "b3").tolist())
    assert set(a2b.test.tolist()) | set(b2a.test.tolist()) == shared_idx and set(a2b.test.tolist()).isdisjoint(b2a.test.tolist())
    with pytest.raises(ValueError):
        D.method_splits(behavior, np.array(["DE"] * 100))


def _planted_classes(n=300, d=40, K=4, seed=0, scale=2.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    means = rng.normal(0, scale, (K, d))
    X = rng.normal(0, 1, (n, d)) + means[y]
    return X, y


def test_fit_eval_recovers_planted_signal():
    X, y = _planted_classes()
    behavior = np.array([f"b{k}" for k in y])
    splits = D.random_splits(behavior, 5, seed=0)
    res = D.fit_eval(X, y, np.ones(len(y), bool), splits, "behavior", n_pcs=16, seed=0, keep_proba=True)
    m = D.metrics("multiclass", res.target, res.pred, None, res.proba, res.tested)
    assert m["bal_acc"] > 0.9 and m["auc_macro"] > 0.95 and res.tested.all() and res.n_classes == 4
    yp = np.random.default_rng(1).permutation(y)
    res_p = D.fit_eval(X, yp, np.ones(len(y), bool), splits, "behavior", n_pcs=16, seed=0)
    assert D.metrics("multiclass", res_p.target, res_p.pred, None, tested=res_p.tested)["bal_acc"] < 0.45
    # variant within: the evaluated target is y minus the training mean of its behaviour
    rng = np.random.default_rng(2)
    offset = {f"b{k}": 10.0 * k for k in range(4)}
    X2 = rng.normal(0, 1, (len(y), 20))                                   # class-free features: the offsets are not decodable
    w = rng.normal(0, 1, X2.shape[1])
    yr = np.array([offset[b] for b in behavior]) + X2 @ w
    res_w = D.fit_eval(X2, yr, np.ones(len(yr), bool), splits, "elicitation", behavior=behavior, variant="within", n_pcs=None, seed=0)
    for sp in splits:
        means = {b: yr[sp.train][behavior[sp.train] == b].mean() for b in set(behavior)}
        expect = yr[sp.test] - np.array([means[b] for b in behavior[sp.test]])
        assert np.allclose(res_w.target[sp.test], expect)
    assert D.metrics("ridge", res_w.target, res_w.pred, None, tested=res_w.tested)["r2"] > 0.8
    res_r = D.fit_eval(X2, yr, np.ones(len(yr), bool), splits, "elicitation", variant="raw", n_pcs=None, seed=0)
    assert D.metrics("ridge", res_r.target, res_r.pred, None, tested=res_r.tested)["r2"] < 0.5    # raw: offsets dominate


def test_binary_and_ridge():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 20))
    w = rng.normal(0, 1, 20)
    y = X @ w + rng.normal(0, 0.3, 300)
    valid = np.ones(300, bool)
    valid[:15] = False
    yv = y.copy()
    yv[:15] = np.nan
    behavior = np.array([f"b{i % 3}" for i in range(300)])
    splits = D.random_splits(behavior, 5, seed=0)
    res = D.fit_eval(X, yv, valid, splits, "elicitation", n_pcs=None, seed=0)
    assert D.metrics("ridge", res.target, res.pred, None, tested=res.tested)["r2"] > 0.8
    assert not res.tested[:15].any() and np.isnan(res.pred[:15]).all() and res.tested[15:].all()
    yb = (X @ w > 0).astype(float)
    res_b = D.fit_eval(X, yb, valid, splits, "miss_sol", n_pcs=None, seed=0)
    m = D.metrics("binary", res_b.target, res_b.pred, res_b.score, tested=res_b.tested)
    assert m["auc"] > 0.9 and m["bal_acc"] > 0.8 and not res_b.tested[:15].any()
    assert D.primary_metric("binary", yb[15:], res_b.pred[15:], res_b.score[15:]) == pytest.approx(m["auc"])


def test_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    pred = y.copy()
    flip = rng.random(200) < 0.3
    pred[flip] = rng.integers(0, 4, int(flip.sum()))
    point, lo, hi = D.bootstrap_ci("multiclass", y, pred, None, B=300, seed=0)
    assert lo <= point <= hi and point == pytest.approx(D._bal_acc(y, pred))
    same = D.paired_bootstrap("multiclass", y, pred, None, pred, None, B=200, seed=0)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0 and same["p_boot"] == 1
    rand = rng.integers(0, 4, 200)
    better = D.paired_bootstrap("multiclass", y, pred, None, rand, None, B=300, seed=0)
    assert better["diff"] > 0.3 and better["lo"] > 0 and better["p_boot"] < 0.05
    yb = rng.integers(0, 2, 200).astype(float)
    s = yb + rng.normal(0, 0.5, 200)
    point, lo, hi = D.bootstrap_ci("binary", yb, (s > 0.5).astype(float), s, B=200, seed=0)
    assert lo <= point <= hi and 0.5 < point <= 1.0
    yr = rng.normal(0, 1, 200)
    point, lo, hi = D.bootstrap_ci("ridge", yr, yr + rng.normal(0, 0.5, 200), None, B=200, seed=0)
    assert lo <= point <= hi and point > 0.5


# ----------------------------------------------------------------------------- 11-12: angles, rank test

def test_direction_angles():
    rng = np.random.default_rng(0)
    n, d, K = 400, 200, 4
    y = rng.integers(0, K, n)
    X = rng.normal(0, 1, (n, d))
    classes = [f"b{k}" for k in range(K)]
    ana = D.angle_analysis(X, y, classes, d=d, perms=60, seed=0)
    null_mean = ana["null"]["mean_angle_null"]
    assert np.percentile(null_mean, 2.5) - 3 <= ana["mean_angle_massmean"] <= np.percentile(null_mean, 97.5) + 3
    assert ana["n_aligned"] == 0 and ana["n_opposed"] == 0 and ana["n_pairs"] == 6
    n1 = D.n1_reference(d)
    assert n1["q500"] == pytest.approx(90.0, abs=0.01) and n1["q025"] < 90.0 < n1["q975"]
    u = rng.normal(0, 1, d)
    u /= np.linalg.norm(u)
    Xp = X.copy()
    Xp[y == 0] += 3 * u
    Xp[y == 1] += 3 * u
    Xp[y == 2] -= 3 * u
    splits = D.random_splits(np.array([f"b{k}" for k in y]), 5, seed=0)
    anp = D.angle_analysis(Xp, y, classes, d=d, perms=60, seed=0, splits=splits)
    assert anp["n_folds"] == 5 and anp["massmean_folds"].shape == (5, K, K) and len(anp["fold_spearman"]) == 5
    assert min(anp["fold_spearman"]) > 0.7 and 0 <= anp["fold_pair_sd_mean"] < 15 and np.isfinite(anp["fold_mean_angle"])
    assert anp["massmean_folds"][:, 0, 1].max() < anp["massmean_folds"][:, 0, 2].min()   # held-out: aligned pair stays below the opposed pair
    assert ana["n_folds"] == 0 and ana["massmean_folds"] is None and ana["fold_spearman"] == []
    flag, q025 = anp["null"]["flag"], anp["null"]["q025"]
    assert anp["massmean"][0, 1] < q025[0, 1] and flag[0, 1] == -1          # aligned
    assert anp["massmean"][0, 2] > anp["null"]["q975"][0, 2] and flag[0, 2] == 1   # opposed
    assert "b1" in [c for c, _ in anp["nearest"]["b0"]] and anp["merges"][0][0] < 45
    assert D.effective_rank(np.eye(4, 50)) == pytest.approx(4.0, abs=0.05)
    assert D.effective_rank(np.tile(rng.normal(0, 1, 50), (4, 1))) == pytest.approx(1.0, abs=1e-6)
    A = D.angle_matrix(np.eye(3, 10))
    assert A.shape == (3, 3) and np.allclose(np.diag(A), 0) and A[0, 1] == pytest.approx(90.0)


def test_rank_test():
    rng = np.random.default_rng(0)
    n, d, K = 400, 50, 4
    y = rng.integers(0, K, n)
    W = rng.normal(0, 1, (3, d))
    W /= np.linalg.norm(W, axis=1, keepdims=True)                           # 3 random directions (correlated dims,
    S = np.zeros((K, 3))                                                     # so the per-dim standardization keeps them)
    S[1, 0], S[2, 1], S[3, 2] = 8.0, 8.0, 8.0                                # 3-dim planted signal
    X = rng.normal(0, 1, (n, d)) + S[y] @ W
    splits = D.random_splits(np.array([f"b{k}" for k in y]), 5, seed=0)
    df = D.rank_test(X, y, np.ones(n, bool), splits, ks=(1, 2, 4, 8, 16, 32), seed=0, boot=50)
    assert list(df.columns) == ["k", "bal_acc", "ci_lo", "ci_hi", "n_test"] and df.attrs["k95"] <= 4
    upto = df[df.k <= df.attrs["k95"]].bal_acc.to_numpy()
    assert (np.diff(upto) >= -0.02).all() and df.bal_acc.iloc[-1] > 0.9


# ----------------------------------------------------------------------------- 13-15: raw files, synthetic layout, baselines

def _manifest_of(raw_dir, behaviors):
    files = {}
    for path in ["scores.csv", "behaviors.json"] + [f"datasets/{b}.jsonl.xz" for b in behaviors]:
        local = D.raw_file(raw_dir, path)
        files[path] = (os.path.getsize(local), D.file_sha256(local))
    return files


def test_verify_raw(synthetic_dir, tmp_path):
    behaviors = D.load_behaviors(synthetic_dir)
    files = _manifest_of(synthetic_dir, behaviors)
    status = D.verify_raw(synthetic_dir, files)
    assert all(s["ok"] for s in status.values()) and len(status) == 2 + len(behaviors)
    copy = str(tmp_path / "copy")
    shutil.copytree(synthetic_dir, copy)
    with open(os.path.join(copy, "scores.csv"), "r+b") as f:
        f.seek(10)
        c = f.read(1)
        f.seek(10)
        f.write(b"X" if c != b"X" else b"Y")
    with pytest.raises(ValueError, match="scores.csv"):
        D.verify_raw(copy, files)
    assert D.verify_raw(copy, files, strict=False)["scores.csv"]["ok"] is False
    os.remove(os.path.join(copy, "behaviors.json"))
    with pytest.raises(FileNotFoundError):
        D.verify_raw(copy, files)
    flat = str(tmp_path / "flat")
    shutil.copytree(synthetic_dir, flat)
    b0 = behaviors[0]
    os.replace(os.path.join(flat, "datasets", f"{b0}.jsonl.xz"), os.path.join(flat, f"datasets_{b0}.jsonl.xz"))
    assert D.raw_file(flat, f"datasets/{b0}.jsonl.xz").endswith(f"datasets_{b0}.jsonl.xz")
    assert D.verify_raw(flat, files)[f"datasets/{b0}.jsonl.xz"]["ok"]
    assert len(D.build_rows(flat, n_rows=4).dataset_ids) == 40


def test_synthetic_layout(synthetic_dir):
    df = pd.read_csv(os.path.join(synthetic_dir, "scores.csv"))
    assert list(df.columns) == D.SCORE_COLUMNS and len(df) == 40
    with open(os.path.join(synthetic_dir, "behaviors.json")) as f:
        beh = json.load(f)
    assert isinstance(beh, list) and all("behavior_id" in b for b in beh) and len(beh) == 4
    scores = D.load_scores(synthetic_dir)
    rows = D.build_rows(synthetic_dir, n_rows=8)
    assert list(scores.index) == list(rows.dataset_ids) and len(rows.rows) == 320
    assert set(scores.mc_missed_sol.unique()) == {0.0, 1.0} and set(scores.method) == {"DE", "PT-noterse"}
    assert D.shared_behaviors(rows.behavior, rows.method) == sorted(set(rows.behavior))
    # planted signal: marker words (end with 'q') -> elicitation score
    density = np.zeros(40)
    for i, p, r in rows.rows:
        words = r.split()
        density[i] += sum(1 for w in words if w.rstrip(".,!").endswith("q")) / max(1, len(words))
    density /= 8
    assert np.corrcoef(density, scores.qwen3_1_7b_opus.to_numpy())[0, 1] > 0.5
    assert all(D.SMOKE_MARKER in r for _, _, r in rows.rows)
    docs = D.docs_per_dataset(rows)
    assert len(docs) == 40 and all(d.count("\n") == 15 for d in docs)


def test_style_metadata_features(synthetic_dir):
    scores = D.load_scores(synthetic_dir)
    rows = D.build_rows(synthetic_dir, n_rows=8)
    S = D.style_features(rows)
    assert S.shape == (40, 10) and np.isfinite(S).all() and (S[:, 0] == 8).all() and len(D.STYLE_NAMES) == 10
    de = rows.method == "DE"
    assert S[de, 2].mean() > S[~de, 2].mean()                                # DE responses are longer
    classes = D.behavior_classes(scores)
    M, names = D.metadata_features(scores, classes)
    assert M.shape == (40, 11 + len(classes) + 5) and len(names) == M.shape[1] and np.isfinite(M).all()
    assert names[:11] == D.METADATA_COLUMNS and not any("missed" in n or "surprising" in n for n in names)
    y, valid = D.task_target(scores, "behavior")
    assert valid.all() and set(y.tolist()) == {0, 1, 2, 3}
    y, valid = D.task_target(scores, "miss_sol")
    assert valid.all() and set(y.tolist()) == {0.0, 1.0}


# ----------------------------------------------------------------------------- 16-18: CLI chain, no text leak, budget / verify

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Full CLI chain on synthetic data with the tiny model, every stage in a subprocess with captured stdout + stderr
    (progress bar on, no --quiet) so that the leak test can scan the console output as well as the files."""
    root = tmp_path_factory.mktemp("dsse_root")
    raw = str(root / "raw")
    saved = os.environ.pop("VH_FEATURES", None)              # an inherited feature dir would move the npz out of `root`
    env = {k: v for k, v in os.environ.items() if k != "VH_FEATURES"}
    common = ["--root", str(root), "--raw_dir", raw]
    log = []

    def run(*args):
        r = subprocess.run([sys.executable, "-m", "valueheads_pythia.cli", "dsse-probe", *args], capture_output=True, text=True,
                           env=env, cwd=REPO_ROOT)
        log.append(r.stdout + r.stderr)
        assert r.returncode == 0, (args, r.stdout[-2000:], r.stderr[-2000:])

    run("--stage", "synthetic", "--synthetic_out", raw, "--n_synthetic", "40")
    run("--stage", "extract", "--tiny", "--revisions", "step0", "--n_rows", "8", "--device", "cpu", *common)
    run("--stage", "baselines", "--n_rows", "8", "--boot", "50", "--perms", "20", *common)
    probe = ["dsse-probe", "--stage", "probe", "--tiny", "--revisions", "step0", "--boot", "50", "--perms", "20", "--n_pcs", "16",
             "--n_jobs", "1", *common]
    run(*probe[1:])
    run("--stage", "report", "--note", "SMOKE", *common)
    yield {"root": str(root), "raw": raw, "probe": probe, "log": "\n".join(log)}
    if saved is not None:
        os.environ["VH_FEATURES"] = saved


def test_cli_end_to_end_tiny(tiny_run, capsys):
    root, raw = tiny_run["root"], tiny_run["raw"]
    p = config.paths(root)
    feat = D.feature_path(p, "tiny-gptneox", "step0")
    assert os.path.exists(feat) and os.path.exists(D.manifest_path(feat))
    out = p["dsse"]
    for fn in ("tiny-gptneox_step0_probe.csv", "tiny-gptneox_step0_probe.json", "tiny-gptneox_step0_oof.npz",
               "tiny-gptneox_step0_angles.npz", "tiny-gptneox_step0_angles.csv", "tiny-gptneox_step0_angles.json",
               "tiny-gptneox_step0_rank.csv", "baselines_probe.csv", "baselines_oof.npz", "baselines_angles.npz",
               "baselines_rank.csv", "report.md", "key_table.csv"):
        assert os.path.exists(os.path.join(out, fn)), fn
    for fig in ("layer_curves_behavior.png", "angles_tiny-gptneox.png", "rank_tiny-gptneox.png", "summary_bars.png"):
        assert os.path.exists(os.path.join(out, "figures", fig)), fig
    report = open(os.path.join(out, "report.md")).read()
    assert "SMOKE" in report and "## Kerntabelle" in report and "tiny-gptneox" in report
    df = pd.read_csv(os.path.join(out, "tiny-gptneox_step0_probe.csv"))
    assert set(df.split) == {"random", "method"} and set(df.task) == set(D.TASKS) and set(df.pooling) == set(D.POOLINGS)
    assert set(df.variant) == {"raw", "within"} and (df.version == D.VERSION).all()
    pooled = df[(df.direction == "pooled") & (df.metric == "bal_acc") & (df.task == "behavior")]
    assert pooled.ci_lo.notna().all() and (pooled.chance == 0.25).all() and len(pooled) == 3 * 2 * 2
    bdf = pd.read_csv(os.path.join(out, "baselines_probe.csv"))
    assert set(bdf.features) == set(D.BASELINE_FEATURES) and set(bdf[bdf.features == "metadata"].task) == {"miss_sol"}
    tf = bdf[(bdf.features == "tfidf_svd128") & (bdf.task == "behavior") & (bdf.split == "random") & (bdf.direction == "pooled") & (bdf.metric == "bal_acc")]
    assert float(tf.value.iloc[0]) > 0.6                                     # lexical markers are easy
    with open(os.path.join(out, "tiny-gptneox_step0_probe.json")) as f:
        info = json.load(f)
    assert info["best"]["behavior"] is not None and info["rank"]["k95"] <= 16 and info["angles"]["n_pairs"] == 6
    assert info["angles"]["n_folds"] == 5 and len(info["angles"]["fold_spearman"]) == 5 and "in-sample" in info["angles"]["directions_fit"]
    with np.load(os.path.join(out, "tiny-gptneox_step0_angles.npz")) as z:
        assert z["massmean_folds"].shape == (5, 4, 4) and z["massmean"].shape == (4, 4)
    key = pd.read_csv(os.path.join(out, "key_table.csv"))
    assert bool(key.baselines_paired.iloc[0]) is True and key.d_tfidf.notna().all() and key.pairing_note.isna().all()
    assert "Fold-Stabilitaet" in report and "in-sample" in report and "WARNUNG" not in report
    capsys.readouterr()
    assert cli.main([a for a in tiny_run["probe"] if a != "--quiet"]) == 0            # second call: fingerprint skip
    assert "skip" in capsys.readouterr().out
    assert cli.main(tiny_run["probe"] + ["--force"]) == 0
    assert "skip" not in capsys.readouterr().out


def test_no_text_in_outputs(tiny_run):
    """No file under the run root (outside raw/) and no console output of any stage carries row text: the marker syllable
    that every synthetic prompt and response contains, and the first 12 characters of a prompt and of a response that were
    really sampled (rows.rows[0] is a sampled row, unlike row 0 of the file)."""
    root, raw = tiny_run["root"], tiny_run["raw"]
    rows = D.build_rows(raw, n_rows=8)
    _, prompt0, resp0 = rows.rows[0]
    sentinels = [D.SMOKE_MARKER, prompt0[:12], resp0[:12]]
    assert all(len(s) == 12 for s in sentinels[1:]) and D.SMOKE_MARKER in prompt0.lower() and D.SMOKE_MARKER in resp0.lower()
    docs = "\n".join(D.docs_per_dataset(rows))
    assert all(s in docs for s in sentinels)                                  # the sentinels are in the text the stages saw
    checked = 0
    for dirpath, _, files in os.walk(root):
        if os.path.abspath(dirpath).startswith(os.path.abspath(raw)):
            continue
        for fn in files:
            path = os.path.join(dirpath, fn)
            if fn.endswith((".csv", ".md", ".json", ".txt", ".log")):
                text = open(path, encoding="utf-8").read()
            elif fn.endswith(".npz"):
                with np.load(path) as z:
                    text = " ".join(str(z[k].tolist()) for k in z.files if z[k].dtype.kind in "US")
            elif fn.endswith(".png"):
                text = open(path, "rb").read().decode("latin-1")
            else:
                continue
            checked += 1
            for s in sentinels:
                assert s not in text, (path, s)
    assert checked >= 12
    for s in sentinels:                                                        # progress bar, say lines, warnings, tracebacks
        assert s not in tiny_run["log"], s
    assert "dsse tiny-gptneox_step0" in tiny_run["log"]                        # the progress bar really was on


def test_token_budget_and_cli_verify(synthetic_dir, tmp_path, capsys):
    tok = D.ByteTokenizer()
    stats = D.token_budget(synthetic_dir, tokenizer=tok, n_rows=8)
    assert stats["n_rows"] == 320 and stats["n_datasets"] == 40 and stats["n_tokens"] > 0
    rows = D.build_rows(synthetic_dir, n_rows=8)
    n_cut = sum(1 for _, p, r in rows.rows if any(D.encode_row(tok, p, r)[2:]))
    assert stats["n_cut"] == n_cut and stats["cut_rate"] == pytest.approx(n_cut / 320)
    assert stats["n_tokens"] == sum(len(D.encode_row(tok, p, r)[0]) for _, p, r in rows.rows)
    bad = str(tmp_path / "bad_raw")
    shutil.copytree(synthetic_dir, bad)
    assert cli.main(["dsse-probe", "--stage", "verify", "--raw_dir", bad, "--quiet"]) == 1   # not the DSSE manifest
    capsys.readouterr()


# ----------------------------------------------------------------------------- 19-22: review fixes

def test_project_splits_tfidf_width():
    """The TF-IDF baseline is TruncatedSVD(TFIDF_SVD = 128) whatever n_pcs (the activation PCA width) says (F20)."""
    rng = np.random.default_rng(0)
    vocab = [_pseudo_text(rng, 1) for _ in range(300)]
    docs = [" ".join(rng.choice(vocab, 30)) for _ in range(300)]
    y = rng.integers(0, 4, 300)
    splits = D.random_splits(np.array([f"b{k}" for k in y]), 5, seed=0)
    for n_pcs in (D.N_PCS, 16, None):
        proj = D.project_splits(docs, splits, n_pcs=n_pcs)
        assert {z[0].shape[1] for z in proj.values()} == {D.TFIDF_SVD} and {z[1].shape[1] for z in proj.values()} == {D.TFIDF_SVD}
    X = rng.normal(0, 1, (300, 50))
    assert {z[0].shape[1] for z in D.project_splits(X, splits, n_pcs=16).values()} == {16}
    assert {z[0].shape[1] for z in D.project_splits(X, splits, n_pcs=None).values()} == {50}
    few = D.random_splits(np.array([f"b{k}" for k in y[:40]]), 5, seed=0)
    assert max(z[0].shape[1] for z in D.project_splits(docs[:40], few, n_pcs=D.N_PCS).values()) <= 31     # n_train - 1


def test_collector_rejects_mask_holes():
    model, tok = D.tiny_model(seed=0)
    with D.ResidualCollector(model) as col:
        for mask in ([[1, 0, 1, 1]], [[1, 1, 2, 0]], [[0, 1, 1, 1]], [[0, 0, 0, 0]]):
            with pytest.raises(ValueError):
                col.collect([[5, 6, 7, 8]], mask, [0])
        last, resp, count = col.collect([[5, 6, 7, 8], [5, 6, 0, 0]], [[1, 1, 1, 1], [1, 1, 0, 0]], [1, 0])
        assert last.shape == (2, 3, 64) and count.tolist() == [3, 2]


class _FakeResponse:
    def __init__(self, data):
        self.buf = io.BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self.buf.read(n)


def test_download_timeout_retry_and_cli_exit(tmp_path, monkeypatch, capsys):
    payload = b"dataset_id,behavior_id\n" * 50
    files = {"scores.csv": (len(payload), hashlib.sha256(payload).hexdigest())}
    calls = []

    def flaky(req, timeout=None):                                            # first attempt: network failure, then ok
        calls.append((req.full_url, timeout))
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset")
        return _FakeResponse(payload)

    monkeypatch.setattr(D.urllib.request, "urlopen", flaky)
    raw = str(tmp_path / "raw")
    status = D.download_raw(raw, files=files, say=lambda *a, **k: None)
    assert status["scores.csv"]["ok"] and len(calls) == 2 and all(t == D.DOWNLOAD_TIMEOUT for _, t in calls)
    assert calls[0][0] == D.DSSE_URL.format(revision="main", path="scores.csv") and os.listdir(raw) == ["scores.csv"]

    def not_found(req, timeout=None):                                        # a 4xx (wrong revision) is not retried
        calls.append(("404", timeout))
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(D.urllib.request, "urlopen", not_found)
    raw2 = str(tmp_path / "raw2")
    with pytest.raises(urllib.error.HTTPError):
        D.download_raw(raw2, files=files, say=lambda *a, **k: None)
    assert sum(1 for c in calls if c[0] == "404") == 1 and not os.path.exists(os.path.join(raw2, "scores.csv.part"))
    assert cli.main(["dsse-probe", "--stage", "download", "--raw_dir", raw2, "--dsse_revision", "0" * 40, "--quiet"]) == 1
    assert "download failed: HTTPError" in capsys.readouterr().out

    def stalled(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(D.urllib.request, "urlopen", stalled)
    assert cli.main(["dsse-probe", "--stage", "download", "--raw_dir", str(tmp_path / "raw3"), "--quiet"]) == 1
    assert "download failed: TimeoutError" in capsys.readouterr().out


def test_report_suspends_unpaired_baselines(tiny_run, tmp_path, capsys):
    """Baselines built from another row sample (rows_sha256) than the feature file: no paired deltas, verdict 'nicht
    pruefbar', warning in report.md and on the CLI; the baselines stage warns about a differing --n_rows up front."""
    root, raw = tiny_run["root"], tiny_run["raw"]
    copy = str(tmp_path / "root_copy")
    shutil.copytree(root, copy)
    p = config.paths(copy)
    bj_path = os.path.join(p["dsse"], "baselines_probe.json")
    with open(bj_path) as f:
        bj = json.load(f)
    bj["rows_sha256"], bj["n_rows"] = "0" * 64, 4
    with open(bj_path, "w") as f:
        json.dump(bj, f)
    assert cli.main(["dsse-probe", "--stage", "report", "--root", copy, "--raw_dir", raw, "--boot", "50"]) == 0
    out = capsys.readouterr().out
    assert "WARNING: paired comparisons suspended" in out
    key = pd.read_csv(os.path.join(p["dsse"], "key_table.csv"))
    assert bool(key.baselines_paired.iloc[0]) is False and key.d_tfidf.isna().all() and key.d_miss_meta.isna().all()
    assert key.d_elicitation_raw_tfidf.isna().all() and "anderer Zeilenstichprobe" in key.pairing_note.iloc[0]
    assert key.behavior_method.notna().all()                                  # the unpaired metrics stay
    report = open(os.path.join(p["dsse"], "report.md")).read()
    assert "WARNUNG: gepaarte Vergleiche ausgesetzt" in report and "!= 000000000000 (n_rows 4)" in report
    results = D.collect_results(p["dsse"])
    k2 = D.key_table(results, D.make_oof_loader(p["dsse"]), boot=20, seed=0)
    assert k2.verdict.iloc[0] == "Referenz (step0)"                          # step0 stays the reference ...
    results["json"][("tiny-gptneox", "step0")]["revision"] = "step143000"     # ... a trained row becomes 'nicht pruefbar'
    k3 = D.key_table({**results, "json": {("tiny-gptneox", "step143000"): results["json"][("tiny-gptneox", "step0")]}},
                     D.make_oof_loader(p["dsse"]), boot=20, seed=0)
    assert k3.verdict.iloc[0].startswith("nicht pruefbar (Baselines auf anderer Zeilenstichprobe")
    assert D.same_rows({"manifest": {"rows_sha256": "a" * 64}}, {"rows_sha256": "a" * 64}) == (True, "")
    assert D.same_rows({"manifest": {}}, {"rows_sha256": "a" * 64})[0] is True   # unknown hash: not a mismatch
    assert D.same_ids({"ids": np.array(["a", "b"])}, {"ids": np.array(["a", "c"])})[0] is False
    C._warn_n_rows(SimpleNamespace(n_rows=4), p)
    assert "WARNING: tiny-gptneox_step0.manifest.json was extracted with n_rows = 8" in capsys.readouterr().out
    C._warn_n_rows(SimpleNamespace(n_rows=8), p)
    assert capsys.readouterr().out == ""


# ----------------------------------------------------------------------------- 23-26: second review


def test_staged_publish_and_recovery(synthetic_dir, tmp_path, monkeypatch, capsys):
    """A failed move to the target keeps the staged npz + manifest; recover_staged publishes them without recomputation;
    mismatching or unreadable staged files are discarded."""
    model, tok = D.tiny_model(seed=0)
    rows = D.build_rows(synthetic_dir, n_rows=4)
    stage = str(tmp_path / "stage")
    target = str(tmp_path / "features" / "tiny-gptneox_step0.npz")
    expect = {"version": D.VERSION, "n_rows": 4, "row_seed": D.ROW_SEED, "max_length": 64, "prompt_cap": 32, "layers": [0, 2],
              "hf_id": None, "revision": "step0", "rows_sha256": rows.rows_sha256()}
    calls = {"n": 0}

    def failing_move(src, dst):
        calls["n"] += 1
        open(dst, "wb").write(b"partial")                   # half-copied target, as a Drive quota error leaves it
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(D.shutil, "move", failing_move)
    model.vh_info["revision"] = "step0"
    with pytest.raises(OSError, match="recovered by the next"):
        D.extract_dsse(model, tok, rows, target, layers=[0, 2], batch_size=8, max_tokens_per_batch=512, max_length=64,
                       prompt_cap=32, stage_dir=stage, show_progress=False, say=lambda *a, **k: None)
    tmp = os.path.join(stage, "tiny-gptneox_step0.tmp.npz")
    assert calls["n"] == 1 and os.path.exists(tmp) and os.path.exists(D.staged_manifest_path(tmp))
    assert not os.path.exists(target) and not os.path.exists(D.manifest_path(target))   # partial target removed
    monkeypatch.undo()
    assert D.feature_done(target, expect) is False
    assert D.recover_staged(stage, target, expect, say=print) is True
    out = capsys.readouterr().out
    assert "recovering staged" in out and "40 datasets" in out
    assert os.path.exists(target) and os.path.exists(D.manifest_path(target))
    assert not os.path.exists(tmp) and not os.path.exists(D.staged_manifest_path(tmp))
    assert D.feature_done(target, expect) is True
    meta = D.load_feature_meta(target)
    assert meta["manifest"]["max_length"] == 64 and meta["layers"] == [0, 2] and len(meta["dataset_ids"]) == 40
    assert D.recover_staged(stage, target, expect) is False               # nothing staged any more
    # mismatching staged pair -> discarded
    shutil.copyfile(target, tmp)
    shutil.copyfile(D.manifest_path(target), D.staged_manifest_path(tmp))
    assert D.recover_staged(stage, target, {**expect, "max_length": 128}, say=print) is False
    assert "discarded" in capsys.readouterr().out and not os.path.exists(tmp)
    # unreadable staged npz -> discarded
    open(tmp, "wb").write(b"garbage")
    shutil.copyfile(D.manifest_path(target), D.staged_manifest_path(tmp))
    assert D.recover_staged(stage, target, expect, say=print) is False
    assert "unreadable" in capsys.readouterr().out and not os.path.exists(tmp)
    # CLI: a staged pair is recovered instead of re-extracted (no model load: the say line names the recovery)
    os.remove(target)
    os.remove(D.manifest_path(target))
    p = config.paths(str(tmp_path))
    cli_target = D.feature_path(p, "tiny-gptneox", "step0")
    os.makedirs(os.path.dirname(cli_target), exist_ok=True)
    D.extract_dsse(model, tok, rows, cli_target, layers=[0, 2], batch_size=8, max_tokens_per_batch=512, max_length=64, prompt_cap=32,
                   stage_dir=stage, show_progress=False, say=lambda *a, **k: None)
    shutil.move(cli_target, os.path.join(stage, "tiny-gptneox_step0.tmp.npz"))
    shutil.move(D.manifest_path(cli_target), D.staged_manifest_path(os.path.join(stage, "tiny-gptneox_step0.tmp.npz")))
    assert cli.main(["dsse-probe", "--stage", "extract", "--tiny", "--revisions", "step0", "--n_rows", "4", "--max_length", "64",
                     "--prompt_cap", "32", "--layer_stride", "2", "--device", "cpu", "--no_progress", "--stage_dir", stage,
                     "--root", str(tmp_path), "--raw_dir", synthetic_dir]) == 0
    out = capsys.readouterr().out
    assert "recovering staged" in out and "model tiny" not in out and D.feature_done(cli_target, expect)


def test_probe_fingerprint_follows_reextraction(synthetic_dir, tmp_path, capsys):
    """--stage extract --force with other settings changes the probe fingerprint: the next --stage probe re-runs instead
    of skipping; a feature file without sidecar is skipped by the glob."""
    root = str(tmp_path)
    common = ["--root", root, "--raw_dir", synthetic_dir, "--quiet"]
    probe = ["dsse-probe", "--stage", "probe", "--tiny", "--revisions", "step0", "--boot", "20", "--perms", "10", "--n_pcs", "8",
             "--n_jobs", "1", "--root", root, "--raw_dir", synthetic_dir]
    assert cli.main(["dsse-probe", "--stage", "extract", "--tiny", "--revisions", "step0", "--n_rows", "4", "--device", "cpu",
                     "--no_progress", *common]) == 0
    assert cli.main(probe) == 0
    capsys.readouterr()
    assert cli.main(probe) == 0
    assert "skip" in capsys.readouterr().out                                   # unchanged features -> skip
    assert cli.main(["dsse-probe", "--stage", "extract", "--tiny", "--revisions", "step0", "--n_rows", "4", "--device", "cpu",
                     "--no_progress", "--max_length", "64", "--prompt_cap", "32", "--force", *common]) == 0
    capsys.readouterr()
    assert cli.main(probe) == 0
    out = capsys.readouterr().out
    assert "skip" not in out
    p = config.paths(root)
    with open(os.path.join(p["dsse"], "tiny-gptneox_step0_probe.json")) as f:
        assert json.load(f)["manifest"]["max_length"] == 64
    # glob mode: a stray npz without sidecar is skipped, the real file is (fingerprint-)skipped, exit 0
    junk = os.path.join(p["features"], "stray_step0.npz")
    np.savez(junk, x=np.zeros(3))
    assert cli.main(["dsse-probe", "--stage", "probe", "--boot", "20", "--perms", "10", "--n_pcs", "8", "--n_jobs", "1",
                     "--root", root, "--raw_dir", synthetic_dir]) == 0
    out = capsys.readouterr().out
    assert "stray_step0.npz" in out and "without manifest" in out and "skip " in out


def test_raw_dir_never_on_drive(tmp_path, capsys):
    """stage download refuses a Google-Drive target; the default raw_dir leaves a Drive root for the local disk."""
    assert C.on_drive("/content/drive/MyDrive/ValueHeadsPythia/dsse/dsse_raw") and not C.on_drive("/content/dsse_raw")
    assert cli.main(["dsse-probe", "--stage", "download", "--raw_dir", "/content/drive/MyDrive/x/dsse_raw", "--root", str(tmp_path)]) == 1
    assert "refusing" in capsys.readouterr().out
    args = SimpleNamespace(raw_dir=None)
    os.environ.pop("VH_DSSE_RAW", None)
    local = C.raw_dir_of(args, {"root": str(tmp_path)})
    assert local == os.path.join(str(tmp_path), "dsse_raw")
    drive_root = {"root": "/content/drive/MyDrive/ValueHeadsPythia/dsse"}
    expected = "/content/dsse_raw" if os.path.isdir("/content") else os.path.join(drive_root["root"], "dsse_raw")
    assert C.raw_dir_of(args, drive_root) == expected
    assert C.raw_dir_of(SimpleNamespace(raw_dir="/x/raw"), drive_root) == "/x/raw"


def test_collector_runs_decoder_only(model_tok):
    """ResidualCollector.collect calls the decoder (gpt_neox), not the LM head: identical activations, no logits needed."""
    model, tok = model_tok
    enc = tok(["one two three", "four"], return_tensors="pt", padding=True)
    calls = {"lm": 0}
    orig = model.forward

    def counting_forward(*a, **k):
        calls["lm"] += 1
        return orig(*a, **k)

    model.forward = counting_forward
    try:
        with D.ResidualCollector(model, layers=[0, 1]) as col:
            last, resp, count = col.collect(enc["input_ids"], enc["attention_mask"], np.array([1, 0]))
    finally:
        model.forward = orig
    assert calls["lm"] == 0 and last.shape == (2, 2, model.config.hidden_size) and (count.numpy() == [2, 1]).all()
    ref = model(**enc, output_hidden_states=True).hidden_states
    for slot, l in enumerate([0, 1]):
        assert torch.allclose(last[:, slot], ref[l + 1][torch.arange(2), enc["attention_mask"].sum(1) - 1].float(), atol=1e-4)


# ----------------------------------------------------------------------------- 27-30: trusted weight source (Hub branches with main's weights)

from valueheads_pythia import models as M


def test_choose_weight_files_and_fingerprint():
    single = [("model.safetensors", "a" * 64), ("pytorch_model.bin", "b" * 64)]
    shards = [("model-00001-of-00002.safetensors", "c" * 64), ("model-00002-of-00002.safetensors", "d" * 64), ("model.safetensors.index.json", None)]
    used, amb = M.choose_weight_files(sorted(single))
    assert [p for p, _ in used] == ["model.safetensors"] and amb is False
    used, amb = M.choose_weight_files(sorted(shards + [("pytorch_model.bin", "e" * 64)]))
    assert [p for p, _ in used] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] and amb is False
    used, amb = M.choose_weight_files(sorted(single + shards))                 # the pythia-2.8b step0 layout: shards win
    assert [p for p, _ in used] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] and amb is True
    used, amb = M.choose_weight_files([("pytorch_model.bin", "b" * 64)])
    assert [p for p, _ in used] == ["pytorch_model.bin"] and amb is False
    assert M.choose_weight_files(None) == (None, False) and M.weight_fingerprint(None) is None
    f1 = M.weight_fingerprint([("model-00002-of-00002.safetensors", "d" * 64), ("model-00001-of-00002.safetensors", "c" * 64)])
    f2 = M.weight_fingerprint([("model-00001-of-00002.safetensors", "c" * 64), ("model-00002-of-00002.safetensors", "d" * 64)])
    assert f1 == f2 and len(f1) == 64 and f1 != M.weight_fingerprint(single)
    assert M.weight_fingerprint([("model.safetensors", None)]) is None       # no LFS id -> no fingerprint


def test_manifest_fingerprint_tolerance_and_stale_detection():
    expect = {"version": D.VERSION, "n_rows": 4, "weight_fingerprint": "f" * 64}
    old = {"version": D.VERSION, "n_rows": 4}                                  # file from before the check
    assert D.manifest_mismatch(old, expect) == []                              # tolerated ...
    assert D.stale_weight_source(old, ambiguous=False) is False                # ... and valid from an unambiguous branch
    assert D.stale_weight_source(old, ambiguous=True) is True                  # ... but rebuilt from an ambiguous one
    new_same = {"version": D.VERSION, "n_rows": 4, "weight_fingerprint": "f" * 64}
    new_other = {"version": D.VERSION, "n_rows": 4, "weight_fingerprint": "0" * 64}
    assert D.manifest_mismatch(new_same, expect) == [] and D.manifest_mismatch(new_other, expect) == ["weight_fingerprint"]
    assert D.stale_weight_source(new_same, ambiguous=True) is False and D.stale_weight_source(None, True) is False


def test_key_table_flags_identical_features(tiny_run):
    """A revision whose out-of-fold predictions equal step0's (same weights loaded) is marked ungueltig."""
    root = tiny_run["root"]
    p = config.paths(root)
    out = p["dsse"]
    for ext in ("_probe.csv", "_probe.json", "_oof.npz", "_angles.npz", "_angles.csv", "_angles.json", "_rank.csv"):
        src = os.path.join(out, "tiny-gptneox_step0" + ext)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out, "tiny-gptneox_step9" + ext)
        if ext == "_probe.csv":
            df = pd.read_csv(src); df["revision"] = "step9"; df.to_csv(dst, index=False)
        elif ext == "_probe.json":
            with open(src) as f:
                j = json.load(f)
            j["revision"] = "step9"; j["manifest"]["revision"] = "step9"; j["manifest"]["finished"] = "later"
            with open(dst, "w") as f:
                json.dump(j, f)
        else:
            shutil.copyfile(src, dst)
    try:
        res = D.collect_results(out, models=["tiny-gptneox"])
        key = D.key_table(res, D.make_oof_loader(out), boot=20)
        row = key[key.revision == "step9"].iloc[0]
        assert bool(row.identical_to_step0) is True and row.verdict.startswith("ungueltig") and "identische Vorhersagen" in row.pairing_note
        assert bool(key[key.revision == "step0"].iloc[0].identical_to_step0) is False
    finally:
        for fn in os.listdir(out):
            if fn.startswith("tiny-gptneox_step9"):
                os.remove(os.path.join(out, fn))


def test_hub_weight_files_real_branches():
    """Network: pythia-160m step0 has its own model.safetensors; pythia-2.8b step0 is the ambiguous layout."""
    try:
        f160 = M.hub_weight_files("EleutherAI/pythia-160m", "step0")
        f28 = M.hub_weight_files("EleutherAI/pythia-2.8b", "step0")
        fmain = M.hub_weight_files("EleutherAI/pythia-2.8b", "main")
    except M.HubListingError:
        f160 = None
    if f160 is None or f28 is None or fmain is None:
        pytest.skip("Hub not reachable")
    used, amb = M.choose_weight_files(f160)
    assert [p for p, _ in used] == ["model.safetensors"] and amb is False and M.weight_fingerprint(used)
    used, amb = M.choose_weight_files(f28)
    assert amb is True and len(used) == 2 and all(p.startswith("model-") for p, _ in used)
    main_single = [s for p, s in fmain if p == "model.safetensors"]
    step0_single = [s for p, s in f28 if p == "model.safetensors"]
    assert main_single == step0_single                                         # the documented Hub defect
    assert M.hub_weight_files(str(os.getcwd())) is None                        # local directories: no Hub lookup


# ----------------------------------------------------------------------------- 31-34: review of the weight-source fix


def test_download_shards_allowlist_and_stray_removal(tmp_path, monkeypatch):
    """snapshot_download gets an allow-list (no optimizer.pt), a pre-cached single model.safetensors is removed from the
    snapshot folder, the index must exist; weight_source returns that folder for an ambiguous branch."""
    snap = tmp_path / "snap"; snap.mkdir()
    for fn in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", "config.json"):
        (snap / fn).write_text("x")
    (snap / "model.safetensors").write_text("main weights")                   # planted by an earlier plain load
    (snap / "pytorch_model.bin").write_text("main weights")
    calls = {}

    def fake_snapshot_download(hf_id, revision=None, allow_patterns=None, **kw):
        calls.update(hf_id=hf_id, revision=revision, allow=allow_patterns, kw=kw); return str(snap)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    files = [("model-00001-of-00002.safetensors", "c" * 64), ("model-00002-of-00002.safetensors", "d" * 64),
             ("model.safetensors", "a" * 64), ("model.safetensors.index.json", None), ("pytorch_model.bin", "b" * 64)]
    monkeypatch.setattr(M, "hub_weight_files", lambda hf_id, revision=None, retries=3: files)
    local, used, amb, fp = M.weight_source("EleutherAI/pythia-2.8b", "step0")
    assert local == str(snap) and amb is True and [p for p, _ in used] == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    assert calls["revision"] == "step0" and "ignore_patterns" not in calls["kw"]
    assert set(calls["allow"]) >= {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", "config.json"}
    assert not any("optimizer" in a or a == "*" for a in calls["allow"])
    assert not (snap / "model.safetensors").exists() and not (snap / "pytorch_model.bin").exists()   # stray single files gone
    assert (snap / "model.safetensors.index.json").exists() and fp == M.weight_fingerprint(used)
    (snap / "model.safetensors.index.json").unlink()
    with pytest.raises(RuntimeError, match="index missing"):
        M.download_shards("EleutherAI/pythia-2.8b", "step0", used)


def test_hub_listing_failure_is_not_silent(monkeypatch):
    """Online: a failing listing raises after retries (no silent fall-back to the single file); offline: None."""
    import huggingface_hub
    calls = {"n": 0}

    class FakeApi:
        def list_repo_tree(self, hf_id, revision=None):
            calls["n"] += 1; raise ConnectionError("429")

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(M, "hub_offline", lambda: False)
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda s: None)
    with pytest.raises(M.HubListingError):
        M.hub_weight_files("EleutherAI/pythia-2.8b", "step0", retries=3)
    assert calls["n"] == 3
    monkeypatch.setattr(M, "hub_offline", lambda: True)
    assert M.hub_weight_files("EleutherAI/pythia-2.8b", "step0") is None    # offline: quiet, as before
    # CLI: extraction is refused, not run with an unverified source
    monkeypatch.setattr(M, "hub_offline", lambda: False)
    rc = cli.main(["dsse-probe", "--stage", "extract", "--model", "pythia-160m", "--revisions", "step0", "--n_rows", "2", "--device", "cpu",
                   "--root", "/tmp/claude-0/-home-user-valueheads/3ccf061c-00fb-546b-8a09-3ab673f5e720/scratchpad/_hubfail_root",
                   "--raw_dir", "/nonexistent-raw-dir"])
    assert rc == 1


def test_probe_fingerprint_stable_for_old_manifests():
    """A manifest without weight_fingerprint hashes exactly as before the key was added (valid files are not re-probed)."""
    settings = D.probe_settings(n_pcs=8, folds=2, boot=5, perms=2)
    man = {k: f"v_{k}" for k in D.EXTRACTION_IDENTITY if k != "weight_fingerprint"}
    old_identity = tuple(k for k in D.EXTRACTION_IDENTITY if k != "weight_fingerprint")
    old = D._fingerprint(settings, {"rows_sha256": "r", "scores_sha256": "s", "npz": "x.npz", "extraction": {k: man.get(k) for k in old_identity}})
    new = D._fingerprint(settings, {"rows_sha256": "r", "scores_sha256": "s", "npz": "x.npz",
                                    "extraction": {k: man.get(k) for k in D.EXTRACTION_IDENTITY if k != "weight_fingerprint" or man.get(k)}})
    assert old == new
    man2 = dict(man, weight_fingerprint="f" * 64)
    withfp = D._fingerprint(settings, {"rows_sha256": "r", "scores_sha256": "s", "npz": "x.npz",
                                       "extraction": {k: man2.get(k) for k in D.EXTRACTION_IDENTITY if k != "weight_fingerprint" or man2.get(k)}})
    assert withfp != old


def test_recover_staged_discards_stale_ambiguous_pair(synthetic_dir, tmp_path, capsys):
    """A staged pair without weight fingerprint from an ambiguous branch is discarded instead of published."""
    model, tok = D.tiny_model(seed=0)
    rows = D.build_rows(synthetic_dir, n_rows=4)
    stage = str(tmp_path / "stage"); target = str(tmp_path / "features" / "tiny-gptneox_step0.npz")
    model.vh_info["revision"] = "step0"
    D.extract_dsse(model, tok, rows, target, layers=[0, 2], batch_size=8, max_tokens_per_batch=512, max_length=64, prompt_cap=32,
                   stage_dir=stage, show_progress=False, say=lambda *a, **k: None)
    tmp = os.path.join(stage, "tiny-gptneox_step0.tmp.npz")
    shutil.move(target, tmp); shutil.move(D.manifest_path(target), D.staged_manifest_path(tmp))
    expect = {"version": D.VERSION, "n_rows": 4, "row_seed": D.ROW_SEED, "max_length": 64, "prompt_cap": 32, "layers": [0, 2],
              "hf_id": None, "revision": "step0", "rows_sha256": rows.rows_sha256()}
    assert D.recover_staged(stage, target, expect, say=print, ambiguous=True) is False
    assert "ambiguous Hub branch" in capsys.readouterr().out and not os.path.exists(tmp) and not os.path.exists(target)
