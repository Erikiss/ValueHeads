"""controls.py: layer-matched random head sets, surface / embedding / layer-mean probes (DESIGN 5.7, 5.9)."""
import random

import numpy as np
import pytest

from valueheads_pythia.controls import (best_control, count_by_layer, embedding_probe, last_token_embeddings, last_token_ids,
                                        layer_mean_probe, random_sets_layer_matched, random_sets_uniform, surface_features,
                                        surface_probe)
from valueheads_pythia.probing import group_by_layer

from test_probe_paper_mode import D, H, L, make_features, write_shards

TOPK = {3: [6, 8, 10], 4: [0, 2, 3, 4, 7], 5: [1, 8, 11], 6: [0, 4, 5], 7: [0, 4], 8: [0], 9: [3], 10: [9], 11: [6, 11]}


def test_random_sets_layer_matched():
    sets = random_sets_layer_matched(TOPK, 12, 12, 5, seed=0)
    assert len(sets) == 5
    for s in sets:
        assert count_by_layer(s) == count_by_layer(TOPK)
        for l, hs in s.items():
            assert hs == sorted(hs) and len(set(hs)) == len(hs) and all(0 <= h < 12 for h in hs)
    assert sets == random_sets_layer_matched(TOPK, 12, 12, 5, seed=0)                 # deterministic
    assert sets[:2] == random_sets_layer_matched(TOPK, 12, 12, 2, seed=0)             # prefix-stable in R
    assert sets != random_sets_layer_matched(TOPK, 12, 12, 5, seed=1)
    assert len({tuple((l, tuple(h)) for l, h in s.items()) for s in sets}) == 5       # distinct sets
    assert random_sets_layer_matched({0: [], 1: [2]}, 2, 4, 1)[0] == {1: [random_sets_layer_matched({1: [2]}, 2, 4, 1)[0][1][0]]}
    with pytest.raises(ValueError):
        random_sets_layer_matched({0: list(range(13))}, 12, 12, 1)
    with pytest.raises(ValueError):
        random_sets_layer_matched({12: [0]}, 12, 12, 1)


def test_random_sets_uniform_matches_repo():
    sets = random_sets_uniform(21, 12, 12, 3)
    for seed, s in enumerate(sets):
        assert sum(len(h) for h in s.values()) == 21
        assert s == group_by_layer(random.Random(seed).sample(range(144), 21), 12)


def test_surface_features_and_probe():
    texts = ["Question: Why? Answer: I have no comment.", "Question: Why? Answer: Because the sky is blue, obviously!!!",
             "short", ""]
    f, table = surface_features(texts)
    assert f.shape == (4, 4) and f[0, 2] == 1.0 and f[1, 2] == 0.0 and f[3, 0] == 0.0
    assert np.isclose(f[0, 0], np.log1p(11)) and f[1, 3] > f[0, 3] > 0 and f[3, 3] == 0.0   # 11 word / punctuation tokens
    assert table[None] < min(v for k, v in table.items() if k is not None)
    f2, _ = surface_features(["unknownword"], table)
    assert np.isclose(f2[0, 1], table[None])
    # label = long text -> the surface probe separates the classes
    rng = np.random.RandomState(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    long = [" ".join(rng.choice(words, 25)) + "." for _ in range(60)]
    short = [" ".join(rng.choice(words, 5)) + "." for _ in range(60)]
    texts, labels = long + short, np.array([1] * 60 + [0] * 60)
    groups = np.arange(120) // 2
    cv = surface_probe(texts, labels, groups=groups, n_splits=4)
    assert cv["auc"] > 0.95 and cv["n_splits"] == 4 and cv["features"][0] == "log_length" and len(cv["coef"]) == 4
    assert cv["control"] == "surface" and cv["n_train"] == 120 and cv["coef"][0] > 0
    held = surface_probe(texts[:40] + texts[60:100], labels[:40].tolist() + labels[60:100].tolist(),
                         val_texts=texts[40:60] + texts[100:], val_labels=[1] * 20 + [0] * 20)
    assert held["auc"] == 1.0 and held["n_val"] == 40 and held["n_train"] == 80


def test_embedding_probe_lexical_baseline(model_tok):
    model, tok = model_tok
    train = [f"Question: {i} Answer: yes" for i in range(20)] + [f"Question: {i} Answer: no" for i in range(20)]
    y_train = [1] * 20 + [0] * 20
    val = ["Question: What? Answer: Mostly yes", "Question: Why? Answer: Certainly not, no", "Question: x Answer: yes", "Question: y Answer: no"]
    y_val = [1, 0, 1, 0]
    ids = last_token_ids(tok, val)
    assert ids.shape == (4,) and ids[0] == ids[2] and ids[1] == ids[3] and ids[0] != ids[1]
    emb = last_token_embeddings(model, tok, val)
    assert emb.shape == (4, model.config.hidden_size) and np.allclose(emb[0], emb[2])
    res = embedding_probe(train, y_train, val, y_val, model=model, tokenizer=tok)
    assert res["auc"] == 1.0 and res["acc"] == 1.0 and res["dim"] == model.config.hidden_size and res["control"] == "embedding"
    res2 = embedding_probe(last_token_embeddings(model, tok, train), y_train, emb, y_val)
    assert res2["auc"] == res["auc"]
    with pytest.raises(ValueError):
        embedding_probe(train, y_train, val, y_val)


def test_layer_mean_probe_and_best_control(tmp_path):
    rng = np.random.RandomState(5)
    tr, va = str(tmp_path / "syn_train"), str(tmp_path / "syn_val")
    a_tr, y_tr, g_tr = make_features(rng, 300)
    a_va, y_va, g_va = make_features(rng, 120)
    write_shards(tr, a_tr, y_tr, g_tr)
    write_shards(va, a_va, y_va, g_va)
    df = layer_mean_probe(tr, va)
    assert list(df.columns) == ["layer", "auc", "acc", "bal_acc", "majority", "pos_rate", "n_train", "n_val"]
    assert len(df) == L and df["n_train"].iloc[0] == 300 and df["n_val"].iloc[0] == 120
    # the planted signal in head (1, 2) leaks into the mean of layer 1 (1/H of the shift)
    assert df.loc[df.layer == 1, "auc"].item() > 0.8 > df.loc[df.layer == 0, "auc"].item()
    assert df.attrs["control"] == "layer_mean"
    sub = layer_mean_probe(tr, va, layers=[2])
    assert sub["layer"].tolist() == [2]
    best = best_control(surface={"auc": 0.6}, embedding={"auc": 0.65}, layer_mean=df)
    assert best["best_name"] == "layer_mean" and best["best"] == df["auc"].max() and best["embedding"] == 0.65
    assert best_control()["best_name"] is None and best_control(surface={"auc": 0.7})["best"] == 0.7
    # manifest geometry sanity: the synthetic shards are (N, H * D)
    assert a_tr.shape[2] == H * D


def test_paired_control_bootstrap_and_heads_vs_controls(tmp_path):
    """DESIGN 5.7: a head is value-sensitive when AUC(head) - AUC(best control) has a positive bootstrap CI (paired on the
    same val items, groups resampled); identical scores never 'beat', a planted strong head does."""
    from sklearn.metrics import roc_auc_score
    from valueheads_pythia.controls import heads_vs_controls, paired_control_bootstrap
    rng = np.random.RandomState(0)
    N = 400
    y = rng.randint(0, 2, N)
    groups = np.arange(N) // 4
    control = 0.6 * (2 * y - 1) + rng.randn(N)                # a decent control (AUC ~0.8)
    strong = 2.0 * (2 * y - 1) + rng.randn(N)                 # clearly better
    noise = rng.randn(N)                                      # chance
    heads = np.vstack([strong, control, noise, np.full(N, np.nan)])
    out = paired_control_bootstrap(heads, control, y, groups, B=200, seed=0)
    assert list(out.columns) == ["auc_head", "auc_control", "delta", "ci_lo", "ci_hi", "beats"] and len(out) == 4
    assert np.isclose(out.loc[0, "auc_head"], roc_auc_score(y, strong)) and np.isclose(out.loc[0, "auc_control"], roc_auc_score(y, control))
    assert out.loc[0, "beats"] and out.loc[0, "ci_lo"] > 0 and out.loc[0, "delta"] > 0.1
    assert out.loc[1, "delta"] == 0.0 and out.loc[1, "ci_lo"] == 0.0 and out.loc[1, "ci_hi"] == 0.0 and not out.loc[1, "beats"]
    assert out.loc[2, "delta"] < 0 and not out.loc[2, "beats"] and out.loc[2, "ci_hi"] < 0
    assert np.isnan(out.loc[3, "auc_head"]) and not out.loc[3, "beats"]
    assert out.attrs["n_groups"] == 100 and out.attrs["B"] == 200
    assert out.loc[0, "ci_lo"] <= out.loc[0, "delta"] <= out.loc[0, "ci_hi"]
    res = heads_vs_controls(heads, y, groups, {"surface": noise, "embedding": control}, B=100, seed=1)
    assert res["best_name"] == "embedding" and np.isclose(res["best_auc"], roc_auc_score(y, control))
    assert res["beating_heads"] == [0] and res["n_beats"] == 1 and res["n_beats_by_control"]["surface"] >= 2
    with pytest.raises(ValueError):
        paired_control_bootstrap(heads, control[:10], y, groups)
    with pytest.raises(ValueError):
        paired_control_bootstrap(heads, control, np.ones(N), groups)
    with pytest.raises(ValueError):
        heads_vs_controls(heads, y, groups, {})
    # the control probes hand back their val scores for this comparison
    rng = np.random.RandomState(5)
    tr, va = str(tmp_path / "syn_train"), str(tmp_path / "syn_val")
    a_tr, y_tr, g_tr = make_features(rng, 200)
    a_va, y_va, g_va = make_features(rng, 80)
    write_shards(tr, a_tr, y_tr, g_tr)
    write_shards(va, a_va, y_va, g_va)
    lm = layer_mean_probe(tr, va, return_scores=True)
    assert set(lm.attrs["scores"]) == set(range(L)) and lm.attrs["scores"][1].shape == (80,)
    assert np.isclose(roc_auc_score(y_va, lm.attrs["scores"][1]), lm.loc[lm.layer == 1, "auc"].item())
    best = best_control(layer_mean=lm)
    assert best["best_layer"] == 1 and best["best_name"] == "layer_mean"
    words = ["alpha", "beta", "gamma", "delta"]
    texts = [" ".join(rng.choice(words, 25)) + "." for _ in range(30)] + [" ".join(rng.choice(words, 5)) + "." for _ in range(30)]
    sp = surface_probe(texts[:20] + texts[30:50], [1] * 20 + [0] * 20, val_texts=texts[20:30] + texts[50:], val_labels=[1] * 10 + [0] * 10,
                       return_scores=True)
    assert sp["scores"].shape == (20,) and np.isclose(roc_auc_score([1] * 10 + [0] * 10, sp["scores"]), sp["auc"])
    emb = embedding_probe(rng.randn(40, 8), [1, 0] * 20, rng.randn(10, 8), [1, 0] * 5, return_scores=True)
    assert emb["scores"].shape == (10,)
