"""probing_stats: exact / weighted AUC, cluster bootstrap, permutation null with the batched torch logistic regression
(verified against sklearn), selection / confirmation split, split stability, significant set and the JSON round trip
of run_stats on synthetic feature directories (no model needed)."""
import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from valueheads_pythia import probing_stats as ps
from valueheads_pythia.probing import fit_head_probes, load_head_scores, save_head_scores, select_topk

from test_probe_paper_mode import D, H, L, make_features, write_shards


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    rng = np.random.RandomState(3)
    base = tmp_path_factory.mktemp("stats")
    tr, va = str(base / "honesty_train"), str(base / "honesty_val")
    a_tr, y_tr, g_tr = make_features(rng, 240)
    a_va, y_va, g_va = make_features(rng, 120)
    write_shards(tr, a_tr, y_tr, g_tr)
    write_shards(va, a_va, y_va, g_va)
    df, scores = fit_head_probes(tr, va, feature_mode="head", score="auc", class_weight="balanced", n_jobs=1, verbose=False, return_scores=True)
    path = str(base / "synthetic_none_honesty_pythia.json")
    save_head_scores(df, path, meta={"model": "synthetic", "revision": "none", "value": "honesty", "mode": "pythia", "feature_tag": None})
    return {"train": tr, "val": va, "df": df, "scores": scores, "y": y_va, "g": g_va, "json": path, "a_tr": a_tr, "y_tr": y_tr, "a_va": a_va}


def test_auc_matrix_exact_and_weighted():
    rng = np.random.RandomState(0)
    N, K = 300, 5
    y = rng.randint(0, 2, N)
    groups = rng.randint(0, 60, N)
    S = rng.randn(K, N) + 0.8 * y[None]
    S[0] = np.round(S[0] * 2) / 2                                   # ties
    S[1, :10] = S[1, 10]                                            # a tie block with mixed labels
    a = ps.auc_matrix(S, y)
    for k in range(K):
        assert abs(a[k] - roc_auc_score(y, S[k])) < 1e-12
    assert ps.auc_matrix(S[0], y).shape == (1,)
    assert np.isnan(ps.auc_matrix(S, np.ones(N))).all()
    prep = ps.WeightedAUC(S, y)
    assert np.allclose(prep.auc(np.ones(N)), a)
    for _ in range(5):                                              # integer weights == expanded sample
        w = rng.multinomial(60, np.full(60, 1 / 60))[groups]
        idx = np.repeat(np.arange(N), w)
        wa = prep.auc(w)
        for k in range(K):
            assert abs(wa[k] - roc_auc_score(y[idx], S[k][idx])) < 1e-12
    acc = ps.acc_matrix(S, y)
    assert acc.shape == (K,) and np.all((acc >= 0) & (acc <= 1))


def test_cluster_bootstrap(synth):
    scores, y, g = synth["scores"], synth["y"], synth["g"]
    boot = ps.cluster_bootstrap(scores, y, g, B=300, seed=1)
    assert list(boot.columns) == ["auc", "ci_lo", "ci_hi", "boot_mean", "boot_std", "n_groups", "B"]
    assert len(boot) == L * H and boot.attrs["B"] == 300 and boot["n_groups"].iloc[0] == len(np.unique(g))
    assert (boot.ci_lo <= boot.auc).all() and (boot.auc <= boot.ci_hi).all()
    assert np.allclose(boot.auc.to_numpy(), synth["df"].sort_values("flat_idx").auc.to_numpy())
    planted = 1 * H + 2
    assert boot.loc[planted, "ci_lo"] > 0.8 and boot.loc[planted, "boot_std"] < 0.1
    # NaN rows (layers not probed) are carried through; reproducible with the seed
    sc = scores.copy()
    sc[0] = np.nan
    b2 = ps.cluster_bootstrap(sc, y, g, B=50, seed=1)
    assert np.isnan(b2.loc[0, "auc"]) and not np.isnan(b2.loc[planted, "ci_lo"])
    assert np.allclose(ps.cluster_bootstrap(scores, y, g, B=50, seed=1).ci_lo, ps.cluster_bootstrap(scores, y, g, B=50, seed=1).ci_lo)
    with pytest.raises(ValueError):
        ps.cluster_bootstrap(scores, y[:-1], g, B=5)


def test_permute_labels_within_group_and_stratified():
    rng = np.random.RandomState(0)
    g = np.repeat(np.arange(100), [1, 2, 3, 2] * 25)          # 200 items, group sizes 1-3
    y = rng.randint(0, 2, len(g))
    yp = ps.permute_labels(y, g, rng, within_group=True)
    for gi in np.unique(g):
        assert y[g == gi].sum() == yp[g == gi].sum()
    assert not np.array_equal(y, yp)
    yg = ps.permute_labels(y, g, rng, within_group=False)
    _, inv, sizes = np.unique(g, return_inverse=True, return_counts=True)
    for s in np.unique(sizes[inv]):
        assert y[sizes[inv] == s].sum() == yg[sizes[inv] == s].sum()
    sets = ps.permutation_label_sets(y, g, 4, seed=0)
    assert sets.shape == (4, len(g)) and np.array_equal(sets, ps.permutation_label_sets(y, g, 4, seed=0))


def test_torch_logreg_matches_sklearn():
    rng = np.random.RandomState(1)
    Hh, Dd, Ntr, Nva = 6, 16, 400, 150
    Xtr, Xva = rng.randn(Hh, Ntr, Dd).astype(np.float32), rng.randn(Hh, Nva, Dd).astype(np.float32)
    ytr, yva = rng.randint(0, 2, Ntr), rng.randint(0, 2, Nva)
    for h in range(Hh):
        Xtr[h, :, 0] += 0.3 * h * (2 * ytr - 1)
        Xva[h, :, 0] += 0.3 * h * (2 * yva - 1)
    Xtr *= 5
    Xva *= 5
    for cw in ("balanced", None):
        sc = ps.torch_batched_logreg(Xtr, ytr, Xva, class_weight=cw, max_iter=200)
        assert sc.shape == (Hh, Nva)
        for h in range(Hh):
            clf = LogisticRegression(random_state=42, max_iter=1000, class_weight=cw).fit(Xtr[h], ytr)
            assert abs(roc_auc_score(yva, clf.decision_function(Xva[h])) - roc_auc_score(yva, sc[h])) < 0.02
    # several label sets at once share the features; the result matches a single-set fit
    Y = ps.permutation_label_sets(ytr, np.arange(Ntr), 3, seed=0)
    sc3, (W, b) = ps.torch_batched_logreg(Xtr, Y, Xva, class_weight="balanced", return_params=True)
    assert sc3.shape == (3, Hh, Nva) and W.shape == (3, Hh, Dd) and b.shape == (3, Hh)
    single = ps.torch_batched_logreg(Xtr, Y[1], Xva, class_weight="balanced")
    assert abs(roc_auc_score(yva, sc3[1][5]) - roc_auc_score(yva, single[5])) < 0.01
    with pytest.raises(ValueError):
        ps.torch_batched_logreg(Xtr, ytr[:-1], Xva)


def test_permutation_null_and_p_values(synth):
    perm = ps.permutation_null(synth["train"], P=8, within_group=True, seed=0, val_dir=synth["val"], feature_mode="head",
                               class_weight="balanced", verbose=False)
    assert perm["dist"].shape == (8,) and perm["null_by_head"].shape == (8, L * H) and not np.isnan(perm["null_by_head"]).any()
    assert 0.5 <= perm["threshold"] < 0.8 and perm["P"] == 8 and perm["statistic"] == "max_auc"
    assert np.all((perm["null_by_head"] >= 0) & (perm["null_by_head"] <= 1))
    obs = synth["df"].sort_values("flat_idx").auc.to_numpy()
    assert obs[1 * H + 2] > perm["threshold"]                     # planted head beats the max-over-heads null
    p_adj = ps.westfall_young_p(obs, perm["dist"])
    assert p_adj[1 * H + 2] == 1 / 9 and p_adj.max() <= 1.0 and p_adj.min() >= 1 / 9
    p_head = ps.per_head_p(obs, perm["null_by_head"])
    assert p_head.shape == (L * H,) and (p_head <= p_adj + 1e-12).all()
    # stratified global permutation and accuracy statistic, restricted layers -> NaN elsewhere
    perm2 = ps.permutation_null(synth["train"], P=3, within_group=False, statistic="max_acc", seed=1, layers=[1], verbose=False)
    assert np.isnan(perm2["null_by_head"][:, :H]).all() and not np.isnan(perm2["null_by_head"][:, H:2 * H]).any()
    assert np.array_equal(perm["dist"], ps.permutation_null(synth["train"], P=8, seed=0, verbose=False)["dist"])
    with pytest.raises(ValueError):
        ps.permutation_null(synth["train"], P=2, statistic="max_f1", verbose=False)


def test_selection_confirmation_split(synth):
    scores, y, g = synth["scores"], synth["y"], synth["g"]
    conf = ps.selection_confirmation_split(scores, y, g, seed=0, ratio=0.25)
    assert conf["n_groups_A"] + conf["n_groups_B"] == len(np.unique(g)) and conf["n_items_A"] + conf["n_items_B"] == len(y)
    assert conf["k"] == int(0.25 * L * H) and len(conf["A_to_B"]["topk"]) == conf["k"]
    assert conf["A_to_B"]["top1"] == 1 * H + 2 and conf["B_to_A"]["top1"] == 1 * H + 2
    assert conf["A_to_B"]["top1_conf"] > 0.8 and conf["top1_conf_mean"] > 0.8
    assert np.allclose(conf["auc_full"], ps.auc_matrix(scores, y))
    # items of a group never straddle the halves
    rng = np.random.RandomState(0)
    perm = rng.permutation(np.unique(g))
    half_a = np.isin(g, perm[:len(perm) // 2])
    assert half_a.sum() == conf["n_items_A"]


def test_split_stability_and_select_significant(synth):
    df = synth["df"]
    rho, jac, k = ps.spearman_jaccard(df, df, 0.15)
    assert rho == 1.0 and jac == 1.0 and k == int(0.15 * L * H)
    rng = np.random.RandomState(0)
    df2 = df.copy()
    df2.attrs.update(df.attrs)
    df2["auc"] = df["auc"].to_numpy()[rng.permutation(len(df))]
    st = ps.split_stability([df, df, df2], names=["a", "b", "c"])
    assert st["spearman"].shape == (3, 3) and st["spearman"][0, 1] == 1.0 and st["spearman"][0, 2] < 0.9
    assert st["jaccard"][0, 1] == 1.0 and 0 <= st["jaccard"][0, 2] <= 1 and st["n"] == 3 and len(st["pairs"]) == 3
    assert st["spearman_min"] <= st["spearman_mean"] <= 1 and set(st["mean_scores"].columns) >= {"flat_idx", "auc_mean", "auc_std", "rank"}
    assert st["mean_scores"]["rank"].iloc[0] == 1 and st["mean_scores"]["flat_idx"].iloc[0] in select_topk(df, 0.15)
    single = ps.split_stability([df])
    assert np.isnan(single["spearman_mean"]) and len(single["pairs"]) == 0
    # significant set: CI lower bound above the threshold, in rank order
    d = df.copy()
    d.attrs.update(df.attrs)
    d["ci_lo"] = d["auc"] - 0.05
    sig = ps.select_significant(d, perm_threshold=0.7)
    assert sig == [int(i) for i in d.sort_values("rank").flat_idx if d.set_index("flat_idx").loc[i, "ci_lo"] > 0.7]
    assert sig[0] == 1 * H + 2
    with pytest.raises(ValueError):
        ps.select_significant(d)


def test_val_predictions_cache_and_run_stats(synth, tmp_path):
    cache = str(tmp_path / "vs.npz")
    df, scores, y, g = ps.val_predictions(synth["train"], synth["val"], class_weight="balanced", n_jobs=1, cache_path=cache)
    assert os.path.exists(cache) and scores.shape == (L * H, 120) and np.array_equal(y, synth["y"]) and np.array_equal(g, synth["g"])
    df2, scores2, _, _ = ps.val_predictions(synth["train"], synth["val"], class_weight="balanced", cache_path=cache)   # cached
    assert np.array_equal(scores, scores2) and df2 is not None and len(df2) == L * H
    _, scores3, _, _ = ps.val_predictions(synth["train"], synth["val"], class_weight=None, cache_path=cache)            # settings differ -> refit
    assert not np.array_equal(scores, scores3)
    # run_stats augments the JSON written by probe, is idempotent and reloads cleanly
    out = ps.run_stats(synth["json"], synth["train"], synth["val"], B=60, P=4, seed=0, n_jobs=1, verbose=False)
    for col in ("ci_lo", "ci_hi", "boot_std", "p_adj", "p_head", "signif", "auc_half_A", "auc_half_B"):
        assert col in out.columns
    assert out.attrs["n_signif"] >= 1 and (1 * H + 2) in out.attrs["signif_heads"] and out.attrs["perm_P"] == 4
    df_re, meta = load_head_scores(synth["json"])
    assert meta["stats_done"] and meta["bootstrap_B"] == 60 and len(meta["perm_dist"]) == 4 and meta["confirmation"]["k"] == int(0.15 * L * H)
    assert "auc_A" not in meta["confirmation"] and df_re["signif"].sum() == out.attrs["n_signif"]
    assert os.path.exists(ps.valscores_path(synth["json"])) and meta["val_auc_max_deviation"] < 1e-9
    again = ps.run_stats(synth["json"], synth["train"], synth["val"], B=10, P=1, verbose=False)          # skipped
    assert again.attrs.get("bootstrap_B", meta["bootstrap_B"]) == 60 and load_head_scores(synth["json"])[1]["bootstrap_B"] == 60
    forced = ps.run_stats(synth["json"], synth["train"], synth["val"], B=10, P=0, force=True, verbose=False)   # P=0: no permutation
    assert forced.attrs["bootstrap_B"] == 10 and "perm_threshold" not in forced.attrs and "signif" not in forced.columns
    with open(synth["json"]) as f:
        assert json.load(f)["meta"]["bootstrap_B"] == 10
