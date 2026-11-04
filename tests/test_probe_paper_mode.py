"""mode=paper probing reproduces the reference repo's probing.py exactly: on identical features the per-head
accuracies of `fit_head_probes(feature_mode="head+mean", score="acc", class_weight=None)` equal
`valueheads.probing.train_head_probes(..., layer_bias=True)`, and head-only mode equals layer_bias=False.

The reference repo is imported from VH_REPO (default /home/user/sihuo-design/valueheads); the test is skipped
when it is absent."""
import json
import os
import sys

import numpy as np
import pytest

from valueheads_pythia.extract import DONE, MANIFEST, load_features, shard_path
from valueheads_pythia.probing import (fit_head_probes, group_by_layer, load_head_scores, rank_heads, save_head_scores,
                                       scores_vector, select_topk, top_heads_from_file)

REPO = os.environ.get("VH_REPO", "/home/user/sihuo-design/valueheads")
N_TR, N_VA, L, H, D = 240, 90, 3, 4, 8


def make_features(rng, n):
    """float16 features (n, L, H*D) with a planted linear signal in a few heads plus labels / groups."""
    y = rng.randint(0, 2, n)
    acts = rng.randn(n, L, H * D).astype(np.float32)
    acts[:, 1, 2 * D:3 * D] += 1.0 * (2 * y[:, None] - 1) * (0.5 + rng.rand(D))    # head (1, 2) carries a strong signal
    acts[:, 2, 0:D] += 0.15 * (2 * y[:, None] - 1)                                 # head (2, 0) a weak one
    return acts.astype(np.float16), y, np.arange(n) // 2


def write_shards(out_dir, acts, labels, groups):
    os.makedirs(out_dir, exist_ok=True)
    for l in range(L):
        np.save(shard_path(out_dir, l), acts[:, l])
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "groups.npy"), groups)
    man = {"n": len(acts), "layers": L, "heads": H, "head_dim": D, "width": H * D, "dtype": "float32", "attn_impl": "eager",
           "hf_id": "synthetic", "revision": "none", "model": "synthetic", "feature_dtype": "float16"}
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(man, f)
    open(os.path.join(out_dir, DONE), "w").close()


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    rng = np.random.RandomState(7)
    base = tmp_path_factory.mktemp("feat")
    tr, va = str(base / "synthetic_train"), str(base / "synthetic_val")
    a_tr, y_tr, g_tr = make_features(rng, N_TR)
    a_va, y_va, g_va = make_features(rng, N_VA)
    write_shards(tr, a_tr, y_tr, g_tr)
    write_shards(va, a_va, y_va, g_va)
    return {"train": tr, "val": va, "a_tr": a_tr, "y_tr": y_tr, "a_va": a_va, "y_va": y_va}


def repo_probing():
    if not os.path.isdir(os.path.join(REPO, "valueheads")):
        pytest.skip(f"reference repo not found at {REPO}")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import valueheads.probing as rp
    return rp


def test_paper_mode_matches_repo_exactly(dirs):
    rp = repo_probing()
    ref = rp.train_head_probes(dirs["a_tr"], dirs["y_tr"], dirs["a_va"], dirs["y_va"], H, seed=42, layer_bias=True)
    df = fit_head_probes(dirs["train"], dirs["val"], feature_mode="head+mean", score="acc", class_weight=None, seed=42,
                         n_jobs=2, verbose=False)
    ours = scores_vector(df, "acc")
    assert ours.shape == ref.shape == (L * H,)
    assert np.array_equal(ours, ref), np.c_[ours, ref]
    # head-only equals the repo with layer_bias=False
    ref0 = rp.train_head_probes(dirs["a_tr"], dirs["y_tr"], dirs["a_va"], dirs["y_va"], H, seed=42, layer_bias=False)
    df0 = fit_head_probes(dirs["train"], dirs["val"], feature_mode="head", score="acc", class_weight=None, seed=42, n_jobs=1, verbose=False)
    assert np.array_equal(scores_vector(df0, "acc"), ref0)
    # the planted head is found and the repo-style top-k agrees with ours when there are no ties at the cut
    k = int(0.15 * L * H)
    repo_top = list(np.argsort(ref)[::-1][:k])
    ours_top = select_topk(df, 0.15, "acc")
    assert len(ours_top) == k == 1 and ours_top[0] == 1 * H + 2
    assert ref[repo_top[0]] == ours[ours_top[0]]


def test_auc_and_columns(dirs):
    df, scores = fit_head_probes(dirs["train"], dirs["val"], feature_mode="head", score="auc", class_weight="balanced",
                                 seed=42, n_jobs=1, verbose=False, return_scores=True)
    assert list(df.columns) == ["layer", "head", "flat_idx", "auc", "acc", "bal_acc", "majority", "pos_rate", "n_train", "n_val", "rank"]
    assert len(df) == L * H and sorted(df["rank"]) == list(range(1, L * H + 1))
    assert scores.shape == (L * H, N_VA) and not np.isnan(scores).any()
    from sklearn.metrics import roc_auc_score
    for row in df.itertuples():
        assert abs(roc_auc_score(dirs["y_va"], scores[row.flat_idx]) - row.auc) < 1e-12
    assert df.loc[df.flat_idx == 1 * H + 2, "auc"].item() > 0.9
    assert df.attrs["feature_mode"] == "head" and df.attrs["class_weight"] == "balanced"
    assert np.isclose(df["majority"].iloc[0], max(dirs["y_va"].mean(), 1 - dirs["y_va"].mean()))
    assert np.isclose(df["pos_rate"].iloc[0], dirs["y_va"].mean())


def test_ranking_is_deterministic_with_tie_break():
    import pandas as pd
    df = pd.DataFrame({"flat_idx": [0, 1, 2, 3], "layer": [0, 0, 0, 0], "head": [0, 1, 2, 3],
                       "auc": [0.7, 0.9, 0.9, 0.9], "bal_acc": [0.6, 0.55, 0.65, 0.65], "acc": [0.6, 0.6, 0.7, 0.7]})
    df.attrs.update(layers=1, heads=4)
    assert rank_heads(df, "auc").tolist() == [2, 3, 1, 0]       # ties on auc -> bal_acc, then flat_idx
    assert select_topk(df, 0.5, "auc") == [2, 3]
    assert select_topk(df, 0.15, "auc", total_heads=20) == [2, 3, 1]
    assert group_by_layer([5, 1, 17, 4], 4) == {0: [1], 1: [0, 1], 4: [1]}


def test_save_and_load_head_scores(dirs, tmp_path):
    df = fit_head_probes(dirs["train"], dirs["val"], feature_mode="head", score="auc", class_weight="balanced", n_jobs=1, verbose=False)
    path = str(tmp_path / "heads.json")
    payload = save_head_scores(df, path, meta={"model": "synthetic"}, ratio=0.5)
    assert payload["k"] == L * H // 2 and payload["topk"] == select_topk(df, 0.5, "auc")
    assert list(payload["scores_by_flat_idx"])[0] == payload["topk"][0]
    df2, meta = load_head_scores(path)
    assert meta["model"] == "synthetic" and meta["score"] == "auc"
    assert np.allclose(df2.sort_values("flat_idx")["auc"].to_numpy(), df.sort_values("flat_idx")["auc"].to_numpy())
    assert top_heads_from_file(path, 0.5) == payload["topk"]
    acts, labels, groups, man = load_features(dirs["train"])
    assert acts[1].shape == (N_TR, H * D) and acts.head(1, 2).shape == (N_TR, D) and labels.shape == (N_TR,)
    with pytest.raises(KeyError):
        load_features(dirs["train"], layers=[0])[0][2]


def test_paper_ranking_reproduces_repo_argsort_including_ties(dirs, tmp_path):
    """ranking='paper' returns np.argsort(accs)[::-1] exactly, so the top-k set equals the repo's at accuracy ties."""
    rp = repo_probing()
    ref = rp.train_head_probes(dirs["a_tr"], dirs["y_tr"], dirs["a_va"], dirs["y_va"], H, seed=42, layer_bias=True)
    df = fit_head_probes(dirs["train"], dirs["val"], feature_mode="head+mean", score="acc", class_weight=None, seed=42,
                         n_jobs=1, verbose=False, ranking="paper")
    assert df.attrs["ranking"] == "paper" and np.array_equal(scores_vector(df, "acc"), ref)
    for k in (1, 3, 6, L * H):
        assert select_topk(df, k=k, score="acc") == [int(i) for i in np.argsort(ref)[::-1][:k]]
    path = str(tmp_path / "paper.json")
    save_head_scores(df, path, ratio=0.5)
    assert top_heads_from_file(path, 0.5) == [int(i) for i in np.argsort(ref)[::-1][:L * H // 2]]
    assert load_head_scores(path)[1]["ranking"] == "paper"
    # crafted ties at the cut: the stable rule orders by bal_acc, the paper rule follows numpy's argsort order
    import pandas as pd
    tied = pd.DataFrame({"flat_idx": [0, 1, 2, 3, 4, 5], "layer": [0] * 6, "head": list(range(6)),
                         "acc": [0.7, 0.9, 0.9, 0.9, 0.8, 0.9], "bal_acc": [0.6, 0.50, 0.65, 0.55, 0.7, 0.60], "auc": [0.5] * 6})
    tied.attrs.update(layers=1, heads=6)
    accs = tied["acc"].to_numpy()
    assert rank_heads(tied, "acc", ranking="paper").tolist() == np.argsort(accs)[::-1].tolist()
    assert rank_heads(tied, "acc", ranking="stable").tolist() == [2, 5, 3, 1, 4, 0]
    assert select_topk(tied, k=2, score="acc", ranking="paper") == np.argsort(accs)[::-1][:2].tolist()
    tied.attrs["ranking"] = "paper"
    assert select_topk(tied, k=2, score="acc") == np.argsort(accs)[::-1][:2].tolist()     # the stored attr is the default
    with pytest.raises(ValueError):
        rank_heads(tied, "acc", ranking="nope")


def test_single_class_split_is_refused(dirs, tmp_path):
    """A val (or train) split with one class must raise instead of yielding an all-NaN AUC ranking."""
    import pandas as pd
    rng = np.random.RandomState(3)
    one_dir = str(tmp_path / "oneclass_val")
    a, y, g = make_features(rng, 40)
    write_shards(one_dir, a, np.ones_like(y), g)
    with pytest.raises(ValueError, match="single class"):
        fit_head_probes(dirs["train"], one_dir, n_jobs=1, verbose=False)
    with pytest.raises(ValueError, match="single class"):
        fit_head_probes(one_dir, dirs["val"], n_jobs=1, verbose=False)
    # a table that somehow holds NaN scores cannot be ranked or used for head selection
    df = pd.DataFrame({"flat_idx": [0, 1, 2], "layer": [0] * 3, "head": [0, 1, 2], "auc": [np.nan, np.nan, np.nan],
                       "acc": [0.5, 0.5, 0.5], "bal_acc": [0.5, 0.5, 0.5]})
    df.attrs.update(layers=1, heads=3)
    with pytest.raises(ValueError, match="NaN"):
        select_topk(df, 0.5, "auc")
    path = str(tmp_path / "nan.json")
    with open(path, "w") as f:
        json.dump({"meta": {"score": "auc", "layers": 1, "heads": 3},
                   "heads": [{"flat_idx": 0, "rank": 1, "auc": None}, {"flat_idx": 1, "rank": 2, "auc": None}]}, f)
    with pytest.raises(ValueError, match="null"):
        top_heads_from_file(path, 0.5)


def test_cv_head_probes_grouped_folds(dirs):
    """Grouped stratified K-fold over train + val: out-of-fold scores for every pooled item, groups never straddle folds."""
    from sklearn.metrics import roc_auc_score
    from valueheads_pythia.probing import cv_folds, cv_head_probes
    df, scores = cv_head_probes(dirs["train"], dirs["val"], n_splits=4, feature_mode="head", score="auc", class_weight="balanced",
                                n_jobs=1, verbose=False, return_scores=True)
    N = N_TR + N_VA
    assert df.attrs["cv_splits"] == 4 and df["n_train"].iloc[0] == N == df["n_val"].iloc[0] and scores.shape == (L * H, N)
    assert np.isfinite(scores).all() and sorted(df["rank"]) == list(range(1, L * H + 1))
    y = np.concatenate([dirs["y_tr"], dirs["y_va"]])
    for row in df.itertuples():
        assert abs(roc_auc_score(y, scores[row.flat_idx]) - row.auc) < 1e-6
    assert df.loc[df.flat_idx == 1 * H + 2, "auc"].item() > 0.9 and df.loc[df.flat_idx == 1 * H + 2, "rank"].item() == 1
    groups = np.concatenate([np.arange(N_TR) // 2, np.arange(N_VA) // 2 + N_TR // 2])
    folds, n = cv_folds(y, groups, 4, seed=42)
    assert n == 4 and sorted(np.concatenate([va for _, va in folds]).tolist()) == list(range(N))
    for tr, va in folds:
        assert not (set(groups[tr]) & set(groups[va]))
    assert cv_folds(np.array([0, 1, 0, 1, 0, 1]), np.arange(6), 5)[1] == 3          # clamped to the smaller class


def test_extra_ratio_sets_in_json(dirs, tmp_path):
    from valueheads_pythia import config
    df = fit_head_probes(dirs["train"], dirs["val"], feature_mode="head", score="auc", class_weight="balanced", n_jobs=1, verbose=False)
    path = str(tmp_path / "ratios.json")
    payload = save_head_scores(df, path, ratio=0.5)
    ranked = [int(i) for i in rank_heads(df, "auc")]
    assert config.EXTRA_RATIOS == [0.05, 0.10]
    for r in config.EXTRA_RATIOS + [0.5]:
        key = f"{int(round(r * 100))}pct"
        k = int(L * H * r)
        assert payload[f"k_{key}"] == k and payload[f"topk_{key}"] == ranked[:k]
        assert top_heads_from_file(path, r) == ranked[:k]
    assert payload["topk_50pct"] == payload["topk"]
