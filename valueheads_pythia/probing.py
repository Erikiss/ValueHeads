"""Step 2: one logistic-regression probe per (layer, head), ranking and top-k selection (DESIGN 2, 5.1-5.3, 5.8).

Adapted from sihuo-design/valueheads `probing.py` (MIT). Paper/repo settings are reproduced exactly with
feature_mode="head+mean", score="acc", class_weight=None, ranking="paper" (LogisticRegression(random_state=seed,
max_iter=1000), features = [head activation ; mean over the heads of the layer], accuracy on val, top-k =
`np.argsort(accs)[::-1][:k]` including its tie order). The `pythia` defaults are feature_mode="head", score="auc",
class_weight="balanced", ranking="stable" (rounded score, then balanced accuracy, then flat index).
`cv_head_probes` is the grouped K-fold variant over train + val (DESIGN 5.3, HelpSteer2 values).
"""
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .extract import load_features

FEATURE_MODES = ["head", "head+mean"]
SCORES = ["auc", "acc"]
RANKINGS = ["stable", "paper"]
SCORE_COLUMNS = ["auc", "acc", "bal_acc", "majority", "pos_rate"]


def sibling_val_dir(train_dir):
    """{...}/{dataset}_train -> {...}/{dataset}_val."""
    if not train_dir.rstrip("/").endswith("_train"):
        raise ValueError(f"cannot derive the val directory from {train_dir!r}; pass val_dir")
    return train_dir.rstrip("/")[:-len("_train")] + "_val"


def head_features(layer_acts, num_heads, head, feature_mode):
    """(N, D) or (N, 2D) float32 features of one head from a (N, H * D) layer array."""
    x = np.asarray(layer_acts, dtype=np.float32).reshape(len(layer_acts), num_heads, -1)
    f = x[:, head]
    if feature_mode == "head+mean":
        f = np.concatenate([f, x.mean(axis=1)], axis=-1)
    return f


def _metrics(y_va, pred, scores):
    return {
        "acc": accuracy_score(y_va, pred),
        "auc": roc_auc_score(y_va, scores) if len(np.unique(y_va)) == 2 else float("nan"),
        "bal_acc": balanced_accuracy_score(y_va, pred),
    }


def pca_project(f_tr, f_va, n_components, seed=42):
    """PCA fitted on the train features of one head and applied to train and val (DESIGN 3.5: probe-capacity
    equalization, e.g. pythia-1b's 256-dim heads -> 64 dims like 160m / 410m). n is clamped to min(D, N_train)."""
    from sklearn.decomposition import PCA
    n = int(min(n_components, f_tr.shape[1], len(f_tr)))
    pca = PCA(n_components=n, random_state=seed).fit(f_tr)
    return pca.transform(f_tr).astype(np.float32), pca.transform(f_va).astype(np.float32)


def _fit_one(f_tr, y_tr, f_va, y_va, seed, C, max_iter, class_weight, want_scores, pca=None):
    if pca:
        f_tr, f_va = pca_project(f_tr, f_va, pca, seed)
    clf = LogisticRegression(random_state=seed, max_iter=max_iter, C=C, class_weight=class_weight).fit(f_tr, y_tr)
    pred = clf.predict(f_va)
    scores = clf.decision_function(f_va)
    return _metrics(y_va, pred, scores), (scores.astype(np.float32) if want_scores else None)


def _fit_cv_one(f, y, folds, seed, C, max_iter, class_weight, want_scores, pca=None):
    """Out-of-fold decision scores of one head over the given folds; metrics on the pooled predictions."""
    scores, pred = np.zeros(len(y)), np.zeros(len(y), dtype=np.int64)
    for tr, va in folds:
        f_tr, f_va = (pca_project(f[tr], f[va], pca, seed) if pca else (f[tr], f[va]))
        clf = LogisticRegression(random_state=seed, max_iter=max_iter, C=C, class_weight=class_weight).fit(f_tr, y[tr])
        scores[va] = clf.decision_function(f_va)
        pred[va] = clf.predict(f_va)
    return _metrics(y, pred, scores), (scores.astype(np.float32) if want_scores else None)


def _labels_and_mask(labels, name):
    if labels is None:
        raise ValueError(f"{name} has no labels (prompt sets cannot be probed)")
    labels = np.asarray(labels)
    mask = labels >= 0
    return labels[mask].astype(np.int64), mask


def _check_two_classes(y, name):
    """Probes need both classes on every split: a single-class val split gives NaN AUCs and a silently wrong ranking."""
    if len(np.unique(y)) < 2:
        raise ValueError(f"{name} holds a single class (n={len(y)}, n_pos={int((y == 1).sum())}, n_neg={int((y == 0).sum())}); "
                         "probes need both classes on train and val (check the subsample / label rule)")


def _check_manifests(man_tr, man_va):
    for key in ("layers", "heads", "head_dim", "dtype", "attn_impl", "hf_id", "revision"):
        if man_tr.get(key) != man_va.get(key):
            raise ValueError(f"train/val feature mismatch on {key}: {man_tr.get(key)!r} vs {man_va.get(key)!r}")


def _finish_table(rows, attrs, score, ranking):
    df = pd.DataFrame(rows)
    df = df[["layer", "head", "flat_idx"] + SCORE_COLUMNS + ["n_train", "n_val"]]
    df.attrs.update(attrs)
    rank_of = {int(flat): r for r, flat in enumerate(rank_heads(df, score, ranking=ranking), start=1)}
    df["rank"] = df["flat_idx"].map(rank_of).astype(int)
    return df


def fit_head_probes(features_dir, val_dir=None, value=None, feature_mode="head", score="auc", class_weight=None,
                    seed=42, n_jobs=-1, layers=None, C=1.0, max_iter=1000, return_scores=False, verbose=True, ranking="stable", pca=None):
    """Fits one probe per (layer, head) on the train features and scores it on the val features.

    features_dir: train feature directory (val defaults to the sibling `_val` directory); `value` picks
    labels_{value}.npy of the HelpSteer2 extract-once directories (rows labeled -1 are dropped).
    Returns a DataFrame with one row per head: layer, head, flat_idx (= layer * H + head), auc, acc, bal_acc,
    majority (majority-class accuracy on val), pos_rate (val), n_train, n_val, rank (1 = best under `score`;
    ranking "stable" = rounded score, then bal_acc, then flat_idx; "paper" = the repo's np.argsort order, DESIGN 2).
    Settings are kept in df.attrs. With return_scores, also returns the val decision scores as an array
    (L * H, N_val) (NaN rows for layers not probed). Raises when a split holds a single class. `pca` (int) projects
    every head's features onto their first `pca` train principal components before the fit (DESIGN 3.5).
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}")
    if ranking not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}")
    val_dir = val_dir or sibling_val_dir(features_dir)
    acts_tr, y_tr, _, man_tr = load_features(features_dir, value=value)
    acts_va, y_va, _, man_va = load_features(val_dir, value=value)
    y_tr, m_tr = _labels_and_mask(y_tr, features_dir)
    y_va, m_va = _labels_and_mask(y_va, val_dir)
    _check_two_classes(y_tr, f"train split {features_dir}")
    _check_two_classes(y_va, f"val split {val_dir}")
    _check_manifests(man_tr, man_va)
    L, H = man_tr["layers"], man_tr["heads"]
    layers = list(range(L)) if layers is None else sorted(layers)
    majority = max(y_va.mean(), 1 - y_va.mean())
    pos_rate = y_va.mean()
    rows, score_rows = [], {}
    for layer in layers:
        x_tr = np.asarray(acts_tr[layer], dtype=np.float32)[m_tr]
        x_va = np.asarray(acts_va[layer], dtype=np.float32)[m_va]
        feats = [(head_features(x_tr, H, h, feature_mode), head_features(x_va, H, h, feature_mode)) for h in range(H)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(f_tr, y_tr, f_va, y_va, seed, C, max_iter, class_weight, return_scores, pca) for f_tr, f_va in feats)
        for h, (res, sc) in enumerate(results):
            flat = layer * H + h
            rows.append({"layer": layer, "head": h, "flat_idx": flat, **res, "majority": majority, "pos_rate": pos_rate,
                         "n_train": int(len(y_tr)), "n_val": int(len(y_va))})
            if return_scores:
                score_rows[flat] = sc
        if verbose:
            best = max(r[score] for r in rows[-H:])
            print(f"  layer {layer:2d}: best {score} = {best:.3f}", flush=True)
    attrs = dict(feature_mode=feature_mode, score=score, class_weight=class_weight, seed=seed, C=C, max_iter=max_iter,
                 ranking=ranking, train_dir=features_dir, val_dir=val_dir, value=value, layers=L, heads=H,
                 head_dim=man_tr["head_dim"], probed_layers=layers, model=man_tr.get("model"), revision=man_tr.get("revision"),
                 dtype=man_tr.get("dtype"), attn_impl=man_tr.get("attn_impl"), pca=(int(pca) if pca else None))
    df = _finish_table(rows, attrs, score, ranking)
    if return_scores:
        scores = np.full((L * H, len(y_va)), np.nan, dtype=np.float32)
        for flat, sc in score_rows.items():
            scores[flat] = sc
        return df, scores
    return df


def cv_folds(y, groups, n_splits=5, seed=42):
    """Stratified group K-fold splits over (y, groups); n_splits is clamped to the number of groups and the smaller class."""
    y, groups = np.asarray(y), np.asarray(groups)
    n_splits = max(2, min(int(n_splits), len(np.unique(groups)), int(np.bincount(y).min())))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros((len(y), 1)), y, groups)), n_splits


def cv_head_probes(features_dir, val_dir=None, value=None, n_splits=5, feature_mode="head", score="auc", class_weight=None,
                   seed=42, n_jobs=-1, layers=None, C=1.0, max_iter=1000, return_scores=False, verbose=True, ranking="stable", pca=None):
    """Grouped stratified K-fold over the pooled train + val features (DESIGN 5.3: HelpSteer2 values are additionally
    ranked by a grouped 5-fold CV over train + val, group = prompt).

    Every head probe is fitted n_splits times (same settings as fit_head_probes) and scored out of fold; auc / acc /
    bal_acc are computed on the pooled out-of-fold predictions of all N = N_train + N_val items (n_train = n_val = N
    in the table, df.attrs["cv_splits"] = folds used). Val groups are offset so that they never collide with train
    groups. With return_scores, also returns the out-of-fold decision scores (L * H, N).
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}")
    val_dir = val_dir or sibling_val_dir(features_dir)
    acts_tr, y_tr, g_tr, man_tr = load_features(features_dir, value=value)
    acts_va, y_va, g_va, man_va = load_features(val_dir, value=value)
    y_tr, m_tr = _labels_and_mask(y_tr, features_dir)
    y_va, m_va = _labels_and_mask(y_va, val_dir)
    _check_manifests(man_tr, man_va)
    g_tr = np.asarray(g_tr)[m_tr] if g_tr is not None else np.arange(m_tr.sum())
    g_va = np.asarray(g_va)[m_va] if g_va is not None else np.arange(m_va.sum())
    g_va = g_va + (int(g_tr.max()) + 1 if len(g_tr) else 0)
    y = np.concatenate([y_tr, y_va])
    groups = np.concatenate([g_tr, g_va])
    _check_two_classes(y, f"pooled train + val of {features_dir}")
    folds, n_splits = cv_folds(y, groups, n_splits, seed)
    L, H = man_tr["layers"], man_tr["heads"]
    layers = list(range(L)) if layers is None else sorted(layers)
    majority = max(y.mean(), 1 - y.mean())
    rows, score_rows = [], {}
    for layer in layers:
        x = np.concatenate([np.asarray(acts_tr[layer], dtype=np.float32)[m_tr], np.asarray(acts_va[layer], dtype=np.float32)[m_va]])
        feats = [head_features(x, H, h, feature_mode) for h in range(H)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_cv_one)(f, y, folds, seed, C, max_iter, class_weight, return_scores, pca) for f in feats)
        for h, (res, sc) in enumerate(results):
            flat = layer * H + h
            rows.append({"layer": layer, "head": h, "flat_idx": flat, **res, "majority": majority, "pos_rate": y.mean(),
                         "n_train": int(len(y)), "n_val": int(len(y))})
            if return_scores:
                score_rows[flat] = sc
        if verbose:
            best = max(r[score] for r in rows[-H:])
            print(f"  layer {layer:2d} (cv{n_splits}): best {score} = {best:.3f}", flush=True)
    attrs = dict(feature_mode=feature_mode, score=score, class_weight=class_weight, seed=seed, C=C, max_iter=max_iter,
                 ranking=ranking, train_dir=features_dir, val_dir=val_dir, value=value, layers=L, heads=H,
                 head_dim=man_tr["head_dim"], probed_layers=layers, model=man_tr.get("model"), revision=man_tr.get("revision"),
                 dtype=man_tr.get("dtype"), attn_impl=man_tr.get("attn_impl"), cv_splits=int(n_splits), cv_requested=int(n_splits),
                 n_groups=int(len(np.unique(groups))), pca=(int(pca) if pca else None))
    df = _finish_table(rows, attrs, score, ranking)
    if return_scores:
        scores = np.full((L * H, len(y)), np.nan, dtype=np.float32)
        for flat, sc in score_rows.items():
            scores[flat] = sc
        return df, scores
    return df


def rank_heads(df, score="auc", digits=6, ranking=None):
    """flat_idx values in rank order (best first).

    ranking "stable" (default, DESIGN 5.8): descending rounded `score`, then descending bal_acc, then flat_idx.
    ranking "paper" (DESIGN 2): `np.argsort(scores)[::-1]` on the (L * H,) score vector, the reference repo's
    exact call including its order among tied scores (restricted to the probed heads). `ranking` defaults to
    df.attrs["ranking"]. Raises when `score` holds NaN (single-class val split): such a table must not be ranked.
    """
    ranking = ranking or df.attrs.get("ranking") or "stable"
    if ranking not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}")
    s = df[score].to_numpy(dtype=np.float64)
    if np.isnan(s).any():
        raise ValueError(f"{int(np.isnan(s).sum())} of {len(s)} heads have no {score} score (NaN: single-class val split?); "
                         "refusing to rank them")
    if ranking == "paper":
        order = np.argsort(scores_vector(df, score))[::-1]
        probed = set(int(i) for i in df["flat_idx"])
        return np.array([int(i) for i in order if int(i) in probed], dtype=np.int64)
    primary = -np.round(s, digits)
    secondary = -np.round(df["bal_acc"].to_numpy(dtype=np.float64), digits)
    order = np.lexsort((df["flat_idx"].to_numpy(), secondary, primary))
    return df["flat_idx"].to_numpy()[order]


def select_topk(df, ratio=0.15, score="auc", total_heads=None, k=None, ranking=None):
    """Flat indices of the top k = int(total_heads * ratio) heads (paper: 15 % of all L * H heads)."""
    if total_heads is None:
        total_heads = df.attrs.get("layers", 0) * df.attrs.get("heads", 0) or len(df)
    k = int(total_heads * ratio) if k is None else int(k)
    return [int(i) for i in rank_heads(df, score, ranking=ranking)[:k]]


def group_by_layer(flat_heads, num_heads):
    """[flat_idx, ...] -> {layer: [head, ...]} (sorted)."""
    groups = {}
    for idx in flat_heads:
        groups.setdefault(int(idx) // num_heads, []).append(int(idx) % num_heads)
    return {l: sorted(hs) for l, hs in sorted(groups.items())}


def scores_vector(df, score="acc"):
    """(L * H,) array of `score` indexed by flat_idx (NaN where not probed), for repo-style consumers."""
    n = df.attrs.get("layers", 0) * df.attrs.get("heads", 0) or int(df["flat_idx"].max()) + 1
    v = np.full(n, np.nan)
    v[df["flat_idx"].to_numpy()] = df[score].to_numpy()
    return v


def ratio_key(ratio):
    """0.05 -> 'topk_5pct' key suffix ('5pct')."""
    return f"{int(round(float(ratio) * 100))}pct"


def save_head_scores(df, path, meta=None, ratio=0.15, extra_ratios=None):
    """Writes the ranked head table as JSON (results/heads/{model}_{revision}_{value}_{mode}.json).

    Keys: meta (settings + extra meta), k (= int(L * H * ratio)), topk (flat indices in rank order),
    heads (rows sorted by rank), scores_by_flat_idx ({flat_idx: primary score} in rank order, the repo's format),
    and for `ratio` plus config.EXTRA_RATIOS (DESIGN 5.8: 5 % and 10 %) the sets k_{r}pct / topk_{r}pct.
    """
    score = df.attrs.get("score", "auc")
    ranked = df.sort_values("rank")
    total = df.attrs.get("layers", 0) * df.attrs.get("heads", 0) or len(df)
    k = int(total * ratio)
    payload = {
        "meta": {**{k_: v for k_, v in df.attrs.items()}, **(meta or {}), "ratio": ratio},
        "k": k,
        "topk": [int(i) for i in ranked["flat_idx"].head(k)],
        "heads": json.loads(ranked.to_json(orient="records")),
        "scores_by_flat_idx": {int(i): float(s) for i, s in zip(ranked["flat_idx"], ranked[score])},
    }
    for r in sorted(set(config.EXTRA_RATIOS if extra_ratios is None else extra_ratios) | {ratio}):
        payload[f"k_{ratio_key(r)}"] = int(total * r)
        payload[f"topk_{ratio_key(r)}"] = [int(i) for i in ranked["flat_idx"].head(int(total * r))]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=_json_default)
    return payload


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_head_scores(path):
    """(DataFrame with df.attrs restored, meta dict) from save_head_scores output."""
    with open(path) as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["heads"]).sort_values("flat_idx").reset_index(drop=True)
    df.attrs.update(payload["meta"])
    return df, payload["meta"]


def top_heads_from_file(path, ratio=0.15, k=None, score=None):
    """Top-k flat indices stored in a head-score JSON (uses the stored ranking; refuses tables whose primary score
    holds NaN / null entries). With `score` (a column such as "auc_half_A" written by probe-stats), the heads are
    re-ranked by that column with the stable tie-break (rounded score, bal_acc, flat_idx; DESIGN 5.8): ranking on
    half A of the selection / confirmation split keeps head selection and `eval-likelihood --eval_groups B` disjoint
    (DESIGN 4.1 / 5.5)."""
    with open(path) as f:
        payload = json.load(f)
    meta, rows = payload["meta"], payload["heads"]
    primary = meta.get("score", "auc")
    for col in {primary, score or primary}:
        bad = [r.get("flat_idx") for r in rows if col not in r or r[col] is None or (isinstance(r[col], float) and np.isnan(r[col]))]
        if bad:
            hint = "run `probe-stats` first" if col.startswith("auc_half") else "null; single-class val split?"
            raise ValueError(f"{path}: {len(bad)} heads have no {col} score ({hint}); refusing to select heads")
    if k is None:
        total = meta["layers"] * meta["heads"]
        k = int(total * ratio)
        stored = payload.get(f"topk_{ratio_key(ratio)}")
        if score is None and stored is not None and len(stored) == k:
            return [int(i) for i in stored]
    if score is None:
        ranked = [row["flat_idx"] for row in sorted(rows, key=lambda r: r["rank"])]
    else:
        ranked = [row["flat_idx"] for row in sorted(rows, key=lambda r: (-round(float(r[score]), 6), -round(float(r.get("bal_acc", 0)), 6), int(r["flat_idx"])))]
    return [int(i) for i in ranked[:k]]
