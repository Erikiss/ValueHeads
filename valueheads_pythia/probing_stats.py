"""Advanced probing statistics (DESIGN 5.3-5.8): cluster-bootstrap CIs, permutation null of the max-over-heads
score (Westfall-Young), selection/confirmation split, split stability and the significant head set.

Builds on `probing.fit_head_probes` (which refits the probes and returns the per-item val decision scores) and
never changes its public API. Val scores are cached next to the head JSON (`*_valscores.npz`) so that the
bootstrap and the confirmation split run without any refit; only the permutation null refits (all heads of a
layer and all permutations at once with a batched torch logistic regression, L-BFGS, verified against sklearn).

Terminology: K = number of heads (rows of a score matrix), N = number of val items, P = permutations,
B = bootstrap resamples, flat_idx = layer * H + head.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from .extract import load_features
from .probing import fit_head_probes, load_head_scores, rank_heads, save_head_scores, select_topk

STATISTICS = ["max_auc", "max_acc"]


# ----------------------------------------------------------------------------- exact AUC helpers
def auc_matrix(scores, y):
    """Exact ROC-AUC of every row of `scores` (K, N) against binary labels `y` (N,), ties counted 1/2.

    Equals sklearn.metrics.roc_auc_score row by row (Mann-Whitney form with average ranks).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[None]
    y = np.asarray(y)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(len(scores), np.nan)
    ranks = rankdata(scores, axis=1)
    return (ranks[:, pos].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def acc_matrix(scores, y):
    """Accuracy of the sign rule (score > 0 -> class 1) for every row of `scores` (K, N)."""
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores[None]
    return ((scores > 0).astype(np.int64) == np.asarray(y)[None]).mean(axis=1)


class WeightedAUC:
    """Weighted ROC-AUC of K fixed score rows under changing item weights (bootstrap counts), exact with ties.

    Sorting and tie bookkeeping happen once; `auc(w)` then costs one gather + one cumsum per row.
    With integer weights this equals the unweighted AUC on the expanded (repeated) sample.
    """

    def __init__(self, scores, y):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores[None]
        self.y = np.asarray(y).astype(np.float64)
        K, N = scores.shape
        self.order = np.argsort(scores, axis=1, kind="stable")                    # ascending per row
        s_sorted = np.take_along_axis(scores, self.order, axis=1)
        self.y_sorted = self.y[self.order]
        new = np.ones((K, N), dtype=bool)
        new[:, 1:] = s_sorted[:, 1:] != s_sorted[:, :-1]
        idx = np.broadcast_to(np.arange(N), (K, N))
        self.start = np.maximum.accumulate(np.where(new, idx, 0), axis=1)       # first index of the tie group
        is_end = np.ones((K, N), dtype=bool)
        is_end[:, :-1] = new[:, 1:]
        end = np.where(is_end, idx, N - 1)
        self.end = np.minimum.accumulate(end[:, ::-1], axis=1)[:, ::-1]          # last index of the tie group

    def auc(self, w):
        """AUC per row for item weights `w` (N,) (NaN rows when a class has zero total weight)."""
        w_s = np.asarray(w, dtype=np.float64)[self.order]
        neg_w = w_s * (1.0 - self.y_sorted)
        pos_w = w_s * self.y_sorted
        c_incl = np.cumsum(neg_w, axis=1)
        c_excl = c_incl - neg_w
        neg_less = np.take_along_axis(c_excl, self.start, axis=1)
        neg_eq = np.take_along_axis(c_incl, self.end, axis=1) - neg_less
        w_pos, w_neg = pos_w.sum(axis=1), neg_w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (pos_w * (neg_less + 0.5 * neg_eq)).sum(axis=1) / (w_pos * w_neg)


# ----------------------------------------------------------------------------- val predictions (refit once, cache)
def valscores_path(heads_json_path):
    """results/heads/{...}.json -> results/heads/{...}_valscores.npz"""
    base = heads_json_path[:-5] if heads_json_path.endswith(".json") else heads_json_path
    return base + "_valscores.npz"


def save_valscores(path, scores, labels, groups, meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, scores=np.asarray(scores, dtype=np.float32), labels=np.asarray(labels),
                        groups=np.asarray(groups), meta=json.dumps(meta or {}))
    return path


def load_valscores(path):
    """(scores (K, N) float32, labels (N,), groups (N,), meta dict)."""
    with np.load(path) as z:
        return z["scores"], z["labels"], z["groups"], json.loads(str(z["meta"]))


def val_predictions(train_dir, val_dir=None, value=None, feature_mode="head", class_weight="balanced", seed=42,
                    C=1.0, max_iter=1000, n_jobs=-1, layers=None, cache_path=None, force=False, verbose=False, pca=None):
    """Per-item val decision scores of every head probe: (df, scores (L * H, N_val), labels (N_val,), groups (N_val,)).

    The core `probe` command stores only per-head summaries, so the probes are refitted once with identical
    settings (deterministic lbfgs -> identical AUCs) and the scores are cached at `cache_path` (npz); a cached
    file whose settings match is loaded instead of refitting (`force` refits anyway).
    """
    settings = {"train_dir": train_dir, "val_dir": val_dir, "value": value, "feature_mode": feature_mode,
                "class_weight": class_weight, "seed": seed, "C": C, "max_iter": max_iter, "layers": layers, "pca": pca}
    if cache_path and os.path.exists(cache_path) and not force:
        scores, labels, groups, meta = load_valscores(cache_path)
        same = all(meta.get("settings", {}).get(k) == v for k, v in settings.items() if k not in ("train_dir", "val_dir"))
        if same:
            df = pd.DataFrame(meta["df"]) if "df" in meta else None
            if df is not None:
                df.attrs.update(meta.get("attrs", {}))
            return df, scores, labels, groups
    df, scores = fit_head_probes(train_dir, val_dir, value=value, feature_mode=feature_mode, score="auc",
                                 class_weight=class_weight, seed=seed, n_jobs=n_jobs, layers=layers, C=C,
                                 max_iter=max_iter, return_scores=True, verbose=verbose, pca=pca)
    _, labels, groups, _ = load_features(df.attrs["val_dir"], value=value)
    labels = np.asarray(labels)
    mask = labels >= 0
    labels = labels[mask].astype(np.int64)
    groups = (np.asarray(groups)[mask] if groups is not None else np.arange(len(labels))).astype(np.int64)
    if cache_path:
        meta = {"settings": settings, "df": json.loads(df.to_json(orient="records")),
                "attrs": {k: v for k, v in df.attrs.items() if isinstance(v, (str, int, float, bool, list, type(None)))}}
        save_valscores(cache_path, scores, labels, groups, meta)
    return df, scores, labels, groups


# ----------------------------------------------------------------------------- cluster bootstrap (DESIGN 5.3)
def cluster_bootstrap(scores_per_item, labels, groups, B=1000, seed=0, alpha=0.05):
    """Percentile CIs of the per-head AUC from B group-level resamples of the val predictions (no refit).

    scores_per_item: (K, N) decision scores (rows = heads, NaN rows allowed), labels (N,) in {0, 1},
    groups (N,) cluster ids (question / dialogue pair / prompt). Groups are drawn with replacement; an item's
    weight is the number of times its group was drawn. Returns a DataFrame indexed by row: auc (point estimate),
    ci_lo, ci_hi ((alpha/2, 1 - alpha/2) percentiles), boot_mean, boot_std, n_groups, B.
    """
    scores = np.asarray(scores_per_item, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[None]
    labels, groups = np.asarray(labels), np.asarray(groups)
    if scores.shape[1] != len(labels) or len(groups) != len(labels):
        raise ValueError(f"scores {scores.shape}, labels {labels.shape}, groups {groups.shape} do not match")
    valid = ~np.isnan(scores).any(axis=1)
    rows = np.flatnonzero(valid)
    _, group_idx = np.unique(groups, return_inverse=True)
    n_groups = int(group_idx.max()) + 1
    rng = np.random.RandomState(seed)
    counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=B)        # (B, n_groups)
    prep = WeightedAUC(scores[rows], labels)
    boot = np.empty((B, len(rows)))
    for b in range(B):
        boot[b] = prep.auc(counts[b][group_idx])
    out = pd.DataFrame(index=np.arange(len(scores)), columns=["auc", "ci_lo", "ci_hi", "boot_mean", "boot_std"], dtype=float)
    out.loc[rows, "auc"] = auc_matrix(scores[rows], labels)
    out.loc[rows, "ci_lo"] = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    out.loc[rows, "ci_hi"] = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    out.loc[rows, "boot_mean"] = np.nanmean(boot, axis=0)
    out.loc[rows, "boot_std"] = np.nanstd(boot, axis=0, ddof=1)
    out["n_groups"] = n_groups
    out["B"] = B
    out.attrs.update(B=B, seed=seed, alpha=alpha, n_groups=n_groups)
    return out


# ----------------------------------------------------------------------------- permutations (DESIGN 5.4)
def permute_labels(labels, groups, rng, within_group=True):
    """One label permutation: within each group (default) or globally but stratified by group size.

    Within-group permutation keeps every group's label counts (TruthfulQA question, HH pair); the stratified
    global variant (HelpSteer2, DESIGN 5.4) permutes labels among all items whose group has the same size.
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    if not within_group:
        _, inv, sizes = np.unique(groups, return_inverse=True, return_counts=True)
        groups = sizes[inv]                                   # stratum = group size
    order = np.lexsort((np.arange(len(labels)), groups))      # block order, ascending index inside a block
    shuffled = np.lexsort((rng.random_sample(len(labels)), groups))
    out = np.empty_like(labels)
    out[order] = labels[shuffled]
    return out


def permutation_label_sets(labels, groups, P, seed=0, within_group=True):
    """(P, N) int array of permuted label vectors (seeded, reproducible)."""
    rng = np.random.RandomState(seed)
    return np.stack([permute_labels(labels, groups, rng, within_group) for _ in range(P)])


def _sample_weights(y, class_weight):
    """sklearn semantics: 'balanced' -> n / (2 * count(class)); None -> 1. y: (P, N) tensor in {0, 1}."""
    if class_weight is None or class_weight == "none":
        return torch.ones_like(y, dtype=torch.float32)
    if class_weight != "balanced":
        raise ValueError("class_weight must be None or 'balanced'")
    n = y.shape[1]
    n_pos = y.sum(dim=1, keepdim=True).clamp(min=1)
    n_neg = (n - y.sum(dim=1, keepdim=True)).clamp(min=1)
    return torch.where(y > 0.5, n / (2.0 * n_pos), n / (2.0 * n_neg)).to(torch.float32)


def torch_batched_logreg(x_tr, y_tr, x_va, class_weight="balanced", C=1.0, max_iter=200, device=None,
                         history_size=20, tolerance_grad=1e-6, return_params=False):
    """L2 logistic regression for all heads (and all label sets) at once, L-BFGS, sklearn objective.

    x_tr: (H, N, D) train features of H heads; y_tr: (N,) labels or (P, N) label sets (permutations) sharing
    the features; x_va: (H, M, D) val features. Minimizes, per (label set p, head h),
        C * sum_i sw_i * log(1 + exp(-t_i (w . x_i + b))) + 0.5 ||w||^2         (sw = sklearn sample weights)
    jointly with torch.optim.LBFGS (strong-Wolfe line search, at most `max_iter` iterations). Returns the val
    decision scores (P, H, M) ((H, M) for 1-D y_tr); with return_params also (W (P, H, D), b (P, H)).
    """
    device = torch.device(device or "cpu")
    X = torch.as_tensor(np.asarray(x_tr, dtype=np.float32), device=device)
    Xv = torch.as_tensor(np.asarray(x_va, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(y_tr), device=device).to(torch.float32)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[None]
    P, N = y.shape
    H, N2, D = X.shape
    if N2 != N:
        raise ValueError(f"x_tr has {N2} items, y_tr {N}")
    sw = _sample_weights(y, class_weight)[:, None, :]                       # (P, 1, N)
    t = (2.0 * y - 1.0)[:, None, :]                                          # (P, 1, N)
    W = torch.zeros((P, H, D), device=device, requires_grad=True)
    b = torch.zeros((P, H), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=history_size, line_search_fn="strong_wolfe",
                            tolerance_grad=tolerance_grad, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        z = torch.einsum("hnd,phd->phn", X, W) + b[..., None]
        loss = C * (sw * torch.nn.functional.softplus(-t * z)).sum() + 0.5 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        scores = (torch.einsum("hmd,phd->phm", Xv, W) + b[..., None]).cpu().numpy()
        params = (W.detach().cpu().numpy(), b.detach().cpu().numpy())
    if squeeze:
        scores, params = scores[0], (params[0][0], params[1][0])
    return (scores, params) if return_params else scores


def _layer_features(acts, layer, mask, num_heads, feature_mode):
    """(H, N, D) or (H, N, 2D) float32 features of all heads of one layer."""
    x = np.asarray(acts[layer], dtype=np.float32)[mask]
    x = x.reshape(len(x), num_heads, -1)
    feats = np.transpose(x, (1, 0, 2))
    if feature_mode == "head+mean":
        mean = np.broadcast_to(x.mean(axis=1)[None], feats.shape)
        feats = np.concatenate([feats, mean], axis=-1)
    return np.ascontiguousarray(feats)


def _pca_heads(x_tr, x_va, n_components, seed=42):
    """Per-head PCA (fitted on train) of (H, N, D) feature stacks -> (H, N, n) (same projection as probing.pca_project)."""
    from .probing import pca_project
    pairs = [pca_project(x_tr[h], x_va[h], n_components, seed) for h in range(len(x_tr))]
    return np.stack([a for a, _ in pairs]), np.stack([b for _, b in pairs])


def permutation_null(features_dir, P=100, within_group=True, statistic="max_auc", seed=0, val_dir=None, value=None,
                     feature_mode="head", class_weight="balanced", C=1.0, layers=None, device=None, max_iter=200,
                     perm_batch=25, verbose=True, pca=None, probe_seed=42):
    """Null distribution of max_h score(h) over P label permutations (Westfall-Young, FWER control, DESIGN 5.4).

    Train and val labels are each permuted within their groups (or globally stratified by group size), every
    probe is refitted on the permuted train labels (batched torch L-BFGS, all heads of a layer and up to
    `perm_batch` permutations at once) and scored on the permuted val labels; the statistic is the max over
    heads of the val AUC ('max_auc') or accuracy ('max_acc'). Returns a dict with threshold (95th percentile),
    dist (P,), null_by_head (P, L * H) (NaN for layers not probed), argmax_head (P,), and the settings.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}")
    t0 = time.time()
    val_dir = val_dir or features_dir.rstrip("/")[:-len("_train")] + "_val"
    acts_tr, y_tr, g_tr, man = load_features(features_dir, value=value)
    acts_va, y_va, g_va, _ = load_features(val_dir, value=value)
    y_tr, y_va = np.asarray(y_tr), np.asarray(y_va)
    m_tr, m_va = y_tr >= 0, y_va >= 0
    g_tr = np.asarray(g_tr)[m_tr] if g_tr is not None else np.arange(m_tr.sum())
    g_va = np.asarray(g_va)[m_va] if g_va is not None else np.arange(m_va.sum())
    y_tr, y_va = y_tr[m_tr].astype(np.int64), y_va[m_va].astype(np.int64)
    L, H = man["layers"], man["heads"]
    layers = list(range(L)) if layers is None else sorted(layers)
    perm_tr = permutation_label_sets(y_tr, g_tr, P, seed, within_group)
    perm_va = permutation_label_sets(y_va, g_va, P, seed + 1, within_group)
    null = np.full((P, L * H), np.nan)
    score_fn = auc_matrix if statistic == "max_auc" else acc_matrix
    for layer in layers:
        x_tr = _layer_features(acts_tr, layer, m_tr, H, feature_mode)
        x_va = _layer_features(acts_va, layer, m_va, H, feature_mode)
        if pca:
            x_tr, x_va = _pca_heads(x_tr, x_va, pca, probe_seed)
        for start in range(0, P, perm_batch):
            ys = perm_tr[start:start + perm_batch]
            scores = torch_batched_logreg(x_tr, ys, x_va, class_weight=class_weight, C=C, max_iter=max_iter, device=device)
            for p in range(len(ys)):
                null[start + p, layer * H:(layer + 1) * H] = score_fn(scores[p], perm_va[start + p])
        if verbose:
            print(f"  perm layer {layer:2d}: null max {np.nanmax(null[:, layer * H:(layer + 1) * H]):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    dist = np.nanmax(null, axis=1)
    return {"threshold": float(np.percentile(dist, 95)), "dist": dist, "null_by_head": null,
            "argmax_head": np.nanargmax(null, axis=1), "P": P, "within_group": within_group, "statistic": statistic,
            "seed": seed, "feature_mode": feature_mode, "class_weight": class_weight, "C": C, "layers": layers, "pca": pca,
            "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "seconds": round(time.time() - t0, 1)}


def westfall_young_p(observed, dist):
    """FWER-adjusted p-value per head: (1 + #{p : max_p >= obs_h}) / (1 + P). NaN observations stay NaN."""
    observed, dist = np.asarray(observed, dtype=float), np.asarray(dist, dtype=float)
    p = (1 + (dist[None, :] >= observed[:, None]).sum(axis=1)) / (1 + len(dist))
    return np.where(np.isnan(observed), np.nan, p)


def per_head_p(observed, null_by_head):
    """Unadjusted per-head permutation p-value: (1 + #{p : null_hp >= obs_h}) / (1 + P)."""
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null_by_head, dtype=float)
    p = (1 + (null >= observed[None, :]).sum(axis=0)) / (1 + len(null))
    return np.where(np.isnan(observed) | np.isnan(null).all(axis=0), np.nan, p)


# ----------------------------------------------------------------------------- selection (DESIGN 5.5, 5.8)
def select_significant(df, ci_lo=None, perm_threshold=None):
    """Flat indices (rank order) of heads whose bootstrap CI lower bound exceeds the permutation threshold.

    `ci_lo` defaults to df['ci_lo'], `perm_threshold` to df.attrs['perm_threshold']; both must be available.
    """
    lo = np.asarray(df["ci_lo"] if ci_lo is None else ci_lo, dtype=float)
    thr = df.attrs.get("perm_threshold") if perm_threshold is None else perm_threshold
    if thr is None or np.isnan(thr):
        raise ValueError("perm_threshold is required (run permutation_null first)")
    keep = pd.Series(lo > float(thr), index=df.index)
    ranked = rank_heads(df, df.attrs.get("score", "auc"))
    flat_ok = set(df.loc[keep, "flat_idx"].astype(int))
    return [int(i) for i in ranked if int(i) in flat_ok]


def _rank_by_auc(auc, flat_idx, digits=6):
    primary = np.where(np.isnan(auc), np.inf, -np.round(auc, digits))
    order = np.lexsort((flat_idx, primary))
    return flat_idx[order]


def selection_confirmation_split(scores_per_item, labels, groups, seed=0, ratio=0.15, flat_idx=None):
    """Groupwise half split of val: rank heads on half A, report the AUC of the selected heads on half B.

    Returns a dict with per-head arrays (auc_A, auc_B, auc_full), both directions (A->B and B->A) of
    top1 / top-k (k = int(ratio * K)) with their selection and confirmation AUCs, their mean, and the
    winner's-curse gap auc_sel(top1) - auc_conf(top1). Groups are split by a seeded permutation.
    """
    scores = np.asarray(scores_per_item, dtype=np.float64)
    labels, groups = np.asarray(labels), np.asarray(groups)
    K = len(scores)
    flat_idx = np.arange(K) if flat_idx is None else np.asarray(flat_idx)
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(uniq)                 # confirmation_halves() reproduces this split from (groups, seed)
    half_a = np.isin(groups, perm[:len(perm) // 2])
    half_b = ~half_a
    auc_a, auc_b = auc_matrix(scores[:, half_a], labels[half_a]), auc_matrix(scores[:, half_b], labels[half_b])
    auc_full = auc_matrix(scores, labels)
    k = max(1, int(ratio * K))
    n_a = len(perm) // 2
    out = {"seed": seed, "ratio": ratio, "k": k, "n_groups_A": int(n_a), "n_groups_B": int(len(perm) - n_a),
           "n_items_A": int(half_a.sum()), "n_items_B": int(half_b.sum()),
           "auc_A": auc_a, "auc_B": auc_b, "auc_full": auc_full, "flat_idx": flat_idx}
    for name, sel, conf in (("A_to_B", auc_a, auc_b), ("B_to_A", auc_b, auc_a)):
        ranked = _rank_by_auc(sel, flat_idx)
        pos = {int(f): i for i, f in enumerate(flat_idx)}
        top = [pos[int(f)] for f in ranked[:k]]
        out[name] = {"top1": int(flat_idx[top[0]]), "top1_sel": float(sel[top[0]]), "top1_conf": float(conf[top[0]]),
                     "topk": [int(flat_idx[i]) for i in top], "topk_sel_mean": float(np.nanmean(sel[top])),
                     "topk_conf_mean": float(np.nanmean(conf[top])), "topk_conf_min": float(np.nanmin(conf[top]))}
    out["top1_conf_mean"] = float(np.mean([out["A_to_B"]["top1_conf"], out["B_to_A"]["top1_conf"]]))
    out["topk_conf_mean"] = float(np.mean([out["A_to_B"]["topk_conf_mean"], out["B_to_A"]["topk_conf_mean"]]))
    out["winners_curse_gap"] = float(np.mean([out["A_to_B"]["top1_sel"] - out["A_to_B"]["top1_conf"],
                                              out["B_to_A"]["top1_sel"] - out["B_to_A"]["top1_conf"]]))
    return out


def confirmation_halves(groups, seed=0):
    """(half A group ids, half B group ids) of the selection / confirmation split: the same seeded permutation of the
    unique group ids as selection_confirmation_split (A = first half)."""
    uniq = np.unique(np.asarray(groups))
    perm = np.random.RandomState(seed).permutation(uniq)
    n_a = len(perm) // 2
    return set(int(g) for g in perm[:n_a]), set(int(g) for g in perm[n_a:])


def confirmation_groups(heads_json):
    """(half A, half B) group ids of the confirmation split stored by probe-stats for a head table (reads the split seed
    from the JSON meta and the val groups from the cached *_valscores.npz). Lets eval-likelihood evaluate on half B
    only while the heads are ranked on half A (DESIGN 4.1: head selection and evaluation disjoint)."""
    with open(heads_json) as f:
        meta = json.load(f)["meta"]
    conf = meta.get("confirmation")
    if not conf:
        raise ValueError(f"{heads_json} has no confirmation split (run `probe-stats` first)")
    cache = meta.get("valscores") or valscores_path(heads_json)
    if not os.path.exists(cache):
        raise FileNotFoundError(f"val scores of {heads_json} not found at {cache} (run `probe-stats` first)")
    _, _, groups, _ = load_valscores(cache)
    a, b = confirmation_halves(groups, int(conf.get("seed", 0)))
    if len(a) != int(conf.get("n_groups_A", len(a))) or len(b) != int(conf.get("n_groups_B", len(b))):
        raise ValueError(f"{heads_json}: recomputed confirmation split ({len(a)}/{len(b)} groups) differs from the stored one "
                         f"({conf.get('n_groups_A')}/{conf.get('n_groups_B')})")
    return a, b


# ----------------------------------------------------------------------------- stability (DESIGN 5.6)
def aligned_scores(df_a, df_b, score="auc"):
    """(scores_a, scores_b) aligned on the flat indices present in both tables."""
    a = df_a.set_index("flat_idx")[score]
    b = df_b.set_index("flat_idx")[score]
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(dtype=float), b.loc[common].to_numpy(dtype=float)


def spearman_jaccard(df_a, df_b, ratio=0.15, score="auc"):
    """(Spearman rho of the score vectors, Jaccard of the top-k sets, k) between two head tables."""
    a, b = aligned_scores(df_a, df_b, score)
    ok = ~(np.isnan(a) | np.isnan(b))
    rho = float(spearmanr(a[ok], b[ok]).correlation) if ok.sum() > 2 else float("nan")
    top_a, top_b = set(select_topk(df_a, ratio, score)), set(select_topk(df_b, ratio, score))
    k = len(top_a)
    jac = len(top_a & top_b) / len(top_a | top_b) if (top_a | top_b) else float("nan")
    return rho, float(jac), k


def split_stability(dfs, ratio=0.15, score="auc", names=None):
    """Ranking stability across splits / seeds: pairwise Spearman and Jaccard@k plus the mean-score ranking.

    dfs: list of head tables (fit_head_probes output or load_head_scores) of the same model on different
    question / prompt splits or probe seeds. Returns {"pairs": DataFrame(a, b, spearman, jaccard, k),
    "spearman": (n, n) matrix, "jaccard": (n, n) matrix, "spearman_mean", "spearman_min", "jaccard_mean",
    "jaccard_min", "k", "mean_scores": DataFrame(flat_idx, layer, head, score mean/std over splits, rank)}.
    """
    n = len(dfs)
    names = list(names) if names is not None else [str(i) for i in range(n)]
    sp, jc = np.eye(n), np.eye(n)
    rows, k = [], int(ratio * (dfs[0].attrs.get("layers", 0) * dfs[0].attrs.get("heads", 0) or len(dfs[0])))
    for i in range(n):
        for j in range(i + 1, n):
            rho, jac, k = spearman_jaccard(dfs[i], dfs[j], ratio, score)
            sp[i, j] = sp[j, i] = rho
            jc[i, j] = jc[j, i] = jac
            rows.append({"a": names[i], "b": names[j], "spearman": rho, "jaccard": jac, "k": k})
    pairs = pd.DataFrame(rows, columns=["a", "b", "spearman", "jaccard", "k"])
    stacked = pd.concat([d.set_index("flat_idx")[score].rename(nm) for d, nm in zip(dfs, names)], axis=1)
    mean = pd.DataFrame({"flat_idx": stacked.index, f"{score}_mean": stacked.mean(axis=1).to_numpy(),
                         f"{score}_std": stacked.std(axis=1, ddof=1).to_numpy() if n > 1 else np.nan})
    H = dfs[0].attrs.get("heads") or int(dfs[0]["head"].max()) + 1
    mean["layer"], mean["head"] = mean["flat_idx"] // H, mean["flat_idx"] % H
    order = _rank_by_auc(mean[f"{score}_mean"].to_numpy(), mean["flat_idx"].to_numpy())
    mean["rank"] = mean["flat_idx"].map({int(f): r for r, f in enumerate(order, start=1)}).astype(int)
    off = ~np.eye(n, dtype=bool)
    return {"pairs": pairs, "spearman": sp, "jaccard": jc, "k": k, "n": n, "names": names,
            "spearman_mean": float(sp[off].mean()) if n > 1 else float("nan"),
            "spearman_min": float(sp[off].min()) if n > 1 else float("nan"),
            "jaccard_mean": float(jc[off].mean()) if n > 1 else float("nan"),
            "jaccard_min": float(jc[off].min()) if n > 1 else float("nan"),
            "mean_scores": mean.sort_values("rank").reset_index(drop=True)}


# ----------------------------------------------------------------------------- orchestration
STATS_COLUMNS = ["ci_lo", "ci_hi", "boot_std", "p_adj", "p_head", "signif", "auc_half_A", "auc_half_B"]
STATS_ATTRS = ["bootstrap_B", "bootstrap_seed", "bootstrap_alpha", "n_groups", "perm_threshold", "perm_P", "perm_within_group",
               "perm_statistic", "perm_seed", "perm_dist", "perm_seconds", "signif_heads", "n_signif", "confirmation", "stats_done",
               "stats_seconds", "valscores", "val_auc_max_deviation"]


def attach_stats(df, boot=None, perm=None, confirmation=None, ratio=0.15):
    """Adds ci_lo / ci_hi / boot_std / p_adj / p_head / signif columns and the stats meta to a head table (in place)."""
    if boot is not None:
        for col in ("ci_lo", "ci_hi", "boot_std"):
            df[col] = df["flat_idx"].map(boot[col]).astype(float)
        df.attrs.update(bootstrap_B=int(boot.attrs.get("B", 0)), bootstrap_seed=boot.attrs.get("seed"),
                        bootstrap_alpha=boot.attrs.get("alpha"), n_groups=int(boot.attrs.get("n_groups", 0)))
    if perm is not None:
        obs = df["auc" if perm["statistic"] == "max_auc" else "acc"].to_numpy(dtype=float)
        df["p_adj"] = westfall_young_p(obs, perm["dist"])
        df["p_head"] = per_head_p(obs, perm["null_by_head"][:, df["flat_idx"].to_numpy()])
        df.attrs.update(perm_threshold=float(perm["threshold"]), perm_P=int(perm["P"]), perm_within_group=bool(perm["within_group"]),
                        perm_statistic=perm["statistic"], perm_seed=perm["seed"], perm_dist=[float(x) for x in perm["dist"]],
                        perm_seconds=perm.get("seconds"))
    if boot is not None and perm is not None:
        signif = select_significant(df)
        df["signif"] = df["flat_idx"].isin(signif)
        df.attrs.update(signif_heads=signif, n_signif=len(signif))
    if confirmation is not None:
        keep = {k: v for k, v in confirmation.items() if k not in ("auc_A", "auc_B", "auc_full", "flat_idx")}
        df.attrs.update(confirmation=keep)
        pos = {int(f): i for i, f in enumerate(confirmation["flat_idx"])}
        idx = df["flat_idx"].map(pos)
        df["auc_half_A"] = confirmation["auc_A"][idx.to_numpy()]
        df["auc_half_B"] = confirmation["auc_B"][idx.to_numpy()]
    df.attrs["ratio"] = ratio
    return df


def run_stats(heads_json, train_dir, val_dir=None, value=None, B=1000, P=100, within_group=True, statistic="max_auc",
              seed=0, device=None, n_jobs=-1, max_iter_torch=200, force=False, verbose=True):
    """Computes bootstrap CIs, the permutation null, the significant set and the confirmation split for one
    head-score JSON written by the `probe` command and rewrites it with the added columns / meta.

    Probe settings (feature mode, class weight, seed, C, max_iter) are taken from the JSON meta so that the
    refitted val scores reproduce the stored AUCs. Returns the augmented DataFrame.
    """
    t0 = time.time()
    df0, meta = load_head_scores(heads_json)
    if meta.get("stats_done") and not force:
        if verbose:
            print(f"[stats] skip {heads_json}: stats present (n_signif={meta.get('n_signif')})")
        return df0
    fm, cw, pseed = meta.get("feature_mode", "head"), meta.get("class_weight"), int(meta.get("seed", 42))
    C, mi, layers = float(meta.get("C", 1.0)), int(meta.get("max_iter", 1000)), meta.get("probed_layers")
    pca = meta.get("pca") or None
    cache = valscores_path(heads_json)
    df, scores, labels, groups = val_predictions(train_dir, val_dir, value=value, feature_mode=fm, class_weight=cw, seed=pseed,
                                                 C=C, max_iter=mi, n_jobs=n_jobs, layers=layers, cache_path=cache,
                                                 force=force, verbose=False, pca=pca)
    if df is None:
        df = df0
    # to_numeric: a JSON null (NaN AUC of an older single-class table) must not turn the column into object dtype
    stored = pd.to_numeric(df0.set_index("flat_idx")["auc"], errors="coerce")
    recomputed = pd.Series(auc_matrix(scores, labels), index=np.arange(len(scores))).reindex(stored.index)
    if stored.isna().all():
        raise ValueError(f"{heads_json} has no AUC values (single-class val split?); statistics are undefined")
    max_dev = float(np.nanmax(np.abs(recomputed.to_numpy() - stored.to_numpy())))
    if max_dev > 1e-3 and verbose:
        print(f"[stats] warning: refitted val AUCs deviate from the stored ones by up to {max_dev:.4f}")
    df = df0.drop(columns=[c for c in STATS_COLUMNS if c in df0.columns])
    df.attrs.clear()                                   # pandas propagates attrs; drop the stale stats keys
    df.attrs.update({k: v for k, v in meta.items() if k not in STATS_ATTRS})
    boot = cluster_bootstrap(scores, labels, groups, B=B, seed=seed)
    perm = permutation_null(train_dir, P=P, within_group=within_group, statistic=statistic, seed=seed, val_dir=val_dir,
                            value=value, feature_mode=fm, class_weight=cw, C=C, layers=layers, device=device,
                            max_iter=max_iter_torch, verbose=verbose, pca=pca, probe_seed=pseed) if P > 0 else None
    conf = selection_confirmation_split(scores, labels, groups, seed=seed, ratio=float(meta.get("ratio", 0.15)),
                                        flat_idx=np.arange(len(scores)))
    attach_stats(df, boot, perm, conf, ratio=float(meta.get("ratio", 0.15)))
    extra = {"stats_done": True, "stats_seconds": round(time.time() - t0, 1), "valscores": cache,
             "val_auc_max_deviation": max_dev}
    save_head_scores(df, heads_json, meta=extra, ratio=float(meta.get("ratio", 0.15)))
    if verbose:
        best = df.sort_values("rank").iloc[0]
        thr = df.attrs.get("perm_threshold")
        print(f"[stats] {os.path.basename(heads_json)}: top L{int(best.layer)}.H{int(best["head"])} auc={best.auc:.3f} "
              f"CI [{best.ci_lo:.3f}, {best.ci_hi:.3f}], perm thr={thr if thr is None else round(thr, 3)}, "
              f"n_signif={df.attrs.get('n_signif')}, confirmation top1 {conf['top1_conf_mean']:.3f} "
              f"(gap {conf['winners_curse_gap']:+.3f}); {time.time() - t0:.1f}s")
    return df
