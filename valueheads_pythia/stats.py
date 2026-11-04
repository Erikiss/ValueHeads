"""Statistics for the causal tests and the ranking-stability analysis (DESIGN 5.6, 5.9, section 2 "Signifikanz").

  empirical_p        one-sided empirical p-value of the TopK metric against R random-head controls
  paired_delta_ci    paired per-item effect (intervention - base) with a (cluster) bootstrap percentile CI
  spearman_jaccard   ranking stability between two head rankings (Spearman rho + Jaccard of the top-k sets)
  ttest_paper_mode   the reference repo's one-sided one-sample t-test (analysis/significance.py, Table 10)
"""
import warnings

import numpy as np
from scipy.stats import spearmanr, ttest_1samp


def empirical_p(random_metrics, topk_metric, direction="le"):
    """One-sided empirical p-value p = (1 + #{r : metric_r <= metric_topk}) / (1 + R) (DESIGN 5.9).

    direction "le": "lower = stronger" (masking lowers an accuracy); "ge": "higher = stronger" (steering raises
    an accuracy, or an intervention raises the perplexity). NaN random entries are dropped from R. With R = 0
    the result is NaN.
    """
    random_metrics = np.asarray([m for m in np.atleast_1d(random_metrics) if not np.isnan(m)], dtype=np.float64)
    R = len(random_metrics)
    if R == 0 or topk_metric is None or np.isnan(topk_metric):
        return float("nan")
    if direction == "le":
        hits = int(np.sum(random_metrics <= topk_metric))
    elif direction == "ge":
        hits = int(np.sum(random_metrics >= topk_metric))
    else:
        raise ValueError("direction must be 'le' or 'ge'")
    return (1 + hits) / (1 + R)


def paired_delta_ci(deltas, B=1000, seed=0, groups=None, weights=None, ci=0.95):
    """Mean paired difference with a bootstrap percentile CI (DESIGN 5.9: item bootstrap, B = 1,000).

    deltas:  per-item differences (intervention - base)
    groups:  optional cluster ids (e.g. the question of a TruthfulQA pair) -> clusters are resampled instead of
             items (cluster bootstrap, DESIGN 5.3)
    weights: optional per-item weights; the statistic is then sum(w * d) / sum(w) (token-weighted perplexity
             windows); the bootstrap resamples items (or clusters) together with their weights
    Returns {"mean", "lo", "hi", "n", "n_groups", "B", "ci", "se"} with NaNs for empty input.
    """
    d = np.asarray(deltas, dtype=np.float64)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(w) != len(d):
        raise ValueError("weights must have one entry per delta")
    if groups is not None and len(np.asarray(groups)) != len(d):
        raise ValueError("groups must have one entry per delta")
    keep = np.isfinite(d) & np.isfinite(w)          # NaN items are dropped together with their group / weight
    d, w = d[keep], w[keep]
    groups = None if groups is None else np.asarray(groups)[keep]
    n = len(d)
    out = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(n), "n_groups": int(n), "B": int(B),
           "ci": ci, "se": float("nan")}
    if n == 0:
        return out

    def stat(idx):
        return float(np.sum(w[idx] * d[idx]) / np.sum(w[idx]))

    out["mean"] = stat(np.arange(n))
    rng = np.random.RandomState(seed)
    if groups is None:
        units = [np.array([i]) for i in range(n)]
    else:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        units = [np.flatnonzero(inv == u) for u in range(inv.max() + 1)]
    out["n_groups"] = len(units)
    if len(units) < 2 or B <= 0:
        return out
    boots = np.empty(B)
    for b in range(B):
        pick = rng.randint(0, len(units), len(units))
        idx = np.concatenate([units[u] for u in pick])
        boots[b] = stat(idx)
    alpha = (1 - ci) / 2
    out["lo"], out["hi"] = float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))
    out["se"] = float(np.std(boots, ddof=1))
    return out


def _ordering(x, scores):
    """Flat head indices in rank order (best first) from an ordering, a {flat_idx: score} dict or a score vector."""
    if isinstance(x, dict):
        return np.array(sorted(x, key=lambda k: (-float(x[k]), k)), dtype=np.int64)
    x = np.asarray(x)
    if scores:
        # stable descending argsort on rounded scores (DESIGN 5.8 tie-breaking; NaN ranks last)
        s = np.round(x.astype(np.float64), 6)
        s = np.where(np.isnan(s), -np.inf, s)
        return np.argsort(-s, kind="stable")
    return x.astype(np.int64)


def spearman_jaccard(rank_a, rank_b, k, scores=False):
    """Ranking stability between two head rankings (DESIGN 5.6).

    rank_a, rank_b: orderings of the same flat head indices (best first, e.g. probing.rank_heads output), or
    {flat_idx: score} dicts, or, with scores=True, per-head score vectors indexed by flat_idx.
    Returns {"spearman": rho of the rank positions, "jaccard": |topk_a & topk_b| / |topk_a | topk_b|,
    "overlap": |topk_a & topk_b|, "k": k, "n": number of heads}.
    """
    a, b = _ordering(rank_a, scores), _ordering(rank_b, scores)
    if sorted(a.tolist()) != sorted(b.tolist()):
        raise ValueError("rank_a and rank_b must rank the same set of heads")
    n = len(a)
    pos_a = np.empty(n, dtype=np.int64)
    pos_b = np.empty(n, dtype=np.int64)
    pos_a[a] = np.arange(n)
    pos_b[b] = np.arange(n)
    rho = float(spearmanr(pos_a, pos_b).statistic) if n > 1 else float("nan")
    top_a, top_b = set(a[:k].tolist()), set(b[:k].tolist())
    inter, union = len(top_a & top_b), len(top_a | top_b)
    return {"spearman": rho, "jaccard": inter / union if union else float("nan"), "overlap": inter, "k": int(k), "n": int(n)}


def ttest_paper_mode(random_scores, topk_score):
    """The reference repo's significance test (analysis/significance.py; paper App. G, Table 10).

    One-sided one-sample t-test of the random-head masking scores against the TopK score, H1: mean(random) > topk.
    Returns {"p", "random_mean", "random_std" (ddof=1), "n_random", "topk", "t"}; p is NaN with fewer than two
    random scores.
    """
    r = np.asarray(random_scores, dtype=np.float64)
    out = {"random_mean": float(r.mean()) if len(r) else float("nan"),
           "random_std": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
           "n_random": int(len(r)), "topk": float(topk_score), "p": float("nan"), "t": float("nan")}
    if len(r) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # identical random scores -> scipy precision warning
            res = ttest_1samp(r, topk_score, alternative="greater")
        out["p"], out["t"] = float(res.pvalue), float(res.statistic)
    return out


def summarize_random(random_metrics):
    """mean / std (ddof=1) / min / max / n of the random-control metrics (NaNs dropped)."""
    r = np.asarray([m for m in np.atleast_1d(random_metrics) if m is not None and not np.isnan(m)], dtype=np.float64)
    return {"mean": float(r.mean()) if len(r) else float("nan"), "std": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
            "min": float(r.min()) if len(r) else float("nan"), "max": float(r.max()) if len(r) else float("nan"), "n": int(len(r))}
