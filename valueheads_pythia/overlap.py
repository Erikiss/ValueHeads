"""Overlap of value heads with the in-context-learning circuit of Qwen3 (Gao & Gao, arXiv 2611.05344).

Question: are the heads whose attention output (the o_proj input, last token) encodes a value -- the top-k of a probe
ranking, k = 15 % of all heads as in arXiv 2611.02503 -- the heads to which the ICL census / named circuit assigns a
role? Raw overlap counts are no evidence: value heads and ICL roles both sit in the middle-to-late layers, so every
test conditions on the layer profile of the value heads, and a verdict is only given after the null and Holm.

Statistics, per model variant x ranking x value x ICL role, restricted to the role's test layers T
(icl_sets.RoleSet.test_layers: the layers in which membership was decided exhaustively):

(a) primary: top-k overlap O = |V_T & R| and Jaccard J = O / |V_T | R| (V_T = value top-k inside T, R = role heads)
    against a layer-matched permutation null that keeps the value set's layer profile. One-sided p-values with the +1
    correction
        p_enr = (1 + #{O* >= O}) / (1 + N)     (enrichment)      p_dep = (1 + #{O* <= O}) / (1 + N)     (depletion)
    and the two-sided p = min(1, 2 min(p_enr, p_dep)) that enters Holm. J is monotone in O under the fixed margins
    |V_T|, |R|, so its p-values equal those of O (both are reported). Two nulls:
      - head-exchangeable ("layer"): a draw puts n_l random distinct heads into every layer l of T (n_l = number of
        value top-k heads in l); exact distribution = sum over layers of independent hypergeometrics.
      - KV-group-preserving ("kv", GQA only): the heads_per_kv query heads of a KV group share one v_proj, so value heads
        and ICL roles both cluster by group, and the head-exchangeable null then underestimates the variance of O
        (anti-conservative: type-I 0.29-0.36 at 0.05 for group-clustered value sets in the review's simulation). The
        KV null permutes the KV groups of each layer and the heads inside each group, i.e. it draws uniformly from all
        head sets with the value set's per-layer multiset of group counts -- an exact randomisation test under
        exchangeability of groups (and of heads within a group), conditional on the clustering.
    For GQA models the KV null is the primary null (`null` = "kv": verdict, Holm); the head-exchangeable one is reported
    for comparison (`*_layer`). Roles with unit "kv_group" (value_kv_groups: the source names KV groups) are tested at
    group level: O = number of role groups holding >= 1 value top-k head, null = the value-hit groups of every layer
    permuted over the layer's groups (`null` = "kv_group"; exact = hypergeometrics over groups).
(d) Holm family (pre-specified): per ranking, every value x role pair with RoleSet.family == "primary" (census roles,
    named circuit L20-27, query-type heads L20-27, L0-19 additions, value KV groups); "descriptive" roles (2-5-head
    fragments, unions, subsets of other roles) get p-values but no verdict. Pairs whose null is degenerate (one
    attainable overlap) are no tests. Tarone's rule then drops the tests that cannot reach significance: with p_min =
    the smallest p-value attainable under the null (from the attainable support and the null histogram), K = the
    smallest k with #{p_min <= alpha / k} <= k, and Holm runs over {p_min <= alpha / K}. The selection uses only the
    null distributions, never the observed overlap, so FWER <= alpha holds. `p_holm` = per ranking (per model: no
    correction across the model variants), `p_holm_all` = both rankings of the model as one family.
(b) rank-level: Spearman rho between the per-head value score (AUC | accuracy) and each census effect (dret, dR, dP and
    |dret|) over the census heads; null = the value scores shuffled within each layer (the layer profiles of both
    stay, so the null mean is the between-layer part of rho). Two-sided p = min(1, 2 min(p_ge, p_le)); Holm over
    value x effect per ranking (a separate, secondary family).
(c) per value: the share of its top-k (and of its top-k inside the role band) that falls in each listed role, next to
    the share expected under the layer-matched null. For partial lists (icl_sets: explicit="partial") it is a lower
    bound; they are never tested.
Ties: the paper ranking (accuracy on small val sets, np.argsort order) decides part of its top-k by index order among
equal scores; every table reports how many top-k slots go to heads tied with the k-th score (`ties_in_topk` of
`ties_at_cutoff`).

Everything is numpy / scipy; no model is loaded. `analyze` works on in-memory tables (tests), `run` in
commands/overlap.py resolves the head tables of results/heads/ and writes JSON + CSV + a German markdown summary.
"""
import hashlib
import json
import math
import os
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, rankdata

from . import config

RANKINGS = {
    "pythia": {"mode": "pythia", "score": "auc", "label": "Head-only-AUC (Modus pythia)"},
    "paper": {"mode": "paper", "score": "acc", "label": "[Head ⊕ Layer-Mittel]-Accuracy (Modus paper)"},
}
EFFECTS = ("dret", "dR", "dP", "abs_dret")
EFFECT_LABELS = {"dret": "dret (Symbol+rare-Marge)", "dR": "dR (Demonstrations-Teil)", "dP": "dP (Wortbedeutungs-Teil)",
                 "abs_dret": "abs(dret) (Beteiligung, vorzeichenfrei)"}
ALPHA = 0.05
CHUNK = 10000


def _rng(seed, *keys):
    """Generator seeded by `seed` and stable string keys (independent of the order in which tables are processed)."""
    return np.random.default_rng([int(seed)] + [zlib.crc32(str(k).encode("utf-8")) for k in keys])


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# value-head tables
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class ValueTable:
    """Top-k and per-head scores of one probe table (results/heads/*.json)."""
    value: str
    ranking: str
    score: str
    layers: int
    heads: int
    topk: tuple                       # ((layer, head), ...) in rank order
    scores: np.ndarray                # (layers * heads,) primary score per flat index, NaN where not probed
    path: str = ""
    n_train: int = 0
    n_val: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def k(self):
        return len(self.topk)


def load_value_table(path, value, ranking, ratio=config.TOP_RATIO):
    """ValueTable from a head-score JSON written by `probe` (top-k = the stored ranking, k = int(ratio * L * H))."""
    from .probing import load_head_scores, top_heads_from_file
    df, meta = load_head_scores(path)
    L, H = int(meta["layers"]), int(meta["heads"])
    score = meta.get("score") or RANKINGS[ranking]["score"]
    flat = top_heads_from_file(path, ratio=ratio)
    scores = np.full(L * H, np.nan)
    scores[df["flat_idx"].to_numpy(dtype=np.int64)] = pd.to_numeric(df[score], errors="coerce").to_numpy(dtype=np.float64)
    return ValueTable(value, ranking, score, L, H, tuple((int(i) // H, int(i) % H) for i in flat), scores, path,
                      int(df["n_train"].iloc[0]), int(df["n_val"].iloc[0]),
                      {**{k: meta.get(k) for k in ("model", "revision", "mode", "feature_mode", "class_weight", "feature_tag",
                                                   "train_dir", "val_dir", "dtype", "attn_impl", "n_signif", "perm_threshold",
                                                   "stats_done")}, "rank_rule": meta.get("ranking"),
                       **cutoff_ties(scores, flat, secondary=_stable_secondary(df, L * H, meta.get("ranking")),
                                     decimals=6 if (meta.get("ranking") or "stable") == "stable" else 12)})


def _stable_secondary(df, n, rank_rule):
    """bal_acc per flat index for the stable ranking (its tie-break before the index), else None."""
    if (rank_rule or "stable") != "stable" or "bal_acc" not in df:
        return None
    out = np.full(n, np.nan)
    out[df["flat_idx"].to_numpy(dtype=np.int64)] = pd.to_numeric(df["bal_acc"], errors="coerce").to_numpy(dtype=np.float64)
    return out


def cutoff_ties(scores, topk_flat, secondary=None, decimals=12):
    """Ties at the top-k cutoff that only the head index resolves: {'ties_at_cutoff': heads whose sort key equals the
    k-th head's key, 'ties_in_topk': how many of them are in the top-k}. Sort key = the score rounded to `decimals`
    (paper ranking: np.argsort on the raw accuracies), plus `secondary` (stable ranking: bal_acc, its tie-break before
    the flat index). ties_at_cutoff > ties_in_topk means the index order decided ties_in_topk slots of the top-k."""
    flat = np.asarray(topk_flat, dtype=np.int64)
    if not len(flat) or not np.isfinite(scores[flat[-1]]):
        return {"ties_at_cutoff": 0, "ties_in_topk": 0}
    eq = np.round(np.asarray(scores, dtype=np.float64), decimals) == round(float(scores[flat[-1]]), decimals)
    if secondary is not None:
        sec = np.round(np.asarray(secondary, dtype=np.float64), decimals)
        eq &= sec == sec[flat[-1]]
    return {"ties_at_cutoff": int(eq.sum()), "ties_in_topk": int(eq[flat].sum())}


# ----------------------------------------------------------------------------------------------------------------
# nulls
# ----------------------------------------------------------------------------------------------------------------

def layer_counts(heads):
    out = {}
    for l, _ in heads:
        out[int(l)] = out.get(int(l), 0) + 1
    return out


def layer_matched_draws(topk, n_layers, n_heads, n, rng, heads_per_kv=1):
    """(n, n_layers * n_heads) bool: n random head sets with the per-layer counts of `topk`.

    heads_per_kv == 1: in every layer, n_l distinct heads drawn uniformly (the primary null).
    heads_per_kv > 1 (GQA): the KV groups of the layer are permuted at random and, inside every group, the heads; the
    value heads of original group g go to the first c_g heads (in that random order) of g's target group, so the
    multiset of per-group counts -- the clustering of the value set within KV groups -- is kept.
    """
    M = np.zeros((n, n_layers * n_heads), dtype=bool)
    rows = np.arange(n)[:, None]
    by_layer = {}
    for l, h in topk:
        by_layer.setdefault(int(l), []).append(int(h))
    for l, hs in sorted(by_layer.items()):
        base = l * n_heads
        if heads_per_kv <= 1:
            pick = np.argsort(rng.random((n, n_heads)), axis=1)[:, :len(hs)]
            M[rows, base + pick] = True
            continue
        if n_heads % heads_per_kv:
            raise ValueError(f"{n_heads} heads are not divisible into KV groups of {heads_per_kv}")
        G = n_heads // heads_per_kv
        counts = np.bincount(np.asarray(hs) // heads_per_kv, minlength=G)
        target = np.argsort(rng.random((n, G)), axis=1)                    # original group g -> target[:, g]
        within = np.argsort(rng.random((n, G, heads_per_kv)), axis=2)      # random head order inside every group
        for g in np.flatnonzero(counts):
            t = target[:, g]
            hsel = t[:, None] * heads_per_kv + within[np.arange(n), t][:, :counts[g]]
            M[rows, base + hsel] = True
    return M


def null_histograms(topk, role_flat_lists, n_layers, n_heads, draws, rng, heads_per_kv=1, chunk=CHUNK):
    """Histogram (counts of O* = 0 .. |R|) of the null overlap with every role (flat indices), from `draws` draws."""
    hists = [np.zeros(len(idx) + 1, dtype=np.int64) for idx in role_flat_lists]
    idx_arrays = [np.asarray(idx, dtype=np.int64) for idx in role_flat_lists]
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        M = layer_matched_draws(topk, n_layers, n_heads, m, rng, heads_per_kv)
        for j, idx in enumerate(idx_arrays):
            o = M[:, idx].sum(axis=1) if len(idx) else np.zeros(m, dtype=np.int64)
            hists[j] += np.bincount(o, minlength=len(idx) + 1)
        done += m
    return hists


def exact_overlap_pmf(pairs, n_heads):
    """pmf of O* = sum_l X_l, X_l ~ Hypergeom(n_heads, r_l role heads, n_l drawn), for pairs [(n_l, r_l), ...]."""
    pmf = np.array([1.0])
    for n_l, r_l in pairs:
        if n_l <= 0 or r_l <= 0:
            continue
        support = np.arange(0, min(n_l, r_l) + 1)
        pmf = np.convolve(pmf, hypergeom.pmf(support, n_heads, r_l, n_l))
    return pmf


def layer_support(n_l, r_l, n_heads):
    """(min, max) overlap of n_l random heads with r_l role heads in a layer of n_heads (head-exchangeable null)."""
    return max(0, n_l + r_l - n_heads), min(n_l, r_l)


def kv_layer_support(value_heads, role_heads, n_heads, heads_per_kv):
    """(min, max) overlap attainable in one layer under the KV-group-preserving null.

    The null assigns the value set's per-group counts c to the layer's groups by a random permutation; inside a group
    with r role heads, c random heads overlap in [max(0, c + r - s), min(c, r)] (s = heads_per_kv). min(c, r) and
    max(0, c + r - s) are supermodular, so the maximum is reached by the assortative pairing (both sorted descending),
    the minimum by the anti-assortative one."""
    G = n_heads // heads_per_kv
    c = sorted(np.bincount(np.asarray(list(value_heads), dtype=np.int64) // heads_per_kv, minlength=G).tolist(), reverse=True)
    r = sorted(np.bincount(np.asarray(list(role_heads), dtype=np.int64) // heads_per_kv, minlength=G).tolist(), reverse=True)
    hi = sum(min(a, b) for a, b in zip(c, r))
    lo = sum(max(0, a + b - heads_per_kv) for a, b in zip(c, r[::-1]))
    return lo, hi


def min_attainable_p(hist, lo, hi):
    """Smallest two-sided Monte-Carlo p-value attainable for an overlap in [lo, hi] under the null histogram `hist`
    (+1 correction): min(1, 2 min(p_enr(hi), p_dep(lo))). NaN if the null is degenerate (lo == hi)."""
    if lo >= hi:
        return float("nan")
    hist = np.asarray(hist, dtype=np.float64)
    n = hist.sum()
    return min(1.0, 2.0 * min(1.0 + hist[hi:].sum(), 1.0 + hist[:lo + 1].sum()) / (1.0 + n))


def tarone_holm(p, p_min, alpha=ALPHA):
    """Holm on Tarone's reduced family. p, p_min: arrays (NaN = no test). K = the smallest k with
    #{p_min <= alpha / k} <= k; the family is {p_min <= alpha / K} (chosen from the null distributions only, so Holm on it
    keeps FWER <= alpha; the dropped tests could not reach alpha / K). Returns (Holm-adjusted p, NaN outside the family;
    bool mask of the family; K; number of tests before the reduction)."""
    p, p_min = np.asarray(p, dtype=np.float64), np.asarray(p_min, dtype=np.float64)
    ok = ~np.isnan(p) & ~np.isnan(p_min)
    m = int(ok.sum())
    if m == 0:
        return np.full(p.shape, np.nan), np.zeros(p.shape, dtype=bool), 0, 0
    pm = p_min[ok]
    K = next(k for k in range(1, m + 1) if int((pm <= alpha / k).sum()) <= k)
    fam = ok & (p_min <= alpha / K)
    return holm(np.where(fam, p, np.nan)), fam, K, m


def _hist_stats(hist, observed):
    hist = np.asarray(hist, dtype=np.float64)
    n = hist.sum()
    support = np.arange(len(hist))
    cdf = np.cumsum(hist) / n
    return {
        "p_enr": (1.0 + hist[observed:].sum()) / (1.0 + n),
        "p_dep": (1.0 + hist[:observed + 1].sum()) / (1.0 + n),
        "mean": float((support * hist).sum() / n),
        "q05": int(np.searchsorted(cdf, 0.05)),
        "q95": int(np.searchsorted(cdf, 0.95)),
    }


def two_sided(p_a, p_b):
    if p_a is None or p_b is None or (isinstance(p_a, float) and math.isnan(p_a)):
        return float("nan")
    return min(1.0, 2.0 * min(p_a, p_b))


def holm(pvals):
    """Holm step-down adjusted p-values (NaN entries are not part of the family and stay NaN)."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = len(ok)
    running = 0.0
    for i, j in enumerate(ok[np.argsort(p[ok], kind="stable")]):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out


def stratified_spearman(x, y, strata, perms, rng, chunk=2000):
    """(rho, null array): Spearman rho of x and y, null by permuting x within each stratum (perms draws)."""
    x, y, strata = np.asarray(x, float), np.asarray(y, float), np.asarray(strata)
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom == 0:
        return float("nan"), np.full(perms, np.nan)
    rho = float((rx * ry).sum() / denom)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(perms)
    for start in range(0, perms, chunk):
        m = min(chunk, perms - start)
        P = np.empty((m, len(x)))
        for idx in groups:
            P[:, idx] = rx[idx][np.argsort(rng.random((m, len(idx))), axis=1)]
        null[start:start + m] = P @ ry / denom
    return rho, null


# ----------------------------------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------------------------------

def testable_roles(role_sets, names=None):
    """(roles with an overlap row, {role: reason not tested}). Full lists with a test band get a row; only those with
    family == "primary" enter the Holm family (RoleSet.confirmatory), "descriptive" ones are reported without verdict."""
    tested, skipped = [], {}
    for name, r in role_sets.items():
        if names and name not in names:
            continue
        if r.overlap_ready:
            tested.append(r)
        else:
            skipped[name] = ("Liste unvollständig (partial): nur Anteil (c), kein Test" if r.explicit == "partial"
                             else "keine Kopfliste veröffentlicht (nur Population)")
    if names:
        unknown = sorted(set(names) - set(role_sets))
        if unknown:
            raise KeyError(f"unknown ICL roles {unknown}; known: {sorted(role_sets)}")
    return tested, skipped


def _group_heads(heads, heads_per_kv):
    return sorted({(int(l), int(h) // heads_per_kv) for l, h in heads})


def _check_whole_groups(role, heads_per_kv):
    groups = _group_heads(role.heads, heads_per_kv)
    if len(groups) * heads_per_kv != len(role.heads):
        raise ValueError(f"ICL role {role.role} (unit kv_group) does not consist of whole KV groups of {heads_per_kv} heads")
    return groups


def _null_block(hist, observed, lo, hi):
    """p-values, mean, 5/95 % quantiles and the attainable minimum p of one null histogram."""
    st = _hist_stats(hist, observed)
    st["degenerate"] = lo >= hi
    st["p_two"] = float("nan") if st["degenerate"] else two_sided(st["p_enr"], st["p_dep"])
    st["p_min"] = min_attainable_p(hist, lo, hi)
    return st


def _overlap_row(t, role, hists, n_heads, heads_per_kv, draws):
    """One (a) row. hists: {"layer": head-level head-exchangeable histogram, "kv": head-level KV-null histogram or None,
    "kv_group": group-level histogram or None (unit kv_group)}. The primary null is kv_group > kv > layer."""
    band = sorted(set(role.test_layers))
    V = [h for h in t.topk if h[0] in band]
    R = [tuple(h) for h in role.heads if h[0] in band]
    O_h = len(set(V) & set(R))
    nV, nR = layer_counts(V), layer_counts(R)
    pmf_h = exact_overlap_pmf([(nV.get(l, 0), nR.get(l, 0)) for l in band], n_heads)
    sup_h = [layer_support(nV.get(l, 0), nR.get(l, 0), n_heads) for l in band]
    lo_h, hi_h = sum(a for a, _ in sup_h), sum(b for _, b in sup_h)
    layer = _null_block(hists["layer"], O_h, lo_h, hi_h)
    blocks = {"layer": layer}
    if hists.get("kv") is not None:
        lo = hi = 0
        for l in band:
            a, b = kv_layer_support([h for ll, h in V if ll == l], [h for ll, h in R if ll == l], n_heads, heads_per_kv)
            lo, hi = lo + a, hi + b
        blocks["kv"] = _null_block(hists["kv"], O_h, lo, hi)
    if role.unit == "kv_group" and hists.get("kv_group") is not None:
        G = n_heads // heads_per_kv
        VG, RG = _group_heads(V, heads_per_kv), _group_heads(R, heads_per_kv)
        O, a, b = len(set(VG) & set(RG)), len(VG), len(RG)
        gV, gR = layer_counts(VG), layer_counts(RG)
        pmf = exact_overlap_pmf([(gV.get(l, 0), gR.get(l, 0)) for l in band], G)
        sup = [layer_support(gV.get(l, 0), gR.get(l, 0), G) for l in band]
        blocks["kv_group"] = _null_block(hists["kv_group"], O, sum(x for x, _ in sup), sum(y for _, y in sup))
        expected = float(sum(gV.get(l, 0) * gR.get(l, 0) / G for l in band))
        primary, hist = "kv_group", np.asarray(hists["kv_group"])
        overlap_units = " ".join(f"{l}.{g}" for l, g in sorted(set(VG) & set(RG)))
    else:
        O, a, b, pmf = O_h, len(V), len(R), pmf_h
        expected = float(sum(nV.get(l, 0) * nR.get(l, 0) / n_heads for l in band))
        primary = "kv" if "kv" in blocks else "layer"
        hist = np.asarray(hists[primary])
        overlap_units = " ".join(f"{l}:{h}" for l, h in sorted(set(V) & set(R)))
    st = blocks[primary]
    jac = (lambda o: o / (a + b - o) if (a + b - o) > 0 else float("nan"))
    row = {
        "value": t.value, "ranking": t.ranking, "score": t.score, "role": role.role, "role_explicit": role.explicit,
        "family": role.family, "unit": "KV-Gruppe" if primary == "kv_group" else "Head", "null": primary,
        "test_layers": _span(role.test_layers), "k": t.k, "k_in_band": a, "role_n": b, "overlap": O,
        "expected": expected, "null_mean": st["mean"], "null_q05": st["q05"], "null_q95": st["q95"],
        "ratio_obs_exp": (O / st["mean"]) if st["mean"] > 0 else float("nan"),
        "jaccard": jac(O), "jaccard_null_mean": float(sum(jac(o) * c for o, c in enumerate(hist) if c) / max(1, hist.sum())),
        "p_enr": st["p_enr"], "p_dep": st["p_dep"], "p_two": st["p_two"], "p_min": st["p_min"],
        # exact null of the head-exchangeable test (head roles) or of the group test (kv_group roles)
        "p_enr_exact": float(pmf[O:].sum()) if O < len(pmf) else 0.0, "p_dep_exact": float(pmf[:O + 1].sum()),
        "draws": int(draws), "degenerate": bool(st["degenerate"]),
        "overlap_units": overlap_units, "overlap_heads_n": O_h,
        "overlap_heads": " ".join(f"{l}:{h}" for l, h in sorted(set(V) & set(R))),
    }
    # J is monotone in O under fixed margins -> its permutation p-values are those of O (computed, not assumed)
    jn = np.array([jac(o) for o in range(len(hist))])
    if a + b > 0:
        row["p_enr_jaccard"] = float((1 + hist[jn >= row["jaccard"] - 1e-12].sum()) / (1 + hist.sum()))
        row["p_dep_jaccard"] = float((1 + hist[jn <= row["jaccard"] + 1e-12].sum()) / (1 + hist.sum()))
    else:
        row["p_enr_jaccard"] = row["p_dep_jaccard"] = 1.0
    # comparison columns (head level): head-exchangeable null always, KV null for GQA
    for name in ("layer", "kv"):
        if name in blocks:
            b_ = blocks[name]
            row.update({f"null_mean_{name}": b_["mean"], f"p_enr_{name}": b_["p_enr"], f"p_dep_{name}": b_["p_dep"],
                        f"p_two_{name}": b_["p_two"], f"p_min_{name}": b_["p_min"]})
    row["direction"] = "" if st["degenerate"] else ("enrichment" if row["p_enr"] < row["p_dep"] else
                                                    "depletion" if row["p_dep"] < row["p_enr"] else "")
    return row


def _span(ls):
    ls = sorted(ls)
    if not ls:
        return "-"
    if ls == list(range(ls[0], ls[-1] + 1)):
        return f"{ls[0]}-{ls[-1]}" if len(ls) > 1 else str(ls[0])
    return ",".join(map(str, ls))


def analyze(tables, role_sets, census=None, *, draws=20000, perms=10000, seed=0, heads_per_kv=1, kv_null=True,
            roles=None, effects=EFFECTS, alpha=ALPHA, n_layers=None, n_heads=None, verbose=True):
    """Runs (a)-(d) for `tables` (ValueTable list of one model) against `role_sets` ({name: icl_sets.RoleSet}) and the
    census (icl_sets.CensusResult or None: then (b) is skipped). heads_per_kv = query heads per KV group of the model
    (1 = no GQA); with heads_per_kv > 1 and kv_null the KV-group-preserving null is the primary null of head roles
    (kv_null=False: head-exchangeable null, anti-conservative under GQA clustering; kept for comparison runs only).
    Returns a dict with DataFrames topk / spearman / shares and the family definitions."""
    if draws < 1 or perms < 0:
        raise ValueError("draws must be >= 1 and perms >= 0")
    if not tables:
        raise ValueError("no value tables")
    L = n_layers or tables[0].layers
    H = n_heads or tables[0].heads
    for t in tables:
        if (t.layers, t.heads) != (L, H):
            raise ValueError(f"{t.path or t.value}: geometry {t.layers} x {t.heads} != {L} x {H} of the ICL sets")
    if heads_per_kv < 1 or H % heads_per_kv:
        raise ValueError(f"{H} heads are not divisible into KV groups of {heads_per_kv}")
    tested, skipped = testable_roles(role_sets, roles)
    for r in role_sets.values():
        bad = [h for h in r.heads if not (0 <= h[0] < L and 0 <= h[1] < H)]
        if bad:
            raise ValueError(f"ICL role {r.role} has heads outside {L} x {H}: {bad[:3]}")
    use_kv = heads_per_kv > 1 and kv_null
    G = H // heads_per_kv
    role_flat = [[l * H + h for l, h in r.heads if l in set(r.test_layers)] for r in tested]
    group_roles = [j for j, r in enumerate(tested) if r.unit == "kv_group" and heads_per_kv > 1]
    group_flat = {j: [l * G + g for l, g in _check_whole_groups(tested[j], heads_per_kv) if l in set(tested[j].test_layers)]
                  for j in group_roles}

    rows_a = []
    for t in tables:
        if verbose:
            print(f"[overlap] (a) {t.value}/{t.ranking}: k={t.k}, {len(tested)} roles, {draws} draws, primary null "
                  + (f"KV-group-preserving ({heads_per_kv} heads per KV group)" if use_kv else "head-exchangeable"),
                  flush=True)
        hl = null_histograms(t.topk, role_flat, L, H, draws, _rng(seed, t.value, t.ranking, "layer"))
        hk = (null_histograms(t.topk, role_flat, L, H, draws, _rng(seed, t.value, t.ranking, "kv"), heads_per_kv)
              if use_kv else [None] * len(tested))
        hg = {}
        if group_roles:
            vgroups = _group_heads(t.topk, heads_per_kv)
            hists = null_histograms(vgroups, [group_flat[j] for j in group_roles], L, G, draws,
                                    _rng(seed, t.value, t.ranking, "kv_group"))
            hg = dict(zip(group_roles, hists))
        for j, r in enumerate(tested):
            rows_a.append(_overlap_row(t, r, {"layer": hl[j], "kv": hk[j], "kv_group": hg.get(j)}, H, heads_per_kv, draws))
    topk = pd.DataFrame(rows_a)
    families = {"a": {}, "b": {}}
    if len(topk):
        conf = topk["family"].eq("primary").to_numpy()
        p_fam = np.where(conf, topk["p_two"], np.nan)
        pmin_fam = np.where(conf, topk["p_min"], np.nan)
        topk["p_holm"] = np.nan
        topk["in_family"] = False
        topk["p_holm_layer"] = np.nan
        for rk, sub in topk.groupby("ranking"):
            idx = sub.index.to_numpy()
            adj, fam, K, m = tarone_holm(p_fam[idx], pmin_fam[idx], alpha)
            topk.loc[idx, "p_holm"] = adj
            topk.loc[idx, "in_family"] = fam
            # comparison: the same pre-specified family under the head-exchangeable head-level null
            adj_l, _, _, _ = tarone_holm(np.where(conf[idx], sub["p_two_layer"], np.nan),
                                         np.where(conf[idx], sub["p_min_layer"], np.nan), alpha)
            topk.loc[idx, "p_holm_layer"] = adj_l
            prim = sub[conf[idx]]
            dropped = sub[conf[idx] & ~sub["degenerate"].to_numpy() & ~fam]
            families["a"][rk] = {
                "test": "(a) Top-k-Überlappung, layer-gematchte Permutations-Null"
                        + (" (KV-Gruppen-erhaltend; Rollen mit Einheit KV-Gruppe auf Gruppenebene)" if use_kv else
                           " (Heads innerhalb des Layers austauschbar)")
                        + ", zweiseitig p = min(1, 2 min(p_anr, p_abr)), Holm nach Tarone-Reduktion",
                "null": "kv" if use_kv else "layer",
                "values": sorted(prim["value"].unique().tolist()),
                "roles": [r.role for r in tested if r.family == "primary"],
                "descriptive_roles": [r.role for r in tested if r.family != "primary"],
                "m_before_tarone": int(m), "tarone_K": int(K), "m": int(fam.sum()),
                "excluded_tarone": [f"{a}/{b} (p_min {c:.2g})" for a, b, c in dropped[["value", "role", "p_min"]].values],
                "excluded_degenerate": [f"{a}/{b}" for a, b in prim[prim["degenerate"]][["value", "role"]].values],
                "scope": "je Modell und Ranking (keine Korrektur über Modellvarianten); p_holm_all = beide Rankings",
            }
        adj_all, _, _, _ = tarone_holm(p_fam, pmin_fam, alpha)
        topk["p_holm_all"] = adj_all
        topk["verdict"] = [_verdict(r, alpha) for r in topk.itertuples()]

    # (b) rank-level association with the census effects
    rows_b = []
    if census is not None and perms > 0:
        heads = sorted(census.screened_heads)
        eff = {k: census.effects(k) for k in ("dret", "dR", "dP")}
        eff["abs_dret"] = {h: abs(v) for h, v in eff["dret"].items()}
        for t in tables:
            for e in effects:
                xs, ys, ls = [], [], []
                for (l, h) in heads:
                    x, y = t.scores[l * H + h], eff[e].get((l, h), float("nan"))
                    if np.isfinite(x) and np.isfinite(y):
                        xs.append(x), ys.append(y), ls.append(l)
                rho, null = stratified_spearman(xs, ys, ls, perms, _rng(seed, t.value, t.ranking, "spearman", e))
                ok = not math.isnan(rho)
                p_ge = (1 + int((null >= rho - 1e-12).sum())) / (1 + perms) if ok else float("nan")
                p_le = (1 + int((null <= rho + 1e-12).sum())) / (1 + perms) if ok else float("nan")
                rows_b.append({"value": t.value, "ranking": t.ranking, "score": t.score, "effect": e, "n_heads": len(xs),
                               "layers": _span(set(ls)), "rho": rho, "rho_null_mean": float(np.nanmean(null)) if ok else float("nan"),
                               "rho_excess": rho - float(np.nanmean(null)) if ok else float("nan"),
                               "p_ge": p_ge, "p_le": p_le, "p_two": two_sided(p_ge, p_le) if ok else float("nan"),
                               "perms": int(perms)})
    spearman = pd.DataFrame(rows_b)
    if len(spearman):
        spearman["p_holm"] = np.nan
        for rk, sub in spearman.groupby("ranking"):
            spearman.loc[sub.index, "p_holm"] = holm(sub["p_two"].to_numpy())
            families["b"][rk] = {"test": "(b) Spearman Wert-Score vs. Census-Effekt, Null = Permutation innerhalb Layer, zweiseitig",
                                 "values": sorted(sub["value"].unique().tolist()), "effects": list(effects),
                                 "m": int(sub["p_two"].notna().sum())}

    # (c) shares
    rows_c = []
    listed = [r for r in role_sets.values() if r.heads and (not roles or r.role in roles)]
    for t in tables:
        V = set(t.topk)
        for r in listed:
            band = set(r.test_layers)
            R = set(tuple(h) for h in r.heads)
            V_T = {h for h in V if h[0] in band}
            nV, nR = layer_counts(V_T), layer_counts([h for h in R if h[0] in band])
            exp = sum(nV.get(l, 0) * nR.get(l, 0) / H for l in band)
            rows_c.append({"value": t.value, "ranking": t.ranking, "role": r.role, "role_explicit": r.explicit,
                           "lower_bound": r.explicit == "partial", "k": t.k, "in_role": len(V & R),
                           "share_topk": len(V & R) / t.k if t.k else float("nan"),
                           "k_in_band": len(V_T), "share_band": (len(V_T & R) / len(V_T)) if V_T else float("nan"),
                           "share_band_null": (exp / len(V_T)) if V_T else float("nan"),
                           "role_share_of_band": len([h for h in R if h[0] in band]) / (len(band) * H) if band else float("nan")})
    shares = pd.DataFrame(rows_c)
    return {"topk": topk, "spearman": spearman, "shares": shares, "families": families,
            "tested_roles": [r.as_dict() for r in tested], "skipped_roles": skipped,
            "settings": {"draws": int(draws), "perms": int(perms), "seed": int(seed), "heads_per_kv": int(heads_per_kv),
                         "kv_null": bool(use_kv), "alpha": alpha, "layers": L, "heads": H, "effects": list(effects)}}


VERDICTS_SIGNIFICANT = ("angereichert", "abgereichert")


def _verdict(r, alpha):
    if r.family != "primary":
        return "deskriptiv (keine Holm-Familie)"
    if r.degenerate:
        return "kein Test (Null degeneriert)"
    if not r.in_family:
        return "nicht testbar (p_min > α/K, Tarone)"
    if not (r.p_holm < alpha):
        return "kein Nachweis"
    return "angereichert" if r.direction == "enrichment" else "abgereichert"


# ----------------------------------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------------------------------

def _fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def _fmt(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{d}f}"


def _n(x):
    return f"{int(x):,}".replace(",", " ")


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def markdown_summary(result, info):
    """German markdown summary of `analyze` output; `info` = header dict (model, revision, icl, tables, note, ...)."""
    s = result["settings"]
    topk, sp, sh = result["topk"], result["spearman"], result["shares"]
    lines = [f"# Wert-Heads × ICL-Schaltkreis: `{info['model']}` @ {info.get('revision', 'main')}", ""]
    if info.get("note"):
        lines += [f"> **{info['note']}**", ""]
    small = [t for t in info.get("tables", []) if t.get("n_train", 0) < 1000 or t.get("n_val", 0) < 300]
    if small:
        lines += ["> **Kleine Stichproben** (n_train < 1000 oder n_val < 300 in "
                  + ", ".join(f"{t['value']}/{t['ranking']}" for t in small)
                  + "): Rankings und damit alle Tests sind nur eine Funktionsprüfung, kein Befund.", ""]
    icl = info.get("icl", {})
    lines += [
        f"- **Wert-Heads:** Probe je Head auf dem Eingang von `o_proj` am letzten Token ({info.get('hf_id', '')}), "
        f"Top-k = {int(round(100 * info.get('ratio', config.TOP_RATIO)))} % aller {s['layers']} × {s['heads']} Heads "
        f"(k = {info.get('k', '?')}). Rankings: " + "; ".join(f"`{r}` = {RANKINGS[r]['label']}" for r in info.get("rankings", [])) + ".",
        f"- **ICL-Rollen:** `{icl.get('model')}` aus Gao & Gao, arXiv 2611.05344 (Repo-Commit `{str(icl.get('commit', ''))[:12]}`; "
        f"Census `{icl.get('census_file', '')}`, sha256 {'gepinnt und geprüft' if icl.get('pinned') else 'NICHT gepinnt'}). "
        "Census-Rollen nach der Regel von `icl/analyze_D188_scale.py` (Leser: dret-CI < 0, Bremsen: dret-CI > 0, "
        "P-Bremsen: dP-CI > 0, P-Träger: dP-CI < 0); benannte Rollen nur, wo die Quelle die vollständige Liste nennt.",
    ]
    if icl.get("transferred"):
        lines.append(f"- **Übertragen:** Die ICL-Sätze stammen aus `{icl.get('model')}` und werden unverändert auf "
                     f"`{info['model']}` angewandt (gleiche Architektur; das ICL-Paper überträgt seine Basis-Heads ebenso auf "
                     "das Instruct-Modell, main.tex:470). Eine Überlappung hier heißt: Wert-Heads des post-trainierten Modells "
                     "liegen auf Positionen, die im Base-Modell ICL-Rollen tragen.")
    kv = bool(s.get("kv_null", s["heads_per_kv"] > 1))
    if kv:
        null_txt = (f"- **Null (a, primär): KV-Gruppen-erhaltend und layer-gematcht.** In Qwen3 teilen sich {s['heads_per_kv']} "
                    "Query-Heads eine `v_proj`; Wert-Heads und ICL-Rollen häufen sich daher in KV-Gruppen, und eine Null mit "
                    "austauschbaren Heads wäre zu optimistisch (falsch positiv). Je Layer des Testbands werden die KV-Gruppen "
                    "und die Heads in jeder Gruppe permutiert: gleiche Zahl Top-k-Heads je Layer und gleiche Häufung je Gruppe. "
                    "Rollen der Einheit KV-Gruppe (`value_kv_groups`) werden auf Gruppenebene getestet (Überlappung = Zahl "
                    "der Rollen-Gruppen mit ≥ 1 Top-k-Head; Null = Gruppen je Layer permutiert). ")
    else:
        null_txt = "- **Null (a, primär): layer-gematcht** – je Layer des Testbands so viele zufällige Heads wie die Top-k dort hat. "
    lines += [
        null_txt + f"N = {_n(s['draws'])} Ziehungen, p mit +1-Korrektur, einseitig für Anreicherung (p_anr) und Abreicherung "
        "(p_abr), zweiseitig 2·min für Holm. Vergleich: exakte Null mit austauschbaren Heads (Summe von Hypergeometrischen)"
        + (" und dieselbe Null als Monte-Carlo (`*_layer` in der CSV, Holm `p_holm_layer`; bei GQA antikonservativ)" if kv else "") + ".",
        "- **Holm-Familie (a), vorab festgelegt:** Census-Rollen, benannte Schaltkreis-Heads L20-27, Query-Typ-Heads L20-27, "
        "L0-19-Zusätze, Wert-KV-Gruppen (sofern vorhanden); Teil-Listen mit 2-5 Heads, Vereinigungen und Teilmengen anderer "
        "Rollen sind **deskriptiv** (p ohne Befund). Tarone: Tests, deren kleinstes erreichbares p (aus der Null) über α/K liegt, "
        "verlassen die Familie. "
        + "; ".join(f"Ranking `{rk}`: {len(f['values'])} Werte × {len(f['roles'])} Rollen, {f['m_before_tarone']} Tests, "
                    f"nach Tarone (K = {f['tarone_K']}) m = {f['m']}"
                    + (f" (dazu {len(f['excluded_degenerate'])} degenerierte ohne Test)" if f["excluded_degenerate"] else "")
                    for rk, f in result["families"]["a"].items())
        + f". α = {s['alpha']}. `Holm-p` = je Modell und Ranking (keine Korrektur über die Modellvarianten); "
        "`p_holm_all` (CSV) = beide Rankings als eine Familie.",
        "- **Regel:** „angereichert“ / „abgereichert“ nur bei Holm-p < α. Rohe Überlappungszahlen ohne Null sind kein Beleg; "
        "Wert-Heads und ICL-Rollen liegen beide in mittleren bis späten Layern.",
    ]
    tied = [t for t in info.get("tables", []) if t.get("ties_at_cutoff", 0) > t.get("ties_in_topk", 0)]
    if tied:
        lines.append("- **Gleichstände an der Top-k-Grenze** (Sortierschlüssel gleich dem des k-ten Heads: Score, im Ranking "
                     "`pythia` zusätzlich bal_acc; die Index-Reihenfolge entscheidet, wer in die Top-k kommt): " + ", ".join(
                         f"{t['value']}/{t['ranking']} {t['ties_in_topk']} von {t['ties_at_cutoff']}" for t in tied)
                     + ". Befunde dieser Tabellen hängen teilweise an der Tie-Reihenfolge.")
    lines.append("")
    # findings
    lines += ["## Befunde nach Holm (Test a)", ""]
    for rk in info.get("rankings", []):
        sub = topk[(topk["ranking"] == rk)] if len(topk) else topk
        sig = sub[sub["verdict"].isin(VERDICTS_SIGNIFICANT)] if len(sub) else sub
        if not len(sub):
            lines.append(f"- `{rk}`: keine Tabellen.")
        elif not len(sig):
            lines.append(f"- `{rk}`: kein Wert × Rolle-Paar übersteht Holm (m = {result['families']['a'].get(rk, {}).get('m', 0)}); "
                         "weder Überlappung noch Abreicherung nachgewiesen.")
        else:
            for r in sig.itertuples():
                lines.append(f"- `{rk}`: **{r.value} × {r.role}: {r.verdict}** – Überlappung {r.overlap} ({r.unit}) bei "
                             f"Null-Mittel {r.null_mean:.2f} (k im Band {r.k_in_band}, n Rolle {r.role_n}), p_anr {_fmt_p(r.p_enr)}, "
                             f"p_abr {_fmt_p(r.p_dep)}, Holm-p {_fmt_p(r.p_holm)}"
                             + (f"; Vergleich austauschbare Heads: p {_fmt_p(r.p_two_layer)}" if kv else "") + ".")
    lines.append("")
    # (a) full table
    for rk in info.get("rankings", []):
        sub = topk[topk["ranking"] == rk] if len(topk) else topk
        if not len(sub):
            continue
        lines += [f"## (a) Top-k-Überlappung – Ranking `{rk}`", ""]
        hdr = (["Wert", "Rolle", "Einheit", "Band", "k im Band", "n Rolle", "Überl.", "Null-Mittel [5–95 %]", "Jaccard (Null)",
                "p_anr", "p_abr", "p_min", "exakt anr/abr"] + (["austauschbar anr/abr"] if kv else []) + ["Holm-p", "Befund"])
        rows = []
        for r in sub.itertuples():
            row = [r.value, r.role, r.unit, r.test_layers, r.k_in_band, r.role_n, r.overlap,
                   f"{r.null_mean:.2f} [{r.null_q05}–{r.null_q95}]", f"{_fmt(r.jaccard)} ({_fmt(r.jaccard_null_mean)})",
                   _fmt_p(r.p_enr), _fmt_p(r.p_dep), _fmt_p(r.p_min), f"{_fmt_p(r.p_enr_exact)} / {_fmt_p(r.p_dep_exact)}"]
            if kv:
                row.append(f"{_fmt_p(r.p_enr_layer)} / {_fmt_p(r.p_dep_layer)}")
            row += [_fmt_p(r.p_holm), r.verdict]
            rows.append(row)
        lines += _md_table(hdr, rows) + [""]
        lines += ["`exakt` = exakte Null mit austauschbaren Heads (Kopf-Rollen) bzw. über Gruppen (Einheit KV-Gruppe); "
                  "`p_min` = kleinstes unter der primären Null erreichbares p (Tarone).", ""]
    # (b)
    if len(sp):
        lines += ["## (b) Rangkorrelation Wert-Score × Census-Effekt (sekundär)", "",
                  f"Spearman ρ über die Census-Heads (Layer {sp['layers'].iloc[0]}); Null: Wert-Scores innerhalb jedes Layers "
                  f"permutiert (N = {_n(s['perms'])}), das Null-Mittel ist der Anteil zwischen Layern. Negative dret/dR/dP heißen: "
                  "Ausschalten senkt die ICL-Marge (Leser-Richtung). Holm je Ranking über Werte × Effekte (eigene Familie).", ""]
        rows = [[r.value, r.ranking, EFFECT_LABELS.get(r.effect, r.effect), r.n_heads, _fmt(r.rho), _fmt(r.rho_null_mean),
                 _fmt(r.rho_excess), _fmt_p(r.p_two), _fmt_p(r.p_holm)] for r in sp.itertuples()]
        lines += _md_table(["Wert", "Ranking", "Effekt", "n", "ρ", "ρ Null-Mittel", "ρ − Null", "p zweiseitig", "Holm-p"], rows) + [""]
    # (c)
    if len(sh):
        lines += ["## (c) Anteil der Top-k je Rolle (deskriptiv)", "",
                  "`Anteil im Band` = Top-k-Heads in der Rolle / Top-k-Heads im Testband der Rolle; `Null` = derselbe Anteil "
                  "unter der layer-gematchten Null. Teil-Listen (≥) sind Untergrenzen und werden nicht getestet.", ""]
        rows = [[r.value, r.ranking, r.role + (" (≥)" if r.lower_bound else ""), r.k_in_band, r.in_role,
                 f"{100 * r.share_topk:.1f} %", "–" if math.isnan(r.share_band) else f"{100 * r.share_band:.1f} %",
                 "–" if math.isnan(r.share_band_null) else f"{100 * r.share_band_null:.1f} %"] for r in sh.itertuples()]
        lines += _md_table(["Wert", "Ranking", "Rolle", "k im Band", "in Rolle", "Anteil Top-k", "Anteil im Band", "Null"], rows) + [""]
    if result.get("skipped_roles"):
        lines += ["## Nicht getestete Rollen", ""]
        lines += _md_table(["Rolle", "Grund"], sorted(result["skipped_roles"].items())) + [""]
    if info.get("tables"):
        lines += ["## Eingaben", ""]
        lines += _md_table(["Wert", "Ranking", "Score", "n_train", "n_val", "k", "Gleichstand an der Grenze", "Tabelle"],
                           [[t["value"], t["ranking"], t["score"], t["n_train"], t["n_val"], t["k"],
                             f"{t.get('ties_in_topk', 0)} von {t.get('ties_at_cutoff', 0)}", f"`{os.path.basename(t['path'])}`"]
                            for t in info["tables"]]) + [""]
        if info.get("missing"):
            lines += ["Fehlende Tabellen (übersprungen): " + ", ".join(f"`{os.path.basename(m)}`" for m in info["missing"]), ""]
    return "\n".join(lines) + "\n"


def write_outputs(result, info, out_base):
    """{out_base}.json / .md and {out_base}_topk.csv / _spearman.csv / _shares.csv; returns the paths."""
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    paths = {"json": out_base + ".json", "md": out_base + ".md"}
    for key in ("topk", "spearman", "shares"):
        paths[key] = f"{out_base}_{key}.csv"
        result[key].to_csv(paths[key], index=False)
    payload = {"info": info, "settings": result["settings"], "families": result["families"],
               "tested_roles": result["tested_roles"], "skipped_roles": result["skipped_roles"],
               **{k: json.loads(result[k].to_json(orient="records")) for k in ("topk", "spearman", "shares")}}
    tmp = paths["json"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    os.replace(tmp, paths["json"])
    with open(paths["md"], "w") as f:
        f.write(markdown_summary(result, info))
    return paths
