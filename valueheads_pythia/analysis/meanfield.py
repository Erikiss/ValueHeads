"""Mean-field signatures of the angles between value directions over training (docs/MEANFIELD.md; paper App. K, Table 17).

For every (model, checkpoint, value pair, head set) the per-head cosine r(h) = cos angle(dir_a[h], dir_b[h]) between the
mass-mean directions of two values is the signed residual from orthogonality (r = 0 <=> 90 deg, arccos r = the angle of
Table 17). The question is whether these residuals behave like independent noise that averages out over heads
("mean-field-artig": a statistical signature, NOT a claim about the mean-field vs lazy / NTK parametrisation regime of
training) or carry structure. Signatures, each with a bootstrap CI (2000 draws) and a fixed verdict rule:

    (a) location  mean r vs 0 (N1, isotropic unit vectors) and vs step0 (N2, the untrained model on the same heads)
    (b) spread    var(r) - 1/d, sd(r) / sd_N1, share of angles > 100 deg vs its N1 expectation (and vs step0)
    (c) averaging sd of the mean of r over windows of n contiguous heads in layer order vs n (log-log slope); the null is
                  the same windows on randomly permuted heads (= random subsets without replacement, whose slope is -1/2
                  after the finite-population correction BY CONSTRUCTION, so random subsets alone cannot test anything);
                  independent residuals -> -1/2, correlation tied to the layer order (neighbouring heads, layers, a
                  depth trend) -> shallower; n_eff = var(r) / var(window mean). A component shared by ALL heads
                  (equicorrelation) is a constant shift within one realization and leaves the window variances
                  unchanged: (c) cannot see it, only the location test (a) can
    (d) layers    ICC of r by layer (one-way variance components), permutation test over layer labels; lag-1
                  autocorrelation of the layer means (depth structure, descriptive)
    (e) training  Spearman of r(h) with the previous and with the final checkpoint on the same heads, head-permutation null
    (f) shape     excess kurtosis; bimodality = 2-component Gaussian mixture against the better of a single Gaussian
                  and a skew-normal (BIC), plus Ashman's D and minor weight (a skewed but unimodal r is not bimodal);
                  Hartigan's dip is reported when the optional `diptest` package is installed (not used in the verdict,
                  so verdicts do not depend on the environment)
    (g) predict.  out-of-sample predictability of r and of |r| from head covariates (relative depth and its square,
                  log DoM norms of both values, probe scores of both values, r at step0): ridge regression on
                  standardized covariates, penalty by inner CV, 5-fold CV with whole layers held out, held-out R^2;
                  null = permutations that move whole layers (and shuffle heads within layers). The 1/sqrt(n) shrinkage
                  of (c) cannot tell iid noise from a structured term that is not tied to the layer order (both shrink
                  alike); only predictability separates them

Verdict (verdict_of): "mean-field-artig" additionally needs a testable layer test (d), a slope CI of at most
MAX_SLOPE_CI_WIDTH and a computable predictability test (g), else "unterbestimmt" (code m) - typical for the small
shared top-15 % sets; a residual that is predictable (g) is never "mean-field-artig" (label "vorhersagbar", code V).

N1: for unit vectors uniform on S^{d-1}, (r + 1) / 2 ~ Beta((d - 1) / 2, (d - 1) / 2), i.e. density ~ (1 - r^2)^((d-3)/2),
mean 0, variance 1/d (d = 64: angle 90 +- 7.2 deg, d = 128: 90 +- 5.1 deg); quantiles are exact (beta.ppf).
No label-permutation null is possible (the feature shards were deleted); mass-mean estimation noise pushes angles towards
90 deg, so an excess spread over N1 / step0 is conservative.

Inputs (written by `directions` / `probe` / `sweep`, read here only):
    results/directions/{model}_{revision}_{value}.npz   raw mass-mean (Eq. 2, not normalized), (L * H, D),
                                                        flat index = layer * H + head (as flat_idx of the head tables)
    results/heads/{model}_{revision}_{value}_pythia.json  top-15 % = probing.top_heads_from_file (stored ranking)
Outputs per model root: results/meanfield/{model}_meanfield.{csv,md,json}, figures/meanfield/{model}_*.png; with several
models a combined {base}/meanfield_summary_{models}.csv (+ width figure). Idempotent (input fingerprint), CPU only.

    python -m valueheads_pythia.analysis.meanfield --base /content/drive/MyDrive/ValueHeadsPythia
    python -m valueheads_pythia.analysis.meanfield --synthetic /tmp/mf_smoke        # synthetic tree, then the analysis
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
import re
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .. import config

VERSION = "meanfield-v1"
VALUES = ("harmless", "helpful", "honesty")            # pairs: harmless x helpful, harmless x honesty, helpful x honesty
HEADSETS = ("shared_top15", "all")
HEADSET_LABEL = {"shared_top15": "geteilte Top-15 %-Heads", "all": "alle Heads"}
N_BOOT = 2000
N_PERM = 2000
ALPHA = 0.05
MIN_HEADS = 16             # below: no averaging-law slope (needs >= 3 window sizes with >= 4 windows each) -> "unterbestimmt"
MIN_WITHIN_DF = 8          # layer clustering (d) needs >= 8 within-layer degrees of freedom and >= 3 layers with >= 2 heads
# "mean-field-artig" needs a slope CI no wider than this: a wider interval cannot separate -1/2 (independent) from -1/4
# (half-way to fully collective, slope 0); isotropic simulations give ~0.7 for 18 heads, ~0.45 for 40, <= 0.26 for all heads
MAX_SLOPE_CI_WIDTH = 0.5
CONFLICT_DEG = 100.0
BIMODAL_DBIC = 10.0        # BIC(best 1-comp: Gaussian or skew-normal) - BIC(2 Gaussians) > 10 ("very strong", Kass & Raftery)
BIMODAL_ASHMAN = 2.0       # Ashman's D > 2: clean separation of the two components
BIMODAL_MIN_WEIGHT = 0.05  # minor component >= 5 % of the heads (a single outlier head is not a population)
AVG_GRID_POINTS = 8
PRIMARY_REVISION = "step143000"
# (g) predictability of the residual: ridge regression of r / |r| on head covariates, CV with whole layers held out
PRED_FOLDS = 5             # outer folds (whole layers, interleaved over depth); fewer layers than folds -> not computable
# ridge penalty lambda = alpha * n_train on standardized covariates (alpha 100: ~intercept only, 0.001: ~OLS); in
# descending order, so ties of the inner CV go to the stronger penalty
PRED_ALPHAS = (100.0, 10.0, 1.0, 0.1, 0.01, 0.001)
PRED_TARGETS = ("r", "absr")                   # signed residual and its magnitude (heteroscedastic structure)
ABS_COVARIATES = ("r_step0",)                  # enter the |r| model as absolute values
COVARIATE_LABEL = {"depth": "Tiefe", "depth2": "Tiefe²", "log_norm_a": "log‖d_a‖", "log_norm_b": "log‖d_b‖",
                   "score_a": "Probe-Score a", "score_b": "Probe-Score b", "r_step0": "r(step0)", "|r_step0|": "|r(step0)|"}
# p-value tests per row; Holm within each model x checkpoint family over all rows and tests (descriptive)
TESTS = ("loc_n1", "loc_n2", "spread_n1", "spread_n2", "conflict_n1", "slope", "icc", "persist_prev", "persist_final",
         "pred_r", "pred_absr")
# preregistered primary: (a) + (c) on the shared top heads at step143000, per pair, Holm within each model (3 pairs x 3)
PRIMARY_TESTS = ("loc_n1", "loc_n2", "slope")
# IOI below-chance window (Dahiya & Blondin 2026): floor at step2000 for 160m / 410m, step1000 for 1b
IOI_MARKS = (1000, 2000)
IOI_FLOOR = {"pythia-160m": 2000, "pythia-410m": 2000, "pythia-1b": 1000}
LABEL_CODES = {"mean-field-artig": "M", "systematisch": "S", "kollektiv": "K", "vorhersagbar": "V", "bistabil": "B",
               "unterbestimmt": "U", "uneindeutig": "?"}
PERSISTENCE_ONLY_CODE = "P"   # "systematisch" driven by persistence alone (frozen residuals, location ~ 0)
# "unterbestimmt" although location, slope and shape fit the mean-field pattern: the layer test was not testable, the
# slope CI is wider than MAX_SLOPE_CI_WIDTH (sparse shared top heads) or the predictability test (g) was not computable
UNDERDETERMINED_MF_CODE = "m"
SKIP_DIRS = {"features", "features_cache", "hf_cache", "valueheads_pythia", "__pycache__", "results", "figures",
             "notebooks", "node_modules"}


# ----------------------------------------------------------------------------- isotropic null N1

def _n1_a(d):
    if d < 3:
        raise ValueError(f"N1 needs d >= 3, got {d}")
    return (d - 1) / 2.0


def n1_pdf(x, d):
    """Density of r = cos(angle) of two independent uniform unit vectors in R^d: ~ (1 - x^2)^((d-3)/2) on [-1, 1]."""
    a = _n1_a(d)
    return stats.beta.pdf((np.asarray(x, dtype=np.float64) + 1) / 2, a, a) / 2


def n1_cdf(x, d):
    a = _n1_a(d)
    return stats.beta.cdf((np.asarray(x, dtype=np.float64) + 1) / 2, a, a)


def n1_quantile(q, d):
    """Exact quantile of r under N1 ((r + 1) / 2 ~ Beta((d-1)/2, (d-1)/2))."""
    a = _n1_a(d)
    return 2 * stats.beta.ppf(np.asarray(q, dtype=np.float64), a, a) - 1


def n1_angle_quantile(q, d):
    """Quantile of the angle in degrees under N1 (the angle decreases with r, so the angle quantile q is r's 1 - q)."""
    return np.degrees(np.arccos(np.clip(n1_quantile(1 - np.asarray(q, dtype=np.float64), d), -1, 1)))


def n1_frac_beyond(angle_deg, d):
    """P(angle > angle_deg) = P(r < cos angle_deg) under N1."""
    return float(n1_cdf(np.cos(np.radians(angle_deg)), d))


def n1_mean_interval(n, d, level=0.95):
    """Central interval of the mean of n independent N1 cosines (variance 1 / (n d); CLT, exact enough for n d >= 64)."""
    z = stats.norm.ppf(0.5 + level / 2)
    half = z / np.sqrt(max(n, 1) * d)
    return -half, half


def n1_excess_kurtosis(d):
    """Excess kurtosis of r under N1: symmetric Beta(a, a) has -6 / (2a + 3) = -6 / (d + 2)."""
    return -6.0 / (d + 2)


# ----------------------------------------------------------------------------- per-head cosines

def head_cosines(dir_a, dir_b):
    """(L * H,) cos of the angle between the two directions of every head (NaN where one of them is zero)."""
    a = np.asarray(dir_a, dtype=np.float64)
    b = np.asarray(dir_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"direction arrays differ in shape: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(axis=1) / (na * nb)
    r[(na == 0) | (nb == 0)] = np.nan
    return np.clip(r, -1.0, 1.0)


def pair_with_cosines(r, d, rng):
    """Two (n, d) direction arrays whose rows have exactly the cosines r (for synthetic trees and tests)."""
    r = np.asarray(r, dtype=np.float64)
    a = rng.standard_normal((len(r), d))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    u = rng.standard_normal((len(r), d))
    u -= (u * a).sum(axis=1, keepdims=True) * a
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    b = r[:, None] * a + np.sqrt(np.clip(1 - r ** 2, 0, None))[:, None] * u
    return a, b


def holm(pvals):
    """Holm-adjusted p-values (NaN entries are ignored and stay NaN)."""
    p = np.asarray(pvals, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    q = p[ok]
    order = np.argsort(q, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * q[j]))
        adj[j] = running
    out[ok] = adj
    return out


# ----------------------------------------------------------------------------- (a) location

def _cluster_boot_means(x, layers, rng, n_boot):
    """Bootstrap means resampling whole layers (heads of one layer stay together); iid heads when < 3 layers."""
    groups, inv = np.unique(layers, return_inverse=True)
    if len(groups) >= 3:
        sums = np.bincount(inv, weights=x, minlength=len(groups))
        counts = np.bincount(inv, minlength=len(groups)).astype(np.float64)
        draw = rng.integers(0, len(groups), size=(n_boot, len(groups)))
        return sums[draw].sum(axis=1) / counts[draw].sum(axis=1), "layer", len(groups) - 1
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(axis=1), "head", len(x) - 1


def location_test(x, layers, rng, n_boot=N_BOOT):
    """Mean with layer-cluster bootstrap CI; two-sided t-test of mean = 0 with the bootstrap SE (df = clusters - 1)."""
    x = np.asarray(x, dtype=np.float64)
    out = {"mean": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "p": np.nan, "median": np.nan, "frac_pos": np.nan,
           "sign_ok": False, "unit": None}
    if len(x) < 3:
        return out
    m = float(x.mean())
    boots, unit, df = _cluster_boot_means(x, np.asarray(layers), rng, n_boot)
    # the bootstrap variance of a mean over G units is (G - 1) / G times the unbiased one (liberal for ~12 layers)
    se = float(boots.std(ddof=1) * np.sqrt((df + 1) / df))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float(2 * stats.t.sf(abs(m) / se, df)) if se > 0 else np.nan
    med = float(np.median(x))
    out.update(mean=m, lo=float(lo), hi=float(hi), se=se, p=p, median=med, frac_pos=float((x > 0).mean()),
               sign_ok=bool(m != 0 and np.sign(med) == np.sign(m)), unit=unit)
    return out


# ----------------------------------------------------------------------------- (b) spread

def spread_test(x, d, idx, x0=None):
    """Variance vs N1 (chi^2, one-sided excess) and vs step0 on the same heads (paired bootstrap of log sd ratio).

    idx: (B, n) iid head-bootstrap indices shared with the other head-bootstrap statistics of the row. The CI of var(r)
    (and of excess_var, sd_ratio) is the union of the bootstrap percentile interval and the normal-theory chi^2 interval
    (n - 1) s^2 / chi^2_{n-1}: the percentile bootstrap alone under-covers (isotropic simulations: 0.93-0.95 for all heads,
    0.82-0.89 for 18-25 heads), the union covers 0.94-0.98; N1 is slightly platykurtic, so the chi^2 interval is exact
    to conservative under isotropy, and the bootstrap side takes over for heavy-tailed r."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = {k: np.nan for k in ("var_r", "excess_var", "excess_var_lo", "excess_var_hi", "sd_ratio", "sd_ratio_lo",
                               "sd_ratio_hi", "p_spread_n1", "sd_vs_step0", "sd_vs_step0_lo", "sd_vs_step0_hi", "p_spread_n2")}
    if n < 3:
        return out
    var = float(x.var(ddof=1))
    vb = x[idx].var(axis=1, ddof=1)
    out.update(var_r=var, excess_var=var - 1.0 / d, sd_ratio=float(np.sqrt(var * d)),
               p_spread_n1=float(stats.chi2.sf((n - 1) * var * d, n - 1)))
    b_lo, b_hi = np.percentile(vb, [2.5, 97.5])
    c_lo, c_hi = (n - 1) * var / stats.chi2.ppf([0.975, 0.025], n - 1)
    v_lo, v_hi = float(min(b_lo, c_lo)), float(max(b_hi, c_hi))
    out.update(excess_var_lo=v_lo - 1.0 / d, excess_var_hi=v_hi - 1.0 / d, sd_ratio_lo=float(np.sqrt(v_lo * d)),
               sd_ratio_hi=float(np.sqrt(v_hi * d)))
    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64)
        v0 = float(x0.var(ddof=1))
        v0b = x0[idx].var(axis=1, ddof=1)
        if v0 > 0 and np.all(v0b > 0) and np.all(vb > 0):
            lr = 0.5 * np.log(vb / v0b)
            se = float(lr.std(ddof=1))
            obs = 0.5 * np.log(var / v0)
            lo, hi = np.percentile(lr, [2.5, 97.5])
            out.update(sd_vs_step0=float(np.exp(obs)), sd_vs_step0_lo=float(np.exp(lo)), sd_vs_step0_hi=float(np.exp(hi)),
                       p_spread_n2=float(2 * stats.norm.sf(abs(obs) / se)) if se > 0 else np.nan)
    return out


def conflict_test(x, d, x0=None, threshold=CONFLICT_DEG):
    """Share of heads with angle > threshold vs its N1 expectation (binomial, one-sided greater)."""
    x = np.asarray(x, dtype=np.float64)
    out = {"frac_gt100": np.nan, "frac_gt100_n1": n1_frac_beyond(threshold, d), "frac_gt100_step0": np.nan,
           "p_conflict_n1": np.nan}
    if len(x) == 0:
        return out
    angles = np.degrees(np.arccos(np.clip(x, -1, 1)))
    k = int((angles > threshold).sum())
    out["frac_gt100"] = k / len(x)
    out["p_conflict_n1"] = float(stats.binomtest(k, len(x), out["frac_gt100_n1"], alternative="greater").pvalue)
    if x0 is not None and len(x0):
        out["frac_gt100_step0"] = float((np.degrees(np.arccos(np.clip(x0, -1, 1))) > threshold).mean())
    return out


# ----------------------------------------------------------------------------- (c) averaging law

def averaging_grid(n, points=AVG_GRID_POINTS):
    """Log grid of window sizes 1 .. n // 4 (>= 4 windows' worth of heads each); None if fewer than 3 sizes."""
    if n < MIN_HEADS:
        return None
    grid = np.unique(np.round(np.geomspace(1, n // 4, num=points)).astype(int))
    return grid if len(grid) >= 3 else None


def window_variance(X, grid):
    """(B, len(grid)) variance (ddof=1) of the means of all windows of w contiguous entries of every row of X."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    c = np.concatenate([np.zeros((len(X), 1)), np.cumsum(X, axis=1)], axis=1)
    out = np.empty((len(X), len(grid)))
    for j, w in enumerate(grid):
        out[:, j] = ((c[:, w:] - c[:, :-w]) / w).var(axis=1, ddof=1)
    return out


def _wls_slope(logn, logv, weights):
    """Weighted least-squares slope of every row of logv on logn."""
    w = weights / weights.sum()
    xm = float((w * logn).sum())
    ym = logv @ w
    return ((logv - ym[:, None]) * (w * (logn - xm))).sum(axis=1) / float((w * (logn - xm) ** 2).sum())


def _slopes(V, grid, n):
    """Slope of log sd(window mean) vs log n (= half the variance slope) per row of V; weights = n / w - 1
    (independent windows - 1, the inverse variance of a log variance estimate up to a constant)."""
    w = n / grid.astype(np.float64) - 1.0
    return _wls_slope(np.log(grid.astype(np.float64)), np.log(np.maximum(V, 1e-300)), w) / 2


def cluster_block_index(layers, rng, n_boot):
    """(B, n) indices of a layer-block bootstrap: whole layers (heads in flat order) are drawn with replacement and
    concatenated until n heads are reached, then truncated to n (keeps the within-layer structure of the windows)."""
    layers = np.asarray(layers)
    groups, inv = np.unique(layers, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(groups))]
    sizes = np.array([len(m) for m in members])
    n = len(layers)
    need = int(np.ceil(n / sizes.min())) + 1
    draws = rng.integers(0, len(groups), size=(n_boot, need))
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        parts, total = [], 0
        for g in draws[b]:
            parts.append(members[g])
            total += sizes[g]
            if total >= n:
                break
        out[b] = np.concatenate(parts)[:n]
    return out


def averaging_law(x, layers, rng, perms, n_boot=N_BOOT):
    """Averaging law of r over contiguous heads in layer order (see module docstring, (c)).

    x must be in flat-index order. perms: (P, n) random permutations (the null: windows on permuted heads = random
    subsets without replacement). The raw slope is bias-corrected by the null mean (log of a variance estimate is
    biased, more so for large windows): slope = raw - (null mean + 0.5), so that independent residuals give -0.5;
    CI = layer-block bootstrap spread re-centred on the corrected slope, each half widened to at least 1.96 x the sd of
    the permutation null (the sampling sd of the slope under independence; the bootstrap alone covers -0.5 in only
    91-95 % of isotropic simulations, the widened interval in 95-97 %); p = one-sided permutation p (shallower)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    keys = ("slope_raw", "slope_null_mean", "slope_null_sd", "slope", "slope_lo", "slope_hi", "p_slope", "n_eff",
            "n_eff_null", "neff_ratio", "neff_window", "avg_window_max")
    out = {k: np.nan for k in keys}
    out.update(slope_ci_has_half=False, avg_n="", avg_sd_obs="", avg_sd_null="")
    grid = averaging_grid(n)
    if grid is None or not np.isfinite(x).all() or x.var() == 0:
        return out
    v_obs = window_variance(x, grid)
    raw = float(_slopes(v_obs, grid, n)[0])
    v_null = window_variance(x[perms], grid)
    s_null = _slopes(v_null, grid, n)
    null_mean, null_sd = float(s_null.mean()), float(s_null.std(ddof=1))
    corrected = raw - (null_mean + 0.5)
    s_boot = _slopes(window_variance(x[cluster_block_index(layers, rng, n_boot)], grid), grid, n)
    med = float(np.median(s_boot))
    q_lo, q_hi = np.percentile(s_boot, [2.5, 97.5])
    # n_eff = var(r) / var(window mean) at the largest window that still leaves >= 16 independent windows (the largest
    # window of the grid rests on ~4 and is too noisy); equals the window size for independent heads (up to the
    # overlap bias, hence the ratio to the same quantity on permuted heads)
    j = int(np.flatnonzero(grid <= max(1, n // 16))[-1])
    var = float(x.var(ddof=1))
    n_eff = var / float(v_obs[0, j])
    n_eff_null = float(np.mean(var / v_null[:, j]))
    z = stats.norm.ppf(0.975)
    lo = corrected + min(q_lo - med, -z * null_sd)
    hi = corrected + max(q_hi - med, z * null_sd)
    out.update(slope_raw=raw, slope_null_mean=null_mean, slope_null_sd=null_sd, slope=corrected, slope_lo=float(lo),
               slope_hi=float(hi), slope_ci_has_half=bool(lo <= -0.5 <= hi),
               p_slope=float((1 + (s_null >= raw).sum()) / (len(s_null) + 1)), n_eff=n_eff, n_eff_null=n_eff_null,
               neff_ratio=n_eff / n_eff_null, neff_window=int(grid[j]), avg_window_max=int(grid[-1]),
               avg_n=" ".join(str(int(g)) for g in grid),
               avg_sd_obs=" ".join(f"{v:.6g}" for v in np.sqrt(v_obs[0])),
               avg_sd_null=" ".join(f"{v:.6g}" for v in np.sqrt(v_null.mean(axis=0))))
    return out


def random_subset_sd(x, n_values, rng, draws=2000):
    """sd of the mean of r over random subsets without replacement, and the finite-population prediction
    S * sqrt((1 - n / N) / n): equal for ANY fixed set of values, i.e. slope -1/2 after the correction by construction
    (documented reference; the test (c) therefore uses contiguous windows in layer order against this null)."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    s = x.std(ddof=1)
    rows = []
    for n in n_values:
        means = np.array([x[rng.permutation(N)[:n]].mean() for _ in range(draws)])
        rows.append((int(n), float(means.std(ddof=1)), float(s * np.sqrt((1 - n / N) / n))))
    return rows


# ----------------------------------------------------------------------------- (d) layer structure

def layer_structure(x, layers, rng, perms):
    """One-way variance components of r by layer: ICC(1), between / within variance, F; permutation p over layer
    labels (one-sided, larger F) and the F-test p; lag-1 autocorrelation of the layer means with a layer-order null."""
    x = np.asarray(x, dtype=np.float64)
    out = {k: np.nan for k in ("icc", "var_between", "var_within", "F_layer", "p_icc", "p_icc_F", "layer_lag1", "p_lag1")}
    groups, inv = np.unique(np.asarray(layers), return_inverse=True)
    G, N = len(groups), len(x)
    sizes = np.bincount(inv, minlength=G).astype(np.float64)
    # a sparse head set (mostly one head per layer) leaves almost no within-layer degrees of freedom: no test then
    if G < 2 or N - G < MIN_WITHIN_DF or int((sizes >= 2).sum()) < 3:
        return out
    m = x.mean()
    sst = float(((x - m) ** 2).sum())
    if sst == 0:
        return out
    sums = np.bincount(inv, weights=x, minlength=G)
    ssb = float((sums ** 2 / sizes).sum() - N * m ** 2)
    msb, msw = ssb / (G - 1), (sst - ssb) / (N - G)
    if msw <= 0:
        return out
    n0 = (N - (sizes ** 2).sum() / N) / (G - 1)
    F = msb / msw
    onehot = np.zeros((N, G))
    onehot[np.arange(N), inv] = 1.0
    s_perm = x[perms] @ onehot
    ssb_perm = (s_perm ** 2 / sizes).sum(axis=1) - N * m ** 2
    F_perm = (ssb_perm / (G - 1)) / ((sst - ssb_perm) / (N - G))
    out.update(icc=float((msb - msw) / (msb + (n0 - 1) * msw)), var_between=float(max((msb - msw) / n0, 0.0)),
               var_within=float(msw), F_layer=float(F), p_icc=float((1 + (F_perm >= F - 1e-12).sum()) / (len(F_perm) + 1)),
               p_icc_F=float(stats.f.sf(F, G - 1, N - G)))
    if G >= 4:
        lm = sums / sizes
        if lm.std() > 0:
            lag = float(np.corrcoef(lm[:-1], lm[1:])[0, 1])
            order = np.argsort(rng.random((len(perms), G)), axis=1)
            lp = lm[order]
            a, b = lp[:, :-1] - lp[:, :-1].mean(axis=1, keepdims=True), lp[:, 1:] - lp[:, 1:].mean(axis=1, keepdims=True)
            null = (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
            out.update(layer_lag1=lag, p_lag1=float((1 + (null >= lag).sum()) / (len(null) + 1)))
    return out


# ----------------------------------------------------------------------------- (e) persistence

def persistence_test(x, y, rng, perms=None, n_perm=N_PERM):
    """Spearman rho of r(h) at two checkpoints on the same heads; one-sided head-permutation p (rho > 0)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan, np.nan
    if not ok.all() or perms is None:
        perms = np.argsort(rng.random((n_perm, int(ok.sum()))), axis=1)
    rx, ry = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan
    zx, zy = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    rho = float((zx * zy).mean())
    null = (zy[perms] * zx).mean(axis=1)
    return rho, float((1 + (null >= rho - 1e-12).sum()) / (len(null) + 1))


# ----------------------------------------------------------------------------- (f) shape

def excess_kurtosis_rows(X):
    """Unbiased sample excess kurtosis (G2, = scipy.stats.kurtosis(bias=False)) of every row of X."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n = X.shape[1]
    c = X - X.mean(axis=1, keepdims=True)
    m2, m4 = (c ** 2).mean(axis=1), (c ** 4).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        g2 = m4 / m2 ** 2 - 3
    return ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))


def skewnorm_loglik(x):
    """Maximum log-likelihood of a one-component skew-normal fit to x (3 parameters). Multi-start Nelder-Mead on the
    standardized data (method-of-moments start, the normal a = 0, and two strongly skewed starts), so the fit does not
    depend on the scale of r; the normal is a special case, hence the result is >= the Gaussian maximum log-likelihood."""
    x = np.asarray(x, dtype=np.float64)
    m, s = float(x.mean()), float(x.std())
    z = (x - m) / s

    def nll(p):
        return -float(stats.skewnorm.logpdf(z, p[2], p[0], np.exp(p[1])).sum())

    b = np.sqrt(2 / np.pi)
    g = abs(float(np.clip(stats.skew(z), -0.99, 0.99))) ** (2 / 3)
    delta = float(np.clip(np.sign(stats.skew(z)) * np.sqrt(np.pi / 2 * g / (g + ((4 - np.pi) / 2) ** (2 / 3))), -0.995, 0.995))
    sc = 1 / np.sqrt(1 - (b * delta) ** 2)
    best = -nll([0.0, 0.0, 0.0])
    for p0 in ([-sc * b * delta, np.log(sc), delta / np.sqrt(1 - delta ** 2)], [0.0, 0.0, 0.0], [-0.8, np.log(1.3), 3.0],
               [0.8, np.log(1.3), -3.0]):
        res = optimize.minimize(nll, p0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
        if np.isfinite(res.fun):
            best = max(best, -float(res.fun))
    return best - len(x) * np.log(s)


def _mixture_modes(mu, var, w):
    """Number of modes of a fitted two-component Gaussian mixture density (descriptive)."""
    sd = np.sqrt(var)
    grid = np.linspace(mu.min() - 3 * sd.max(), mu.max() + 3 * sd.max(), 4001)
    dens = (w[None, :] * stats.norm.pdf(grid[:, None], mu[None, :], sd[None, :])).sum(axis=1)
    dd = np.diff(dens)
    return int(((dd[:-1] > 0) & (dd[1:] <= 0)).sum())


def bimodality(x):
    """Two-component Gaussian mixture of r against the best one-component alternative: Ashman's D, minor weight,
    component angles, number of modes of the fitted mixture; Hartigan's dip if the optional `diptest` package is
    installed. gmm_dbic = BIC(1 Gaussian) - BIC(2 Gaussians) (as before, descriptive); gmm_dbic_skew = min(BIC(1
    Gaussian), BIC(skew-normal)) - BIC(2 Gaussians). bimodal = n >= MIN_HEADS and gmm_dbic_skew > BIMODAL_DBIC and
    D > 2 and minor weight >= 5 %. Against a single Gaussian alone, a skewed but unimodal r (skew-normal, gamma,
    lognormal) is split into a narrow core and a wide tail with D just above 2 and flagged in up to ~50 % of runs."""
    x = np.asarray(x, dtype=np.float64)
    out = {"gmm_dbic": np.nan, "gmm_dbic_skew": np.nan, "gmm_ashman_d": np.nan, "gmm_w_minor": np.nan,
           "gmm_angle_major": np.nan, "gmm_angle_minor": np.nan, "gmm_modes": np.nan, "bimodal": False, "dip": np.nan,
           "p_dip": np.nan}
    if len(x) < 8 or x.std() == 0:
        return out
    from sklearn.mixture import GaussianMixture
    X = x.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g1 = GaussianMixture(1, random_state=0).fit(X)
        g2 = GaussianMixture(2, n_init=5, random_state=0).fit(X)
    mu, var, w = g2.means_.ravel(), g2.covariances_.ravel(), g2.weights_
    minor = int(np.argmin(w))
    d_ash = float(np.sqrt(2) * abs(mu[0] - mu[1]) / np.sqrt(var[0] + var[1]))
    bic2 = float(g2.bic(X))
    bic_gauss = float(g1.bic(X))
    bic_skew = float(-2 * skewnorm_loglik(x) + 3 * np.log(len(x)))
    dbic_skew = min(bic_gauss, bic_skew) - bic2
    out.update(gmm_dbic=bic_gauss - bic2, gmm_dbic_skew=dbic_skew, gmm_ashman_d=d_ash, gmm_w_minor=float(w[minor]),
               gmm_angle_major=float(np.degrees(np.arccos(np.clip(mu[1 - minor], -1, 1)))),
               gmm_angle_minor=float(np.degrees(np.arccos(np.clip(mu[minor], -1, 1)))),
               gmm_modes=_mixture_modes(mu, var, w),
               bimodal=bool(len(x) >= MIN_HEADS and dbic_skew > BIMODAL_DBIC and d_ash > BIMODAL_ASHMAN
                            and w[minor] >= BIMODAL_MIN_WEIGHT))
    try:
        import diptest
        dip, p = diptest.diptest(x)
        out.update(dip=float(dip), p_dip=float(p))
    except ImportError:
        pass
    return out


# ----------------------------------------------------------------------------- (g) predictability

def pred_folds(layers, k=PRED_FOLDS):
    """Fold id per head: whole layers, interleaved over depth (rank of the layer mod k, so every fold spans the depth
    range and a depth profile is interpolated, not extrapolated); None with fewer than k layers."""
    groups, inv = np.unique(np.asarray(layers), return_inverse=True)
    return None if len(groups) < k else inv % k


def layer_block_perms(layers, rng, n_perm):
    """(n_perm, n) permutation indices: whole layers are moved onto layers with the same number of heads (a block of
    residuals keeps its layer clustering) and the heads are shuffled within each layer. These permutations form a group
    and are drawn uniformly from it, so the test is exact whenever the residual blocks are exchangeable across layers of
    equal size - in particular for iid residuals, but also for layer offsets that do not depend on the covariates.
    Price: a layer whose head count occurs only once stays in place (only its heads are shuffled), so sparse head sets
    with unevenly filled layers lose power, mostly for covariates that are constant within a layer (depth;
    docs/MEANFIELD.md, (g))"""
    layers = np.asarray(layers)
    n = len(layers)
    groups, inv = np.unique(layers, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(groups))]
    within = np.empty((n_perm, n), dtype=np.int64)
    for m in members:
        within[:, m] = m[np.argsort(rng.random((n_perm, len(m))), axis=1)]
    out = np.empty_like(within)
    rows = np.arange(n_perm)[:, None]
    sizes = np.array([len(m) for m in members])
    for s in np.unique(sizes):
        block = np.stack([members[g] for g in np.flatnonzero(sizes == s)])     # (layers of size s, s) head indices
        src = np.argsort(rng.random((n_perm, len(block))), axis=1)             # target layer j <- block of layer src[:, j]
        for j in range(len(block)):
            out[:, block[j]] = within[rows, block[src[:, j]]]
    return out


def ridge_cv_r2(Y, Z, folds, k=PRED_FOLDS, alphas=PRED_ALPHAS):
    """Held-out R^2 of ridge regressions of every row of Y (P, n) on Z (n, p), fully vectorized over the rows.

    Outer CV over the k folds; for outer fold f the penalty is chosen by an inner CV over the remaining k - 1 folds
    (leave one fold out, training on k - 2), by the smallest summed squared error. The intercept is not penalized (each
    training set is centred). Held-out R^2 = 1 - sum_f SS_res,f / sum_f SS_tot,f with SS_tot,f around the TRAINING mean
    of fold f (the intercept-only prediction), so pure noise gives R^2 <= 0 on average. Every fit only needs per-fold
    sums (n, sum z, Z'Z, sum y, sum y^2, Z'y), so all rows (observed + permutations) cost little more than one.
    Returns {"r2": (P,), "alpha_idx": (k, P) chosen penalty per outer fold, "full_alpha_idx": penalty with the smallest
    outer CV error of row 0, "beta": (p,) coefficients of row 0 fitted on all heads with that penalty}."""
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    Z = np.asarray(Z, dtype=np.float64)
    P, p, A = len(Y), Z.shape[1], len(alphas)
    masks = [folds == f for f in range(k)]
    nf = np.array([m.sum() for m in masks], dtype=np.float64)
    sz = np.stack([Z[m].sum(axis=0) for m in masks])                      # (k, p)
    szz = np.stack([Z[m].T @ Z[m] for m in masks])                        # (k, p, p)
    sy = np.stack([Y[:, m].sum(axis=1) for m in masks])                   # (k, P)
    syy = np.stack([(Y[:, m] ** 2).sum(axis=1) for m in masks])           # (k, P)
    szy = np.stack([Y[:, m] @ Z[m] for m in masks])                       # (k, P, p)

    def fit(train):
        n_t = nf[train].sum()
        mz, my = sz[train].sum(axis=0) / n_t, sy[train].sum(axis=0) / n_t
        czz = szz[train].sum(axis=0) - n_t * np.outer(mz, mz)
        czy = szy[train].sum(axis=0) - n_t * my[:, None] * mz[None, :]
        fits = []
        for a in alphas:
            beta = np.linalg.solve(czz + a * n_t * np.eye(p), czy.T).T     # (P, p)
            fits.append((beta, my - beta @ mz))
        return fits, my

    def sse(v, beta, c):
        return (syy[v] - 2 * (beta * szy[v]).sum(axis=1) - 2 * c * sy[v] + np.einsum("bi,ij,bj->b", beta, szz[v], beta)
                + 2 * c * (beta @ sz[v]) + nf[v] * c ** 2)

    ss_res, ss_tot = np.zeros(P), np.zeros(P)
    outer_err = np.zeros((A, P))
    chosen = np.empty((k, P), dtype=np.int64)
    cols = np.arange(P)
    for f in range(k):
        inner = np.zeros((A, P))
        for j in range(k):
            if j == f:
                continue
            train = np.ones(k, dtype=bool)
            train[[f, j]] = False
            fits, _ = fit(train)
            inner += np.stack([sse(j, beta, c) for beta, c in fits])
        best = np.argmin(inner, axis=0)                                     # first minimum = strongest penalty
        train = np.ones(k, dtype=bool)
        train[f] = False
        fits, my = fit(train)
        err = np.stack([sse(f, beta, c) for beta, c in fits])               # (A, P)
        outer_err += err
        ss_res += err[best, cols]
        ss_tot += syy[f] - 2 * my * sy[f] + nf[f] * my ** 2
        chosen[f] = best
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1 - ss_res / ss_tot
    full = int(np.argmin(outer_err[:, 0]))
    fits, _ = fit(np.ones(k, dtype=bool))
    return {"r2": r2, "alpha_idx": chosen, "full_alpha_idx": full, "beta": fits[full][0][0]}


def predictability(r, layers, covariates, rng, n_perm=N_PERM, perms=None, k=PRED_FOLDS, alphas=PRED_ALPHAS):
    """(g) Out-of-sample predictability of the residual r (signed) and of |r| (magnitude) from head covariates.

    covariates: {name: (n,) array or None}; a covariate is dropped when it is missing, has a non-finite entry on this
    head set or no variance (names in ABS_COVARIATES enter the |r| model as absolute values). Covariates are
    standardized over the head set (label-free: it only fixes the scale of the penalty); the targets are standardized as
    well (R^2 is invariant to that). Held-out R^2 from ridge_cv_r2 with folds = whole layers (pred_folds); null =
    layer_block_perms (whole layers moved onto layers of the same size, heads shuffled within layers, the same draws
    for both targets); p = (1 + #{R^2_perm >= R^2}) / (1 + n_perm), one-sided. Not computable (NaN) with fewer than
    MIN_HEADS heads, fewer than k layers or no usable covariate."""
    out = {f"pred_{t}_{s}": np.nan for t in PRED_TARGETS for s in ("r2", "r2_null", "alpha")}
    out.update({f"p_pred_{t}": np.nan for t in PRED_TARGETS})
    out.update({f"pred_{t}_{s}": "" for t in PRED_TARGETS for s in ("top", "coefs")})
    out.update(pred_n_cov=0, pred_covariates="", pred_folds=np.nan)
    r, layers = np.asarray(r, dtype=np.float64), np.asarray(layers)
    n = len(r)
    folds = pred_folds(layers, k)
    # constancy by the exact range: the float std of a constant vector need not be 0
    if n < MIN_HEADS or folds is None or not np.isfinite(r).all() or np.ptp(r) == 0:
        return out
    names, cols = [], []
    for name, v in (covariates or {}).items():
        if v is None:
            continue
        v = np.asarray(v, dtype=np.float64)
        if v.shape == (n,) and np.isfinite(v).all() and np.ptp(v) > 0:
            names.append(name)
            cols.append(v)
    if not names:
        return out
    if perms is None:
        perms = layer_block_perms(layers, rng, n_perm)
    idx = np.vstack([np.arange(n)[None, :], perms])
    out.update(pred_n_cov=len(names), pred_covariates=" ".join(names), pred_folds=int(k))
    for t in PRED_TARGETS:
        y = r if t == "r" else np.abs(r)
        use = [(f"|{nm}|", np.abs(c)) if (t == "absr" and nm in ABS_COVARIATES) else (nm, c) for nm, c in zip(names, cols)]
        use = [(nm, c) for nm, c in use if np.ptp(c) > 0]
        if np.ptp(y) == 0 or not use:
            continue
        X = np.column_stack([c for _, c in use])
        res = ridge_cv_r2(((y - y.mean()) / y.std())[idx], (X - X.mean(axis=0)) / X.std(axis=0), folds, k, alphas)
        obs, null = float(res["r2"][0]), res["r2"][1:]
        beta = res["beta"]
        top = int(np.argmax(np.abs(beta)))
        out.update({f"pred_{t}_r2": obs, f"pred_{t}_r2_null": float(np.median(null)),
                    f"pred_{t}_alpha": float(np.median(np.asarray(alphas)[res["alpha_idx"][:, 0]])),
                    f"p_pred_{t}": float((1 + (null >= obs - 1e-12).sum()) / (len(null) + 1)),
                    f"pred_{t}_top": use[top][0] if np.abs(beta[top]) > 0 else "",
                    f"pred_{t}_coefs": " ".join(f"{nm}:{b:+.3f}" for (nm, _), b in zip(use, beta))})
    return out


# ----------------------------------------------------------------------------- one head set

def analyze_set(r, layers, d, rng=None, r0=None, r_prev=None, r_final=None, n_boot=N_BOOT, n_perm=N_PERM,
                covariates=None, n_layers_model=None):
    """All signatures (a)-(g) for one head set. r, layers, r0 (step0), r_prev, r_final: per head, in flat-index order.

    covariates: extra head covariates for (g), {name: per-head array in the same order as r, or None} (analyze_model
    passes log_norm_a / log_norm_b / score_a / score_b); relative depth layer / (n_layers_model - 1) (default: the
    largest layer index present), its centred square and r0 (as r_step0) are added here.
    Returns a dict of statistics with p-values p_{test} for the tests in TESTS (NaN where not applicable); Holm and the
    verdict are applied afterwards to a whole family (judge)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    r, layers = np.asarray(r, dtype=np.float64), np.asarray(layers)
    keep = np.isfinite(r)
    others = {}
    for name, arr in (("r0", r0), ("r_prev", r_prev), ("r_final", r_final)):
        others[name] = None if arr is None else np.asarray(arr, dtype=np.float64)[keep]
    r, layers = r[keep], layers[keep]
    n = len(r)
    extra = {name: None if arr is None else np.asarray(arr, dtype=np.float64)[keep] for name, arr in (covariates or {}).items()}
    row = {"d": int(d), "n_heads": n, "n_layers": int(len(np.unique(layers))) if n else 0, "n_dropped_nan": int((~keep).sum())}
    angles = np.degrees(np.arccos(np.clip(r, -1, 1)))
    row.update(angle_n=n, angle_mean_deg=float(angles.mean()) if n else np.nan,
               angle_median_deg=float(np.median(angles)) if n else np.nan,
               angle_std_deg=float(angles.std()) if n else np.nan)      # ddof = 0 as in Table 17 / direction_angles
    for t in TESTS:
        row[f"p_{t}"] = np.nan
    if n < 3:
        return row
    perms = np.argsort(rng.random((n_perm, n)), axis=1)
    boot_idx = rng.integers(0, n, size=(n_boot, n))

    # (a) location vs N1 and vs step0 (paired, same heads)
    loc = location_test(r, layers, rng, n_boot)
    lo1, hi1 = n1_mean_interval(n, d)
    row.update(mean_r=loc["mean"], mean_r_lo=loc["lo"], mean_r_hi=loc["hi"], mean_r_se=loc["se"], median_r=loc["median"],
               frac_pos=loc["frac_pos"], loc_sign_ok=loc["sign_ok"], loc_boot_unit=loc["unit"], n1_mean_lo=lo1,
               n1_mean_hi=hi1, loc_in_n1_interval=bool(lo1 <= loc["mean"] <= hi1), p_loc_n1=loc["p"])
    r0 = others["r0"]
    ok0 = None if r0 is None else np.isfinite(r0)
    if r0 is not None and ok0.sum() >= 3:
        loc2 = location_test(r[ok0] - r0[ok0], layers[ok0], rng, n_boot)
        row.update(mean_r_step0=float(r0[ok0].mean()), delta_r=loc2["mean"], delta_r_lo=loc2["lo"], delta_r_hi=loc2["hi"],
                   delta_sign_ok=loc2["sign_ok"], p_loc_n2=loc2["p"])

    # (b) spread and conflict share
    paired0 = r0 if (r0 is not None and ok0.all()) else None
    row.update(spread_test(r, d, boot_idx, paired0))
    row.update(conflict_test(r, d, None if r0 is None else r0[ok0]))

    # (c) averaging law, (d) layer structure
    row.update(averaging_law(r, layers, rng, perms, n_boot))
    row.update(layer_structure(r, layers, rng, perms))

    # (e) persistence
    for name, key in (("r_prev", "prev"), ("r_final", "final")):
        if others[name] is not None:
            rho, p = persistence_test(r, others[name], rng, perms)
            row.update({f"rho_{key}": rho, f"p_persist_{key}": p})

    # (f) shape
    if n >= 4:
        kb = excess_kurtosis_rows(r[boot_idx])
        kb = kb[np.isfinite(kb)]
        row.update(kurt=float(excess_kurtosis_rows(r)[0]), kurt_n1=n1_excess_kurtosis(d),
                   kurt_lo=float(np.percentile(kb, 2.5)) if len(kb) else np.nan,
                   kurt_hi=float(np.percentile(kb, 97.5)) if len(kb) else np.nan)
    row.update(bimodality(r))

    # (g) predictability of r and |r| from head covariates (last, so the random streams of (a)-(f) are unchanged)
    depth = layers.astype(np.float64) / max((int(n_layers_model) if n_layers_model else int(layers.max()) + 1) - 1, 1)
    cov = {"depth": depth, "depth2": (depth - 0.5) ** 2, **extra, "r_step0": r0}
    row.update(predictability(r, layers, cov, rng, n_perm))
    return row


# ----------------------------------------------------------------------------- Holm + verdict

def _finite(v):
    try:
        return v is not None and bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _text(v):
    """str(v), but "" for None / NaN (empty strings come back from a CSV as NaN)."""
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)


def _p_for_verdict(row, test):
    if bool(row.get("primary")) and test in PRIMARY_TESTS:
        return row.get(f"p_{test}_primary_holm", np.nan)
    return row.get(f"p_{test}_holm", np.nan)


def verdict_of(row, alpha=ALPHA):
    """(labels, drivers, code) of one row after Holm (fixed rules, docs/MEANFIELD.md):

    mean-field-artig: n >= MIN_HEADS, location consistent with N1 and step0 (no Holm-significant test), slope CI contains
                      -0.5, is at most MAX_SLOPE_CI_WIDTH wide and the slope is not significantly shallower, the layer
                      test (d) was testable (finite p_icc) and is not significant, unimodal, and the predictability
                      tests (g) were computable (finite p_pred_r and p_pred_absr) and neither is significant
    systematisch:     location differs from N1 or step0 (Holm, mean and median of the same sign), or significant
                      persistence of r(h) to the previous checkpoint
    kollektiv:        slope significantly shallower than -0.5 (Holm), or significant layer clustering (ICC, Holm)
    vorhersagbar:     r or |r| significantly predictable out of sample from the head covariates (g) (Holm); such a row is
                      never mean-field-artig
    bistabil:         bimodal (mixture rule of `bimodality`)
    unterbestimmt:    n < MIN_HEADS (labels from significant tests are kept), or the mean-field pattern holds but the
                      layer test was not testable, the slope CI is too wide or (g) was not computable (code "m");
    uneindeutig:      none of the above.
    code: one letter per label (LABEL_CODES); "P" instead of "S" when persistence alone made it systematisch, "m" instead
    of "U" for the guarded mean-field pattern."""
    def p(t):
        v = _p_for_verdict(row, t)
        return v if v is not None and np.isfinite(v) else np.nan

    def sig(t):
        return bool(np.isfinite(p(t)) and p(t) < alpha)

    labels, drivers = [], []
    n = int(row.get("n_heads", 0) or 0)
    loc1 = sig("loc_n1") and bool(row.get("loc_sign_ok"))
    loc2 = sig("loc_n2") and bool(row.get("delta_sign_ok"))
    pers = sig("persist_prev")
    if loc1:
        drivers.append(f"Lage≠0 (N1): r̄={row['mean_r']:+.3f}, p_Holm={p('loc_n1'):.2g}")
    if loc2:
        drivers.append(f"Lage≠step0 (N2): Δr̄={row['delta_r']:+.3f}, p_Holm={p('loc_n2'):.2g}")
    if pers:
        drivers.append(f"Persistenz zu {row.get('prev_revision')}: ρ={row['rho_prev']:.2f}, p_Holm={p('persist_prev'):.2g}")
    if loc1 or loc2 or pers:
        labels.append("systematisch")
    slope_shallow = sig("slope") and np.isfinite(row.get("slope", np.nan)) and row["slope"] > -0.5
    clustered = sig("icc")
    if slope_shallow:
        drivers.append(f"Mittelungssteigung {row['slope']:.2f} flacher als -0.5, p_Holm={p('slope'):.2g}")
    if clustered:
        drivers.append(f"Layer-Häufung ICC={row['icc']:.2f}, p_Holm={p('icc'):.2g}")
    if slope_shallow or clustered:
        labels.append("kollektiv")
    pred = {t: sig(f"pred_{t}") for t in PRED_TARGETS}
    for t, name in (("r", "r"), ("absr", "|r|")):
        if pred[t]:
            top = _text(row.get(f"pred_{t}_top"))
            drivers.append(f"Rest vorhersagbar ({name}): R²={_fmt(row.get(f'pred_{t}_r2'), '{:+.2f}')} (Null-Median "
                           f"{_fmt(row.get(f'pred_{t}_r2_null'), '{:+.2f}')}), p_Holm={p(f'pred_{t}'):.2g}"
                           + (f", stärkste Kovariate {COVARIATE_LABEL.get(top, top)}" if top else ""))
    predictable = any(pred.values())
    if predictable:
        labels.append("vorhersagbar")
    if bool(row.get("bimodal")):
        labels.append("bistabil")
        drivers.append(f"bimodal: ΔBIC gegen Normal/Schiefnormal={row['gmm_dbic_skew']:.0f}, D={row['gmm_ashman_d']:.1f}, "
                       f"{row['gmm_w_minor']:.0%} der Heads bei {row['gmm_angle_minor']:.0f}°")
    # the mean-field pattern of (a)-(f); a predictable residual (g) is structured, whatever (a)-(f) say
    pattern_af = (n >= MIN_HEADS and not sig("loc_n1") and not sig("loc_n2") and bool(row.get("slope_ci_has_half"))
                  and not slope_shallow and not clustered and not bool(row.get("bimodal")))
    pattern = pattern_af and not predictable
    if pattern_af and predictable:
        drivers.append("Lage, Steigung, Layer-Test und Form passen zum Mean-Field-Muster, aber der Rest ist vorhersagbar "
                       "(g): nicht mean-field-artig")
    # a mean-field verdict needs the layer test to have been run (a sparse head set leaves no within-layer degrees of
    # freedom: p_icc is NaN, which must not count as "no clustering") and a slope CI narrow enough to tell -1/2 apart
    p_icc_raw = row.get("p_icc", np.nan)
    icc_testable = p_icc_raw is not None and np.isfinite(float(p_icc_raw))
    width = float(row.get("slope_hi", np.nan)) - float(row.get("slope_lo", np.nan))
    precise = bool(np.isfinite(width) and width <= MAX_SLOPE_CI_WIDTH)
    guards = []
    if n >= MIN_HEADS and not icc_testable:
        guards.append(f"Layer-Häufung nicht testbar ({n} Heads in {int(row.get('n_layers', 0) or 0)} Layern; nötig: "
                      f">= {MIN_WITHIN_DF} Freiheitsgrade innerhalb der Layer und >= 3 Layer mit >= 2 Heads)")
    if n >= MIN_HEADS and np.isfinite(width) and not precise:
        guards.append(f"Steigungs-KI zu breit ({width:.2f} > {MAX_SLOPE_CI_WIDTH})")
    # (g) must have been computed for both targets: "not computable" must not count as "not predictable"
    pred_testable = all(_finite(row.get(f"p_pred_{t}")) for t in PRED_TARGETS)
    if n >= MIN_HEADS and not pred_testable:
        guards.append(f"Vorhersagbarkeit des Rests (g) nicht berechenbar (nötig: >= {MIN_HEADS} Heads in >= {PRED_FOLDS} "
                      "Layern und mindestens eine Kovariate)")
    mean_field = pattern and icc_testable and precise and pred_testable
    if mean_field:
        labels.insert(0, "mean-field-artig")
        drivers.insert(0, f"Lage≈0, Steigung {row['slope']:.2f} [{row['slope_lo']:.2f}, {row['slope_hi']:.2f}] ∋ -0.5, "
                          f"keine Layer-Häufung (ICC={row['icc']:.2f}, p_Holm={p('icc'):.2g}), unimodal, kein "
                          f"vorhersagbarer Rest (R² r {_fmt(row.get('pred_r_r2'), '{:+.2f}')}, |r| "
                          f"{_fmt(row.get('pred_absr_r2'), '{:+.2f}')}; p_Holm {_fmt(p('pred_r'), '{:.2g}')} / "
                          f"{_fmt(p('pred_absr'), '{:.2g}')})")
    elif pattern:
        labels.insert(0, "unterbestimmt")
        drivers.insert(0, f"Lage≈0, Steigung {row['slope']:.2f} [{row['slope_lo']:.2f}, {row['slope_hi']:.2f}] ∋ -0.5, "
                          "unimodal, aber nicht mean-field-artig: " + "; ".join(guards))
    elif guards:
        drivers.extend(guards)
    if n < MIN_HEADS:
        labels.append("unterbestimmt")
        drivers.append(f"n={n} < {MIN_HEADS} Heads (keine Mittelungssteigung)")
    if not labels:
        labels.append("uneindeutig")
        drivers.append("keine Regel greift (z. B. Steigungs-KI ohne -0.5, aber nach Holm nicht signifikant)")

    def code_of(label):
        if label == "systematisch" and not (loc1 or loc2):
            return PERSISTENCE_ONLY_CODE
        if label == "unterbestimmt" and pattern:
            return UNDERDETERMINED_MF_CODE
        return LABEL_CODES[label]

    return labels, drivers, "".join(code_of(l) for l in labels)


def judge(df, alpha=ALPHA, primary_revision=PRIMARY_REVISION):
    """Adds p_{test}_holm (Holm within model x checkpoint over all rows and tests), the preregistered primary family
    (shared top heads at primary_revision, tests PRIMARY_TESTS, Holm within each model), verdict and drivers."""
    df = df.copy().reset_index(drop=True)
    for t in TESTS:
        if f"p_{t}" not in df:
            df[f"p_{t}"] = np.nan
        df[f"p_{t}_holm"] = np.nan
    for _, idx in df.groupby(["model", "revision"]).groups.items():
        cells = [(i, t) for i in idx for t in TESTS if np.isfinite(df.at[i, f"p_{t}"])]
        for (i, t), a in zip(cells, holm([df.at[i, f"p_{t}"] for i, t in cells])):
            df.at[i, f"p_{t}_holm"] = a
    df["primary"] = (df["headset"] == "shared_top15") & (df["revision"] == primary_revision)
    for t in PRIMARY_TESTS:
        df[f"p_{t}_primary_holm"] = np.nan
    for _, sub in df[df["primary"]].groupby("model"):
        cells = [(i, t) for i in sub.index for t in PRIMARY_TESTS if np.isfinite(df.at[i, f"p_{t}"])]
        for (i, t), a in zip(cells, holm([df.at[i, f"p_{t}"] for i, t in cells])):
            df.at[i, f"p_{t}_primary_holm"] = a
    verdicts, codes, drivers = [], [], []
    for _, row in df.iterrows():
        labels, why, code = verdict_of(row.to_dict(), alpha)
        verdicts.append("+".join(labels))
        codes.append(code)
        drivers.append("; ".join(why))
    df["verdict"], df["verdict_code"], df["drivers"] = verdicts, codes, drivers
    df["status"] = np.where(df["primary"], "primaer (vorab festgelegt)", "deskriptiv")
    return df


# ----------------------------------------------------------------------------- discovery + loading

def parse_step(revision):
    m = re.fullmatch(r"step(\d+)", str(revision))
    return int(m.group(1)) if m else None


def sort_revisions(revisions):
    return sorted(revisions, key=lambda r: (parse_step(r) is None, parse_step(r) or 0, str(r)))


def geometry(model):
    """(layers, heads, head_dim) from the registry."""
    info = config.MODELS[model]
    return int(info["layers"]), int(info["heads"]), int(info["head_dim"])


def nominal_params(model):
    """Nominal parameter count from the Pythia name (pythia-160m -> 1.6e8, pythia-2.8b -> 2.8e9)."""
    m = re.search(r"(\d+(?:\.\d+)?)([mb])$", model)
    return float(m.group(1)) * (1e6 if m.group(2) == "m" else 1e9) if m else np.nan


def _direction_regex():
    names = sorted((re.escape(m) for m in config.MODELS), key=len, reverse=True)
    return re.compile(rf"^({'|'.join(names)})_(step\d+|main)_({'|'.join(VALUES)})\.npz$")


def candidate_roots(base, max_depth=2):
    """Directories at `base` or up to max_depth levels below that hold results/directions/ (sorted)."""
    roots = []

    def walk(d, depth):
        if os.path.isdir(os.path.join(d, "results", "directions")):
            roots.append(d)
        if depth >= max_depth:
            return
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        for e in entries:
            if e.is_dir() and not e.name.startswith(".") and e.name not in SKIP_DIRS:
                walk(e.path, depth + 1)

    if not os.path.isdir(base):
        raise FileNotFoundError(f"base directory not found: {base}")
    walk(os.path.abspath(base), 0)
    return roots


def discover(base, models=None, values=VALUES, max_depth=2):
    """{model: {"root", "manifest", "revisions": {rev: {value: {"directions", "heads"}}}, "notes"}} for every model with
    direction files under base (the root may be base itself, e.g. pythia-160m in MyDrive/ValueHeadsPythia, or a
    subfolder). A model found in several roots is taken from the root with the most files (noted)."""
    rx = _direction_regex()
    found = {}
    for root in candidate_roots(base, max_depth):
        p = config.paths(root)
        per_model = {}
        for name in sorted(os.listdir(p["directions"])):
            m = rx.match(name)
            if not m:
                continue
            model, rev, value = m.groups()
            if (models and model not in models) or value not in values:
                continue
            heads = config.heads_path(p, model, rev, value, "pythia")
            per_model.setdefault(model, {}).setdefault(rev, {})[value] = {
                "directions": os.path.join(p["directions"], name), "heads": heads if os.path.exists(heads) else None}
        for model, revs in per_model.items():
            n_files = sum(len(v) for v in revs.values())
            entry = {"root": root, "manifest": p["manifest"] if os.path.exists(p["manifest"]) else None,
                     "revisions": {r: revs[r] for r in sort_revisions(revs)}, "notes": [], "n_files": n_files}
            if model in found:
                keep, other = (entry, found[model]) if n_files > found[model]["n_files"] else (found[model], entry)
                keep["notes"].append(f"{model} auch unter {other['root']} ({other['n_files']} Dateien) gefunden; "
                                     f"verwendet: {keep['root']}")
                found[model] = keep
            else:
                found[model] = entry
    return {m: found[m] for m in sorted(found, key=lambda m: (nominal_params(m), m))}


def discovery_table(found):
    """One row per model: root, checkpoints, values, head tables present, geometry."""
    rows = []
    for model, e in found.items():
        L, H, D = geometry(model)
        revs = list(e["revisions"])
        vals = sorted({v for r in revs for v in e["revisions"][r]})
        n_heads = sum(1 for r in revs for v in e["revisions"][r].values() if v["heads"])
        n_dirs = sum(len(e["revisions"][r]) for r in revs)
        rows.append({"model": model, "root": e["root"], "manifest": bool(e["manifest"]), "checkpoints": len(revs),
                     "first": revs[0] if revs else None, "last": revs[-1] if revs else None, "values": ",".join(vals),
                     "directions": n_dirs, "head_tables": n_heads, "L": L, "H": H, "d": D})
    return pd.DataFrame(rows)


def head_scores(payload, n):
    """(n,) stored ranking score (meta "score", e.g. auc) of every head of a head-table payload by flat_idx; NaN where a
    head is missing or has no score."""
    col = payload["meta"].get("score", "auc")
    out = np.full(n, np.nan)
    for row in payload["heads"]:
        i, v = int(row["flat_idx"]), row.get(col)
        if 0 <= i < n and v is not None:
            out[i] = float(v)
    return out


def load_model_data(model, entry):
    """{rev: {"dirs": {value: (L * H, D) float32}, "top": {value: [flat_idx] or None}, "scores": {value: (L * H,) stored
    ranking score or None}}} plus notes; geometry checked against the registry (directions shape, heads-table meta
    layers / heads)."""
    from ..directions import load_directions
    from ..probing import top_heads_from_file
    L, H, D = geometry(model)
    data, notes = {}, list(entry.get("notes", []))
    for rev, files in entry["revisions"].items():
        dirs, top, scores = {}, {}, {}
        for value, f in files.items():
            arr, _, meta = load_directions(f["directions"])
            if arr.shape != (L * H, D):
                notes.append(f"{rev} {value}: Richtungen haben Form {arr.shape}, erwartet {(L * H, D)}; übersprungen")
                continue
            dirs[value] = arr
            top[value] = scores[value] = None
            if f["heads"]:
                try:
                    with open(f["heads"]) as fh:
                        payload = json.load(fh)
                    hmeta = payload["meta"]
                    if (int(hmeta["layers"]), int(hmeta["heads"])) != (L, H):
                        raise ValueError(f"layers/heads {hmeta['layers']}/{hmeta['heads']} != {L}/{H}")
                    top[value] = top_heads_from_file(f["heads"], ratio=config.TOP_RATIO)
                    scores[value] = head_scores(payload, L * H)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                    notes.append(f"{rev} {value}: Head-Tabelle nicht nutzbar ({e}); nur 'alle Heads'")
            else:
                notes.append(f"{rev} {value}: keine Head-Tabelle; nur 'alle Heads'")
        data[rev] = {"dirs": dirs, "top": top, "scores": scores}
    return data, notes


def _row_seed(seed, *parts):
    return np.random.default_rng([int(seed), zlib.crc32("|".join(str(p) for p in parts).encode())])


def analyze_model(model, entry, n_boot=N_BOOT, n_perm=N_PERM, seed=0, say=None):
    """DataFrame (one row per checkpoint x pair x head set, Holm + verdicts applied) and loading notes."""
    from . import direction_angles as ang
    say = say or (lambda *a, **k: None)
    L, H, D = geometry(model)
    k15 = int(config.TOP_RATIO * L * H)
    data, notes = load_model_data(model, entry)
    revs = [r for r in sort_revisions(data) if len(data[r]["dirs"]) >= 2]
    if not revs:
        return pd.DataFrame(), notes
    final = PRIMARY_REVISION if PRIMARY_REVISION in revs else revs[-1]
    pairs = [p for p in itertools.combinations(VALUES, 2) if any(set(p) <= set(data[r]["dirs"]) for r in revs)]
    cos = {r: {p: head_cosines(data[r]["dirs"][p[0]], data[r]["dirs"][p[1]]) for p in pairs if set(p) <= set(data[r]["dirs"])}
           for r in revs}
    norms = {r: {v: np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1) for v, x in data[r]["dirs"].items()} for r in revs}
    rows = []
    t0 = time.time()
    for i, rev in enumerate(revs):
        prev = next((q for q in reversed(revs[:i]) if any(pp in cos[q] for pp in cos[rev])), None)
        for pair in pairs:
            if pair not in cos[rev]:
                continue
            a, b = pair
            pd_dirs = {a: data[rev]["dirs"][a], b: data[rev]["dirs"][b]}
            ref_all = ang.all_head_angle_table(pd_dirs, model, rev).iloc[0]
            tops = {v: data[rev]["top"].get(v) for v in pair}
            for headset in HEADSETS:
                if headset == "all":
                    idx = np.arange(L * H)
                    ref = {"n": int(ref_all["n"]), "mean": ref_all["mean_deg"], "median": ref_all["median_deg"],
                           "std": ref_all["std_deg"]}
                else:
                    if tops[a] is None or tops[b] is None:
                        continue
                    table = ang.angle_table(pd_dirs, tops, model, rev).iloc[0]
                    idx = np.array(table["shared_heads"], dtype=np.int64)
                    ref = {"n": int(table["n_shared"]), "mean": table["mean_deg"], "median": table["median_deg"],
                           "std": table["std_deg"]}
                idx = np.sort(idx)
                r = cos[rev][pair][idx] if len(idx) else np.array([])

                def on_heads(q):
                    return None if (q is None or q == rev or pair not in cos[q]) else cos[q][pair][idx]

                # (g) head covariates: log DoM norms (raw mass-mean, Eq. 2) and stored probe scores of both values
                cov = {}
                for key, v in (("a", a), ("b", b)):
                    with np.errstate(divide="ignore"):
                        cov[f"log_norm_{key}"] = np.log(norms[rev][v][idx])
                    s = data[rev]["scores"].get(v)
                    cov[f"score_{key}"] = None if s is None else s[idx]
                row = analyze_set(r, idx // H, D, _row_seed(seed, model, rev, pair, headset), r0=on_heads("step0" if "step0" in cos else None),
                                  r_prev=on_heads(prev), r_final=on_heads(final), n_boot=n_boot, n_perm=n_perm,
                                  covariates=cov, n_layers_model=L)
                # Table-17 numbers must agree with analysis/direction_angles (same heads, same ddof = 0)
                if ref["n"] and row["n_dropped_nan"] == 0:
                    if ref["n"] != row["angle_n"] or not np.allclose([ref["mean"], ref["median"], ref["std"]],
                                                                     [row["angle_mean_deg"], row["angle_median_deg"], row["angle_std_deg"]],
                                                                     atol=1e-6):
                        raise AssertionError(f"{model} {rev} {pair} {headset}: angle statistics disagree with direction_angles")
                row.update(model=model, revision=rev, step=parse_step(rev), pair=f"{a} x {b}", value_a=a, value_b=b,
                           headset=headset, L=L, H=H, k15=k15, n_top_a=len(tops[a]) if tops[a] is not None else np.nan,
                           n_top_b=len(tops[b]) if tops[b] is not None else np.nan, prev_revision=prev, final_revision=final,
                           heads=" ".join(str(int(h)) for h in idx) if headset != "all" else "all")
                rows.append(row)
        say(f"[meanfield] {model} {rev}: {len(rows)} Zeilen, {time.time() - t0:.1f}s")
    df = judge(pd.DataFrame(rows))
    return df[[c for c in COLUMNS if c in df] + [c for c in df if c not in COLUMNS]], notes


COLUMNS = ["model", "revision", "step", "pair", "value_a", "value_b", "headset", "status", "verdict", "verdict_code", "drivers",
           "d", "L", "H", "k15", "n_top_a", "n_top_b", "n_heads", "n_layers", "n_dropped_nan",
           "angle_n", "angle_mean_deg", "angle_median_deg", "angle_std_deg",
           "mean_r", "mean_r_lo", "mean_r_hi", "mean_r_se", "median_r", "frac_pos", "loc_sign_ok", "loc_boot_unit",
           "n1_mean_lo", "n1_mean_hi", "loc_in_n1_interval", "p_loc_n1",
           "mean_r_step0", "delta_r", "delta_r_lo", "delta_r_hi", "delta_sign_ok", "p_loc_n2",
           "var_r", "excess_var", "excess_var_lo", "excess_var_hi", "sd_ratio", "sd_ratio_lo", "sd_ratio_hi", "p_spread_n1",
           "sd_vs_step0", "sd_vs_step0_lo", "sd_vs_step0_hi", "p_spread_n2",
           "frac_gt100", "frac_gt100_n1", "frac_gt100_step0", "p_conflict_n1",
           "slope", "slope_lo", "slope_hi", "slope_ci_has_half", "slope_raw", "slope_null_mean", "slope_null_sd", "p_slope",
           "n_eff", "n_eff_null", "neff_ratio", "neff_window", "avg_window_max",
           "icc", "var_between", "var_within", "F_layer", "p_icc", "p_icc_F", "layer_lag1", "p_lag1",
           "prev_revision", "rho_prev", "p_persist_prev", "final_revision", "rho_final", "p_persist_final",
           "kurt", "kurt_lo", "kurt_hi", "kurt_n1", "gmm_dbic", "gmm_dbic_skew", "gmm_ashman_d", "gmm_w_minor",
           "gmm_angle_major", "gmm_angle_minor", "gmm_modes", "bimodal", "dip", "p_dip",
           "pred_n_cov", "pred_covariates", "pred_folds", "pred_r_r2", "pred_r_r2_null", "pred_r_alpha", "pred_r_top",
           "pred_r_coefs", "p_pred_r", "pred_absr_r2", "pred_absr_r2_null", "pred_absr_alpha", "pred_absr_top",
           "pred_absr_coefs", "p_pred_absr", "primary"]


# ----------------------------------------------------------------------------- figures

def _figure(*args, **kw):
    from matplotlib.figure import Figure          # object API: no pyplot, no backend switch inside notebooks
    return Figure(*args, **kw)


def _palette():
    from .heatmaps import CATEGORICAL
    return CATEGORICAL


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)


def _pair_slug(pair):
    return pair.replace(" x ", "_x_")


def _floats(text):
    return np.array([float(v) for v in str(text).split()]) if isinstance(text, str) and text.strip() else np.array([])


def plot_trajectory(df, model, pair, out_path):
    """Mean r, sd ratio, averaging slope and verdict over the checkpoints (x = step + 1, log), both head sets."""
    pal = _palette()
    sub = df[df["pair"] == pair]
    fig = _figure(figsize=(8.5, 10))
    axes = fig.subplots(4, 1, sharex=True)
    labels = list(LABEL_CODES)
    for j, hs in enumerate(HEADSETS):
        s = sub[sub["headset"] == hs].sort_values("step")
        if s.empty:
            continue
        x = s["step"].to_numpy(dtype=np.float64) + 1
        c = pal[j]
        for ax, (y, lo, hi) in zip(axes[:3], [("mean_r", "mean_r_lo", "mean_r_hi"), ("sd_ratio", "sd_ratio_lo", "sd_ratio_hi"),
                                              ("slope", "slope_lo", "slope_hi")]):
            ax.fill_between(x, s[lo].astype(float), s[hi].astype(float), color=c, alpha=0.15, lw=0)
            ax.plot(x, s[y].astype(float), color=c, lw=2, marker="o", ms=4, label=f"{HEADSET_LABEL[hs]}")
        for yi, lab in enumerate(labels):
            hit = s[s["verdict"].str.split("+").apply(lambda ls: lab in ls)]
            axes[3].plot(hit["step"].to_numpy(dtype=np.float64) + 1, np.full(len(hit), yi + (j - 0.5) * 0.25), ls="none",
                         marker="o" if j == 0 else "s", ms=8, color=c, label=HEADSET_LABEL[hs] if yi == 0 else None)
    all_rows = sub[sub["headset"] == "all"].sort_values("step")
    if not all_rows.empty:
        x = all_rows["step"].to_numpy(dtype=np.float64) + 1
        axes[0].fill_between(x, all_rows["n1_mean_lo"].astype(float), all_rows["n1_mean_hi"].astype(float), color="#9a9994",
                             alpha=0.25, lw=0, label="N1: 95 % des Mittels (alle Heads)")
    axes[0].axhline(0, color="#52514e", lw=0.8)
    axes[0].set_ylabel("mittleres r = cos θ")
    axes[1].axhline(1, color="#52514e", lw=0.8)
    axes[1].set_ylabel("sd(r) / sd_N1")
    axes[2].axhline(-0.5, color="#52514e", lw=0.8, ls="--")
    axes[2].set_ylabel("Mittelungssteigung\n(-0.5 = unabhängig)")
    axes[3].set_yticks(range(len(labels)))
    axes[3].set_yticklabels(labels)
    axes[3].set_ylim(-0.6, len(labels) - 0.4)
    axes[3].set_ylabel("Urteil")
    floor = IOI_FLOOR.get(model)
    for ax in axes:
        _style(ax)
        for s in IOI_MARKS:
            ax.axvline(s + 1, color="#9a9994", lw=2.0 if s == floor else 0.8, ls="--")
    steps = sorted(sub["step"].dropna().unique())
    axes[3].set_xscale("log")
    axes[3].set_xticks([s + 1 for s in steps])
    axes[3].set_xticklabels([str(int(s)) for s in steps], rotation=45, fontsize=8)
    axes[3].minorticks_off()
    axes[3].set_xlabel("Trainingsschritt (log-Achse über Schritt + 1); gestrichelt: step1000 / step2000 (IOI-Tief, "
                       "Dahiya & Blondin 2026" + (f"; Boden dieses Modells bei step{floor}" if floor else "") + ")")
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    axes[3].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"{model}: {pair} – Residuen der Werterichtungen über das Training", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return out_path


def plot_histograms(df, model, revision, cos_by_pair, d, out_path):
    """Histogram of r over all heads and the shared top heads per pair, with the exact N1 density."""
    pal = _palette()
    sub = df[df["revision"] == revision]
    pairs = list(dict.fromkeys(sub["pair"]))
    fig = _figure(figsize=(4.2 * max(len(pairs), 1), 3.6))
    axes = np.atleast_1d(fig.subplots(1, max(len(pairs), 1)))
    for ax, pair in zip(axes, pairs):
        r_all = cos_by_pair.get(pair, np.array([]))
        r_all = r_all[np.isfinite(r_all)]
        lim = max(0.5, float(np.nanmax(np.abs(r_all))) * 1.05) if len(r_all) else 0.5
        bins = np.linspace(-lim, lim, 41)
        ax.hist(r_all, bins=bins, density=True, color=pal[1], alpha=0.55, label=f"alle Heads (n={len(r_all)})")
        shared = sub[(sub["pair"] == pair) & (sub["headset"] == "shared_top15")]
        if not shared.empty and shared["heads"].iloc[0] not in ("", "all"):
            idx = np.array([int(h) for h in shared["heads"].iloc[0].split()])
            ax.hist(r_all[idx[idx < len(r_all)]] if len(r_all) else [], bins=bins, density=True, color=pal[0], alpha=0.65,
                    label=f"geteilte Top-15 % (n={len(idx)})")
        xs = np.linspace(-lim, lim, 400)
        ax.plot(xs, n1_pdf(xs, d), color="#2b2a27", lw=2, label=f"N1 (d={d})")
        ax.axvline(np.cos(np.radians(CONFLICT_DEG)), color="#9a9994", lw=0.8, ls="--")
        ax.set_title(pair, fontsize=10)
        ax.set_xlabel("r = cos θ  (gestrichelt: θ = 100°)")
        _style(ax)
    axes[0].set_ylabel("Dichte")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"{model} @ {revision}: Verteilung der Residuen r gegen die isotrope Null", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return out_path


def plot_averaging(df, model, revision, out_path):
    """sd of the window mean vs window size (log-log): layer order (observed) vs randomly permuted heads (null)."""
    pal = _palette()
    sub = df[df["revision"] == revision]
    pairs = list(dict.fromkeys(sub["pair"]))
    fig = _figure(figsize=(4.2 * max(len(pairs), 1), 3.6))
    axes = np.atleast_1d(fig.subplots(1, max(len(pairs), 1)))
    for ax, pair in zip(axes, pairs):
        drawn = False
        for j, hs in enumerate(HEADSETS):
            s = sub[(sub["pair"] == pair) & (sub["headset"] == hs)]
            if s.empty:
                continue
            n, obs, null = (_floats(s[c].iloc[0]) for c in ("avg_n", "avg_sd_obs", "avg_sd_null"))
            if not len(n):
                continue
            ax.plot(n, obs, color=pal[j], lw=2, marker="o", ms=5, label=f"{HEADSET_LABEL[hs]}: Layer-Reihenfolge "
                                                                         f"(Steigung {s['slope'].iloc[0]:.2f})")
            ax.plot(n, null, color=pal[j], lw=1.5, ls="--", label=f"{HEADSET_LABEL[hs]}: zufällige Teilmengen (Null)")
            drawn = True
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(pair, fontsize=10)
        ax.set_xlabel("Fenstergröße n (benachbarte Heads)")
        if not drawn:
            ax.text(0.5, 0.5, "zu wenige Heads", transform=ax.transAxes, ha="center", color="#52514e")
        else:
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
        _style(ax)
    axes[0].set_ylabel("sd des Mittels von r")
    fig.suptitle(f"{model} @ {revision}: Mittelungsgesetz (unabhängig: Steigung -1/2)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return out_path


def plot_width(summary, out_path, revision=PRIMARY_REVISION):
    """Excess spread sd(r) / sd_N1 vs nominal model size at the final checkpoint, per pair; one panel per head set."""
    pal = _palette()
    sub = summary[summary["revision"] == revision].copy()
    if sub.empty or sub["model"].nunique() < 2:
        return None
    sub["params"] = sub["model"].map(nominal_params)
    fig = _figure(figsize=(9, 3.8))
    axes = fig.subplots(1, len(HEADSETS), sharey=True)
    pairs = [f"{a} x {b}" for a, b in itertools.combinations(VALUES, 2)]
    for ax, hs in zip(axes, HEADSETS):
        for j, pair in enumerate(pairs):
            s = sub[(sub["headset"] == hs) & (sub["pair"] == pair)].sort_values("params")
            if s.empty:
                continue
            y = s["sd_ratio"].astype(float)
            err = np.vstack([y - s["sd_ratio_lo"].astype(float), s["sd_ratio_hi"].astype(float) - y])
            ax.errorbar(s["params"], y, yerr=np.clip(err, 0, None), color=pal[j], lw=2, marker="o", ms=5, capsize=3, label=pair)
        ax.axhline(1, color="#52514e", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(HEADSET_LABEL[hs], fontsize=10)
        ax.set_xlabel("Modellgröße (nominelle Parameter)")
        _style(ax)
    axes[0].set_ylabel("sd(r) / sd_N1 (1 = isotrop)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Breiten-Achse: Überschuss-Streuung der Residuen bei {revision}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return out_path


# ----------------------------------------------------------------------------- report (German)

def _fmt(v, f="{:.3f}"):
    try:
        return "–" if v is None or not np.isfinite(float(v)) else f.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def _ci(row, key, f="{:+.3f}"):
    return f"{_fmt(row.get(key), f)} [{_fmt(row.get(key + '_lo'), f)}, {_fmt(row.get(key + '_hi'), f)}]"


def _md_cell(text):
    """Escapes the pipes of |r| labels inside a Markdown table cell."""
    return str(text).replace("|", "\\|")


def _pred_cell(row):
    """'R² r (p_Holm) / R² |r| (p_Holm)' of (g) for the primary table."""
    if not any(_finite(row.get(f"p_pred_{t}")) for t in PRED_TARGETS):
        return "nicht berechenbar"
    return " / ".join(f"{_fmt(row.get(f'pred_{t}_r2'), '{:+.2f}')} ({_fmt(row.get(f'p_pred_{t}_holm'), '{:.2g}')})"
                      for t in PRED_TARGETS)


def render_markdown(df, model, entry, notes, figures, settings):
    """German summary: primary result, Table-17 comparison, trajectory of the verdicts, drivers, caveats."""
    L, H, D = geometry(model)
    k15 = int(config.TOP_RATIO * L * H)
    revs = sort_revisions(df["revision"].unique())
    final = PRIMARY_REVISION if PRIMARY_REVISION in revs else revs[-1]
    q_lo, q_hi = n1_angle_quantile([0.025, 0.975], D)
    out = [f"# Mean-Field-Signaturen der Werterichtungen: {model}", "",
           f"Erzeugt von `valueheads_pythia/analysis/meanfield.py` ({VERSION}) am {datetime.date.today().isoformat()} aus "
           f"`{entry['root']}` (Massenmittel-Richtungen `results/directions/`, Head-Tabellen `results/heads/`). "
           f"Geometrie {L} Layer × {H} Heads, Head-Dimension d = {D}, Top-15 % = {k15} Heads je Wert. "
           f"Checkpoints ({len(revs)}): {', '.join(revs)}. Bootstrap {settings['n_boot']}, Permutationen {settings['n_perm']}.",
           "",
           f"**Isotrope Null N1 (d = {D}):** Winkel 90° ± {np.degrees(1 / np.sqrt(D)):.1f}° (exaktes zentrales 95 %-Intervall "
           f"[{q_lo:.1f}°, {q_hi:.1f}°]); r = cos θ hat Mittel 0 und Varianz 1/d = {1 / D:.4f}; Anteil der Winkel > 100° unter "
           f"N1: {n1_frac_beyond(CONFLICT_DEG, D):.1%}. Belastbar ist nur der **Überschuss** über N1 und über step0 "
           "(untrainiertes Modell, gleiche Heads).", ""]
    prim = df[df["primary"]]
    out += [f"## Primärbefund (vorab festgelegt): geteilte Top-15 %-Heads bei {PRIMARY_REVISION}", ""]
    if prim.empty:
        out += [f"{PRIMARY_REVISION} (oder dessen Head-Tabellen) fehlt: kein Primärbefund; alles Folgende ist deskriptiv.", ""]
    else:
        out += ["Getestet werden je Wertepaar (a) die Lage gegen N1 und gegen step0 und (c) die Mittelungssteigung; Holm über "
                "diese 3 Paare × 3 Tests. Alle anderen Zahlen sind deskriptiv.", "",
                f"„mean-field-artig“ verlangt zusätzlich einen durchführbaren Layer-Test (d), ein Steigungs-KI von "
                f"höchstens {MAX_SLOPE_CI_WIDTH} Breite und eine berechenbare Vorhersagbarkeit des Rests (g); sonst lautet "
                "das Urteil „unterbestimmt“ (Kürzel m), auch wenn Lage, Steigung und Form zum Mean-Field-Muster passen. Ist "
                "der Rest vorhersagbar (g, deskriptive Holm-Familie), heißt die Zeile „vorhersagbar“ (V) und nie "
                "„mean-field-artig“.", "",
                "| Paar | n (Layer) | Winkel Mittel ± sd (Median) | r̄ [95 %-KI] | N1-Intervall von r̄ | Δr̄ zu step0 [KI] | "
                "Steigung [KI] | Holm-p: Lage N1 / Lage step0 / Steigung | Layer-Test (d) | Vorhersagbarkeit (g): R² r / |r| "
                "(p_Holm) | Urteil |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in prim.iterrows():
            icc = (f"ICC {_fmt(r.get('icc'), '{:.2f}')}, p_Holm {_fmt(r.get('p_icc_holm'), '{:.2g}')}"
                   if np.isfinite(float(r.get("p_icc", np.nan))) else "nicht testbar")
            out.append(f"| {r['pair']} | {int(r['n_heads'])} ({int(r['n_layers'])}) | {_fmt(r['angle_mean_deg'], '{:.1f}')}° ± "
                       f"{_fmt(r['angle_std_deg'], '{:.1f}')} ({_fmt(r['angle_median_deg'], '{:.1f}')}°) | {_ci(r, 'mean_r')} | "
                       f"[{_fmt(r['n1_mean_lo'], '{:+.3f}')}, {_fmt(r['n1_mean_hi'], '{:+.3f}')}] | {_ci(r, 'delta_r')} | "
                       f"{_ci(r, 'slope', '{:.2f}')} | {_fmt(r.get('p_loc_n1_primary_holm'), '{:.2g}')} / "
                       f"{_fmt(r.get('p_loc_n2_primary_holm'), '{:.2g}')} / {_fmt(r.get('p_slope_primary_holm'), '{:.2g}')} | "
                       f"{icc} | {_pred_cell(r)} | **{r['verdict']}** |")
        out.append("")
    fin = df[df["revision"] == final]
    out += [f"## Vergleich mit Tabelle 17 (Winkel in Grad, {final})", "",
            "Wie im Paper (App. K, Tab. 17): n, Mittel, Median, sd (Populations-sd) der Winkel; zum Vergleich alle Heads und "
            "das Verhältnis der Streuung zu N1.", "",
            "| Paar | Head-Satz | n | Mittel | Median | sd | sd(r)/sd_N1 [KI] | Anteil > 100° (N1) | Urteil |",
            "|---|---|---|---|---|---|---|---|---|"]
    for _, r in fin.iterrows():
        out.append(f"| {r['pair']} | {HEADSET_LABEL[r['headset']]} | {int(r['angle_n'])} | {_fmt(r['angle_mean_deg'], '{:.1f}')} | "
                   f"{_fmt(r['angle_median_deg'], '{:.1f}')} | {_fmt(r['angle_std_deg'], '{:.1f}')} | "
                   f"{_ci(r, 'sd_ratio', '{:.2f}')} | {_fmt(r['frac_gt100'], '{:.1%}')} ({_fmt(r['frac_gt100_n1'], '{:.1%}')}) | "
                   f"{r['verdict']} |")
    out += ["", f"## Vorhersagbarkeit des Rests (g) bei {final}", "",
            "Lässt sich r (Vorzeichen) bzw. |r| (Betrag) aus Eigenschaften der Heads **außerhalb der Stichprobe** "
            "vorhersagen? Ridge-Regression auf standardisierten Kovariaten (relative Tiefe und ihr Quadrat, log-Normen "
            "der beiden Massenmittel-Differenzen, gespeicherte Probe-Scores beider Werte, r bei step0), Strafe per innerer "
            f"CV, {PRED_FOLDS}-fache CV mit ganzen Layern als Testblöcken; R² der zurückgehaltenen Heads (≤ 0: nicht "
            f"vorhersagbar). Null: {settings['n_perm']} Permutationen, die ganze Layer auf Layer gleicher Head-Zahl "
            "verschieben und Heads innerhalb der Layer mischen. Das Mittelungsgesetz (c) kann unabhängiges Rauschen nicht von einem strukturierten Rest "
            "unterscheiden, der nicht an die Layer-Reihenfolge gebunden ist (beide mitteln sich wie 1/√n heraus); nur "
            "die Vorhersagbarkeit trennt sie.", "",
            "| Paar | Head-Satz | n | Kovariaten | R² r (Null-Median) | p_Holm r | R² \\|r\\| (Null-Median) | p_Holm \\|r\\| | "
            "stärkste Kovariate r / \\|r\\| |",
            "|---|---|---|---|---|---|---|---|---|"]
    def label(name):
        return COVARIATE_LABEL.get(_text(name), _text(name))

    for _, r in fin.iterrows():
        names = ", ".join(label(c) for c in _text(r.get("pred_covariates")).split()) or "–"
        tops = " / ".join(label(r.get(f"pred_{t}_top")) or "–" for t in PRED_TARGETS)
        out.append(f"| {r['pair']} | {HEADSET_LABEL[r['headset']]} | {int(r['n_heads'])} | {_md_cell(names)} | "
                   f"{_fmt(r.get('pred_r_r2'), '{:+.3f}')} ({_fmt(r.get('pred_r_r2_null'), '{:+.3f}')}) | "
                   f"{_fmt(r.get('p_pred_r_holm'), '{:.2g}')} | {_fmt(r.get('pred_absr_r2'), '{:+.3f}')} "
                   f"({_fmt(r.get('pred_absr_r2_null'), '{:+.3f}')}) | {_fmt(r.get('p_pred_absr_holm'), '{:.2g}')} | "
                   f"{_md_cell(tops)} |")
    out += ["", "## Verlauf über das Training", "",
            "Kürzel: M = mean-field-artig, S = systematisch (Lage), P = systematisch nur durch Persistenz (eingefrorene "
            "Residuen), K = kollektiv, V = vorhersagbar (Rest aus Head-Eigenschaften vorhersagbar, (g)), B = bistabil, "
            f"U = unterbestimmt (n < {MIN_HEADS}), m = unterbestimmt, obwohl Lage, Steigung und Form zum "
            f"Mean-Field-Muster passen (Layer-Häufung nicht testbar, Steigungs-KI breiter als {MAX_SLOPE_CI_WIDTH} oder "
            "Vorhersagbarkeit (g) nicht berechenbar), ? = uneindeutig; je Zelle "
            "`geteilte Top-15 % / alle Heads`, dahinter r̄ der geteilten Heads. "
            "step1000 / step2000 = IOI-Tief (Dahiya & Blondin 2026"
            + (f"; Boden dieses Modells bei step{IOI_FLOOR[model]}" if model in IOI_FLOOR else "") + ").", ""]
    pairs = list(dict.fromkeys(df["pair"]))
    out += ["| Checkpoint | " + " | ".join(pairs) + " |", "|---|" + "---|" * len(pairs)]
    for rev in revs:
        cells = []
        for pair in pairs:
            s = df[(df["revision"] == rev) & (df["pair"] == pair)].set_index("headset")
            sh = s.loc["shared_top15"] if "shared_top15" in s.index else None
            al = s.loc["all"] if "all" in s.index else None
            cells.append(f"{sh['verdict_code'] if sh is not None else '–'} / {al['verdict_code'] if al is not None else '–'}"
                         f" (r̄ {_fmt(sh['mean_r'], '{:+.3f}') if sh is not None else '–'})")
        mark = " (IOI)" if parse_step(rev) in IOI_MARKS else ""
        out.append(f"| {rev}{mark} | " + " | ".join(cells) + " |")
    out += ["", f"## Treiber der Urteile bei {final}", ""]
    for _, r in fin.iterrows():
        out.append(f"- **{r['pair']}, {HEADSET_LABEL[r['headset']]}** ({r['status']}): {r['verdict']} – {r['drivers']}")
    out += ["", "## Lesehilfe und Vorbehalte", "",
            "- *mean-field-artig* heißt hier nur: die Residuen r(h) verhalten sich wie unabhängiges Rauschen um 0, das sich "
            "über Heads herausmittelt (Steigung −1/2). Das ist eine statistische Signatur, **keine** Aussage über das "
            "Mean-Field- vs. Lazy/NTK-Regime der Trainings-Parametrisierung.",
            "- Das Mittelungsgesetz (c) sieht nur Korrelation, die an die Layer-Reihenfolge gebunden ist (Nachbar-Heads, "
            "Layer, Tiefentrend). Ein Anteil, den **alle** Heads teilen, verschiebt nur die Lage und wird allein von (a) "
            "erfasst; ist die Verschiebung klein, kann eine solche Zeile trotzdem *mean-field-artig* heißen.",
            "- Dünne Head-Sätze (meist ein Head je Layer, typisch für die geteilten Top-15 %-Heads) erlauben keinen "
            f"Layer-Test (d), und ihr Steigungs-KI ist breiter als {MAX_SLOPE_CI_WIDTH}: Sie heißen dann *unterbestimmt* "
            "(Kürzel m), nicht *mean-field-artig*.",
            "- Dass sich r über Heads wie 1/√n herausmittelt, unterscheidet unabhängiges Rauschen **nicht** von einem "
            "strukturierten Rest, der von Head-Eigenschaften abhängt (beide mitteln sich gleich heraus). Das leistet nur "
            "die Vorhersagbarkeit (g): R² der zurückgehaltenen Layer ≈ 0 bei Rauschen, deutlich > 0 bei Struktur. "
            "*vorhersagbar* (V) schließt *mean-field-artig* aus. Mit r(step0) als Kovariate gilt auch ein aus der "
            "Initialisierung eingefrorener Rest als vorhersagbar; die Spalte `pred_*_coefs` zeigt, welche Kovariate "
            "trägt. Eine Abhängigkeit des Betrags |r| von den Normen der Massenmittel-Differenzen kann teilweise "
            "Schätzrauschen sein (Heads mit kleiner Norm haben verrauschtere Richtungen).",
            "- *systematisch* allein durch Persistenz (Lage ≈ 0) bedeutet eingefrorene, aber über Heads ausmittelnde Residuen; "
            "frühe Checkpoints sind fast immer persistent (das Netz ändert sich kaum, und die Richtungen stammen aus denselben "
            "Texten, ihr Schätzrauschen ist also über Checkpoints korreliert).",
            "- Rauschen der Massenmittel-Schätzung zieht Winkel Richtung 90°: Überschuss-Streuung ist konservativ. Eine "
            "Label-Permutations-Null ist nicht möglich (Feature-Shards gelöscht); step0 ersetzt sie als empirische "
            "Referenz (Anisotropie des Aktivierungsraums, Daten-Artefakte wie gemeinsames HH-RLHF-Format).",
            "- p-Werte: Holm je Modell × Checkpoint über alle Paare, Head-Sätze und Tests (deskriptiv); nur der Primärbefund "
            "ist konfirmatorisch. Permutations-p haben die Auflösung 1/(P+1).",
            "- Die geteilten Top-15 %-Heads wechseln von Checkpoint zu Checkpoint (Top-Mengen je Checkpoint); Vergleiche mit "
            "step0 und Persistenz laufen auf denselben Heads.", "", "Details: `docs/MEANFIELD.md`.", ""]
    if figures:
        out += ["## Abbildungen", ""] + [f"- `{os.path.relpath(f, entry['root'])}`" for f in figures] + [""]
    if notes:
        out += ["## Hinweise beim Laden", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- run

def fingerprint(entry, settings):
    """sha256 over the content of every input file (relative path + bytes), the settings and this module's source (a code
    fix pulled into the notebook recomputes the outputs)."""
    h = hashlib.sha256(json.dumps({**settings, "version": VERSION}, sort_keys=True).encode())
    with open(os.path.abspath(__file__), "rb") as f:
        h.update(hashlib.sha256(f.read()).digest())
    for rev in sort_revisions(entry["revisions"]):
        for value in sorted(entry["revisions"][rev]):
            for kind in ("directions", "heads"):
                path = entry["revisions"][rev][value].get(kind)
                if path:
                    h.update(os.path.relpath(path, entry["root"]).encode())
                    with open(path, "rb") as f:
                        h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def output_paths(model, root):
    res, fig = os.path.join(root, "results", "meanfield"), os.path.join(root, "figures", "meanfield")
    return {"csv": os.path.join(res, f"{model}_meanfield.csv"), "md": os.path.join(res, f"{model}_meanfield.md"),
            "json": os.path.join(res, f"{model}_meanfield.json"), "figures": fig}


def run_model(model, entry, n_boot=N_BOOT, n_perm=N_PERM, seed=0, figures=True, force=False, say=print):
    """Analysis + outputs of one model; skipped (CSV re-read) when the input fingerprint and settings are unchanged."""
    settings = {"n_boot": int(n_boot), "n_perm": int(n_perm), "seed": int(seed), "figures": bool(figures)}
    out = output_paths(model, entry["root"])
    fp = fingerprint(entry, settings)
    if not force and all(os.path.exists(out[k]) for k in ("csv", "md", "json")):
        with open(out["json"]) as f:
            meta = json.load(f)
        if meta.get("fingerprint") == fp:
            say(f"[meanfield] {model}: unverändert (Fingerabdruck {fp[:12]}), übersprungen -> {out['csv']}")
            return pd.read_csv(out["csv"]), out, True
    t0 = time.time()
    df, notes = analyze_model(model, entry, n_boot, n_perm, seed, say=say)
    if df.empty:
        say(f"[meanfield] {model}: keine Checkpoints mit >= 2 Werten")
        return df, out, False
    os.makedirs(os.path.dirname(out["csv"]), exist_ok=True)
    figs = []
    if figures:
        for pair in dict.fromkeys(df["pair"]):
            figs.append(plot_trajectory(df, model, pair, os.path.join(out["figures"], f"{model}_{_pair_slug(pair)}_trajektorie.png")))
        final = df.loc[df["step"].idxmax(), "revision"] if PRIMARY_REVISION not in set(df["revision"]) else PRIMARY_REVISION
        data, _ = load_model_data(model, {"revisions": {final: entry["revisions"][final]}, "notes": []})
        cos = {f"{a} x {b}": head_cosines(data[final]["dirs"][a], data[final]["dirs"][b])
               for a, b in itertools.combinations(VALUES, 2) if {a, b} <= set(data[final]["dirs"])}
        figs.append(plot_histograms(df, model, final, cos, geometry(model)[2], os.path.join(out["figures"], f"{model}_{final}_r_hist.png")))
        figs.append(plot_averaging(df, model, final, os.path.join(out["figures"], f"{model}_{final}_mittelung.png")))
    df.to_csv(out["csv"], index=False)
    with open(out["md"], "w") as f:
        f.write(render_markdown(df, model, entry, notes, figs, settings))
    with open(out["json"], "w") as f:
        json.dump({"version": VERSION, "fingerprint": fp, "settings": settings, "created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "rows": int(len(df)), "notes": notes, "figures": figs, "root": entry["root"]}, f, indent=1)
    say(f"[meanfield] {model}: {len(df)} Zeilen, {len(figs)} Abbildungen, {time.time() - t0:.1f}s -> {out['csv']}")
    return df, out, False


def run_all(base, models=None, n_boot=N_BOOT, n_perm=N_PERM, seed=0, figures=True, force=False, found=None, say=print):
    """Every model found under base; with >= 2 models also {base}/meanfield_summary_{models}.csv (+ width figure)."""
    found = found if found is not None else discover(base, models)
    if not found:
        raise SystemExit(f"[meanfield] keine Richtungen unter {base} (gesucht: results/directions/*_{{{','.join(VALUES)}}}.npz)")
    results = {}
    for model, entry in found.items():
        df, out, skipped = run_model(model, entry, n_boot, n_perm, seed, figures, force, say)
        results[model] = {"df": df, "paths": out, "skipped": skipped, "root": entry["root"]}
    combined = width = None
    frames = [r["df"] for r in results.values() if not r["df"].empty]
    if len(frames) >= 2:
        tag = "_".join(m for m in results if not results[m]["df"].empty)
        combined = os.path.join(base, f"meanfield_summary_{tag}.csv")
        summary = pd.concat(frames, ignore_index=True)
        summary.to_csv(combined, index=False)
        if figures:
            width = plot_width(summary, os.path.join(base, f"meanfield_breite_{tag}.png"))
        say(f"[meanfield] kombiniert: {combined} ({len(summary)} Zeilen)")
    return {"models": results, "combined": combined, "width_figure": width}


def merge_summaries(paths):
    """Concatenates several meanfield_summary_*.csv / per-model CSVs (e.g. of account A and B), one copy per model."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # copy(): read_csv (pandas 3) keeps one block per column; consolidate so that later column inserts do not warn
    return df.drop_duplicates(subset=["model", "revision", "pair", "headset"], keep="last").reset_index(drop=True).copy()


# ----------------------------------------------------------------------------- synthetic tree (smoke / tests)

def make_synthetic_tree(base, models=("pythia-160m", "pythia-1b"), revisions=("step0", "step1000", "step143000"), seed=0):
    """Tiny fake Drive tree with known structure (no texts, no features). The first model lives in base itself (like
    pythia-160m in MyDrive/ValueHeadsPythia), the others in base/{model}. Directions: independent Gaussian 'init'
    vectors per value and head (step0 ~ N1), a small drift that grows with training, and a per-head component shared by
    harmless and helpful that grows to r ~ 0.35 at step143000 (systematic); honesty stays independent of both (isotropic).
    Head tables: AUC from a per-head relevance shared by all values (overlapping top-15 % sets)."""
    from ..directions import save_directions
    from ..probing import rank_heads, save_head_scores
    roots = []
    for i, model in enumerate(models):
        root = base if i == 0 else os.path.join(base, model)
        roots.append(root)
        p = config.paths(root)
        L, H, D = geometry(model)
        rng = np.random.default_rng([seed, i])
        init = {v: rng.standard_normal((L * H, D)) for v in VALUES}
        drift = {v: rng.standard_normal((L * H, D)) for v in VALUES}
        shared = rng.standard_normal((L * H, D))
        relevance = rng.random(L * H)
        for rev in revisions:
            t = np.log1p(parse_step(rev)) / np.log1p(143000)
            for v in VALUES:
                vec = init[v] + 0.3 * t * drift[v] + (0.75 * t * shared if v in ("harmless", "helpful") else 0.0)
                save_directions(config.directions_path(p, model, rev, v), vec.astype(np.float32), {v: np.ones(L * H, dtype=np.float32)},
                                {"model": model, "revision": rev, "value": v, "mode": "pythia", "synthetic": True,
                                 "stored_directions": "raw mass-mean (synthetic)", "unit_norm": False})
                auc = 0.5 + 0.3 * relevance + 0.03 * rng.random(L * H)
                df = pd.DataFrame({"layer": np.arange(L * H) // H, "head": np.arange(L * H) % H, "flat_idx": np.arange(L * H),
                                   "auc": auc, "acc": auc - 0.05, "bal_acc": auc - 0.03, "majority": 0.55, "pos_rate": 0.45,
                                   "n_train": 200, "n_val": 100})
                df.attrs.update(layers=L, heads=H, head_dim=D, score="auc", ranking="stable", feature_mode="head",
                                class_weight="balanced", model=model, revision=rev, value=v, mode="pythia", synthetic=True)
                rank_of = {int(f): k for k, f in enumerate(rank_heads(df, "auc"), start=1)}
                df["rank"] = df["flat_idx"].map(rank_of).astype(int)
                save_head_scores(df, config.heads_path(p, model, rev, v, "pythia"),
                                 meta={"model": model, "revision": rev, "value": v, "mode": "pythia", "feature_tag": None},
                                 ratio=config.TOP_RATIO)
        os.makedirs(p["results"], exist_ok=True)
        with open(p["manifest"], "w") as f:
            json.dump({"model": model, "synthetic": True, "revisions": list(revisions)}, f, indent=1)
    return roots


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mean-field signatures of value-direction angles (docs/MEANFIELD.md)")
    ap.add_argument("--base", default=None, help="Drive base (MyDrive/ValueHeadsPythia); searched up to 2 levels deep")
    ap.add_argument("--synthetic", default=None, help="create a synthetic tree in this directory and analyze it")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--perms", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    base = args.base
    if args.synthetic:
        make_synthetic_tree(args.synthetic)
        base = base or args.synthetic
    if not base:
        ap.error("--base or --synthetic required")
    found = discover(base, args.models)
    print(discovery_table(found).to_string(index=False))
    res = run_all(base, args.models, args.boot, args.perms, args.seed, not args.no_figures, args.force, found=found)
    for model, r in res["models"].items():
        df = r["df"]
        if df.empty:
            continue
        fin = df[df["revision"] == df.loc[df["step"].idxmax(), "revision"]]
        print(fin[["model", "revision", "pair", "headset", "n_heads", "angle_mean_deg", "angle_std_deg", "mean_r", "sd_ratio",
                   "slope", "icc", "pred_r_r2", "pred_absr_r2", "verdict"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
