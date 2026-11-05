"""Statistics and German verdict texts of the causal ablation test (docs/ABLATION_DESIGN.md, part B, 6.5-6.6).

  cluster_bootstrap        paired mean change with a vectorised cluster bootstrap (multinomial cluster counts)
  cluster_bootstrap_ratio  ratio of means with a joint cluster bootstrap (M5 self-repair index)
  paired_auc_delta         change of the mean probe AUC (M2 monitors), cluster bootstrap on item weights
  vs_random                empirical p of a named set against its layer/KV-matched random sets (one- and two-sided, z)
  primary_holm             Holm over the preregistered primary family (M1 under "mean" for C33 and O, per model)
  floor_check              clean pair accuracy / margin with CIs and the length baseline (F25)
  gate_decision            positive-control gate on the ICL battery (F23)
  tercile_labels, sign_change, PREREG / direction, verdict_lines (German markdown, 6.6)

Bootstrap semantics equal `stats.paired_delta_ci` (statistic sum(w * d) / sum(w); clusters resampled with replacement;
percentile CI; NaN items dropped together with their weight and cluster), but the B resamples are drawn as multinomial
cluster counts in chunks from one RandomState(seed), so the result does not depend on the chunk size. No model, no data.
"""
import math

import numpy as np

from . import overlap
from .ablation_sets import SET_ORDER
from .probing_stats import auc_matrix
from .stats import empirical_p

ALPHA = 0.05
DIRECTIONS = ("le", "ge", "two")


# ----------------------------------------------------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------------------------------------------------

def _cluster_index(clusters, n):
    """(n,) int cluster index (0 .. G-1) from cluster labels (ints, strings, or rows of a 2-D array such as
    (concept, seq)); None -> every item its own cluster."""
    if clusters is None:
        return np.arange(n)
    c = np.asarray(clusters)
    if len(c) != n:
        raise ValueError(f"clusters must have one entry per item ({len(c)} vs {n})")
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if c.ndim > 1:
        _, inv = np.unique(c.reshape(n, -1), axis=0, return_inverse=True)
    else:
        _, inv = np.unique(c, return_inverse=True)
    return np.asarray(inv).ravel().astype(np.int64)


def _count_chunks(G, B, seed, chunk):
    """Yields (m, G) float arrays of multinomial cluster counts; the stream is independent of `chunk`."""
    rs = np.random.RandomState(seed)
    p = np.full(G, 1.0 / G)
    for start in range(0, int(B), int(chunk)):
        m = min(int(chunk), int(B) - start)
        yield rs.multinomial(G, p, size=m).astype(np.float64)


def _rowdot(C, v):
    """C @ v row by row with numpy's pairwise sum (unlike BLAS, the result of a row does not depend on how many rows
    the chunk holds, so the bootstrap is exactly chunk-invariant)."""
    return (C * v).sum(axis=1)


def _percentiles(boots, ci):
    alpha = (1 - ci) / 2
    finite = boots[np.isfinite(boots)]
    if not len(finite):
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(finite, [100 * alpha, 100 * (1 - alpha)])
    se = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    return float(lo), float(hi), se


def cluster_bootstrap(values, clusters, *, B=10000, seed=0, weights=None, ci=0.95, chunk=500):
    """Mean paired change sum(w * d) / sum(w) with a cluster-bootstrap percentile CI.

    values:   per-item changes (condition - clean); NaN items are dropped with their weight and cluster
    clusters: per-item cluster labels (prompt cluster, (concept, seq), PPL window); None -> item bootstrap
    weights:  optional per-item weights (token-weighted NLL)
    Returns {"mean", "lo", "hi", "se", "n", "n_clusters", "B", "ci"} (lo / hi / se NaN with < 2 clusters or B = 0).
    """
    d = np.asarray(values, dtype=np.float64).ravel()
    w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if len(w) != len(d):
        raise ValueError("weights must have one entry per value")
    inv = _cluster_index(clusters, len(d))
    keep = np.isfinite(d) & np.isfinite(w)
    d, w, inv = d[keep], w[keep], inv[keep]
    out = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "se": float("nan"), "n": int(len(d)),
           "n_clusters": 0, "B": int(B), "ci": ci}
    if not len(d):
        return out
    out["mean"] = float(np.sum(w * d) / np.sum(w))
    _, inv = np.unique(inv, return_inverse=True)
    inv = np.asarray(inv).ravel()
    G = int(inv.max()) + 1
    out["n_clusters"] = G
    if G < 2 or B <= 0:
        return out
    sw = np.bincount(inv, weights=w, minlength=G)
    swd = np.bincount(inv, weights=w * d, minlength=G)
    boots = np.empty(int(B))
    pos = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for C in _count_chunks(G, B, seed, chunk):
            boots[pos:pos + len(C)] = _rowdot(C, swd) / _rowdot(C, sw)
            pos += len(C)
    out["lo"], out["hi"], out["se"] = _percentiles(boots, ci)
    return out


def cluster_bootstrap_ratio(num, den, clusters, *, B=10000, seed=0, ci=0.95, chunk=500):
    """Ratio of means mean(num) / mean(den) with a joint cluster bootstrap (same resamples for both).

    M5: num = TE_i, den = DE_i -> SRI = 1 - ratio. The ratio (and its CI) is NaN ("defined": False) when the CI of
    mean(den) contains 0. Items with a non-finite num or den are dropped.
    Returns {"ratio", "lo", "hi", "se", "defined", "num": {mean, lo, hi}, "den": {mean, lo, hi}, "n", "n_clusters",
    "B", "ci"}."""
    a = np.asarray(num, dtype=np.float64).ravel()
    b = np.asarray(den, dtype=np.float64).ravel()
    if len(a) != len(b):
        raise ValueError("num and den must have the same length")
    inv = _cluster_index(clusters, len(a))
    keep = np.isfinite(a) & np.isfinite(b)
    a, b, inv = a[keep], b[keep], inv[keep]
    nan = float("nan")
    out = {"ratio": nan, "lo": nan, "hi": nan, "se": nan, "defined": False, "num": {"mean": nan, "lo": nan, "hi": nan},
           "den": {"mean": nan, "lo": nan, "hi": nan}, "n": int(len(a)), "n_clusters": 0, "B": int(B), "ci": ci}
    if not len(a):
        return out
    out["num"]["mean"], out["den"]["mean"] = float(a.mean()), float(b.mean())
    _, inv = np.unique(inv, return_inverse=True)
    inv = np.asarray(inv).ravel()
    G = int(inv.max()) + 1
    out["n_clusters"] = G
    if G < 2 or B <= 0:
        return out
    cnt = np.bincount(inv, minlength=G).astype(np.float64)
    sa, sb = np.bincount(inv, weights=a, minlength=G), np.bincount(inv, weights=b, minlength=G)
    bn, bd, br = np.empty(int(B)), np.empty(int(B)), np.empty(int(B))
    pos = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for C in _count_chunks(G, B, seed, chunk):
            m = len(C)
            n_items, ca, cb = _rowdot(C, cnt), _rowdot(C, sa), _rowdot(C, sb)
            bn[pos:pos + m] = ca / n_items
            bd[pos:pos + m] = cb / n_items
            br[pos:pos + m] = ca / cb
            pos += m
    for key, boots in (("num", bn), ("den", bd)):
        out[key]["lo"], out[key]["hi"], _ = _percentiles(boots, ci)
    dlo, dhi = out["den"]["lo"], out["den"]["hi"]
    if not (np.isfinite(dlo) and np.isfinite(dhi)) or dlo <= 0 <= dhi:
        return out
    out["defined"] = True
    out["ratio"] = float(a.sum() / b.sum())
    out["lo"], out["hi"], out["se"] = _percentiles(br, ci)
    return out


def paired_auc_delta(scores_ref, scores_cond, labels, clusters, *, B=1000, seed=0, ci=0.95):
    """Change of the mean probe AUC between two score matrices on the same items (M2), cluster bootstrap.

    scores_ref / scores_cond: (K, N) or (N,) decision scores of K probes on the same N texts (clean / ablated);
    statistic = mean over rows of AUC(cond) - AUC(ref) (exact AUC with ties; probing_stats.auc_matrix). Rows that are
    entirely non-finite in either matrix (probes excluded because their head is ablated) are dropped, then items with
    a non-finite score in a remaining row. The bootstrap reweights items by multinomial cluster counts c (equal to
    resampling clusters). With item weights w_i = c_cluster(i), the weighted AUC of row k (probing_stats.WeightedAUC)
    is sum_{i pos, j neg} w_i w_j h^k_ij / (W_pos W_neg), h = [s_i > s_j] + 0.5 [s_i == s_j]; the denominator is the same
    for every row, so the bootstrap mean-over-rows delta is exactly c' D c / (W_pos(c) W_neg(c)) with the cluster-
    aggregated G x G matrix D = sum over cluster pairs of mean_k (h^k_cond - h^k_ref) (one O(K N_pos N_neg) pass, then
    one quadratic form per resample instead of 2 K weighted AUCs).
    Returns {"auc_ref", "auc_cond", "delta", "lo", "hi", "se", "B", "k", "n", "n_clusters", "ci"}."""
    R = np.asarray(scores_ref, dtype=np.float64)
    Cn = np.asarray(scores_cond, dtype=np.float64)
    R, Cn = (R[None] if R.ndim == 1 else R), (Cn[None] if Cn.ndim == 1 else Cn)
    if R.shape != Cn.shape:
        raise ValueError(f"score shapes differ: {R.shape} vs {Cn.shape}")
    y = np.asarray(labels).ravel()
    inv = _cluster_index(clusters, R.shape[1])
    rows = np.isfinite(R).any(axis=1) & np.isfinite(Cn).any(axis=1)
    R, Cn = R[rows], Cn[rows]
    cols = np.isfinite(R).all(axis=0) & np.isfinite(Cn).all(axis=0) if len(R) else np.zeros(R.shape[1], dtype=bool)
    R, Cn, y, inv = R[:, cols], Cn[:, cols], y[cols], inv[cols]
    nan = float("nan")
    out = {"auc_ref": nan, "auc_cond": nan, "delta": nan, "lo": nan, "hi": nan, "se": nan, "B": int(B), "k": int(len(R)),
           "n": int(R.shape[1]), "n_clusters": 0, "ci": ci}
    if not len(R) or not R.shape[1]:
        return out
    a_ref, a_cond = auc_matrix(R, y), auc_matrix(Cn, y)
    out["auc_ref"], out["auc_cond"] = float(np.mean(a_ref)), float(np.mean(a_cond))
    out["delta"] = float(np.mean(a_cond - a_ref))
    _, inv = np.unique(inv, return_inverse=True)
    inv = np.asarray(inv).ravel()
    G = int(inv.max()) + 1
    out["n_clusters"] = G
    pos, neg = y == 1, y != 1
    if G < 2 or B <= 0 or not pos.any() or not neg.any():
        return out
    D = _pair_delta_matrix(R, Cn, pos, neg, inv, G)
    n_pos = np.bincount(inv[pos], minlength=G).astype(np.float64)
    n_neg = np.bincount(inv[neg], minlength=G).astype(np.float64)
    boots = np.empty(int(B))
    at = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for C in _count_chunks(G, B, seed, 500):
            boots[at:at + len(C)] = _rowdot(C @ D, C) / (_rowdot(C, n_pos) * _rowdot(C, n_neg))
            at += len(C)
    out["lo"], out["hi"], out["se"] = _percentiles(boots, ci)
    return out


def _pair_delta_matrix(R, Cn, pos, neg, inv, G, chunk_rows=16):
    """(G, G): sum over positive items of cluster g and negative items of cluster g' of mean_k (h^k_cond - h^k_ref),
    h_ij = [s_i > s_j] + 0.5 [s_i == s_j] (the Mann-Whitney kernel of the AUC)."""
    def kernel_sum(S):
        sp, sn = S[:, pos], S[:, neg]
        M = np.zeros((sp.shape[1], sn.shape[1]))
        for a in range(0, len(S), chunk_rows):
            p, n = sp[a:a + chunk_rows, :, None], sn[a:a + chunk_rows, None, :]
            M += (p > n).sum(axis=0) + 0.5 * (p == n).sum(axis=0)
        return M
    M = (kernel_sum(Cn) - kernel_sum(R)) / len(R)                              # (N_pos, N_neg)
    P = np.zeros((G, int(pos.sum())))
    P[inv[pos], np.arange(P.shape[1])] = 1.0
    Q = np.zeros((G, int(neg.sum())))
    Q[inv[neg], np.arange(Q.shape[1])] = 1.0
    return P @ M @ Q.T


# ----------------------------------------------------------------------------------------------------------------
# tests against random sets, Holm
# ----------------------------------------------------------------------------------------------------------------

def vs_random(observed, random_values, direction):
    """Named set vs its random-set distribution.

    p_le / p_ge = stats.empirical_p (+1 correction, NaN randoms dropped; NaN with R = 0); p_one = the p of the
    preregistered direction ("le" | "ge"; "two" -> NaN); p_two = overlap.two_sided(p_le, p_ge) = min(1, 2 min);
    z = (observed - mean) / std (ddof=1; NaN with R < 2 or std 0); n_le / n_ge = #{random <= / >= observed}.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    r = np.asarray([v for v in np.atleast_1d(np.asarray(random_values, dtype=np.float64)) if np.isfinite(v)])
    R = len(r)
    obs = float("nan") if observed is None else float(observed)
    p_le, p_ge = empirical_p(r, obs, "le"), empirical_p(r, obs, "ge")
    nan = float("nan")
    std = float(np.std(r, ddof=1)) if R > 1 else nan
    mean = float(r.mean()) if R else nan
    z = (obs - mean) / std if R > 1 and std > 0 and np.isfinite(obs) else nan
    ok = R > 0 and np.isfinite(obs)
    return {"observed": obs, "p_le": p_le, "p_ge": p_ge,
            "p_one": {"le": p_le, "ge": p_ge}.get(direction, nan), "p_two": overlap.two_sided(p_le, p_ge),
            "z": float(z), "rand_mean": mean, "rand_std": std, "rand_min": float(r.min()) if R else nan,
            "rand_max": float(r.max()) if R else nan, "rand_q025": float(np.quantile(r, 0.025)) if R else nan,
            "rand_q975": float(np.quantile(r, 0.975)) if R else nan, "R": int(R),
            "n_le": int((r <= obs).sum()) if ok else 0, "n_ge": int((r >= obs).sum()) if ok else 0,
            "direction": direction}


def primary_holm(tests, alpha=ALPHA):
    """Holm (overlap.holm) over the primary family. tests: [{"set", "p_one", "observed", ...}] -> copies with
    "p_holm" and "significant" (= p_holm < alpha; a NaN p is not part of the family and never significant)."""
    p = [float("nan") if t.get("p_one") is None else float(t["p_one"]) for t in tests]
    adj = overlap.holm(p) if len(p) else np.zeros(0)
    out = []
    for t, ph in zip(tests, adj):
        row = dict(t)
        row["p_holm"] = float(ph)
        row["significant"] = bool(np.isfinite(ph) and ph < alpha)
        row["alpha"] = alpha
        out.append(row)
    return out


# ----------------------------------------------------------------------------------------------------------------
# floor and gate
# ----------------------------------------------------------------------------------------------------------------

def _f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def floor_check(win, margin, len_hi_gt_lo, clusters, *, B=10000, seed=0):
    """Floor check of the clean model (F25).

    win: per-item [margin_mean > 0] (ties are losses); margin: per-item clean margin_mean; len_hi_gt_lo: per-item
    [n_pos > n_neg] (length baseline = accuracy of "the longer response is chosen"); clusters: prompt clusters.
    floor = the 95 % cluster CI of the accuracy contains 0.5 (or is undefined); inverted = its upper end < 0.5.
    Returns {"acc", "acc_lo", "acc_hi", "margin", "margin_lo", "margin_hi", "len_baseline", "len_baseline_lo",
    "len_baseline_hi", "n", "n_clusters", "B", "floor", "inverted", "text_de"}."""
    acc = cluster_bootstrap(np.asarray(win, dtype=np.float64), clusters, B=B, seed=seed)
    mar = cluster_bootstrap(margin, clusters, B=B, seed=seed)
    lb = cluster_bootstrap(np.asarray(len_hi_gt_lo, dtype=np.float64), clusters, B=B, seed=seed)
    lo, hi = acc["lo"], acc["hi"]
    defined = np.isfinite(lo) and np.isfinite(hi)
    inverted = bool(defined and hi < 0.5)
    floor = bool(not defined or lo <= 0.5 <= hi)
    out = {"acc": acc["mean"], "acc_lo": lo, "acc_hi": hi, "margin": mar["mean"], "margin_lo": mar["lo"],
           "margin_hi": mar["hi"], "len_baseline": lb["mean"], "len_baseline_lo": lb["lo"], "len_baseline_hi": lb["hi"],
           "n": acc["n"], "n_clusters": acc["n_clusters"], "B": int(B), "floor": floor, "inverted": inverted}
    acc_txt = f"{_f(acc['mean'])} [{_f(lo)}, {_f(hi)}]"
    tail = (f" Saubere Marge {_f(mar['mean'], sign=True)} [{_f(mar['lo'], sign=True)}, {_f(mar['hi'], sign=True)}] Nats; "
            f"Längen-Baseline (die längere Antwort gewinnt) {_f(lb['mean'])} [{_f(lb['lo'])}, {_f(lb['hi'])}]; "
            f"n = {acc['n']} Paare in {acc['n_clusters']} Prompt-Clustern.")
    if floor:
        text = (f"**Boden:** Die saubere Paargenauigkeit ({acc_txt}) ist nicht von 0,5 unterscheidbar. M1 kann eine "
                "verringerte Abhängigkeit nicht zeigen; interpretierbar sind nur M2 und M3.")
    elif inverted:
        text = (f"**Warnung:** Die saubere Präferenz ist invertiert (Paargenauigkeit {acc_txt} < 0,5): das Modell "
                "bevorzugt im Mittel die abgelehnte Antwort. M1-Änderungen beziehen sich auf diese invertierte Präferenz.")
    else:
        text = f"Kein Boden: Die saubere Paargenauigkeit ({acc_txt}) liegt über 0,5."
    out["text_de"] = text + tail
    return out


def gate_decision(delta_dret, clusters, random_means, *, calibration=None, B=10000, seed=0, advisory=False):
    """Positive-control gate (F23) from the CR20 knock on the ICL battery.

    delta_dret:   per-item change of the ICL margin (sym + rare items), CR20 knocked jointly at the query
    clusters:     (concept, seq) cluster per item
    random_means: mean delta of every random layer/KV-matched 20-set (R_SEC)
    calibration:  icl_battery.calibration_check(...) dict (or a bool); None = skipped (smoke) and not blocking
    PASS <=> (a) calibration ok (when given) and (b) the 95 % cluster CI of the mean delta lies entirely below 0 and
    (c) the mean delta is below every random set (R >= 1). advisory=True (smoke) only records the outcome:
    "proceed" = passed or advisory.
    Returns {"passed", "advisory", "proceed", "criteria": {"calibration", "ci_below_zero", "below_all_random"},
    "delta", "lo", "hi", "n", "n_clusters", "B", "random": vs_random(...), "calibration", "reason_de"}."""
    bs = cluster_bootstrap(delta_dret, clusters, B=B, seed=seed)
    rnd = vs_random(bs["mean"], random_means, "le")
    if calibration is None:
        calib_ok = None
    elif isinstance(calibration, dict):
        calib_ok = bool(calibration.get("ok"))
    else:
        calib_ok = bool(calibration)
    ci_below = bool(np.isfinite(bs["hi"]) and bs["hi"] < 0)
    below_all = bool(rnd["R"] > 0 and np.isfinite(bs["mean"]) and bs["mean"] < rnd["rand_min"])
    passed = bool(ci_below and below_all and calib_ok is not False)
    reasons = []
    if calib_ok is False:
        diffs = calibration.get("diffs", {}) if isinstance(calibration, dict) else {}
        tol = calibration.get("tol", 0.15) if isinstance(calibration, dict) else 0.15
        bad = ", ".join(f"{k} {v:+.3f}" for k, v in diffs.items()
                        if isinstance(v, (int, float)) and np.isfinite(v) and abs(v) > tol)
        reasons.append(f"Kalibrierung (Seed 1) weicht um mehr als {tol} Nats von D188_summary ab"
                       + (f" ({bad})" if bad else ""))
    if not ci_below:
        reasons.append(f"95-%-CI von Δdret [{_f(bs['lo'], sign=True)}, {_f(bs['hi'], sign=True)}] liegt nicht "
                       "vollständig unter 0")
    if not below_all:
        if rnd["R"] == 0:
            reasons.append("keine Zufallsmengen zum Vergleich")
        else:
            reasons.append(f"Δdret {_f(bs['mean'], sign=True)} liegt nicht unter allen {rnd['R']} Zufallsmengen "
                           f"(Minimum {_f(rnd['rand_min'], sign=True)}, {rnd['n_le']} gleich oder darunter)")
    if passed:
        reason = (f"bestanden: CR20 Δdret {_f(bs['mean'], sign=True)} [{_f(bs['lo'], sign=True)}, "
                  f"{_f(bs['hi'], sign=True)}] Nats < 0 und unter allen {rnd['R']} layer- und KV-gematchten "
                  f"Zufallsmengen (Mittel {_f(rnd['rand_mean'], sign=True)}, p einseitig {_fmt_p(rnd['p_le'])})"
                  + ("; Kalibrierung übersprungen" if calib_ok is None else "; Kalibrierung ok"))
    else:
        reason = "; ".join(reasons)
    if advisory:
        reason += " (beratend: Smoke-Lauf, kein Abbruch)"
    return {"passed": passed, "advisory": bool(advisory), "proceed": bool(passed or advisory),
            "criteria": {"calibration": calib_ok, "ci_below_zero": ci_below, "below_all_random": below_all},
            "delta": bs["mean"], "lo": bs["lo"], "hi": bs["hi"], "n": bs["n"], "n_clusters": bs["n_clusters"],
            "B": int(B), "random": rnd, "calibration": calibration if isinstance(calibration, dict) else calib_ok,
            "reason_de": reason}


# ----------------------------------------------------------------------------------------------------------------
# covariates, trajectory, preregistration
# ----------------------------------------------------------------------------------------------------------------

def tercile_labels(x):
    """(labels, edges): edges = np.quantile(finite x, [1/3, 2/3]); labels = np.searchsorted(edges, x, 'right') in
    {0, 1, 2} (a value equal to an edge goes up; heavy ties can leave a tercile empty); NaN -> -1."""
    x = np.asarray(x, dtype=np.float64).ravel()
    fin = np.isfinite(x)
    labels = np.full(len(x), -1, dtype=np.int64)
    if not fin.any():
        return labels, np.full(2, np.nan)
    edges = np.quantile(x[fin], [1 / 3, 2 / 3])
    labels[fin] = np.searchsorted(edges, x[fin], side="right")
    return labels, edges


def sign_change(a, b):
    """Base -> instruct comparison of two effects {"mean", "lo", "hi"}: "Vorzeichenwechsel" if both CIs exclude 0 with
    opposite signs, "gleiche Richtung" if both exclude 0 with the same sign, else "unentschieden"."""
    def side(e):
        if e is None:
            return 0
        lo, hi = e.get("lo", float("nan")), e.get("hi", float("nan"))
        if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
            return 0
        return 1 if lo > 0 else -1 if hi < 0 else 0
    sa, sb = side(a), side(b)
    if sa == 0 or sb == 0:
        return "unentschieden"
    return "gleiche Richtung" if sa == sb else "Vorzeichenwechsel"


def _prereg():
    """F29: preregistered direction per (metric, set)."""
    m3 = {"CR20": "le", "CensusReaders": "le", "C33": "le", "CB24": "ge", "O": "two", "P_band": "two",
          "TopK_all": "two", "WholeBand": "two"}
    out = {}
    for s in SET_ORDER:
        out[("M1", s)] = "le"
        out[("M2", s)] = "le"
        out[("M4", s)] = "ge"
        out[("M3", s)] = m3.get(s, "two")
        if s in ("C33_readers", "C33_brakes"):
            for m in ("M1", "M2", "M3", "M4"):
                out[(m, s)] = "two"
    return out


PREREG = _prereg()


def direction(metric, set_name):
    """Preregistered direction of (metric, set); fixed-identity sets ("O_base") follow their base set; unknown -> "two"."""
    base = set_name[:-5] if str(set_name).endswith("_base") else set_name
    return PREREG.get((metric, base), "two")


# ----------------------------------------------------------------------------------------------------------------
# verdict texts (6.6)
# ----------------------------------------------------------------------------------------------------------------

TRANSFER_MARK = "(Kopf-Identitäten aus dem Base-Modell übertragen)"
TRANSFER_MARK_O = "(C33-Identitäten übertragen, Top-k aus der eigenen Probe-Tabelle)"


def transfer_mark(set_name):
    """Marker of a primary row on the instruct model: C33 carries the Base identities, O = C33 ∩ the instruct model's own
    top-k carries only its C33 part from Base (its frozen identity is the instruct model, F11 / F14)."""
    return TRANSFER_MARK_O if str(set_name) == "O" else TRANSFER_MARK


SMOKE_LINE = "SMOKE: nur Funktionsprüfung, kein Befund."


def stop_text(reason):
    return (f"**STOPP:** Die Positivkontrolle (CR20, ICL-Marge) ist nicht bestanden ({reason}). Ohne bestandene "
            "Positivkontrolle wird kein Helpfulness-Befund berichtet.")


def verdict_lines(primary, floor, gate, *, model, transferred=False, smoke=False):
    """German markdown lines of "Ergebnis in einem Satz" (6.6).

    primary: primary_holm(...) rows, each {"set", "observed" (mean Δ margin_mean), "lo", "hi", "p_one", "p_holm",
             "significant"} plus optionally "R" and "n_le" (from vs_random) for "stärker als k von R";
    floor:   floor_check(...) dict (or None); gate: gate_decision(...) dict (None = missing -> STOPP).
    Gate failed (not advisory) -> only the STOPP text. Floor -> every primary row "nicht interpretierbar (Boden)".
    """
    head = [SMOKE_LINE] if smoke else []
    if gate is None or (not gate.get("passed") and not gate.get("advisory")):
        reason = "Gate-Ergebnis fehlt" if gate is None else gate.get("reason_de") or "Kriterien nicht erfüllt"
        return head + [stop_text(reason)]
    lines = list(head)
    if not gate.get("passed"):
        lines.append(f"Hinweis: Die Positivkontrolle ist nicht bestanden ({gate.get('reason_de', '')}); im beratenden "
                     "Modus wird trotzdem weitergerechnet.")
    is_floor = bool(floor and floor.get("floor"))
    if is_floor:
        lines.append(f"**Boden:** Die saubere Paargenauigkeit ({_f(floor.get('acc'))} [{_f(floor.get('acc_lo'))}, "
                     f"{_f(floor.get('acc_hi'))}]) ist nicht von 0,5 unterscheidbar. M1 kann eine verringerte Abhängigkeit "
                     "nicht zeigen; interpretierbar sind nur M2 und M3.")
    elif floor and floor.get("inverted"):
        lines.append(f"**Warnung:** Die saubere Präferenz ist invertiert (Paargenauigkeit {_f(floor.get('acc'))} "
                     f"[{_f(floor.get('acc_lo'))}, {_f(floor.get('acc_hi'))}] < 0,5).")
    o_row = None
    for row in primary:
        name = row.get("set")
        mark = f" {transfer_mark(name)}" if transferred else ""
        if name == "O":
            o_row = row
        d, lo, hi = row.get("observed"), row.get("lo"), row.get("hi")
        ci = f"{_f(d, sign=True)} [{_f(lo, sign=True)}, {_f(hi, sign=True)}]"
        p = _fmt_p(row.get("p_holm"))
        if is_floor:
            lines.append(f"**{name}: nicht interpretierbar (Boden)** (Δ {ci}, Holm-p {p}){mark}.")
        elif row.get("significant"):
            R = row.get("R")
            n_le = row.get("n_le")
            if R is not None and n_le is not None:
                beaten = f"{int(R) - int(n_le)} von {int(R)}"
            elif R is not None and row.get("p_one") is not None and np.isfinite(row["p_one"]):
                beaten = f"{int(R) - int(round(row['p_one'] * (int(R) + 1) - 1))} von {int(R)}"
            else:
                beaten = "den"
            if d is not None and np.isfinite(d) and d < 0:
                what = f"senkt die Helpful-Marge um {_f(-d)} Nats (Δ {ci})"
            else:
                what = (f"verschiebt die Helpful-Marge um Δ {ci} Nats (kein absolutes Absinken; nur niedriger als die "
                        "Zufallsmengen)")
            lines.append(f"**{name}: Abhängigkeit nachgewiesen** – mean-Ablation {what}, stärker als {beaten} layer- und "
                         f"KV-gematchten Zufallsmengen (Holm-p {p}){mark}.")
        else:
            lines.append(f"**{name}: keine Abhängigkeit nachgewiesen** (Δ {ci}, Holm-p {p}){mark}.")
    if o_row is not None:
        prefix = f"`{model}`: "
        if is_floor:
            lines.append(prefix + "Helpful ist in den Überlappungs-Heads dekodierbar; über eine Abhängigkeit der Präferenz "
                         "sagt M1 wegen des Bodens nichts.")
        elif o_row.get("significant"):
            lines.append(prefix + "Helpful ist in den Überlappungs-Heads dekodierbar **und** das Modell stützt seine "
                         "Präferenz kausal darauf (Expression + Reliance).")
        else:
            lines.append(prefix + "Helpful ist in den Überlappungs-Heads dekodierbar, eine Abhängigkeit der Präferenz ist "
                         "nicht nachgewiesen (Expression ohne nachgewiesene Reliance).")
    return lines
