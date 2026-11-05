"""Sequence builders and metric computations of the causal ablation test (docs/ABLATION_DESIGN.md, part D, 8.2).

Everything here sits on top of part A (`ablation.py`: `Seq`, `run_sequences`, collectors) and turns HH items, WikiText
windows and probe weights into the per-item quantities of the metrics:

  M1  behaviour: per item LL / mean-token LL of chosen (pos), rejected (neg) and an unrelated response (unrel, F17);
      margin_mean, margin_sum, win_mean, win_sum, threeway, len_hi_gt_lo (`m1_values`)
  M2  internal readout monitors (F15): fixed logistic head probes at the readout layers and on the post-final-norm
      residual, an MLP on the residual (L3), all trained once on clean means_probe activations; refitted (L2)
      out-of-fold logistic probes on the ablated EVAL activations (`train_probes`, `probe_scores`, `m2_summary`,
      `refit_m2`)
  M4  WikiText windows as `Seq`s (`ppl_windows`, `ppl_sequences`), token-weighted NLL per window
  M5  direct effect per item from `ablation.direct_effect` (`dla_values`)
  M7  covariates: prompt reuse of chosen / rejected and the length difference (`reuse_covariates`)

Variants (F18): hh3 = (pos, neg, unrel) per item, hh2 = (pos, neg), hhshuf = (spos, sneg): chosen and rejected after
the context of another dialogue (F17). The sequences of a variant are ordered in role blocks (all pos, then all neg,
...), item k of every block is EVAL item k. No text is ever printed or written: only token ids live in memory, and the
outputs hold ids, hashes and numbers.
"""
import hashlib
import io
import json
import os
import unicodedata
import warnings
import zipfile

import numpy as np

from . import overlap
from .ablation import Seq
from .eval_likelihood import encode_pair
from .probing_stats import auc_matrix

HH_VARIANTS = {"hh3": ("pos", "neg", "unrel"), "hh2": ("pos", "neg"), "hhshuf": ("spos", "sneg")}
MAX_LENGTH = 4096          # F1: no truncation (longest HH helpful sequence ~1.3k tokens); a cut raises
PROBE_SEED = 42


def one(x):
    """The single continuation of an eval_pairs field (one-element list) or a plain string."""
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"expected exactly one continuation, got {len(x)}")
        return x[0]
    return x


# ----------------------------------------------------------------------------------------------------------------
# HH sequences
# ----------------------------------------------------------------------------------------------------------------

def derangement(clusters, seed, max_tries=1000):
    """F17: permutation pi of the active EVAL items with cluster(pi(i)) != cluster(i) for every i, drawn as
    overlap._rng(seed, "ablation-derangement-v1", n).permutation(n), repeated until valid (at most `max_tries`)."""
    c = np.asarray(clusters)
    n = len(c)
    if n < 2:
        raise ValueError("a derangement needs at least two items")
    rng = overlap._rng(seed, "ablation-derangement-v1", n)
    for _ in range(int(max_tries)):
        perm = rng.permutation(n)
        if bool(np.all(c[perm] != c)):
            return perm
    raise ValueError(f"no cluster-disjoint derangement of {n} items in {max_tries} tries (too few prompt clusters)")


class Encoder:
    """Cached `eval_likelihood.encode_pair(tok, context, continuation, max_length=4096)` (F1: raw text, no BOS, joint
    tokenisation, boundary = context length). A truncated sequence raises (F1 asserts cut == 0)."""

    def __init__(self, tok, max_length=MAX_LENGTH):
        self.tok = tok
        self.max_length = int(max_length)
        self.cache = {}

    def __call__(self, context, continuation):
        key = (context, continuation)
        hit = self.cache.get(key)
        if hit is None:
            ids, boundary, _, cut = encode_pair(self.tok, context, continuation, max_length=self.max_length)
            if cut:
                raise ValueError(f"sequence longer than {self.max_length} tokens (cut {cut}); F1 forbids truncation")
            hit = (tuple(int(i) for i in ids), int(boundary))
            self.cache[key] = hit
        return hit


def hh_sequences(enc, items, variant, partner=None):
    """Seqs of `variant` (hh3 | hh2 | hhshuf) in role blocks; item id = the HH group id.

    hh3 unrel = context_i + chosen of partner[i]; hhshuf spos / sneg = context of partner[i] + chosen_i / rejected_i."""
    roles = HH_VARIANTS[variant]
    if any(r in ("unrel", "spos", "sneg") for r in roles) and partner is None:
        raise ValueError(f"variant {variant} needs the derangement (partner)")
    out = []
    for role in roles:
        for k, it in enumerate(items):
            if role == "pos":
                ctx, cont = it["context"], one(it["pos"])
            elif role == "neg":
                ctx, cont = it["context"], one(it["neg"])
            elif role == "unrel":
                ctx, cont = it["context"], one(items[int(partner[k])]["pos"])
            elif role == "spos":
                ctx, cont = items[int(partner[k])]["context"], one(it["pos"])
            else:
                ctx, cont = items[int(partner[k])]["context"], one(it["neg"])
            ids, b = enc(ctx, cont)
            out.append(Seq(ids, b, int(it["group"]), role))
    return out


def seqs_sha(seqs):
    """sha256 (16 hex) over ids, boundary, item and role of every sequence, in order."""
    h = hashlib.sha256()
    for s in seqs:
        h.update(np.asarray(s.ids, dtype=np.int64).tobytes())
        h.update(f"|{s.boundary}|{s.item}|{s.role};".encode())
    return h.hexdigest()[:16]


def ids_sha(values):
    return hashlib.sha256(json.dumps([int(v) for v in values]).encode()).hexdigest()[:16]


def hh_item_arrays(seqs, sums, counts, variant, n_items):
    """{"id", "{role}_sum", "{role}_n"} per item from a run over hh_sequences(...) (role-block order checked)."""
    roles = HH_VARIANTS[variant]
    if len(seqs) != len(roles) * n_items:
        raise ValueError(f"{len(seqs)} sequences for {n_items} items x {len(roles)} roles")
    out = {"id": [int(s.item) for s in seqs[:n_items]]}
    for j, role in enumerate(roles):
        block = seqs[j * n_items:(j + 1) * n_items]
        if any(s.role != role for s in block) or [s.item for s in block] != out["id"]:
            raise ValueError(f"role block {role} is out of order")
        out[f"{role}_sum"] = [float(x) for x in np.asarray(sums)[j * n_items:(j + 1) * n_items]]
        out[f"{role}_n"] = [int(x) for x in np.asarray(counts)[j * n_items:(j + 1) * n_items]]
    return out


def m1_values(items):
    """Per-item M1 quantities (8.2) from an items dict of hh_item_arrays: margin_mean, margin_sum, win_mean, win_sum
    (ties are losses), len_hi_gt_lo, threeway (hh3: LLbar(pos) - max(LLbar(neg), LLbar(unrel)))."""
    a, b = ("pos", "neg") if "pos_sum" in items else ("spos", "sneg")
    ps, pn = np.asarray(items[f"{a}_sum"], float), np.asarray(items[f"{a}_n"], float)
    ns, nn = np.asarray(items[f"{b}_sum"], float), np.asarray(items[f"{b}_n"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp, ln = ps / pn, ns / nn
        out = {"margin_mean": lp - ln, "margin_sum": ps - ns, "win_mean": (lp - ln > 0).astype(float),
               "win_sum": (ps - ns > 0).astype(float), "len_hi_gt_lo": (pn > nn).astype(float)}
        if "unrel_sum" in items:
            lu = np.asarray(items["unrel_sum"], float) / np.asarray(items["unrel_n"], float)
            out["threeway"] = lp - np.maximum(ln, lu)
    return out


def m1_summary(items):
    v = m1_values(items)
    out = {"margin_mean": float(np.mean(v["margin_mean"])), "margin_sum": float(np.mean(v["margin_sum"])),
           "acc_mean": float(np.mean(v["win_mean"])), "acc_sum": float(np.mean(v["win_sum"])),
           "len_baseline": float(np.mean(v["len_hi_gt_lo"])), "n_items": int(len(v["margin_mean"]))}
    if "threeway" in v:
        out["threeway_mean"] = float(np.mean(v["threeway"]))
    return out


# ----------------------------------------------------------------------------------------------------------------
# M7 covariates
# ----------------------------------------------------------------------------------------------------------------

def _content_token(tok, tid, cache):
    hit = cache.get(tid)
    if hit is None:
        s = tok.decode([int(tid)]).strip()
        hit = bool(s) and not all(unicodedata.category(c)[0] in "PS" for c in s)
        cache[tid] = hit
    return hit


def token_set(tok, text, cache):
    """U(text): token ids (no special tokens) whose decoded string is neither whitespace-only nor punctuation-only
    (punctuation = Unicode categories P* and S*)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return {int(i) for i in ids if _content_token(tok, i, cache)}


def reuse_covariates(tok, items):
    """M7 per item: reuse(pos), reuse(neg) with reuse(resp) = |U(resp) & U(q)| / max(1, |U(resp)|), q = the Human turn
    (data.first_qa(context + " ")[0], tokenised as " " + q). Returns {"reuse_pos", "reuse_neg"} lists."""
    from .data import first_qa
    cache = {}
    rp, rn = [], []
    for it in items:
        q = first_qa(it["context"] + " ")[0] or ""
        uq = token_set(tok, " " + q, cache)
        for cont, out in ((one(it["pos"]), rp), (one(it["neg"]), rn)):
            u = token_set(tok, cont, cache)
            out.append(len(u & uq) / max(1, len(u)))
    return {"reuse_pos": rp, "reuse_neg": rn}


# ----------------------------------------------------------------------------------------------------------------
# M4 WikiText windows
# ----------------------------------------------------------------------------------------------------------------

def ppl_windows(n_tokens, window=1024, stride=512, min_context=None):
    """(begin, end, target_start) windows exactly as eval_likelihood.PPLMetric (default min_context = window - stride)."""
    if not 0 < stride <= window:
        raise ValueError(f"need 0 < stride <= window, got stride={stride}, window={window}")
    min_context = int(window - stride if min_context is None else min_context)
    out, prev_end = [], 0
    for begin in range(0, int(n_tokens), stride):
        end = min(begin + window, int(n_tokens))
        target_start = max(prev_end, begin + 1, begin + min_context)
        if target_start < end:
            out.append((begin, end, target_start))
        prev_end = end
        if end == n_tokens:
            break
    if not out:
        raise ValueError(f"no token to score: {n_tokens} tokens, window {window}, stride {stride}")
    return out


def ppl_sequences(ids, windows):
    """One Seq per window (role "ppl", item = window index, boundary = target_start - begin)."""
    ids = [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    return [Seq(tuple(ids[b:e]), t - b, k, "ppl") for k, (b, e, t) in enumerate(windows)]


def ppl_summary(sums, counts):
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    nll = float(-sums.sum() / counts.sum()) if counts.sum() else float("nan")
    return {"nll": nll, "ppl": float(np.exp(nll)), "n_tokens_scored": int(counts.sum()), "n_windows": int(len(sums))}


# ----------------------------------------------------------------------------------------------------------------
# M2 probes
# ----------------------------------------------------------------------------------------------------------------

def save_npz_stable(path, arrays, meta=None):
    """npz as a stored zip with fixed timestamps (identical content -> identical bytes), atomic; returns sha256."""
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {k: (lambda a: a if a.flags.c_contiguous else np.ascontiguousarray(a))(np.asarray(v)) for k, v in arrays.items()}
    if meta is not None:
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True, default=str))
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name in sorted(arrays):
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                buf = io.BytesIO()
                np.lib.format.write_array(buf, arrays[name], allow_pickle=False)
                zf.writestr(info, buf.getvalue())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return overlap.file_sha256(path)


def load_npz(path):
    """({name: array}, meta dict) of a save_npz_stable file (no pickle)."""
    with np.load(str(path), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"][()])) if "meta" in z.files else {}
    return arrays, meta


def train_probes(head_x, resid_x, labels, layers, n_heads, head_dim, seed=PROBE_SEED):
    """M2 probes (F15), trained on clean means_probe activations at ln - 1.

    head_x: (N, len(layers), H*D) o_proj inputs; resid_x: (N, hidden) post-final-norm residual; labels: (N,) 1 = chosen.
    Head probes and the residual probe: LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
    random_state=42), no scaling (probing._fit_one, mode pythia). L3: StandardScaler + MLPClassifier((64,), relu, adam,
    alpha 1e-2, lr 1e-3, max_iter 500, random_state 42). Returns the weights as arrays (no pickle)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("M2 probe training needs both labels")
    head_x = np.asarray(head_x, dtype=np.float32)
    N, nL = head_x.shape[0], len(layers)
    H, D = int(n_heads), int(head_dim)
    coef = np.zeros((nL, H, D), np.float32)
    icpt = np.zeros((nL, H), np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for li in range(nL):
            X = head_x[:, li].reshape(N, H, D)
            for h in range(H):
                clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed).fit(X[:, h], y)
                coef[li, h], icpt[li, h] = clf.coef_[0], clf.intercept_[0]
        R = np.asarray(resid_x, dtype=np.float32)
        rclf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed).fit(R, y)
        scaler = StandardScaler().fit(R)
        mlp = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-2,
                            learning_rate_init=1e-3, max_iter=500, random_state=seed).fit(scaler.transform(R), y)
    return {"layers": np.asarray(layers, dtype=np.int64), "head_coef": coef, "head_intercept": icpt,
            "resid_coef": rclf.coef_[0].astype(np.float32), "resid_intercept": np.float32(rclf.intercept_[0]),
            "mlp_mean": scaler.mean_.astype(np.float32), "mlp_scale": scaler.scale_.astype(np.float32),
            "mlp_W0": mlp.coefs_[0].astype(np.float32), "mlp_b0": mlp.intercepts_[0].astype(np.float32),
            "mlp_W1": mlp.coefs_[1].astype(np.float32), "mlp_b1": mlp.intercepts_[1].astype(np.float32)}


def probe_scores(probes, head_x, resid_x, exclude_flat=(), n_heads=None):
    """Decision scores of the fixed probes on new activations: (head (len(layers) * H, N) with NaN rows for readout
    heads in `exclude_flat` (never read out at ablated heads), resid (N,), mlp probability (N,))."""
    from scipy.special import expit
    layers = [int(l) for l in np.asarray(probes["layers"]).ravel()]
    coef, icpt = np.asarray(probes["head_coef"], np.float64), np.asarray(probes["head_intercept"], np.float64)
    H, D = coef.shape[1], coef.shape[2]
    head_x = np.asarray(head_x, dtype=np.float32)
    N = head_x.shape[0]
    ex = {int(f) for f in exclude_flat}
    head = np.full((len(layers) * H, N), np.nan)
    for li, l in enumerate(layers):
        X = head_x[:, li].reshape(N, H, D).astype(np.float64)
        s = np.einsum("nhd,hd->hn", X, coef[li]) + icpt[li][:, None]
        for h in range(H):
            if l * H + h not in ex:
                head[li * H + h] = s[h]
    R = np.asarray(resid_x, dtype=np.float64)
    resid = R @ np.asarray(probes["resid_coef"], np.float64) + float(np.ravel(probes["resid_intercept"])[0])
    xs = (R - np.asarray(probes["mlp_mean"], np.float64)) / np.asarray(probes["mlp_scale"], np.float64)
    hid = np.maximum(xs @ np.asarray(probes["mlp_W0"], np.float64) + np.asarray(probes["mlp_b0"], np.float64), 0.0)
    mlp = expit(hid @ np.asarray(probes["mlp_W1"], np.float64) + np.asarray(probes["mlp_b1"], np.float64)).ravel()
    return head, resid, mlp


def _auc_rows(scores, labels):
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    out = np.full(len(scores), np.nan)
    ok = np.isfinite(scores).all(axis=1)
    if ok.any():
        out[ok] = auc_matrix(scores[ok], labels)
    return out


def m2_summary(head, resid, mlp, labels):
    """{"head_auc_mean" (over non-excluded readout heads), "head_auc" (list, NaN = excluded), "resid_auc", "mlp_auc"}."""
    ha = _auc_rows(head, labels)
    return {"head_auc_mean": float(np.nanmean(ha)) if np.isfinite(ha).any() else float("nan"),
            "head_auc": [None if not np.isfinite(a) else float(a) for a in ha],
            "resid_auc": float(_auc_rows(resid, labels)[0]), "mlp_auc": float(_auc_rows(mlp, labels)[0]),
            "n_heads_read": int(np.isfinite(ha).sum())}


def refit_m2(head_x, resid_x, labels, groups, layers, n_heads, exclude_flat=(), device=None, seed=PROBE_SEED):
    """L2 (information survival): logistic probes refitted on the (ablated) EVAL activations, out of fold.

    5-fold StratifiedGroupKFold(shuffle=True, random_state=42), groups = prompt cluster (fewer folds with fewer than 5
    clusters); probing_stats.torch_batched_logreg(C=1.0, class_weight="balanced", max_iter=200), all non-excluded
    readout heads at once and the residual. Returns {"refit_head_auc_mean", "refit_resid_auc", "n_folds"}."""
    from sklearn.model_selection import StratifiedGroupKFold

    from .probing_stats import torch_batched_logreg
    y = np.asarray(labels, dtype=np.int64)
    g = np.asarray(groups)
    nan = float("nan")
    k = min(5, len(np.unique(g)))
    if k < 2 or len(np.unique(y)) < 2:
        return {"refit_head_auc_mean": nan, "refit_resid_auc": nan, "n_folds": 0}
    head_x = np.asarray(head_x, dtype=np.float32)
    N = head_x.shape[0]
    H = int(n_heads)
    D = head_x.shape[2] // H
    ex = {int(f) for f in exclude_flat}
    keep = [(li, h) for li, l in enumerate(layers) for h in range(H) if int(l) * H + h not in ex]
    X = np.stack([head_x[:, li, h * D:(h + 1) * D] for li, h in keep]) if keep else None      # (K, N, D)
    R = np.asarray(resid_x, dtype=np.float32)[None]
    oof_h = np.full((len(keep), N), np.nan)
    oof_r = np.full(N, np.nan)
    folds = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(N), y, g)
    n_folds = 0
    for tr, va in folds:
        if len(np.unique(y[tr])) < 2 or not len(va):
            continue
        n_folds += 1
        if X is not None:
            oof_h[:, va] = torch_batched_logreg(X[:, tr], y[tr], X[:, va], class_weight="balanced", C=1.0, max_iter=200,
                                                device=device)
        oof_r[va] = torch_batched_logreg(R[:, tr], y[tr], R[:, va], class_weight="balanced", C=1.0, max_iter=200,
                                         device=device)[0]
    ha = _auc_rows(oof_h, y) if len(keep) else np.zeros(0)
    return {"refit_head_auc_mean": float(np.nanmean(ha)) if np.isfinite(ha).any() else nan,
            "refit_resid_auc": float(_auc_rows(oof_r, y)[0]), "n_folds": n_folds}


# ----------------------------------------------------------------------------------------------------------------
# M5
# ----------------------------------------------------------------------------------------------------------------

def dla_values(items):
    """DE_i: per-item mean-token margin change of the direct effect (de_pos / n_pos - de_neg / n_neg)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.asarray(items["de_pos"], float) / np.asarray(items["pos_n"], float)
                - np.asarray(items["de_neg"], float) / np.asarray(items["neg_n"], float))
