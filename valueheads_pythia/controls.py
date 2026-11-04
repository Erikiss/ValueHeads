"""Controls for the head localization (DESIGN 5.7, 5.9): layer-matched random head sets for the causal tests, and
three baselines that bound what a per-head probe can read without any "value" semantics:

  surface_probe      logistic regression on four surface statistics of the text (log length, mean token
                     log-frequency under a unigram table of the probing texts, refusal flag, punctuation share)
  embedding_probe    probe on the input embedding of the last token (pure lexical baseline, no forward pass)
  layer_mean_probe   probe on the mean over the H heads of a layer (the "layer bias" of the repo's features)

A head counts as value-sensitive only when its AUC beats the best control in a paired cluster bootstrap
(DESIGN 5.7: `paired_control_bootstrap` / `heads_vs_controls` on the cached val decision scores, no refit). All
probes use the pythia-mode settings of probing.py (C = 1, lbfgs, max_iter 1000, class_weight balanced) plus
standardization, because the surface features live on very different scales; with return_scores they also hand
back their val decision scores for that paired comparison.
"""
import random
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .extract import load_features
from .probing import group_by_layer

SURFACE_FEATURES = ["log_length", "mean_token_logfreq", "refusal_flag", "punct_share"]

# Refusal / hedging phrases (TruthfulQA "I have no comment", assistant-style refusals); matched case-insensitively.
REFUSAL_RE = re.compile(
    r"\b(I cannot|I can't|I can not|I'm sorry|I am sorry|I apologize|as an AI|I don't know|I do not know|I have no comment|"
    r"I'm not able|I am not able|I won't|I will not|not appropriate|I'm unable|I am unable|I'm not sure|I am not sure)\b", re.I)
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PUNCT = set(string.punctuation)


# ----------------------------------------------------------------------------- random head sets
def random_sets_layer_matched(topk_by_layer, L, H, R, seed=0):
    """R random head sets with the same number of heads per layer as `topk_by_layer` ({layer: [heads]}).

    One RandomState(seed) draws the sets in sequence, so the first R sets are identical for any larger R
    (R = 20 on T4 extends to R = 50 on A100 without changing the first 20). Returns [{layer: [heads]}, ...].
    """
    counts = {int(l): len(hs) for l, hs in topk_by_layer.items() if len(hs)}
    for l, c in counts.items():
        if not 0 <= l < L:
            raise ValueError(f"layer {l} out of range for {L} layers")
        if c > H:
            raise ValueError(f"{c} heads requested in layer {l} but the model has {H}")
    rng = np.random.RandomState(seed)
    sets = []
    for _ in range(R):
        sets.append({l: sorted(int(h) for h in rng.choice(H, counts[l], replace=False)) for l in sorted(counts)})
    return sets


def random_sets_uniform(k, L, H, R, seed0=0):
    """Paper-mode control: R uniform random sets of k heads, seeds seed0 .. seed0 + R - 1
    (`random.Random(seed).sample(range(L * H), k)` as in mask_heads.py of the reference repo). Returns [{layer: [heads]}]."""
    return [group_by_layer(random.Random(seed0 + r).sample(range(L * H), k), H) for r in range(R)]


def count_by_layer(heads_by_layer):
    """{layer: number of heads} of a {layer: [heads]} set."""
    return {int(l): len(hs) for l, hs in heads_by_layer.items()}


# ----------------------------------------------------------------------------- probe helper
def _fit_probe(f_tr, y_tr, f_va, y_va, seed=42, class_weight="balanced", C=1.0, max_iter=1000):
    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, max_iter=max_iter, C=C, class_weight=class_weight))
    clf.fit(f_tr, y_tr)
    return clf, _scores_to_metrics(y_va, clf.decision_function(f_va), clf.predict(f_va))


def _scores_to_metrics(y, scores, pred):
    y = np.asarray(y)
    return {"auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan"),
            "acc": float(accuracy_score(y, pred)), "bal_acc": float(balanced_accuracy_score(y, pred)),
            "majority": float(max(y.mean(), 1 - y.mean())), "pos_rate": float(y.mean()), "n_val": int(len(y))}


def _cv_metrics(f, y, groups, seed, n_splits, class_weight):
    """Out-of-fold decision scores from a (group-)stratified K-fold -> metrics on the pooled predictions."""
    y = np.asarray(y)
    f = np.asarray(f, dtype=np.float32)
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(f, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(f, y, np.asarray(groups))
    scores, pred = np.zeros(len(y)), np.zeros(len(y), dtype=np.int64)
    for tr, va in folds:
        clf, _ = _fit_probe(f[tr], y[tr], f[va], y[va], seed, class_weight)
        scores[va] = clf.decision_function(f[va])
        pred[va] = clf.predict(f[va])
    out = _scores_to_metrics(y, scores, pred)
    out["n_splits"] = n_splits
    return out


# ----------------------------------------------------------------------------- surface probe
def unigram_table(texts):
    """Add-one smoothed log relative frequencies of the word tokens of `texts`: {token: log p}; key None = unseen."""
    counts = Counter()
    for t in texts:
        counts.update(_WORD_RE.findall(t.lower()))
    total, vocab = sum(counts.values()), len(counts)
    table = {w: float(np.log((c + 1) / (total + vocab + 1))) for w, c in counts.items()}
    table[None] = float(np.log(1 / (total + vocab + 1)))
    return table


def surface_features(texts, unigram=None):
    """(N, 4) float32 array of [log(1 + #word tokens), mean token log-frequency, refusal flag, punctuation share].

    The unigram table is built over `texts` when not given (fit it on the training texts and reuse it on val).
    Returns (features, unigram table).
    """
    if unigram is None:
        unigram = unigram_table(texts)
    unseen = unigram[None]
    out = np.zeros((len(texts), 4), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = _WORD_RE.findall(t.lower())
        out[i, 0] = np.log1p(len(toks))
        out[i, 1] = np.mean([unigram.get(w, unseen) for w in toks]) if toks else unseen
        out[i, 2] = 1.0 if REFUSAL_RE.search(t) else 0.0
        out[i, 3] = (sum(ch in _PUNCT for ch in t) / len(t)) if t else 0.0
    return out, unigram


def surface_probe(texts, labels, groups=None, val_texts=None, val_labels=None, seed=42, n_splits=5, class_weight="balanced",
                  return_scores=False):
    """AUC (plus acc / bal_acc) of a logistic regression on the four surface features (DESIGN 5.7 control ii).

    With val_texts / val_labels: fit on (texts, labels), score on val (unigram table from the training texts);
    return_scores adds the val decision scores ("scores", in val order) for the paired comparison with the heads.
    Without: out-of-fold predictions of a stratified (group-)K-fold over (texts, labels).
    Returns {"auc", "acc", "bal_acc", "majority", "pos_rate", "n_val", "features", "coef" (standardized), ...}.
    """
    f_tr, table = surface_features(texts)
    y_tr = np.asarray(labels)
    if val_texts is not None:
        f_va, _ = surface_features(val_texts, table)
        clf, out = _fit_probe(f_tr, y_tr, f_va, np.asarray(val_labels), seed, class_weight)
        out["coef"] = clf[-1].coef_[0].astype(float).tolist()
        out["n_train"] = int(len(y_tr))
        if return_scores:
            out["scores"] = clf.decision_function(f_va).astype(np.float32)
    else:
        out = _cv_metrics(f_tr, y_tr, groups, seed, n_splits, class_weight)
        clf, _ = _fit_probe(f_tr, y_tr, f_tr, y_tr, seed, class_weight)
        out["coef"] = clf[-1].coef_[0].astype(float).tolist()
        out["n_train"] = int(len(y_tr))
    out["features"] = list(SURFACE_FEATURES)
    out["control"] = "surface"
    return out


# ----------------------------------------------------------------------------- embedding probe
def last_token_ids(tokenizer, texts):
    """Id of the last token of every text (the token whose o_proj inputs are probed)."""
    enc = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return np.array([seq[-1] if seq else tokenizer.eos_token_id for seq in enc], dtype=np.int64)


def last_token_embeddings(model, tokenizer, texts):
    """(N, hidden) float32 input embeddings of the last token of each text (no forward pass)."""
    ids = last_token_ids(tokenizer, texts)
    weight = model.get_input_embeddings().weight.detach()
    return weight[torch_as_tensor(ids, weight.device)].float().cpu().numpy()


def torch_as_tensor(ids, device):
    import torch
    return torch.as_tensor(ids, dtype=torch.long, device=device)


def embedding_probe(train, y_train, val, y_val, model=None, tokenizer=None, seed=42, class_weight="balanced", return_scores=False):
    """Probe on the last-token input embedding (DESIGN 5.7 control iii): lexical identity of the probed token.

    train / val: (N, d) embedding arrays, or lists of texts (then model + tokenizer embed their last token).
    Returns the metrics dict of the val split (with return_scores also "scores" = val decision scores).
    """
    if isinstance(train[0], str) or isinstance(val[0], str):
        if model is None or tokenizer is None:
            raise ValueError("texts need model and tokenizer to embed the last token")
        train = last_token_embeddings(model, tokenizer, train)
        val = last_token_embeddings(model, tokenizer, val)
    val = np.asarray(val, dtype=np.float32)
    clf, out = _fit_probe(np.asarray(train, dtype=np.float32), np.asarray(y_train), val, np.asarray(y_val), seed, class_weight)
    out["n_train"] = int(len(y_train))
    out["dim"] = int(np.asarray(train).shape[1])
    out["control"] = "embedding"
    if return_scores:
        out["scores"] = clf.decision_function(val).astype(np.float32)
    return out


# ----------------------------------------------------------------------------- layer-mean probe
def layer_mean_probe(train_dir, val_dir, value=None, layers=None, seed=42, class_weight="balanced", return_scores=False):
    """Probe on the mean over the H heads of each layer (D dims; DESIGN 5.7 control iv) from extracted features.

    Returns a DataFrame with one row per layer: layer, auc, acc, bal_acc, majority, pos_rate, n_train, n_val; with
    return_scores, df.attrs["scores"] = {layer: val decision scores (N_val,)}.
    `value` selects labels_{value}.npy of HelpSteer2 extract-once directories (rows labeled -1 are dropped).
    """
    acts_tr, y_tr, _, man = load_features(train_dir, value=value)
    acts_va, y_va, _, _ = load_features(val_dir, value=value)
    if y_tr is None or y_va is None:
        raise ValueError("layer_mean_probe needs labeled feature directories")
    y_tr, y_va = np.asarray(y_tr), np.asarray(y_va)
    m_tr, m_va = y_tr >= 0, y_va >= 0
    H = man["heads"]
    rows, scores = [], {}
    for layer in (range(man["layers"]) if layers is None else sorted(layers)):
        f_tr = np.asarray(acts_tr[layer], dtype=np.float32).reshape(len(y_tr), H, -1).mean(axis=1)[m_tr]
        f_va = np.asarray(acts_va[layer], dtype=np.float32).reshape(len(y_va), H, -1).mean(axis=1)[m_va]
        clf, out = _fit_probe(f_tr, y_tr[m_tr], f_va, y_va[m_va], seed, class_weight)
        rows.append({"layer": int(layer), **out, "n_train": int(m_tr.sum())})
        if return_scores:
            scores[int(layer)] = clf.decision_function(f_va).astype(np.float32)
    df = pd.DataFrame(rows)[["layer", "auc", "acc", "bal_acc", "majority", "pos_rate", "n_train", "n_val"]]
    df.attrs.update(control="layer_mean", train_dir=train_dir, val_dir=val_dir, value=value, class_weight=class_weight, seed=seed)
    if return_scores:
        df.attrs["scores"] = scores
    return df


def best_control(surface=None, embedding=None, layer_mean=None):
    """{"surface": auc, "embedding": auc, "layer_mean": max auc over layers, "best": max, "best_name": ...,
    "best_layer": layer of the layer-mean maximum (when given)}; point AUCs only (the paired test is heads_vs_controls)."""
    aucs = {}
    out = {}
    if surface is not None:
        aucs["surface"] = float(surface["auc"])
    if embedding is not None:
        aucs["embedding"] = float(embedding["auc"])
    if layer_mean is not None:
        lm = layer_mean["auc"].to_numpy(dtype=float)
        aucs["layer_mean"] = float(np.nanmax(lm))
        out["best_layer"] = int(layer_mean["layer"].iloc[int(np.nanargmax(lm))])
    if not aucs:
        return {"best": float("nan"), "best_name": None}
    name = max(aucs, key=aucs.get)
    return {**aucs, **out, "best": aucs[name], "best_name": name}


# ----------------------------------------------------------------------------- paired bootstrap heads vs control (DESIGN 5.7)
def paired_control_bootstrap(head_scores, control_scores, labels, groups, B=1000, seed=0, alpha=0.05):
    """Paired cluster bootstrap of AUC(head) - AUC(control) on the same val items (no refit).

    head_scores: (K, N) val decision scores of the heads (NaN rows allowed), control_scores: (N,) val decision
    scores of a control probe, labels (N,) in {0, 1}, groups (N,) cluster ids. Groups are resampled with
    replacement (B times, seeded); both AUCs are recomputed on every resample, so the difference accounts for the
    shared items. Returns a DataFrame indexed by head row: auc_head, auc_control, delta, ci_lo, ci_hi, beats
    (= ci_lo > 0, the head is value-sensitive beyond the control at the 1 - alpha level), plus attrs.
    """
    from .probing_stats import WeightedAUC, auc_matrix
    head_scores = np.asarray(head_scores, dtype=np.float64)
    if head_scores.ndim == 1:
        head_scores = head_scores[None]
    control_scores = np.asarray(control_scores, dtype=np.float64).reshape(-1)
    labels, groups = np.asarray(labels), np.asarray(groups)
    if head_scores.shape[1] != len(labels) or len(control_scores) != len(labels) or len(groups) != len(labels):
        raise ValueError(f"head scores {head_scores.shape}, control scores {control_scores.shape}, labels {labels.shape}, "
                         f"groups {groups.shape} do not match")
    if len(np.unique(labels)) < 2:
        raise ValueError("paired bootstrap needs both classes on val")
    valid = ~np.isnan(head_scores).any(axis=1)
    rows = np.flatnonzero(valid)
    stacked = np.vstack([head_scores[rows], control_scores[None]])
    prep = WeightedAUC(stacked, labels)
    _, group_idx = np.unique(groups, return_inverse=True)
    n_groups = int(group_idx.max()) + 1
    rng = np.random.RandomState(seed)
    counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=B)
    boot = np.empty((B, len(rows)))
    for b in range(B):
        a = prep.auc(counts[b][group_idx])
        boot[b] = a[:-1] - a[-1]
    point = auc_matrix(stacked, labels)
    out = pd.DataFrame(index=np.arange(len(head_scores)), columns=["auc_head", "auc_control", "delta", "ci_lo", "ci_hi"], dtype=float)
    out.loc[rows, "auc_head"] = point[:-1]
    out.loc[rows, "auc_control"] = point[-1]
    out.loc[rows, "delta"] = point[:-1] - point[-1]
    if B > 0 and n_groups > 1:
        out.loc[rows, "ci_lo"] = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
        out.loc[rows, "ci_hi"] = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    out["beats"] = out["ci_lo"] > 0
    out.attrs.update(B=B, seed=seed, alpha=alpha, n_groups=n_groups, n_items=int(len(labels)))
    return out


def heads_vs_controls(head_scores, labels, groups, control_scores, B=1000, seed=0, alpha=0.05):
    """DESIGN 5.7 verdict per head against the best control: {"best_name", "best_auc", "paired" (DataFrame of
    paired_control_bootstrap vs the best control), "n_beats", "beating_heads" (row indices with ci_lo > 0),
    "n_beats_by_control": {name: n}}. control_scores: {name: (N,) val decision scores} (e.g. "surface",
    "embedding", "layer_mean" = the best layer's scores)."""
    if not control_scores:
        raise ValueError("no control scores given")
    from .probing_stats import auc_matrix
    aucs = {name: float(auc_matrix(np.asarray(s, dtype=np.float64)[None], labels)[0]) for name, s in control_scores.items()}
    best_name = max(aucs, key=aucs.get)
    per_control = {name: paired_control_bootstrap(head_scores, s, labels, groups, B, seed, alpha) for name, s in control_scores.items()}
    paired = per_control[best_name]
    beating = [int(i) for i in paired.index[paired["beats"].fillna(False).to_numpy(dtype=bool)]]
    return {"best_name": best_name, "best_auc": aucs[best_name], "control_aucs": aucs, "paired": paired,
            "n_beats": len(beating), "beating_heads": beating,
            "n_beats_by_control": {name: int(df["beats"].fillna(False).sum()) for name, df in per_control.items()},
            "B": B, "seed": seed, "alpha": alpha}
