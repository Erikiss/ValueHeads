"""Value directions and steering vectors (paper Sec. 5, Eq. 2; DESIGN 2 "Steering-Richtung", 3.3, decisions.json).

Adapted from sihuo-design/valueheads `directions.py` (MIT).

Single value:  dir_l^h = E_pos[x_l^h] - E_neg[x_l^h]                                       (Eq. 2)
Multi value:   dir_{l,V}^h = sum_k lambda_k dir_{l,k}^h over the values whose top heads contain h, sum |lambda_k| <= 1
Both are normalized to unit length and scaled by sigma_l^h = std(acts @ dir, ddof=1) of a std feature set;
at inference x_l^h <- x_l^h + alpha * sigma_l^h * dir_l^h.
mode=paper pools the probing train + val splits for the directions (repo), mode=pythia uses train only so that
directions, sigma (val) and evaluation stay disjoint.
"""
import json
import os

import numpy as np
import torch

from .extract import LayerShards, load_features


def _as_list(x):
    return [x] if isinstance(x, (str, os.PathLike)) else list(x)


def mass_mean_directions(features_dirs, value=None, layers=None, chunk=512):
    """Positive-minus-negative mean activation of every head, pooled over the given feature directories.

    features_dirs: one directory or a list (e.g. [train, val] in mode=paper); `value` selects labels_{value}.npy
    (rows with label -1 are ignored). float64 accumulation in chunks; returns (L * H, D) float32.
    """
    pos_sum = neg_sum = None
    n_pos = n_neg = 0
    geometry = None
    for d in _as_list(features_dirs):
        acts, labels, _, man = load_features(d, value=value)
        if labels is None:
            raise ValueError(f"{d} has no labels")
        labels = np.asarray(labels)
        L, H, D = man["layers"], man["heads"], man["head_dim"]
        if geometry is None:
            geometry = (L, H, D)
            pos_sum, neg_sum = np.zeros((L, H * D)), np.zeros((L, H * D))
        elif geometry != (L, H, D):
            raise ValueError(f"geometry mismatch across feature dirs: {geometry} vs {(L, H, D)}")
        use_layers = range(L) if layers is None else layers
        for l in use_layers:
            shard = acts[l]
            for start in range(0, len(shard), chunk):
                x = np.asarray(shard[start:start + chunk], dtype=np.float64)
                y = labels[start:start + chunk]
                pos_sum[l] += x[y == 1].sum(axis=0)
                neg_sum[l] += x[y == 0].sum(axis=0)
        n_pos += int((labels == 1).sum())
        n_neg += int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"need both classes for a mass-mean direction (n_pos={n_pos}, n_neg={n_neg})")
    L, H, D = geometry
    direction = pos_sum / n_pos - neg_sum / n_neg          # (L, H * D)
    return direction.reshape(L * H, D).astype(np.float32)


def unit_directions(directions):
    """Row-normalized copy (zero rows stay zero)."""
    directions = np.asarray(directions, dtype=np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return np.where(norms > 0, directions / np.maximum(norms, 1e-12), 0.0).astype(np.float32)


def _shards(std_features):
    if isinstance(std_features, LayerShards):
        return std_features
    return load_features(std_features)[0]


def sigma_along(directions, std_features, heads=None, ddof=1):
    """std (ddof=1) of the std-set activations of each head projected onto its unit direction; (L * H,) float32.

    std_features: feature directory path or LayerShards; `heads` restricts to flat indices (others NaN).
    """
    shards = _shards(std_features)
    H, D = shards.num_heads, shards.head_dim
    unit = unit_directions(directions)
    out = np.full(len(unit), np.nan, dtype=np.float32)
    for idx in (range(len(unit)) if heads is None else heads):
        layer, head = divmod(int(idx), H)
        acts = shards.head(layer, head)
        out[idx] = np.std(acts @ unit[idx], ddof=ddof)
    return out


def steering_vectors(value_heads, value_dirs, weights, std_features_dir, num_heads, head_dim):
    """Per-layer steering vectors for one or several values (repo semantics + the paper's lambda constraint).

    value_heads: {value: [flat_idx, ...]} top heads of each value
    value_dirs:  {value: (L * H, D) array} mass-mean directions of each value
    weights:     {value: lambda}; requires the same keys as value_heads and sum |lambda| <= 1
    std_features_dir: feature directory (or LayerShards) whose activations estimate sigma along each direction
    Returns {layer: tensor(H * D)} with sigma * unit composite direction on intervened heads, 0 elsewhere.
    """
    if set(weights) != set(value_heads) or set(value_dirs) != set(value_heads):
        raise ValueError("value_heads, value_dirs and weights must have the same keys")
    total = sum(abs(float(w)) for w in weights.values())
    assert total <= 1.0 + 1e-9, f"sum |lambda_k| = {total:.4f} > 1 (paper requires sum |lambda_k| <= 1)"
    shards = _shards(std_features_dir)
    if (shards.num_heads, shards.head_dim) != (num_heads, head_dim):
        raise ValueError(f"std features have geometry {(shards.num_heads, shards.head_dim)}, expected {(num_heads, head_dim)}")

    head_values = {}
    for value, heads in value_heads.items():
        if weights[value] == 0:
            continue
        for idx in heads:
            head_values.setdefault(int(idx), []).append(value)

    vectors = {}
    for idx, values in sorted(head_values.items()):
        layer, head = divmod(idx, num_heads)
        direction = sum(float(weights[v]) * np.asarray(value_dirs[v][idx], dtype=np.float64) for v in values)
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction = (direction / norm).astype(np.float32)
        acts = shards.head(layer, head)
        sigma = float(np.std(acts @ direction, ddof=1))
        vec = vectors.setdefault(layer, torch.zeros(num_heads * head_dim))
        vec[head * head_dim:(head + 1) * head_dim] = torch.from_numpy(sigma * direction)
    return vectors


def save_directions(path, directions, sigmas=None, meta=None):
    """npz with `directions` (L * H, D) float32, `sigma_{set}` arrays (L * H,) and a JSON `meta` string."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"directions": np.asarray(directions, dtype=np.float32), "meta": json.dumps(meta or {})}
    for name, s in (sigmas or {}).items():
        payload[f"sigma_{name}"] = np.asarray(s, dtype=np.float32)
    np.savez(path, **payload)
    return path


def load_directions(path):
    """(directions (L * H, D) float32, {std_set: sigma (L * H,)}, meta dict)."""
    with np.load(path) as z:
        directions = z["directions"]
        sigmas = {k[len("sigma_"):]: z[k] for k in z.files if k.startswith("sigma_")}
        meta = json.loads(str(z["meta"]))
    return directions, sigmas, meta
