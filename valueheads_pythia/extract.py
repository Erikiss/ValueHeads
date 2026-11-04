"""Step 1: sharded, resumable extraction of last-token head activations (DESIGN 3.1, 6.4).

Layout of a feature directory ({features}/{model}/{revision}/{dataset}_{split}/):
    layer{LL}.npy      (N, H * D) float16, one shard per layer (layer-major, RAM-friendly)
    labels.npy         (N,) int labels, or labels_{value}.npy per value (HelpSteer2 extract-once, -1 = discarded)
    groups.npy         (N,) int group ids
    manifest.json      model / revision / commit / dtype / attn_impl / versions / geometry / counts / progress
    DONE               written last; its presence means the directory is complete

Texts are tokenized once (no BOS for Pythia, truncation to max_length keeping the tail), processed in
descending length order with right padding and a token budget per batch, and gathered at the last valid
token via the attention mask. Interrupted runs resume from the recorded progress.
"""
import json
import os
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from . import config
from .data import texts_fingerprint
from .hooks import ActivationCollector
from .models import model_geometry, num_kv_heads

MANIFEST = "manifest.json"
DONE = "DONE"


def shard_path(out_dir, layer):
    return os.path.join(out_dir, f"layer{layer:02d}.npy")


def is_done(out_dir):
    return os.path.exists(os.path.join(out_dir, DONE))


def read_manifest(out_dir):
    with open(os.path.join(out_dir, MANIFEST)) as f:
        return json.load(f)


def _write_manifest(out_dir, manifest):
    tmp = os.path.join(out_dir, MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, MANIFEST))


def tokenize_all(tokenizer, texts, max_length=config.MAX_LENGTH, chunk=1000):
    """Token id lists per text (no padding; tail-keep truncation to max_length; max_length None / 0 = no truncation, the
    paper-mode tokenization of DESIGN 2). Returns (ids, n_truncated).

    Raises for a text that tokenizes to zero tokens (an empty row would be gathered at a pad column and an all-empty
    batch crashes inside the model).
    """
    ids, n_trunc = [], 0
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        for s in range(0, len(texts), chunk):
            enc = tokenizer(texts[s:s + chunk], truncation=False, add_special_tokens=True)["input_ids"]
            for j, seq in enumerate(enc):
                if len(seq) == 0:
                    raise ValueError(f"text {s + j} tokenizes to zero tokens ({texts[s + j]!r}); drop empty texts before extraction")
                if max_length and len(seq) > max_length:
                    n_trunc += 1
                    seq = seq[-max_length:]
                ids.append(seq)
    finally:
        tokenizer.truncation_side = old_side
    return ids, n_trunc


def plan_batches(lengths, batch_size, max_tokens_per_batch):
    """Deterministic batches (lists of text indices) in descending length order under a token budget."""
    order = np.argsort(-np.asarray(lengths), kind="stable")
    batches, start = [], 0
    while start < len(order):
        longest = lengths[order[start]]
        n = max(1, min(batch_size, max_tokens_per_batch // max(1, longest)))
        batches.append(order[start:start + n].tolist())
        start += n
    return batches


def _pad_batch(ids, idx, pad_id):
    width = max(len(ids[i]) for i in idx)
    input_ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idx), width), dtype=torch.long)
    for r, i in enumerate(idx):
        seq = ids[i]
        input_ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[r, :len(seq)] = 1
    return input_ids, mask


def extract_features(model, tokenizer, texts, out_dir, batch_size=16, max_length=config.MAX_LENGTH, resume=True,
                     labels=None, groups=None, max_tokens_per_batch=16384, flush_every=20, flush_seconds=60,
                     extra_manifest=None, show_progress=True, calibrate_after=100, budget_seconds=45 * 60):
    """Extracts last-token head activations of `texts` into layer shards under `out_dir`; returns the manifest.

    labels: (N,) array, or {value: (N,) array} (written as labels_{value}.npy), or None (prompt sets).
    groups: (N,) array or None. With `resume`, a directory with DONE is skipped and a partial one is continued
    (texts fingerprint and N must match); without it, the directory is rebuilt from scratch. Progress is flushed
    to disk every `flush_every` batches or `flush_seconds` seconds, whichever comes first. After `calibrate_after`
    texts the measured throughput and the ETA of the split are printed (DESIGN 6 / 8: re-calibrate after the first 100
    texts; a warning when the ETA exceeds `budget_seconds`, the 45-min budget per checkpoint). A partial directory is
    only continued with the same dtype, attention backend, revision, max_length, batch_size and max_tokens_per_batch
    (batch composition changes the fp16 numerics by up to ~4e-3; use resume=False / --force otherwise).
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(texts)
    for name, arr in (("groups", groups), *((f"labels[{v}]", a) for v, a in (labels.items() if isinstance(labels, dict) else [("labels", labels)]))):
        if arr is not None and len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} entries for {n} texts")
    L, H, D = model_geometry(model.config)
    width = H * D
    fingerprint = texts_fingerprint(texts)
    if is_done(out_dir) and resume:
        manifest = read_manifest(out_dir)
        if manifest.get("texts_sha256") == fingerprint and manifest.get("n") == n:
            return manifest
        raise RuntimeError(f"{out_dir} is DONE but holds different texts (n={manifest.get('n')}); use resume=False")

    info = getattr(model, "vh_info", {})
    base = {
        "n": n, "layers": L, "heads": H, "head_dim": D, "width": width, "feature_dtype": "float16",
        "max_length": max_length, "truncation_side": "left", "batch_size": batch_size,
        "max_tokens_per_batch": max_tokens_per_batch, "texts_sha256": fingerprint,
        "transformers": info.get("transformers"), "torch": info.get("torch"),
        "model": info.get("model"), "hf_id": info.get("hf_id"), "revision": info.get("revision"),
        "commit_sha": info.get("commit_sha"), "dtype": info.get("dtype"), "attn_impl": info.get("attn_impl"),
        "device": str(model.device), "pad_token_id": tokenizer.pad_token_id, "tokenizer_revision": info.get("tokenizer_revision"),
        "label_sets": sorted(labels) if isinstance(labels, dict) else (["labels"] if labels is not None else []),
        "has_groups": groups is not None,
    }
    kv = num_kv_heads(model.config)
    if kv != H or width != model.config.hidden_size:
        # GQA / head_dim decoupled from hidden (Qwen3, Llama-3): record the KV grouping (query head h reads KV head
        # h // (H / kv_heads)) and the residual width; GPT-NeoX manifests keep their previous keys
        base.update(kv_heads=kv, hidden=model.config.hidden_size)
    base.update(extra_manifest or {})

    manifest_path = os.path.join(out_dir, MANIFEST)
    resumable = resume and os.path.exists(manifest_path) and not is_done(out_dir)
    if resumable:
        manifest = read_manifest(out_dir)
        if manifest.get("texts_sha256") != fingerprint or manifest.get("n") != n:
            raise RuntimeError(f"partial extraction in {out_dir} holds different texts; use resume=False")
        for key in ("dtype", "attn_impl", "hf_id", "revision", "max_length", "batch_size", "max_tokens_per_batch"):
            if manifest.get(key) != base.get(key):
                raise RuntimeError(f"partial extraction in {out_dir} used {key}={manifest.get(key)!r}, now {base.get(key)!r}; "
                                   "resume with the same settings or rebuild with resume=False (CLI: extract --force)")
        missing = [l for l in range(L) if not os.path.exists(shard_path(out_dir, l))]
        if missing:
            raise RuntimeError(f"partial extraction in {out_dir} has an incomplete shard set (missing layer shards {missing}); "
                               "rebuild it with resume=False (CLI: extract --force)")
        n_done = int(manifest.get("n_done", 0))
        mode = "r+"
    else:
        for f in os.listdir(out_dir):
            if f.startswith("layer") or f in (MANIFEST, DONE) or f.startswith("labels") or f == "groups.npy":
                os.remove(os.path.join(out_dir, f))
        manifest = dict(base, n_done=0, created=time.strftime("%Y-%m-%d %H:%M:%S"))
        n_done = 0
        mode = "w+"

    if isinstance(labels, dict):
        for value, arr in labels.items():
            np.save(os.path.join(out_dir, f"labels_{value}.npy"), np.asarray(arr))
    elif labels is not None:
        np.save(os.path.join(out_dir, "labels.npy"), np.asarray(labels))
    if groups is not None:
        np.save(os.path.join(out_dir, "groups.npy"), np.asarray(groups))

    shards = [np.lib.format.open_memmap(shard_path(out_dir, l), mode=mode, dtype=np.float16, shape=(n, width)) for l in range(L)]
    if mode == "w+":
        _write_manifest(out_dir, manifest)

    ids, n_trunc = tokenize_all(tokenizer, texts, max_length)
    lengths = [len(s) for s in ids]
    manifest.update(n_truncated=n_trunc, n_tokens=int(sum(lengths)), max_tokens=int(max(lengths) if lengths else 0))
    batches = plan_batches(lengths, batch_size, max_tokens_per_batch)

    # skip the batches that are already done (progress counts texts in the deterministic batch order)
    done_texts, first_batch = 0, 0
    while first_batch < len(batches) and done_texts + len(batches[first_batch]) <= n_done:
        done_texts += len(batches[first_batch])
        first_batch += 1
    n_done = done_texts
    start_done = n_done
    tokens_total = int(sum(lengths))
    tokens_before = int(sum(lengths[i] for b in batches[:first_batch] for i in b))
    calibrated = False

    t0 = last_flush = time.time()
    tokens_seen = 0
    pbar = tqdm(total=n, initial=n_done, desc=f"extract {os.path.basename(out_dir)}", disable=not show_progress)
    with ActivationCollector(model) as collector:
        for b, idx in enumerate(batches[first_batch:], start=first_batch):
            input_ids, mask = _pad_batch(ids, idx, tokenizer.pad_token_id)
            acts = collector.collect(input_ids, mask).numpy().astype(np.float16)      # (B, L, width)
            for l in range(L):
                shards[l][idx] = acts[:, l]
            n_done += len(idx)
            tokens_seen += int(mask.sum())
            pbar.update(len(idx))
            if not calibrated and calibrate_after and n_done - start_done >= calibrate_after and n_done < n:
                calibrated = True
                rate = tokens_seen / max(time.time() - t0, 1e-9)
                remaining = tokens_total - tokens_before - tokens_seen
                eta = remaining / max(rate, 1e-9)
                manifest["calibration"] = {"after_texts": int(n_done - start_done), "tokens_per_second": round(rate, 1),
                                           "eta_seconds": round(eta, 1), "remaining_tokens": int(remaining)}
                print(f"[extract] after {n_done - start_done} texts: {rate:.0f} tok/s, ETA {eta / 60:.1f} min for the remaining "
                      f"{remaining} tokens of this split", flush=True)
                if eta > budget_seconds:
                    print(f"[extract] warning: ETA exceeds the {budget_seconds / 60:.0f}-min budget per checkpoint (DESIGN 6 / 8): "
                          "reduce --max_samples or choose a smaller tier", flush=True)
            if (b + 1) % flush_every == 0 or time.time() - last_flush >= flush_seconds:
                for s in shards:
                    s.flush()
                manifest["n_done"] = n_done
                _write_manifest(out_dir, manifest)
                last_flush = time.time()
    pbar.close()
    for s in shards:
        s.flush()
    del shards
    elapsed = time.time() - t0
    manifest.update(n_done=n, seconds=round(elapsed, 2), tokens_per_second=round(tokens_seen / elapsed, 1) if elapsed > 0 else None,
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    _write_manifest(out_dir, manifest)
    with open(os.path.join(out_dir, DONE), "w") as f:
        f.write(manifest["finished"] + "\n")
    return manifest


class LayerShards:
    """Lazy per-layer access to the float16 shards of a feature directory: shards[layer] -> memmap (N, H * D)."""

    def __init__(self, out_dir, layers=None, mmap=True):
        self.dir = out_dir
        self.manifest = read_manifest(out_dir)
        self.n = self.manifest["n"]
        self.num_layers, self.num_heads, self.head_dim = self.manifest["layers"], self.manifest["heads"], self.manifest["head_dim"]
        self.layers = list(range(self.num_layers)) if layers is None else sorted(layers)
        self.mmap = mmap
        self._cache = {}

    def __getitem__(self, layer):
        if layer not in self.layers:
            raise KeyError(f"layer {layer} not loaded (layers={self.layers})")
        if layer not in self._cache:
            self._cache[layer] = np.load(shard_path(self.dir, layer), mmap_mode="r" if self.mmap else None)
        return self._cache[layer]

    def head(self, layer, head, dtype=np.float32):
        """(N, D) activations of one head as a dense array."""
        return np.asarray(self[layer][:, head * self.head_dim:(head + 1) * self.head_dim], dtype=dtype)

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter(self.layers)

    def keys(self):
        return list(self.layers)


def load_labels(out_dir, value=None):
    """labels.npy, or labels_{value}.npy when `value` is given and exists; None if absent."""
    candidates = []
    if value is not None:
        candidates.append(os.path.join(out_dir, f"labels_{value}.npy"))
    candidates.append(os.path.join(out_dir, "labels.npy"))
    for p in candidates:
        if os.path.exists(p):
            return np.load(p)
    return None


def load_features(out_dir, layers=None, value=None, mmap=True, require_done=True):
    """(acts, labels, groups, manifest) of a feature directory.

    acts: LayerShards (acts[layer] -> float16 memmap (N, H * D), acts.head(layer, head) -> float32 (N, D));
    labels: int array or None (HelpSteer2 extract-once: pass `value` to pick labels_{value}.npy, -1 = discarded);
    groups: int array or None.
    """
    if require_done and not is_done(out_dir):
        raise FileNotFoundError(f"feature directory {out_dir} is not complete (no DONE marker)")
    acts = LayerShards(out_dir, layers, mmap)
    groups_path = os.path.join(out_dir, "groups.npy")
    groups = np.load(groups_path) if os.path.exists(groups_path) else None
    return acts, load_labels(out_dir, value), groups, acts.manifest
