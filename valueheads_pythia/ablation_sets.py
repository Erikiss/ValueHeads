"""Data split and frozen head sets for the causal ablation test (docs/ABLATION_DESIGN.md, part B, sections 6.1-6.4).

Question of the test: does Qwen3-4B(-Base) *rely* on the heads where the HH-"helpful" probe and the named ICL circuit
L20-27 (Gao & Gao, arXiv 2611.05344) overlap, or is "helpful" only *decodable* there? This module fixes everything that
must be decided before the first EVAL forward pass:

1. HH split (6.1). Items are `data.eval_pairs('hh_pairwise', 'helpful', split='train')` (id = `group` = index in
   `data.hh_pairs('helpful', 'train')`). Three roles -- means_ref (reference means, DoM, P_DoM, donors), means_probe (M2
   probe training) and eval -- are disjoint by prompt cluster (`prompt_key`), disjoint from the probe-train pairs of the
   selection run (by group id and by prompt text) and from the probe-val prompts. `build_split` is pure (testable
   offline); `hh_helpful_split` loads the data. No item text is ever written: the split file holds ids, prompt-key
   hashes and item hashes only.
2. Named head sets (6.2): C33, O = C33 & TopK_all, CminusO, P_band, P_DoM, C33_readers / C33_brakes, WholeBand,
   TopK_all, CR20, CensusReaders, CB24, each with an identity key (ICL sets: the ICL base model, F11; table sets: the
   model that defines them).
3. Random sets (6.3): layer- and KV-group-matched draws exactly like the overlap null (`overlap.layer_matched_draws`),
   one generator per (seed, identity, set, draw index) -> prefix property, and the instruct run shares the C33 draws.
4. Frozen-set file (6.4): written in Stage 0 before any EVAL forward; a later run with different sets refuses
   (`FrozenMismatch`) unless forced.

Heads are (layer, head) tuples; flat index = layer * n_heads + head. Nothing here loads a model.
"""
import copy
import datetime as _dt
import hashlib
import json
import os
import zlib

import numpy as np

from . import overlap

# ----------------------------------------------------------------------------------------------------------------
# HH split (6.1)
# ----------------------------------------------------------------------------------------------------------------

SPLIT_VERSION = 1
SPLIT_KEY = "hh_helpful_split_v1"
SPLIT_ROLES = ("means_ref", "means_probe", "eval")
DROP_REASONS = ("prompt_in_probe_train", "prompt_in_probe_val", "empty_response", "identical")
ASSISTANT_TAG = "\n\nAssistant:"


class SplitMismatch(ValueError):
    """An existing split file differs from the split that would be written."""


def _one(x):
    """The single continuation of an eval_pairs field (a one-element list) or a plain string."""
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"expected exactly one continuation, got {len(x)}")
        return x[0]
    return x


def prompt_key(context):
    """Whitespace-normalised prompt hash: sha256(" ".join(context.split()))[:16] (the cluster id of an item)."""
    return hashlib.sha256(" ".join(str(context).split()).encode("utf-8")).hexdigest()[:16]


def item_sha(item):
    """sha256(context + "\\x1e" + pos + "\\x1e" + neg)[:16] -- verifies an item without storing its text."""
    text = item["context"] + "\x1e" + _one(item["pos"]) + "\x1e" + _one(item["neg"])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chosen_context(chosen):
    """Context of an HH dialogue as eval_pairs builds it: text up to and including the last "\\n\\nAssistant:" (None if
    the tag is missing)."""
    cut = chosen.rfind(ASSISTANT_TAG)
    return None if cut < 0 else chosen[:cut + len(ASSISTANT_TAG)]


def _pair_prompt_keys(pairs):
    """prompt_key of the context of the chosen and of the rejected text of every (chosen, rejected) pair."""
    keys = set()
    for pair in pairs:
        for text in pair:
            ctx = chosen_context(text)
            if ctx is not None:
                keys.add(prompt_key(ctx))
    return keys


def _sha_of(values):
    return hashlib.sha256("\n".join(str(v) for v in values).encode("utf-8")).hexdigest()


def build_split(items, probe_train_groups, probe_train_prompt_keys, probe_val_prompt_keys, *, n_means_ref=256,
                n_means_probe=256, n_eval=1000, seed=42):
    """The three disjoint roles of the HH helpful items (pure; section 6.1).

    items:                    eval_pairs dicts {"group", "context", "pos", "neg"} (pos / neg: one-element lists or str)
    probe_train_groups:       group ids of the probe-train pairs (excluded from the pool)
    probe_train_prompt_keys:  prompt_key of every probe-train context (pool items sharing a prompt are dropped)
    probe_val_prompt_keys:    prompt_key of every probe-val context (likewise)

    1. pool = items whose group is not a probe-train group;
    2. drop for the first matching reason in DROP_REASONS order (counted);
    3. clusters = pool items grouped by prompt_key, visited in a seeded permutation of the sorted keys
       (np.random.default_rng([seed, crc32(SPLIT_KEY)]));
    4. each role (means_ref, means_probe, eval) walks the cluster order from the start and takes every unused cluster
       that still fits (count + size <= target) until the target is met exactly (ValueError otherwise). Items of a
       role keep walk order (cluster order, then group id), so a prefix of a role is a valid smaller role;
    5. the rest is the reserve (count only).
    Returns the split dict (see the file format in 6.1; "source" / probe "rule" fields are added by hh_helpful_split).
    """
    targets = {"means_ref": int(n_means_ref), "means_probe": int(n_means_probe), "eval": int(n_eval)}
    if any(t < 0 for t in targets.values()):
        raise ValueError(f"negative role size in {targets}")
    train_groups = {int(g) for g in probe_train_groups}
    train_keys, val_keys = set(probe_train_prompt_keys), set(probe_val_prompt_keys)
    ids = [int(it["group"]) for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("item group ids are not unique")
    dropped = {r: 0 for r in DROP_REASONS}
    pool = []                                                   # (group, prompt_key, item_sha)
    n_not_train = 0
    for it in items:
        g = int(it["group"])
        if g in train_groups:
            continue
        n_not_train += 1
        key = prompt_key(it["context"])
        pos, neg = _one(it["pos"]), _one(it["neg"])
        if key in train_keys:
            reason = "prompt_in_probe_train"
        elif key in val_keys:
            reason = "prompt_in_probe_val"
        elif not pos.strip() or not neg.strip():
            reason = "empty_response"
        elif pos == neg:
            reason = "identical"
        else:
            reason = None
        if reason:
            dropped[reason] += 1
            continue
        pool.append((g, key, item_sha(it)))
    clusters = {}
    for g, key, _ in pool:
        clusters.setdefault(key, []).append(g)
    for key in clusters:
        clusters[key].sort()
    keys = sorted(clusters)
    rng = np.random.default_rng([int(seed), zlib.crc32(SPLIT_KEY.encode("utf-8"))])
    order = [keys[i] for i in rng.permutation(len(keys))]
    used, roles = set(), {}
    for role in SPLIT_ROLES:
        target, count, take = targets[role], 0, []
        for key in order:
            if count == target:
                break
            if key in used:
                continue
            size = len(clusters[key])
            if count + size <= target:
                take.append(key)
                used.add(key)
                count += size
        if count != target:
            raise ValueError(f"split role {role}: only {count} of {target} items reachable (pool {len(pool)} items in "
                             f"{len(keys)} prompt clusters, {sum(len(clusters[k]) for k in used)} already used)")
        roles[role] = [g for key in take for g in clusters[key]]
    in_roles = {g for r in SPLIT_ROLES for g in roles[r]}
    key_of = {g: key for g, key, _ in pool}
    sha_of = {g: s for g, _, s in pool}
    listed = [g for r in SPLIT_ROLES for g in roles[r]]
    return {
        "version": SPLIT_VERSION, "key": SPLIT_KEY, "seed": int(seed),
        "source": {"dataset": "Anthropic/hh-rlhf", "subset": "helpful-base", "hf_split": "train",
                   "items": "data.eval_pairs('hh_pairwise','helpful',split='train')"},
        "probe_train": {"n_groups": len(train_groups), "groups_sha256": _sha_of(sorted(train_groups)),
                        "n_prompt_keys": len(train_keys), "prompt_keys_sha256": _sha_of(sorted(train_keys))},
        "probe_val": {"n_keys": len(val_keys), "keys_sha256": _sha_of(sorted(val_keys))},
        "counts": {"items": len(items), "not_in_probe_train": n_not_train, "dropped": dropped, "pool": len(pool),
                   "clusters": len(keys), **{r: len(roles[r]) for r in SPLIT_ROLES},
                   "reserve": len(pool) - len(in_roles)},
        "roles": roles,
        "cluster": {str(g): key_of[g] for g in listed},
        "item_sha": {str(g): sha_of[g] for g in listed},
    }


def hh_helpful_split(*, n_means_ref=256, n_means_probe=256, n_eval=1000, seed=42, probe_max_samples=20000,
                     probe_seed=42, verbose=True):
    """(split dict, {item id: item}) for the HH helpful pairs (loads the data; HH-RLHF helpful-base, online or HF cache).

    Inputs (6.1, F6): items = data.eval_pairs('hh_pairwise','helpful',split='train'); probe-train groups =
    data.probing_set('helpful','train','pythia',max_samples=probe_max_samples,seed=probe_seed) (asserted to be
    probe_max_samples texts in probe_max_samples / 2 whole pairs); probe-train prompt keys = the contexts of the
    chosen **and** rejected texts of those pairs (data.hh_pairs('helpful','train')); probe-val prompt keys = the
    contexts of the chosen and rejected texts of every data.hh_pairs('helpful','test') pair, i.e. of all probe-val
    texts of data.probing_set('helpful','val') (700 pairs). Deviation from 6.1: the spec takes the probe-val keys
    from eval_pairs(..., split='val'), which drops the 2 test pairs whose rejected text does not start with the chosen
    context (698 items) -- those 2 probe-val prompts would not be excluded; adding the rejected contexts is a further
    superset (same key for every regular pair). Prints counts only; item texts stay in memory and are never written.
    """
    from . import data       # only here: the rest of the module stays importable without `datasets`

    items = data.eval_pairs("hh_pairwise", "helpful", split="train")
    texts, labels, groups = data.probing_set("helpful", "train", "pythia", max_samples=probe_max_samples,
                                             seed=probe_seed)
    groups = np.asarray(groups, dtype=np.int64)
    uniq, per = np.unique(groups, return_counts=True)
    n_texts = len(texts)
    if probe_max_samples is not None and (n_texts != probe_max_samples or len(uniq) * 2 != probe_max_samples
                                          or not (per == 2).all()):
        raise RuntimeError(f"probe-train reconstruction: {n_texts} texts in {len(uniq)} groups (pair sizes "
                           f"{sorted(set(per.tolist()))}); expected {probe_max_samples} texts in "
                           f"{probe_max_samples // 2} whole pairs (F6)")
    pairs = data.hh_pairs("helpful", "train")
    train_keys = _pair_prompt_keys(pairs[int(g)] for g in uniq.tolist())
    del pairs, texts, labels
    val_pairs = data.hh_pairs("helpful", "test")
    val_keys = _pair_prompt_keys(val_pairs)
    n_val_items = len(data.eval_pairs("hh_pairwise", "helpful", split="val"))
    split = build_split(items, uniq.tolist(), train_keys, val_keys, n_means_ref=n_means_ref,
                        n_means_probe=n_means_probe, n_eval=n_eval, seed=seed)
    split["probe_train"] = {"rule": f"data.probing_set('helpful','train','pythia',max_samples={probe_max_samples},"
                                    f"seed={probe_seed})", "n_texts": int(n_texts), **split["probe_train"]}
    split["probe_val"] = {"rule": "contexts of chosen and rejected of data.hh_pairs('helpful','test') (= the probe-val "
                                  "pairs of data.probing_set('helpful','val'), HF test)",
                          "n_items": len(val_pairs), "n_eval_pairs_val": n_val_items, **split["probe_val"]}
    if verbose:
        c = split["counts"]
        print(f"[split] items {c['items']}, not in probe-train {c['not_in_probe_train']}, dropped "
              + ", ".join(f"{k} {v}" for k, v in c["dropped"].items())
              + f"; pool {c['pool']} in {c['clusters']} prompt clusters -> means_ref {c['means_ref']}, means_probe "
              f"{c['means_probe']}, eval {c['eval']}, reserve {c['reserve']} (probe-train {n_texts} texts / "
              f"{len(uniq)} groups, probe-val {len(val_pairs)} pairs)")
    return split, {int(it["group"]): it for it in items}


def _dumps(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default) + "\n"


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _atomic_write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_split(path, split, force=False):
    """Atomic write of the split file; returns the file's sha256.

    The split holds no wall-clock timestamp, so two writers (both notebooks) produce identical bytes. An existing file
    with identical bytes is left untouched; one with different content raises SplitMismatch (the frozen sets and every
    condition refer to it) unless force=True.
    """
    text = _dumps(split)
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            old = f.read()
        if old == text:
            return overlap.file_sha256(path)
        raise SplitMismatch(f"{path} exists with a different split (counts {json.loads(old).get('counts')} vs "
                            f"{split.get('counts')}); delete it or pass force=True")
    _atomic_write_text(path, text)
    return overlap.file_sha256(path)


def load_split(path):
    """The split dict of a split file (version and key checked; role ids as ints)."""
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    if split.get("version") != SPLIT_VERSION or split.get("key") != SPLIT_KEY:
        raise ValueError(f"{path}: split version/key {split.get('version')}/{split.get('key')} != "
                         f"{SPLIT_VERSION}/{SPLIT_KEY}")
    split["roles"] = {r: [int(g) for g in ids] for r, ids in split["roles"].items()}
    return split


def verify_split(split, items_by_id):
    """Recompute item_sha and prompt_key of every listed id; raises ValueError on a missing or changed item."""
    bad = []
    for role in SPLIT_ROLES:
        for g in split["roles"].get(role, []):
            it = items_by_id.get(int(g))
            if it is None:
                bad.append(f"{g}: missing")
            elif item_sha(it) != split["item_sha"][str(g)]:
                bad.append(f"{g}: item_sha")
            elif prompt_key(it["context"]) != split["cluster"][str(g)]:
                bad.append(f"{g}: prompt_key")
    if bad:
        raise ValueError(f"split verification failed for {len(bad)} items (first: {bad[:5]}); the data source changed")


def split_items(split, items_by_id, role, n=None):
    """Items of `role` (means_ref | means_probe | eval) in walk order; n -> the first n (a prefix, F5)."""
    if role not in SPLIT_ROLES:
        raise ValueError(f"role must be one of {SPLIT_ROLES}")
    ids = split["roles"][role]
    if n is not None:
        if n > len(ids):
            raise ValueError(f"{role}: {n} requested, the split holds {len(ids)}")
        ids = ids[:n]
    return [items_by_id[int(g)] for g in ids]


def cluster_ids(split, ids):
    """Prompt-cluster key (str) of every id, for cluster bootstraps."""
    return np.asarray([split["cluster"][str(int(g))] for g in ids])


# ----------------------------------------------------------------------------------------------------------------
# head sets (6.2)
# ----------------------------------------------------------------------------------------------------------------

BAND = (20, 27)
SET_ORDER = ("C33", "O", "CminusO", "P_band", "P_DoM", "C33_readers", "C33_brakes", "WholeBand", "TopK_all",
             "CR20", "CensusReaders", "CB24")
ICL_SETS = ("C33", "C33_readers", "C33_brakes", "WholeBand", "CR20", "CensusReaders", "CB24")   # identity = ICL base
SMOKE_C33_SIZE = 8
SMOKE_RULE = "SMOKE-ONLY synthetic stand-in (no named circuit for this model)"


def parse_layers(spec):
    """'20-27' -> (20, ..., 27); '28,30-31' -> (28, 30, 31); an iterable of ints is passed through (sorted, unique)."""
    if isinstance(spec, str):
        out = set()
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                if b < a:
                    raise ValueError(f"bad layer range {part!r}")
                out.update(range(a, b + 1))
            else:
                out.add(int(part))
        return tuple(sorted(out))
    return tuple(sorted({int(x) for x in spec}))


def band_layers(band):
    """Layers of a band given as (lo, hi) (inclusive, like BAND and the frozen file's "band") or as a string."""
    if isinstance(band, str):
        return parse_layers(band)
    lo, hi = (int(x) for x in band)
    if hi < lo:
        raise ValueError(f"bad band {band!r}")
    return tuple(range(lo, hi + 1))


def _heads(heads):
    """Sorted unique (layer, head) tuples from (l, h) pairs, a {layer: [heads]} dict or a {(l, h): value} dict."""
    if isinstance(heads, dict):
        if all(isinstance(k, tuple) for k in heads):
            heads = list(heads)
        else:
            heads = [(l, h) for l, hs in heads.items() for h in hs]
    return sorted({(int(l), int(h)) for l, h in heads})


def to_flat(heads, n_heads):
    """[(l, h), ...] -> [l * n_heads + h, ...] (input order)."""
    return [int(l) * int(n_heads) + int(h) for l, h in heads]


def from_flat(flat, n_heads):
    """[flat, ...] -> [(l, h), ...] (input order)."""
    return [(int(f) // int(n_heads), int(f) % int(n_heads)) for f in flat]


def layer_profile(heads):
    """{layer: number of heads}."""
    out = {}
    for l, _ in _heads(heads):
        out[l] = out.get(l, 0) + 1
    return dict(sorted(out.items()))


def kv_profile(heads, heads_per_kv):
    """{layer: tuple of per-KV-group head counts, descending, zeros dropped} (query head h reads group h // hpk)."""
    groups = {}
    for l, h in _heads(heads):
        key = (l, h // int(heads_per_kv))
        groups[key] = groups.get(key, 0) + 1
    out = {}
    for (l, _), c in groups.items():
        out.setdefault(l, []).append(c)
    return {l: tuple(sorted(cs, reverse=True)) for l, cs in sorted(out.items())}


def same_profile(a, b, heads_per_kv):
    """True if a and b have the same per-layer counts and the same per-layer KV-count multisets (equal null)."""
    return layer_profile(a) == layer_profile(b) and kv_profile(a, heads_per_kv) == kv_profile(b, heads_per_kv)


def census_ranked(effects, members, k, largest=False):
    """The first k of `members` sorted by (dret, l, h) ascending (largest=True: by (-dret, l, h)); NaN effects last.

    effects: {(l, h): dret} (icl_sets.CensusResult.effects("dret")); readers -> most negative first (CR20, F22),
    brakes with largest=True -> most positive first (CB24)."""
    def key(lh):
        e = float(effects[lh])
        if np.isnan(e):
            return (1, 0.0, lh[0], lh[1])
        return (0, -e if largest else e, lh[0], lh[1])
    return sorted(_heads(members), key=key)[:int(k)]


def named_c33(icl_identity):
    """The 33 named-circuit heads of layers 20-27 for an ICL identity, or None (e.g. qwen3-0.6b-base: smoke)."""
    from . import icl_sets
    role = icl_sets.NAMED.get(icl_identity, {}).get("named_circuit_L20_27")
    return None if role is None else list(role.heads)


def smoke_c33(n_layers, n_heads, band=BAND, seed=42, size=SMOKE_C33_SIZE):
    """SMOKE-ONLY synthetic stand-in for C33: `size` distinct band heads from overlap._rng(seed, "ablation-smoke-c33")."""
    layers = [l for l in band_layers(band) if 0 <= l < n_layers]
    cand = [(l, h) for l in layers for h in range(n_heads)]
    if size > len(cand):
        raise ValueError(f"smoke C33 of {size} heads does not fit into {len(cand)} band heads")
    pick = overlap._rng(seed, "ablation-smoke-c33").choice(len(cand), size, replace=False)
    return sorted(cand[i] for i in pick)


def select_p_band(topk, c33, o, band, *, seed, identity):
    """P_band: probe top-k heads of the band outside C33, subsampled to |O| with O's per-layer counts where possible.

    1. targets t_l = layer_profile(O); 2. candidates cand_l = sorted((topk & layer l) - C33) for band layers;
    3. rng = overlap._rng(seed, "ablation-pband-v1", identity);
    4. pass 1 (ascending l): min(t_l, |cand_l|) heads from cand_l via rng.choice(len, k, replace=False);
    5. pass 2: every missing slot of a layer with a deficit (ascending l) is filled from the nearest layer l' that still
       has candidates, by (|l' - l|, l') (ties -> lower layer), with a uniform rng.integers; each move (l, l') recorded;
    6. if the candidates run out, P_band is smaller than |O| (info["short_by"]).
    Returns (sorted heads, info)."""
    layers = band_layers(band)
    c33s = set(_heads(c33))
    top = _heads(topk)
    cand = {l: sorted(lh for lh in top if lh[0] == l and lh not in c33s) for l in layers}
    n_cand = sum(len(v) for v in cand.values())
    per_layer_cand = {str(l): len(v) for l, v in cand.items()}
    target = layer_profile(o)
    rng = overlap._rng(seed, "ablation-pband-v1", identity)
    chosen, deficit = [], {}
    for l in sorted(target):
        pool = cand.get(l, [])
        k = min(target[l], len(pool))
        if k:
            idx = {int(i) for i in rng.choice(len(pool), k, replace=False)}
            chosen += [pool[i] for i in sorted(idx)]
            cand[l] = [x for i, x in enumerate(pool) if i not in idx]
        if target[l] > k:
            deficit[l] = target[l] - k
    moves = []
    for l in sorted(deficit):
        for _ in range(deficit[l]):
            avail = [m for m in cand if cand[m]]
            if not avail:
                break
            src = min(avail, key=lambda m: (abs(m - l), m))
            i = int(rng.integers(len(cand[src])))
            chosen.append(cand[src].pop(i))
            moves.append([int(l), int(src)])
    heads = sorted(chosen)
    info = {"candidates": n_cand, "per_layer_candidates": per_layer_cand,
            "target": {str(l): c for l, c in target.items()}, "moves": moves,
            "short_by": int(sum(target.values()) - len(heads))}
    return heads, info


def select_p_dom(dom_norm, n_heads, band, exclude, size):
    """P_DoM (F13): the top-`size` band heads by raw ||DoM||_2 that are not in `exclude` (C33); ties -> lower flat index.

    dom_norm: (L * H,) per-head DoM norms (ablation.dom_norms(refs.dom)); NaN heads are never selected.
    Returns (sorted heads, info) with the selected norms and the scale caveat."""
    dom_norm = np.asarray(dom_norm, dtype=np.float64).ravel()
    n_layers = len(dom_norm) // int(n_heads)
    ex = set(_heads(exclude))
    cand = [(l, h) for l in band_layers(band) if 0 <= l < n_layers for h in range(n_heads)
            if (l, h) not in ex and np.isfinite(dom_norm[l * n_heads + h])]
    ranked = sorted(cand, key=lambda lh: (-dom_norm[lh[0] * n_heads + lh[1]], lh[0] * n_heads + lh[1]))
    heads = sorted(ranked[:int(size)])
    sel = {str(l * n_heads + h): float(dom_norm[l * n_heads + h]) for l, h in heads}
    cutoff = float(dom_norm[ranked[int(size) - 1][0] * n_heads + ranked[int(size) - 1][1]]) if size and len(ranked) >= size \
        else float("nan")
    return heads, {"dom_norm": sel, "candidates": len(cand), "cutoff": cutoff, "short_by": int(size) - len(heads),
                   "note": "raw norm, scale dependent"}


def _entry(heads, n_heads, identity, rule, info=None, synthetic=False):
    heads = _heads(heads)
    e = {"heads": heads, "flat": to_flat(heads, n_heads), "n": len(heads), "identity": identity, "rule": rule,
         "info": info or {}}
    if synthetic:
        e["synthetic"] = True
    return e


def named_sets(*, topk, c33, census_roles, census_effects, dom_norm, n_layers, n_heads, band=BAND, seed=42,
               model_identity, icl_identity, c33_synthetic=False):
    """All named sets of 6.2 as {name: {"heads", "flat", "n", "identity", "rule", "info"[, "synthetic"]}}, SET_ORDER.

    topk:           TopK_all = overlap.load_value_table(table, "helpful", "pythia").topk (model_identity's table)
    c33:            the 33 named-circuit heads (named_c33(icl_identity)); None -> SMOKE-ONLY synthetic stand-in
                    (smoke_c33, identity "smoke", "synthetic": True; also for C33_readers / C33_brakes)
    census_roles:   icl_sets.CensusResult.roles of the ICL identity (None -> census sets omitted)
    census_effects: CensusResult.effects("dret")
    dom_norm:       (L * H,) per-head DoM norms on means_ref (None -> P_DoM omitted)
    Identities (F11): ICL_SETS use icl_identity; TopK_all, O, CminusO, P_band, P_DoM use model_identity.
    """
    H = int(n_heads)
    synthetic = c33 is None or bool(c33_synthetic)
    if c33 is None:
        c33 = smoke_c33(n_layers, H, band, seed)
    c33 = _heads(c33)
    top = _heads(topk)
    c_id = "smoke" if synthetic else icl_identity
    c_rule = SMOKE_RULE if synthetic else "icl_sets named_circuit_L20_27"
    o = sorted(set(c33) & set(top))
    lo, hi = band_layers(band)[0], band_layers(band)[-1]
    sets = {"C33": _entry(c33, H, c_id, c_rule, {"band": [lo, hi]}, synthetic),
            "O": _entry(o, H, model_identity, "C33 & TopK_all", {"n_topk": len(top), "n_c33": len(c33)}),
            "CminusO": _entry(sorted(set(c33) - set(o)), H, model_identity, "C33 - O")}
    pb, pb_info = select_p_band(top, c33, o, band, seed=seed, identity=model_identity)
    sets["P_band"] = _entry(pb, H, model_identity, f"TopK_all in layers {lo}-{hi} minus C33, seeded subsample to |O| "
                            "with O's per-layer counts (nearest-layer fill)", pb_info)
    if dom_norm is not None:
        pd_heads, pd_info = select_p_dom(dom_norm, H, band, c33, len(o))
        sets["P_DoM"] = _entry(pd_heads, H, model_identity, f"top-|O| heads of layers {lo}-{hi} by raw ||DoM|| on "
                               "means_ref, C33 excluded (ties: lower flat index)", pd_info)
    if census_roles is not None:
        readers, brakes = _heads(census_roles.get("readers", ())), _heads(census_roles.get("brakes", ()))
        sets["C33_readers"] = _entry(sorted(set(c33) & set(readers)), H, c_id, "C33 & census readers (dret_hi < 0)",
                                     synthetic=synthetic)
        sets["C33_brakes"] = _entry(sorted(set(c33) & set(brakes)), H, c_id, "C33 & census brakes (dret_lo > 0)",
                                    synthetic=synthetic)
    sets["WholeBand"] = _entry([(l, h) for l in band_layers(band) if 0 <= l < n_layers for h in range(H)], H,
                               icl_identity, f"all heads of layers {lo}-{hi}")
    sets["TopK_all"] = _entry(top, H, model_identity, "overlap.load_value_table(table, 'helpful', 'pythia').topk",
                              {"k": len(top)})
    if census_roles is not None:
        eff = {(int(l), int(h)): float(v) for (l, h), v in census_effects.items()}
        cr = census_ranked(eff, readers, 20)
        cb = census_ranked(eff, brakes, 24, largest=True)
        sets["CR20"] = _entry(cr, H, icl_identity, "20 census readers with the most negative dret (F22)",
                              {"dret": {f"{l}:{h}": eff[(l, h)] for l, h in cr}, "sum_dret": float(sum(eff[x] for x in cr)),
                               "short_by": 20 - len(cr)})
        sets["CensusReaders"] = _entry(readers, H, icl_identity, "all census readers (dret_hi < 0)")
        sets["CB24"] = _entry(cb, H, icl_identity, "24 census brakes with the largest dret",
                              {"dret": {f"{l}:{h}": eff[(l, h)] for l, h in cb}, "sum_dret": float(sum(eff[x] for x in cb)),
                               "short_by": 24 - len(cb)})
    return {name: sets[name] for name in SET_ORDER if name in sets}


# ----------------------------------------------------------------------------------------------------------------
# random sets (6.3)
# ----------------------------------------------------------------------------------------------------------------

def random_draws(heads, *, n_layers, n_heads, heads_per_kv, seed, identity, set_name, R, start=0):
    """[sorted flat list for r in range(start, start + R)]: layer- and KV-group-matched random sets (F10).

    Draw r = overlap.layer_matched_draws(heads, n_layers, n_heads, 1, rng_r, heads_per_kv) with
    rng_r = overlap._rng(seed, "ablation-random-v1", identity, set_name, r): it depends only on (seed, identity, set, r),
    so a longer null extends a shorter one (prefix property) and models sharing an identity share draws. Draws sample
    from all heads (they may repeat or overlap named sets, as in a randomisation test)."""
    hs = _heads(heads)
    out = []
    for r in range(int(start), int(start) + int(R)):
        rng = overlap._rng(seed, "ablation-random-v1", identity, set_name, r)
        M = overlap.layer_matched_draws(hs, int(n_layers), int(n_heads), 1, rng, heads_per_kv=int(heads_per_kv))
        out.append([int(i) for i in np.flatnonzero(M[0])])
    return out


def r_table(r_c33=100, r_o=100, r_small=50, r_sec=20):
    """Number of stored random draws per named set: the largest R any arm of the 8.3 table uses for it."""
    return {"C33": r_c33, "O": r_o, "CminusO": r_small, "P_band": r_small, "P_DoM": r_sec, "C33_readers": r_sec,
            "C33_brakes": r_sec, "WholeBand": 0, "TopK_all": r_sec, "CR20": r_sec, "CensusReaders": r_sec,
            "CB24": r_sec}


def pband_reuses_o(sets, heads_per_kv):
    """F12: P_band shares O's null iff both are non-empty with the same per-layer counts and KV-count multisets."""
    if "P_band" not in sets or "O" not in sets:
        return False
    a, b = sets["P_band"]["heads"], sets["O"]["heads"]
    return len(a) > 0 and len(a) == len(b) and same_profile(a, b, heads_per_kv)


def random_table(sets, R, *, n_layers, n_heads, heads_per_kv, seed):
    """The frozen file's "random" block: {name: {"R", "identity", "draws"}} or {"reuse": "O"} for P_band (F12).

    R: {name: number of draws} (r_table(...)); sets missing from R get 0 draws."""
    out = {}
    for name, s in sets.items():
        if name == "P_band" and pband_reuses_o(sets, heads_per_kv):
            out[name] = {"reuse": "O"}
            continue
        r = int(R.get(name, 0))
        out[name] = {"R": r, "identity": s["identity"],
                     "draws": random_draws(s["heads"], n_layers=n_layers, n_heads=n_heads, heads_per_kv=heads_per_kv,
                                           seed=seed, identity=s["identity"], set_name=name, R=r)}
    return out


# ----------------------------------------------------------------------------------------------------------------
# frozen-set file (6.4)
# ----------------------------------------------------------------------------------------------------------------

FROZEN_VERSION = 1


class FrozenMismatch(RuntimeError):
    """An existing frozen-set file holds different sets / random draws."""


def build_frozen(*, model_short, icl_identity, geometry, band, seed, sets, randoms, inputs, smoke=False):
    """Payload of results/ablation/{short}_frozen_sets.json (6.4).

    geometry: {"layers", "heads", "kv_heads", "heads_per_kv", "head_dim"}; sets: named_sets(...); randoms:
    random_table(...); inputs: {"heads_table", "census", "refs", "split"} with paths / sha256 (D). Every stored draw
    is checked to have its set's layer and KV profile."""
    H, hpk = int(geometry["heads"]), int(geometry["heads_per_kv"])
    out_sets = {}
    for name, s in sets.items():
        heads = _heads(s["heads"])
        e = {"heads": [[l, h] for l, h in heads], "flat": to_flat(heads, H), "n": len(heads),
             "identity": s["identity"], "rule": s["rule"],
             "layers": {str(l): c for l, c in layer_profile(heads).items()},
             "kv": {str(l): list(c) for l, c in kv_profile(heads, hpk).items()}, "info": s.get("info", {})}
        if s.get("synthetic"):
            e["synthetic"] = True
        out_sets[name] = e
    out_rand = {}
    for name, r in randoms.items():
        if name not in sets:
            raise ValueError(f"random draws for unknown set {name!r}")
        if "reuse" in r:
            tgt = r["reuse"]
            if tgt not in sets or not same_profile(sets[name]["heads"], sets[tgt]["heads"], hpk):
                raise ValueError(f"{name}: reuse of {tgt!r} needs the same layer / KV profile")
            out_rand[name] = {"reuse": tgt}
            continue
        prof = (layer_profile(sets[name]["heads"]), kv_profile(sets[name]["heads"], hpk))
        draws = [[int(i) for i in d] for d in r["draws"]]
        for i, d in enumerate(draws):
            hd = from_flat(d, H)
            if (layer_profile(hd), kv_profile(hd, hpk)) != prof:
                raise ValueError(f"{name} draw {i}: layer / KV profile differs from the set's")
        out_rand[name] = {"R": len(draws), "identity": r.get("identity", sets[name]["identity"]), "draws": draws}
        if r.get("extended"):
            out_rand[name]["extended"] = list(r["extended"])
    lo, hi = band_layers(band)[0], band_layers(band)[-1]
    return {"version": FROZEN_VERSION, "model": model_short, "icl_identity": icl_identity, "smoke": bool(smoke),
            "created": _dt.datetime.now().isoformat(timespec="seconds"), "seed": int(seed),
            "geometry": {k: int(v) for k, v in geometry.items()}, "band": [lo, hi], "inputs": inputs,
            "sets": out_sets, "random": out_rand}


def _canon(x):
    return json.loads(json.dumps(x, default=_json_default))


def _compare_frozen(old, new):
    """('same' | 'extends' | 'mismatch', reason). Sets must agree in heads and identity; random draws must be
    prefix-compatible ('extends' = the new payload adds draws; a shorter new payload is 'same': the stored one wins)."""
    old, new = _canon(old), _canon(new)
    if set(old.get("sets", {})) != set(new.get("sets", {})):
        return "mismatch", f"set names differ: {sorted(old.get('sets', {}))} vs {sorted(new.get('sets', {}))}"
    for name, o in old["sets"].items():
        n = new["sets"][name]
        if o["heads"] != n["heads"] or o["identity"] != n["identity"] or bool(o.get("synthetic")) != bool(n.get("synthetic")):
            return "mismatch", f"set {name} differs (heads / identity)"
    if set(old.get("random", {})) != set(new.get("random", {})):
        return "mismatch", "random-draw blocks differ in their sets"
    extends = False
    for name, o in old["random"].items():
        n = new["random"][name]
        if ("reuse" in o) != ("reuse" in n) or o.get("reuse") != n.get("reuse"):
            return "mismatch", f"random {name}: reuse differs"
        if "reuse" in o:
            continue
        if o.get("identity") != n.get("identity"):
            return "mismatch", f"random {name}: identity differs"
        a, b = o["draws"], n["draws"]
        m = min(len(a), len(b))
        if a[:m] != b[:m]:
            return "mismatch", f"random {name}: draws differ"
        extends |= len(b) > len(a)
    return ("extends", "") if extends else ("same", "")


def write_frozen(path, payload, force=False):
    """Atomic write of the frozen-set file; returns its sha256.

    An existing file whose sets / random draws differ raises FrozenMismatch unless force=True. If they agree (or the
    stored draws extend the new ones), the stored file is kept unchanged (frozen = first write wins); if the new
    payload extends the stored draws (extend_random, prefix property), the file is rewritten."""
    if os.path.exists(path) and not force:
        status, reason = _compare_frozen(load_frozen(path), payload)
        if status == "mismatch":
            raise FrozenMismatch(f"{path}: {reason}; the frozen sets were fixed before the evaluation "
                                 "(use --refreeze only if no EVAL condition exists)")
        if status == "same":
            return overlap.file_sha256(path)
    _atomic_write_text(path, _dumps(payload))
    return overlap.file_sha256(path)


def load_frozen(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("version") != FROZEN_VERSION:
        raise ValueError(f"{path}: frozen-set version {payload.get('version')} != {FROZEN_VERSION}")
    return payload


def set_heads(frozen, name):
    """[(l, h)] of a named set of a frozen payload."""
    return [(int(l), int(h)) for l, h in frozen["sets"][name]["heads"]]


def _random_entry(frozen, name):
    entry = frozen["random"][name]
    seen = {name}
    while "reuse" in entry:
        name = entry["reuse"]
        if name in seen:
            raise ValueError(f"reuse cycle at {name}")
        seen.add(name)
        entry = frozen["random"][name]
    return name, entry


def random_R(frozen, name):
    """Number of stored draws of a set (following "reuse")."""
    return len(_random_entry(frozen, name)[1]["draws"])


def random_heads(frozen, name, r):
    """[(l, h)] of random draw r of a set (follows "reuse"; IndexError beyond the stored R -> extend_random)."""
    target, entry = _random_entry(frozen, name)
    draws = entry["draws"]
    if not 0 <= int(r) < len(draws):
        raise IndexError(f"random draw {r} of {name} (stored in {target}): only {len(draws)} frozen "
                         "(extend with --extend_random)")
    return from_flat(draws[int(r)], frozen["geometry"]["heads"])


def extend_random(frozen, name, R_new):
    """A copy of `frozen` whose draws for `name` (or the set it reuses) are extended to R_new (draws r >= stored R are
    appended with the same generator keys, so the stored ones stay; prefix property). Records {"from", "to"} under
    "extended". R_new <= stored R returns an unchanged copy."""
    out = copy.deepcopy(frozen)
    target, entry = _random_entry(out, name)
    R_old = len(entry["draws"])
    if R_new <= R_old:
        return out
    g = out["geometry"]
    entry["draws"] += random_draws(set_heads(out, target), n_layers=g["layers"], n_heads=g["heads"],
                                   heads_per_kv=g["heads_per_kv"], seed=out["seed"], identity=entry["identity"],
                                   set_name=target, R=R_new - R_old, start=R_old)
    entry["R"] = len(entry["draws"])
    entry.setdefault("extended", []).append({"from": R_old, "to": int(R_new)})
    return out
