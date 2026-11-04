"""`extract`: last-token head activations of one (dataset, split) into layer shards (idempotent, resumable).

    python -m valueheads_pythia.cli extract --model pythia-160m --revision step143000 --dataset honesty --split train --tier smoke
    python -m valueheads_pythia.cli extract --model pythia-160m --revision step143000 --dataset helpsteer2 --split val --tier smoke
"""
import os
import time

import numpy as np

from .. import config
from ..cli import add_common_args
from ..data import extract_texts
from ..extract import extract_features, is_done, read_manifest
from ..models import load_model, set_seed


def add_args(parser):
    add_common_args(parser)
    parser.add_argument("--dataset", required=True, choices=config.EXTRACT_DATASETS,
                        help="a value, 'helpsteer2' (extract once, labels per value) or a prompt set (mode paper)")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--tier", default=None, choices=config.TIER_NAMES, help="max_samples defaults (DESIGN 6.3)")
    parser.add_argument("--max_samples", type=int, default=None, help="overrides the tier default (0 = all)")
    parser.add_argument("--split_seed", type=int, default=42, help="TruthfulQA question split seed (honesty only)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens_per_batch", type=int, default=16384)
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH,
                        help="tail-keep truncation (Pythia context 2048); 0 = no truncation (paper-mode tokenization, DESIGN 2)")
    parser.add_argument("--out_dir", default=None, help="override the feature directory")
    parser.add_argument("--force", action="store_true", help="rebuild even if DONE exists")
    parser.add_argument("--no_progress", action="store_true")


def run(args):
    set_seed(args.seed)
    p = config.paths(args.root)
    short, hf_id, _ = config.resolve_model(args.model)
    out_dir = args.out_dir or config.feature_dir(p, short, args.revision, args.dataset, args.split)
    if is_done(out_dir) and not args.force:
        man = read_manifest(out_dir)
        print(f"[extract] skip {out_dir}: DONE (n={man['n']}, {man.get('dtype')}/{man.get('attn_impl')})")
        return 0
    max_samples = args.max_samples if args.max_samples is not None else config.max_samples(args.tier, args.dataset, args.split, short)
    if max_samples == 0:
        max_samples = None
    t0 = time.time()
    texts, labels, groups = extract_texts(args.dataset, args.split, args.mode, max_samples, args.seed, args.split_seed)
    if isinstance(labels, dict):
        label_info = ", ".join(f"{v}: {int((l >= 0).sum())} labeled / {int((l == 1).sum())} pos" for v, l in labels.items())
    elif labels is not None:
        label_info = f"pos={int(np.sum(labels == 1))} neg={int(np.sum(labels == 0))}"
    else:
        label_info = "unlabeled"
    n_groups = len(np.unique(groups)) if groups is not None else 0
    print(f"[extract] {args.dataset}/{args.split} mode={args.mode} tier={args.tier}: {len(texts)} texts, {n_groups} groups "
          f"({label_info}); data {time.time() - t0:.1f}s", flush=True)

    model, tokenizer = load_model(hf_id, revision=args.revision, device=args.device, dtype=args.dtype, attn_impl=args.attn_impl)
    info = model.vh_info
    print(f"[extract] {info['hf_id']}@{info['revision']} ({info['commit_sha']}) dtype={info['dtype']} attn={info['attn_impl']} "
          f"device={info['device']} L/H/D={tuple(info['geometry'].values())}", flush=True)
    extra = {"dataset": args.dataset, "split": args.split, "mode": args.mode, "tier": args.tier, "max_samples": max_samples,
             "seed": args.seed, "split_seed": args.split_seed}
    man = extract_features(model, tokenizer, texts, out_dir, batch_size=args.batch_size, max_length=args.max_length,
                           resume=not args.force, labels=labels, groups=groups, max_tokens_per_batch=args.max_tokens_per_batch,
                           extra_manifest=extra, show_progress=not args.no_progress)
    print(f"[extract] done {out_dir}: n={man['n']} tokens={man['n_tokens']} truncated={man['n_truncated']} "
          f"{man['seconds']}s ({man['tokens_per_second']} tok/s); shards layer00..layer{man['layers'] - 1:02d}.npy "
          f"{os.path.getsize(os.path.join(out_dir, 'layer00.npy')) / 1e6:.1f} MB each")
    return 0
