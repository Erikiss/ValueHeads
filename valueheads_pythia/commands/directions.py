"""`directions`: mass-mean value directions (Eq. 2) and sigma per std set -> results/directions/{model}_{revision}_{value}.npz

    python -m valueheads_pythia.cli directions --model pythia-160m --revision step143000 --value honesty

mode pythia: directions from the train features only, sigma on the value's val features (train/val disjoint);
mode paper: directions pooled over train + val (repo), sigma on the val features of STD_SETS[value].
"""
import os
import time

import numpy as np

from .. import config
from ..cli import add_common_args
from ..data import STD_SETS
from ..directions import mass_mean_directions, save_directions, sigma_along, unit_directions
from ..extract import is_done
from .probe import resolve_feature_dirs


def add_args(parser):
    add_common_args(parser, device=False)
    parser.add_argument("--value", required=True, choices=config.ALL_VALUES)
    parser.add_argument("--std_sets", nargs="*", default=None,
                        help="std sets (values or prompt sets) whose val features give sigma; default: STD_SETS[value]")
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--force", action="store_true")


def std_set_dir(p, model, revision, std_set, mode):
    """Val feature directory of a std set (value -> its probing val set, prompt set -> its own features)."""
    if std_set in config.PROMPT_SETS:
        return config.feature_dir(p, model, revision, std_set, "val")
    return resolve_feature_dirs(p, model, revision, std_set, mode)[1]


def run(args):
    p = config.paths(args.root)
    short, _, _ = config.resolve_model(args.model)
    out = config.directions_path(p, short, args.revision, args.value, args.mode)
    if os.path.exists(out) and not args.force:
        print(f"[directions] skip {out}: exists")
        return 0
    train_dir, val_dir, label_value = resolve_feature_dirs(p, short, args.revision, args.value, args.mode, args.train_dir, args.val_dir)
    dirs = [train_dir] if args.mode == "pythia" else [train_dir, val_dir]
    for d in dirs:
        if not is_done(d):
            raise SystemExit(f"[directions] feature directory not complete: {d}")
    t0 = time.time()
    directions = mass_mean_directions(dirs, value=label_value)
    norms = np.linalg.norm(directions, axis=1)
    std_sets = args.std_sets or [STD_SETS[args.value]]
    sigmas = {}
    for s in std_sets:
        d = std_set_dir(p, short, args.revision, s, args.mode)
        if not is_done(d):
            raise SystemExit(f"[directions] std set {s}: feature directory not complete: {d}")
        sigmas[s] = sigma_along(directions, d)
    # The stored `directions` array is the raw mass-mean difference (Eq. 2); the unit normalization of the repo is
    # applied when the steering vectors are built (directions.steering_vectors / eval_likelihood.steering_vectors_from_directions)
    # and sigma is the std along that unit direction (sigma_along).
    meta = {"model": short, "revision": args.revision or "main", "value": args.value, "mode": args.mode,
            "feature_dirs": dirs, "std_sets": {s: std_set_dir(p, short, args.revision, s, args.mode) for s in std_sets},
            "label_value": label_value, "stored_directions": "raw mass-mean (Eq. 2), not normalized",
            "unit_norm": False, "unit_norm_at_use": True, "sigma_ddof": 1, "sigma_along": "unit direction",
            "seconds": round(time.time() - t0, 1)}
    save_directions(out, directions, sigmas, meta)
    unit = unit_directions(directions)
    print(f"[directions] saved {out}: shape {directions.shape}, raw |dir| mean={norms.mean():.4f} max={norms.max():.4f} "
          f"(unit norm applied at use: {np.allclose(np.linalg.norm(unit, axis=1), 1)}); " +
          "; ".join(f"sigma[{s}] mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f}" for s, v in sigmas.items()) +
          f"; {time.time() - t0:.1f}s")
    return 0
