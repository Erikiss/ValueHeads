"""`probe-stats`: bootstrap CIs, permutation null, significant set and confirmation split for a probed value.

    python -m valueheads_pythia.cli probe-stats --model pythia-160m --revision step143000 --value honesty --bootstrap 1000 --perms 100
    python -m valueheads_pythia.cli probe-stats ... --perms 20 --bootstrap 200          # smoke
    python -m valueheads_pythia.cli probe-stats ... --features head+mean --perm_global  # HelpSteer2-style stratified permutation

Reads results/heads/{model}_{revision}_{value}_{mode}[_{tag}].json written by `probe`, refits the probes once
(cached val scores in *_valscores.npz) and rewrites the JSON with ci_lo / ci_hi / p_adj / p_head / signif columns
and the stats meta (perm_threshold, n_signif, signif_heads, confirmation). Idempotent: skips when stats exist.
"""
import os

from .. import config
from ..cli import add_common_args
from ..extract import is_done
from ..probing_stats import run_stats
from .probe import FEATURE_TAGS, MODE_DEFAULTS, combined_tag, derived_feature_tag, resolve_feature_dirs


def add_args(parser):
    add_common_args(parser, device=False)
    parser.add_argument("--value", required=True, choices=config.ALL_VALUES)
    parser.add_argument("--features", default=None, choices=["head", "head+mean", "both"], help="which probe table(s); default: mode default")
    parser.add_argument("--bootstrap", type=int, default=1000, help="B cluster-bootstrap resamples (DESIGN 5.3)")
    parser.add_argument("--perms", type=int, default=100, help="P label permutations (DESIGN 5.4; smoke: 20; 0 = skip)")
    parser.add_argument("--perm_global", action="store_true",
                        help="permute labels globally, stratified by group size (default for HelpSteer2 values, DESIGN 5.4)")
    parser.add_argument("--perm_within", action="store_true",
                        help="permute labels within groups (default for TruthfulQA / HH values, whose groups hold both labels)")
    parser.add_argument("--statistic", default="max_auc", choices=["max_auc", "max_acc"])
    parser.add_argument("--stats_seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="device of the batched torch logistic regression (cpu | cuda)")
    parser.add_argument("--torch_iters", type=int, default=200, help="L-BFGS iterations of the permutation refits")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--feature_tag", default=None, help="tag of the probe table(s) to augment (see `probe --feature_tag`)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def permute_within_group(value, perm_global=False, perm_within=False):
    """Permutation scheme of DESIGN 5.4: within groups for TruthfulQA questions and HH dialogue pairs (every group holds
    both labels), globally stratified by group size for HelpSteer2 (one label per prompt is common, and subsampled
    smoke runs leave many single-row prompts, where a within-group permutation would be a no-op)."""
    if perm_global and perm_within:
        raise SystemExit("[probe-stats] --perm_global and --perm_within exclude each other")
    if perm_global:
        return False
    if perm_within:
        return True
    return value not in config.HELPSTEER_VALUES_EXT


def stats_targets(p, short, revision, value, mode, features, run_tag=None):
    """[(heads_json, feature_tag)] for the requested feature modes (mode default = primary = untagged file; `run_tag`
    = the tag a `probe --train_dir / --feature_tag` run wrote, combined with the feature-mode tag)."""
    defaults = MODE_DEFAULTS[mode]
    features = features or defaults["features"]
    modes = ["head", "head+mean"] if features == "both" else [features]
    primary = defaults["features"]
    out = []
    for fm in modes:
        tag = combined_tag(run_tag, None if fm == primary else FEATURE_TAGS[fm])
        out.append((config.heads_path(p, short, revision, value, mode, tag), tag))
    return out


def run(args):
    p = config.paths(args.root)
    short, _, _ = config.resolve_model(args.model)
    train_dir, val_dir, label_value = resolve_feature_dirs(p, short, args.revision, args.value, args.mode, args.train_dir, args.val_dir)
    for d in (train_dir, val_dir):
        if not is_done(d):
            raise SystemExit(f"[probe-stats] feature directory not complete: {d} (run `extract` first)")
    device = "cuda" if args.device == "auto" else args.device
    run_tag = derived_feature_tag(args.train_dir, args.value, args.mode, args.feature_tag)
    within_group = permute_within_group(args.value, args.perm_global, args.perm_within)
    rc = 0
    for path, tag in stats_targets(p, short, args.revision, args.value, args.mode, args.features, run_tag):
        if not os.path.exists(path):
            print(f"[probe-stats] missing {path} (run `probe` first)")
            rc = 1
            continue
        run_stats(path, train_dir, val_dir, value=label_value, B=args.bootstrap, P=args.perms, within_group=within_group,
                  statistic=args.statistic, seed=args.stats_seed, device=device, n_jobs=args.n_jobs,
                  max_iter_torch=args.torch_iters, force=args.force, verbose=not args.quiet)
    return rc
