"""`probe`: one logistic-regression probe per head, ranking and top-k, written to results/heads/.

    python -m valueheads_pythia.cli probe --model pythia-160m --revision step143000 --value honesty --features both
    python -m valueheads_pythia.cli probe ... --value coherence_strict --class_weight balanced
    python -m valueheads_pythia.cli probe ... --mode paper            # head+mean features, accuracy, no class weight, argsort ranking
    python -m valueheads_pythia.cli probe ... --value helpfulness --cv 5   # grouped 5-fold CV over train + val -> *_cv5.json

Mode defaults (DESIGN 2, 5.1-5.2, 5.8): pythia = head-only features, AUC, class_weight balanced, stable tie-break
ranking; paper = [head ; layer mean] features, accuracy, no class weight, `np.argsort(accs)[::-1]` ranking
(identical to the reference repo, including the top-k set at accuracy ties).
"""
import os
import re
import time

from .. import config
from ..cli import add_common_args
from ..extract import is_done
from ..probing import cv_head_probes, fit_head_probes, group_by_layer, save_head_scores, select_topk, sibling_val_dir

MODE_DEFAULTS = {
    "pythia": {"features": "head", "score": "auc", "class_weight": "balanced", "ranking": "stable"},
    "paper": {"features": "head+mean", "score": "acc", "class_weight": "none", "ranking": "paper"},
}
FEATURE_TAGS = {"head": "head", "head+mean": "headmean"}


def add_args(parser):
    add_common_args(parser, device=False)
    parser.add_argument("--value", required=True, choices=config.ALL_VALUES)
    parser.add_argument("--features", default=None, choices=["head", "head+mean", "both"], help="default: mode default")
    parser.add_argument("--score", default=None, choices=["auc", "acc"], help="ranking score (default: mode default)")
    parser.add_argument("--class_weight", default=None, choices=["none", "balanced"], help="default: mode default")
    parser.add_argument("--ranking", default=None, choices=["stable", "paper"],
                        help="tie-break rule: stable (rounded score, bal_acc, flat_idx) | paper (np.argsort order); default: mode default")
    parser.add_argument("--cv", type=int, default=0,
                        help="grouped stratified K-fold over train + val instead of the train/val fit (DESIGN 5.3, HelpSteer2 values); "
                             "writes {...}_cv{K}.json next to the primary table")
    parser.add_argument("--pca", type=int, default=None,
                        help="project every head's features onto its first N train principal components before the fit (DESIGN 3.5 "
                             "capacity equalization, e.g. 64 for pythia-1b); result tagged pca{N} unless --feature_tag is given")
    parser.add_argument("--C", type=float, default=1.0, help="inverse L2 regularization of the probes (default 1.0 as in the repo)")
    parser.add_argument("--ratio", type=float, default=config.TOP_RATIO)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="restrict to these layers")
    parser.add_argument("--train_dir", default=None, help="override the train feature directory")
    parser.add_argument("--val_dir", default=None, help="override the val feature directory")
    parser.add_argument("--feature_tag", default=None,
                        help="suffix of the result file ({model}_{rev}_{value}_{mode}_{tag}.json); default: none for the "
                             "standard feature directory, else derived from --train_dir (honesty_split1_train -> split1)")
    parser.add_argument("--force", action="store_true", help="recompute even if the result JSON exists")
    parser.add_argument("--quiet", action="store_true")


def derived_feature_tag(train_dir, value, mode, feature_tag=None):
    """Result-file tag of a probe run: the explicit --feature_tag, else the part of a --train_dir basename that deviates
    from the value's standard directory (honesty_split1_train -> split1, honesty_train -> None), so that an override
    never overwrites the primary head table."""
    if feature_tag:
        return feature_tag
    if not train_dir:
        return None
    base = os.path.basename(os.path.normpath(train_dir))
    if base.endswith("_train"):
        base = base[:-len("_train")]
    dataset = config.probing_dataset_of(value, mode)
    for prefix in (f"{dataset}_", f"{value}_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    if base in ("", dataset, value):
        return None
    return re.sub(r"[^A-Za-z0-9.-]+", "_", base)


def combined_tag(base_tag, fm_tag):
    """Joins the run-level tag (split seed etc.) with the feature-mode tag (headmean) of a non-primary table."""
    return "_".join(t for t in (base_tag, fm_tag) if t) or None


def resolve_feature_dirs(p, model, revision, value, mode, train_dir=None, val_dir=None):
    """(train_dir, val_dir, label_value): HelpSteer2 values use the extract-once `helpsteer2` dirs in mode pythia
    (falling back to per-value dirs when those exist instead); label_value selects labels_{value}.npy."""
    if train_dir:
        label_value = value if value in config.HELPSTEER_VALUES_EXT and os.path.exists(os.path.join(train_dir, f"labels_{value}.npy")) else None
        return train_dir, val_dir or sibling_val_dir(train_dir), label_value
    dataset = config.probing_dataset_of(value, mode)
    tr = config.feature_dir(p, model, revision, dataset, "train")
    va = config.feature_dir(p, model, revision, dataset, "val")
    label_value = value if dataset == "helpsteer2" else None
    if dataset == "helpsteer2" and not (is_done(tr) and is_done(va)):
        alt_tr, alt_va = (config.feature_dir(p, model, revision, value, s) for s in ("train", "val"))
        if is_done(alt_tr) and is_done(alt_va):
            return alt_tr, alt_va, None
    return tr, va, label_value


def run(args):
    p = config.paths(args.root)
    short, _, _ = config.resolve_model(args.model)
    defaults = MODE_DEFAULTS[args.mode]
    features = args.features or defaults["features"]
    score = args.score or defaults["score"]
    class_weight = args.class_weight or defaults["class_weight"]
    class_weight = None if class_weight == "none" else class_weight
    ranking = args.ranking or defaults["ranking"]
    train_dir, val_dir, label_value = resolve_feature_dirs(p, short, args.revision, args.value, args.mode, args.train_dir, args.val_dir)
    for d in (train_dir, val_dir):
        if not is_done(d):
            raise SystemExit(f"[probe] feature directory not complete: {d} (run `extract` first)")
    modes = ["head", "head+mean"] if features == "both" else [features]
    primary = defaults["features"] if features == "both" else features
    run_tag = derived_feature_tag(args.train_dir, args.value, args.mode, args.feature_tag)
    if args.pca and not args.feature_tag:
        run_tag = combined_tag(run_tag, f"pca{int(args.pca)}")
    if args.C != 1.0 and not args.feature_tag:
        run_tag = combined_tag(run_tag, f"C{args.C:g}".replace(".", "p"))
    if args.cv:
        run_tag = combined_tag(run_tag, f"cv{int(args.cv)}")
    outputs = []
    for fm in modes:
        tag = combined_tag(run_tag, None if fm == primary else FEATURE_TAGS[fm])
        path = config.heads_path(p, short, args.revision, args.value, args.mode, tag)
        if os.path.exists(path) and not args.force:
            print(f"[probe] skip {path}: exists")
            outputs.append(path)
            continue
        t0 = time.time()
        print(f"[probe] {args.value} mode={args.mode} features={fm} score={score} class_weight={class_weight} ranking={ranking} C={args.C:g} "
              f"{'pca=' + str(args.pca) + ' ' if args.pca else ''}"
              f"{'cv=' + str(args.cv) + ' (train + val pooled) ' if args.cv else ''}train={train_dir} val={val_dir}", flush=True)
        if args.cv:
            df = cv_head_probes(train_dir, val_dir, value=label_value, n_splits=args.cv, feature_mode=fm, score=score,
                                class_weight=class_weight, seed=args.seed, n_jobs=args.n_jobs, layers=args.layers,
                                verbose=not args.quiet, ranking=ranking, C=args.C, pca=args.pca)
        else:
            df = fit_head_probes(train_dir, val_dir, value=label_value, feature_mode=fm, score=score, class_weight=class_weight,
                                 seed=args.seed, n_jobs=args.n_jobs, layers=args.layers, verbose=not args.quiet, ranking=ranking,
                                 C=args.C, pca=args.pca)
        meta = {"model": short, "revision": args.revision or "main", "value": args.value, "mode": args.mode,
                "feature_tag": tag, "run_tag": run_tag, "seconds": round(time.time() - t0, 1)}
        save_head_scores(df, path, meta=meta, ratio=args.ratio)
        topk = select_topk(df, args.ratio, score)
        best = df.sort_values("rank").head(5)
        H = df.attrs["heads"]
        print(f"[probe] saved {path} ({time.time() - t0:.1f}s): n_train={int(df.n_train.iloc[0])} n_val={int(df.n_val.iloc[0])} "
              f"majority={df.majority.iloc[0]:.3f} pos_rate={df.pos_rate.iloc[0]:.3f}"
              + (f" cv_splits={df.attrs['cv_splits']} groups={df.attrs['n_groups']}" if args.cv else ""))
        print("[probe] top-5: " + ", ".join(f"L{r.layer}.H{r.head} auc={r.auc:.3f} acc={r.acc:.3f} bal={r.bal_acc:.3f}" for r in best.itertuples()))
        print(f"[probe] top-{len(topk)} ({args.ratio:.0%}) by layer: {group_by_layer(topk, H)}")
        print(f"[probe] heads with auc>0.7: {int((df.auc > 0.7).sum())}, acc>0.8: {int((df.acc > 0.8).sum())}")
        outputs.append(path)
    return 0
