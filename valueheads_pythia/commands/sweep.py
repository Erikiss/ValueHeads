"""`sweep`: checkpoint loop extract -> probe -> probe-stats -> directions (-> eval-likelihood) per revision.

    python -m valueheads_pythia.cli sweep --model pythia-160m --tier t4 --values honesty harmless helpful helpfulness complexity verbosity coherence
    python -m valueheads_pythia.cli sweep --model pythia-160m --smoke --values honesty --perms 20 --bootstrap 200   # CPU smoke
    python -m valueheads_pythia.cli sweep ... --eval_revisions step2000 step143000 --eval_args "--metric tqa_pairwise --intervention mask --select topk --random_sets 3"
    python -m valueheads_pythia.cli sweep ... --cleanup_cache        # delete each HF revision from the cache after its probing
    python -m valueheads_pythia.cli sweep ... --cleanup_features     # delete the non-cacheable local train features after each revision

Every step is idempotent (DONE markers, existing JSON / npz files are skipped; a value listed as done in
results/manifest.json is not touched again, so cleaned-up features are not re-extracted), so an interrupted sweep is
resumed by re-running the same command. Revisions default to config.CHECKPOINTS_DEFAULT (DESIGN 3.4); --smoke uses the
three locally cached revisions step0 / step2000 / step143000 with the smoke tier. After the loop the per-value
sweep curves (results/sweep/{model}_{value}_curve.csv + figures/sweep/*.png) and results/manifest.json
(dtype, attn_impl, versions, commit SHAs per revision) are written.
"""
import argparse
import json
import os
import time

from .. import config
from ..cli import add_common_args, load_command
from ..extract import is_done, read_manifest
from ..hf_cache import cleanup_hf_revision  # noqa: F401  (public name kept here; implementation in hf_cache.py)
from ..probing import fit_head_probes, load_head_scores, save_head_scores
from ..probing_stats import split_stability
from .probe import FEATURE_TAGS, MODE_DEFAULTS, resolve_feature_dirs

DEFAULT_VALUES = ["honesty", "harmless", "helpful", "helpfulness", "complexity", "verbosity", "coherence"]


def add_args(parser):
    add_common_args(parser, model=False)
    parser.add_argument("--model", default="pythia-160m", help="registry short name (config.MODELS) or HF id")
    parser.add_argument("--revisions", nargs="*", default=None, help="HF revisions (default: config.CHECKPOINTS_DEFAULT)")
    parser.add_argument("--smoke", action="store_true", help="revisions step0/step2000/step143000, tier smoke, B=200, P=20")
    parser.add_argument("--values", nargs="+", default=None, help=f"values to localize (default: {' '.join(DEFAULT_VALUES)})")
    parser.add_argument("--tier", default=None, choices=config.TIER_NAMES, help="max_samples defaults (DESIGN 6.3); default smoke")
    parser.add_argument("--max_samples_train", type=int, default=None, help="override the tier default (0 = all)")
    parser.add_argument("--max_samples_val", type=int, default=None)
    parser.add_argument("--features", default=None, choices=["head", "head+mean", "both"], help="probe feature mode(s)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--perms", type=int, default=100)
    parser.add_argument("--no_stats", action="store_true", help="skip probe-stats")
    parser.add_argument("--stats_device", default="cpu", help="device of the permutation refits (cpu | cuda)")
    parser.add_argument("--split_seeds", nargs="*", type=int, default=[], help="extra TruthfulQA question-split seeds for honesty (stability)")
    parser.add_argument("--eval_revisions", nargs="*", default=[], help="revisions that also get eval-likelihood (DESIGN: step2000 step16000 step143000)")
    parser.add_argument("--eval_args", default="--metric tqa_pairwise --intervention mask --select topk",
                        help="arguments passed to eval-likelihood for every --eval_revisions x value (must contain --metric)")
    parser.add_argument("--cleanup_cache", action="store_true", help="delete each revision from the HF cache after its steps")
    parser.add_argument("--keep_revisions", nargs="*", default=["main", "step143000"], help="never deleted from the cache / feature disk")
    parser.add_argument("--cleanup_features", action="store_true",
                        help="after a revision's steps delete its local train feature directories that the Drive cache would not keep "
                             "(everything but honesty and the val splits; DESIGN 6.4: features are deleted after probing)")
    parser.add_argument("--no_curves", action="store_true", help="skip the sweep-curve step at the end")
    parser.add_argument("--dry_run", action="store_true", help="print the plan and exit")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute probes / stats / directions (features are kept)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_progress", action="store_true")


# ----------------------------------------------------------------------------- helpers
def call(command, argv):
    """Runs a CLI command module in-process with its own argument parser (same defaults as the command line)."""
    module = load_command(command)
    parser = argparse.ArgumentParser(prog=f"sweep:{command}")
    module.add_args(parser)
    return module.run(parser.parse_args([str(a) for a in argv]))


def command_available(command):
    name = command.replace("-", "_")
    return os.path.exists(os.path.join(os.path.dirname(__file__), name + ".py"))


def revision_commit_sha(hf_id, revision, feature_manifest=None, cache_dir=None):
    """Commit SHA of a revision: extraction manifest, then the HF cache refs file, then the Hub API."""
    if feature_manifest and feature_manifest.get("commit_sha"):
        return feature_manifest["commit_sha"]
    try:
        from huggingface_hub import constants
        cache_dir = cache_dir or constants.HF_HUB_CACHE
        ref = os.path.join(cache_dir, "models--" + hf_id.replace("/", "--"), "refs", revision or "main")
        if os.path.exists(ref):
            with open(ref) as f:
                return f.read().strip()
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi
        for branch in HfApi().list_repo_refs(hf_id).branches:
            if branch.name == (revision or "main"):
                return branch.target_commit
    except Exception:
        return None
    return None


def feature_dirs_of_revision(p, short, revision, values, mode, split_seeds=()):
    """[(dir, dataset, split)] of every feature directory the sweep steps of a revision use."""
    datasets = []
    for v in values:
        d = config.probing_dataset_of(v, mode)
        if d not in datasets:
            datasets.append(d)
    out = [(config.feature_dir(p, short, revision, d, s), d, s) for d in datasets for s in ("train", "val")]
    if "honesty" in values:
        for seed in split_seeds:
            if seed != 42:
                out += [(config.feature_dir(p, short, revision, f"honesty_split{seed}", s), "honesty", s) for s in ("train", "val")]
    return out


def cacheable_features(dataset, split):
    """DESIGN 6.4: the Drive cache keeps honesty (train + val) and every val split; everything else is volatile."""
    return dataset == "honesty" or split == "val"


def cleanup_feature_dirs(p, short, revision, values, mode, split_seeds=(), dry_run=False):
    """Deletes the non-cacheable local feature directories of a revision (train splits of harmless / helpful / helpsteer2;
    DESIGN 6.4 'Features ... werden nach dem Probing geloescht'). Returns {dir: bytes freed}."""
    import shutil
    freed = {}
    for d, dataset, split in feature_dirs_of_revision(p, short, revision, values, mode, split_seeds):
        if cacheable_features(dataset, split) or not os.path.isdir(d):
            continue
        freed[d] = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
        if not dry_run:
            shutil.rmtree(d)
    return freed


def load_manifest(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"revisions": {}}


def save_manifest(path, manifest):
    """Atomic write, merged with the file on disk: another sweep on the same root (e.g. a second Colab session
    covering other revisions of the same model) keeps its revision entries; ours win for the revisions we hold."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        on_disk = load_manifest(path)
    except (OSError, ValueError):
        on_disk = {"revisions": {}}
    revisions = dict(on_disk.get("revisions") or {})
    revisions.update(manifest.get("revisions") or {})
    manifest["revisions"] = revisions
    manifest["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, path)


def plan(args):
    """List of (revision, value, dataset) triples in execution order plus the resolved settings."""
    revisions = args.revisions or (config.CHECKPOINTS_SMOKE if args.smoke else config.CHECKPOINTS_DEFAULT)
    values = args.values or DEFAULT_VALUES
    for v in values:
        if v not in config.ALL_VALUES:
            raise SystemExit(f"unknown value {v!r}; choose from {config.ALL_VALUES}")
        if v == "harmless_pooled" and args.mode != "paper":
            raise SystemExit("harmless_pooled is only defined in --mode paper")
    if args.eval_revisions and "--metric" not in args.eval_args.split():
        raise SystemExit(f"[sweep] --eval_args must name the metric (--metric ...), got {args.eval_args!r}")
    tier = args.tier or "smoke"
    B = 200 if args.smoke and args.bootstrap == 1000 else args.bootstrap
    P = 20 if args.smoke and args.perms == 100 else args.perms
    steps = [(r, v, config.probing_dataset_of(v, args.mode)) for r in revisions for v in values]
    return {"revisions": revisions, "values": values, "tier": tier, "bootstrap": B, "perms": P, "steps": steps}


# ----------------------------------------------------------------------------- per-revision work
def extract_split(args, short, revision, dataset, split, tier, split_seed=None, out_dir=None):
    argv = ["--model", short, "--revision", revision, "--dataset", dataset, "--split", split, "--tier", tier, "--mode", args.mode,
            "--seed", args.seed, "--device", args.device, "--batch_size", args.batch_size]
    if args.root:
        argv += ["--root", args.root]
    if args.dtype:
        argv += ["--dtype", args.dtype]
    if args.attn_impl:
        argv += ["--attn_impl", args.attn_impl]
    ms = args.max_samples_train if split == "train" else args.max_samples_val
    if ms is not None:
        argv += ["--max_samples", ms]
    if split_seed is not None:
        argv += ["--split_seed", split_seed]
    if out_dir:
        argv += ["--out_dir", out_dir]
    if args.no_progress:
        argv += ["--no_progress"]
    return call("extract", argv)


def extra_split_tables(args, p, short, revision, value, tier):
    """Honesty on additional TruthfulQA question-split seeds: features in honesty_split{seed}_{split}, probes saved with
    feature tag split{seed}. Returns the list of head-table paths (main + extra)."""
    paths = [config.heads_path(p, short, revision, value, args.mode)]
    if value != "honesty" or not args.split_seeds:
        return paths
    defaults = MODE_DEFAULTS[args.mode]
    fm = defaults["features"] if args.features in (None, "both") else args.features
    cw = None if defaults["class_weight"] == "none" else defaults["class_weight"]
    for seed in args.split_seeds:
        if seed == 42:
            continue
        dirs = {s: config.feature_dir(p, short, revision, f"honesty_split{seed}", s) for s in ("train", "val")}
        for s, d in dirs.items():
            extract_split(args, short, revision, "honesty", s, tier, split_seed=seed, out_dir=d)
        out = config.heads_path(p, short, revision, value, args.mode, f"split{seed}")
        if os.path.exists(out) and not args.force:
            print(f"[sweep] skip {out}: exists")
        else:
            t0 = time.time()
            df = fit_head_probes(dirs["train"], dirs["val"], feature_mode=fm, score=defaults["score"], class_weight=cw, seed=args.seed,
                                 verbose=False, ranking=defaults["ranking"])
            save_head_scores(df, out, meta={"model": short, "revision": revision, "value": value, "mode": args.mode,
                                            "feature_tag": f"split{seed}", "split_seed": seed, "seconds": round(time.time() - t0, 1)})
            print(f"[sweep] saved {out} (split seed {seed}, {time.time() - t0:.1f}s)")
        paths.append(out)
    return paths


def split_tables_exist(args, p, short, revision, value):
    """True unless `value` is honesty with --split_seeds whose tagged head tables are still missing."""
    if value != "honesty":
        return True
    return all(os.path.exists(config.heads_path(p, short, revision, value, args.mode, f"split{s}")) for s in args.split_seeds if s != 42)


def run_revision(args, p, short, hf_id, revision, values, tier, B, P, manifest):
    """All steps of one revision; records timings and the extraction manifest into `manifest`."""
    entry = manifest["revisions"].setdefault(revision, {})
    entry.pop("error", None)                 # a retried revision drops the error of an earlier run (re-set if it fails again)
    timings = entry.setdefault("timings", {})
    done = set(entry.get("done", []))
    for value in values:
        if value in done and not args.force and split_tables_exist(args, p, short, revision, value):
            print(f"[sweep] skip {revision}/{value}: done (results/manifest.json)")
            continue
        dataset = config.probing_dataset_of(value, args.mode)
        t0 = time.time()
        for split in ("train", "val"):
            extract_split(args, short, revision, dataset, split, tier)
        timings[f"extract:{dataset}"] = round(time.time() - t0, 1)
        train_dir, val_dir, _ = resolve_feature_dirs(p, short, revision, value, args.mode)
        if not entry.get("commit_sha") and is_done(train_dir):
            fm = read_manifest(train_dir)
            entry.update(commit_sha=revision_commit_sha(hf_id, revision, fm), dtype=fm.get("dtype"), attn_impl=fm.get("attn_impl"),
                         transformers=fm.get("transformers"), torch=fm.get("torch"), device=fm.get("device"), hf_id=hf_id)
        common = ["--model", short, "--revision", revision, "--value", value, "--mode", args.mode, "--seed", args.seed] + (["--root", args.root] if args.root else [])
        t0 = time.time()
        call("probe", common + (["--features", args.features] if args.features else []) + (["--force"] if args.force else []) + ["--quiet"])
        timings[f"probe:{value}"] = round(time.time() - t0, 1)
        if not args.no_stats:
            t0 = time.time()
            call("probe-stats", common + ["--bootstrap", B, "--perms", P, "--device", args.stats_device]
                 + (["--features", args.features] if args.features else []) + (["--force"] if args.force else []))
            timings[f"stats:{value}"] = round(time.time() - t0, 1)
        t0 = time.time()
        call("directions", common + (["--force"] if args.force else []))
        timings[f"directions:{value}"] = round(time.time() - t0, 1)
        extra_split_tables(args, p, short, revision, value, tier)
        done.add(value)
        entry["done"] = sorted(done)
        save_manifest(p["manifest"], manifest)
    if revision in args.eval_revisions:
        if command_available("eval-likelihood"):
            for value in values:
                t0 = time.time()
                call("eval-likelihood", ["--model", short, "--revision", revision, "--value", value, "--mode", args.mode]
                     + (["--root", args.root] if args.root else []) + args.eval_args.split())
                timings[f"eval:{value}"] = round(time.time() - t0, 1)
        else:
            print(f"[sweep] eval-likelihood requested for {revision} but the command is not installed; skipped")
            entry["eval_skipped"] = "command not installed"
    if args.cleanup_cache:
        if revision in args.keep_revisions:
            print(f"[sweep] cache: keep {revision} (in --keep_revisions)")
        else:
            res = cleanup_hf_revision(hf_id, revision, protect=tuple(args.keep_revisions))
            entry["cache_cleanup"] = res
            print(f"[sweep] cache cleanup {revision}: {res}")
    if args.cleanup_features:
        if revision in args.keep_revisions:
            print(f"[sweep] features: keep {revision} (in --keep_revisions)")
        else:
            freed = cleanup_feature_dirs(p, short, revision, values, args.mode, args.split_seeds)
            entry["features_cleanup"] = {"freed_bytes": int(sum(freed.values())), "dirs": sorted(freed)}
            print(f"[sweep] feature cleanup {revision}: {len(freed)} dirs, {sum(freed.values()) / 1e9:.2f} GB freed")
    save_manifest(p["manifest"], manifest)


def save_mean_splits(path, mean_scores, names, meta, ratio=0.15):
    """Writes the mean-AUC-over-splits ranking (DESIGN 5.6 "finales Ranking = mittlere AUC über Splits") as a head JSON
    that `top_heads_from_file` / `eval-likelihood --heads_file` can read: meta (model, revision, value, mode, splits,
    layers, heads, score = auc_mean), k, topk (first k by mean AUC) and heads (flat_idx, layer, head, auc_mean, auc_std, rank)."""
    total = int(meta["layers"]) * int(meta["heads"])
    k = int(total * ratio)
    ranked = mean_scores.sort_values("rank")
    payload = {"meta": {**meta, "score": "auc_mean", "ranking": "stable", "splits": list(names), "n_splits": len(names), "ratio": ratio},
               "k": k, "topk": [int(i) for i in ranked["flat_idx"].head(k)],
               "heads": json.loads(ranked.to_json(orient="records"))}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


def build_curves(p, short, values, revisions, mode, split_seeds=()):
    """results/sweep/{model}_{value}_curve.csv (+ figure) per value from the head tables that exist; with extra split
    tables also the stability CSV and, per revision, the mean-AUC-over-splits ranking
    results/heads/{model}_{revision}_{value}_{mode}_meansplits.json (DESIGN 5.6)."""
    from ..analysis.sweep_curves import build_sweep
    out = {}
    for value in values:
        tables = {}
        for r in revisions:
            path = config.heads_path(p, short, r, value, mode)
            if os.path.exists(path):
                tables[r] = load_head_scores(path)[0]
        if len(tables) < 2:
            print(f"[sweep] curves: {value}: only {len(tables)} checkpoint(s) with head tables, skipped")
            continue
        tag = "" if mode == "pythia" else f"_{mode}"
        csv = os.path.join(p["sweep"], f"{short}_{value}{tag}_curve.csv")
        fig = os.path.join(p["figures_sweep"], f"{short}_{value}{tag}.png")
        table = build_sweep(tables, short, value, mode, csv, fig)
        out[value] = table
        print(f"[sweep] curves: wrote {csv} and {fig}")
        print(table[["revision", "top_layer", "top_head", "top_auc", "ci_lo", "ci_hi", "perm_thr", "n_signif", "spearman_to_final", "jaccard_to_final"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if value == "honesty" and split_seeds:
            rows = []
            for r in tables:
                dfs, names = [tables[r]], ["seed42"]
                for s in split_seeds:
                    path = config.heads_path(p, short, r, value, mode, f"split{s}")
                    if os.path.exists(path):
                        dfs.append(load_head_scores(path)[0])
                        names.append(f"seed{s}")
                if len(dfs) > 1:
                    st = split_stability(dfs, names=names)
                    rows.append({"revision": r, "n_splits": st["n"], "spearman_mean": st["spearman_mean"], "spearman_min": st["spearman_min"],
                                 "jaccard_mean": st["jaccard_mean"], "jaccard_min": st["jaccard_min"], "k": st["k"]})
                    mean_path = config.heads_path(p, short, r, value, mode, "meansplits")
                    save_mean_splits(mean_path, st["mean_scores"], names,
                                     {"model": short, "revision": r, "value": value, "mode": mode, "feature_tag": "meansplits",
                                      "layers": tables[r].attrs.get("layers"), "heads": tables[r].attrs.get("heads")})
                    print(f"[sweep] mean-over-splits ranking: wrote {mean_path} (top head flat_idx "
                          f"{int(st['mean_scores']['flat_idx'].iloc[0])}, {len(names)} splits)")
            if rows:
                import pandas as pd
                stab = os.path.join(p["sweep"], f"{short}_{value}{tag}_stability.csv")
                pd.DataFrame(rows).to_csv(stab, index=False)
                print(f"[sweep] stability: wrote {stab}")
    return out


# ----------------------------------------------------------------------------- command
def run(args):
    p = config.paths(args.root)
    short, hf_id, _ = config.resolve_model(args.model)
    pl = plan(args)
    print(f"[sweep] {short} ({hf_id}) mode={args.mode} tier={pl['tier']} revisions={pl['revisions']} values={pl['values']} "
          f"B={pl['bootstrap']} P={pl['perms']} eval={args.eval_revisions or '-'} cleanup_cache={args.cleanup_cache} "
          f"cleanup_features={args.cleanup_features} root={p['root']}")
    if args.dry_run:
        for r, v, d in pl["steps"]:
            print(f"  {r:12s} {v:16s} features {d}_train/_val -> probe -> {'stats -> ' if not args.no_stats else ''}directions")
        return 0
    manifest = load_manifest(p["manifest"])
    manifest.update(model=short, hf_id=hf_id, mode=args.mode, tier=pl["tier"], values=pl["values"], bootstrap=pl["bootstrap"], perms=pl["perms"])
    t_all = time.time()
    failed = []
    for revision in pl["revisions"]:
        t0 = time.time()
        print(f"\n[sweep] ===== {short} @ {revision} =====", flush=True)
        try:
            run_revision(args, p, short, hf_id, revision, pl["values"], pl["tier"], pl["bootstrap"], pl["perms"], manifest)
        except Exception as e:
            if not args.continue_on_error:
                raise
            print(f"[sweep] {revision}: FAILED ({type(e).__name__}: {e}); continuing")
            failed.append(revision)
            manifest["revisions"].setdefault(revision, {})["error"] = f"{type(e).__name__}: {e}"
            save_manifest(p["manifest"], manifest)
        print(f"[sweep] {revision} done in {time.time() - t0:.0f}s")
    if not args.no_curves:
        build_curves(p, short, pl["values"], pl["revisions"], args.mode, args.split_seeds)
    save_manifest(p["manifest"], manifest)
    print(f"\n[sweep] finished {len(pl['revisions']) - len(failed)}/{len(pl['revisions'])} revisions in {time.time() - t_all:.0f}s; "
          f"manifest {p['manifest']}" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0
