"""`eval-likelihood`: judge-free likelihood metrics under Base / TopK / layer-matched Random / Steer (DESIGN 4.1, 5.9).

    python -m valueheads_pythia.cli eval-likelihood --model pythia-160m --revision step143000 --metric tqa_pairwise \\
        --value honesty --intervention mask --select topk --ratio 0.15 --positions all --random_sets 20
    python -m valueheads_pythia.cli eval-likelihood ... --metric wikitext_ppl --max_tokens 50000 --intervention mask
    python -m valueheads_pythia.cli eval-likelihood ... --metric hh_pairwise --eval_value harmless --intervention mask
    python -m valueheads_pythia.cli eval-likelihood ... --intervention steer --alpha 0.3 0.5 1.0 [--values honesty harmless --weights 0.5 0.5]
    python -m valueheads_pythia.cli eval-likelihood ... --metric tqa_pairwise --context lmeval --intervention none   # 6-shot Q:/A: column
    python -m valueheads_pythia.cli eval-likelihood ... --metric tqa_pairwise --disjoint        # heads ranked on half A, eval on half B
    python -m valueheads_pythia.cli eval-likelihood ... --metric hh_pairwise --eval_split train --exclude_groups_from <probing train dir>

Outputs (idempotent: stored conditions are reused unless --force):
    results/eval/{model}_{revision}/{metric[-eval_value][-lmeval]}_none.jsonl      base condition
    results/eval/{model}_{revision}/{metric[-eval_value]}_{mask|steer}-{value}.jsonl  TopK / signif / random / steer conditions
    results/eval/summary.csv                                                       one row per run: base / TopK / random stats,
                                                                                   empirical p-values, paired-delta bootstrap CIs
--value names the heads that are intervened on; --eval_value the evaluation set's value for hh_pairwise
(harmless | helpful, default harmless) and helpsteer2_pairwise (required). DESIGN 4.1 "Head-Auswahl und Eval disjunkt":
--disjoint (= --rank_by auc_half_A --eval_groups B) ranks the heads on half A of the probe-stats confirmation split and
evaluates only the questions / pairs / prompts of half B; --eval_split train --exclude_groups_from <dir> scores the
train pairs that were not in the probing train subsample (>= 1,000 items). Masking: eps = 1e-4 at all positions;
steering: x += alpha * sigma * dir on every top head (sigma from the directions npz, std set = the evaluation
set's value when available, else the head value's own). Random controls are layer-matched to the TopK set, all R
drawn in sequence from one RandomState(--random_seed); every random condition records {random_seed, random_index}
(mode paper: uniform random sets with random.Random(random_seed + r) per set, as in the reference repo).
"""
import json
import os
import time

import torch

from .. import config
from ..cli import add_common_args
from ..controls import random_sets_layer_matched, random_sets_uniform
from ..data import STD_SETS
from ..directions import load_directions, sigma_along, steering_vectors
from ..eval_likelihood import (CONTEXT_STYLES, METRICS, append_condition, append_summary, build_metric, condition_key, flat_heads,
                               format_table, jsonl_path, load_conditions, mask_hook, run_condition, signif_heads_from_file,
                               steer_hook, steering_vectors_from_directions, summary_row)
from ..extract import is_done
from ..models import load_model, model_geometry, set_seed
from ..probing import group_by_layer, top_heads_from_file
from .directions import std_set_dir


def add_args(parser):
    add_common_args(parser)
    parser.add_argument("--metric", required=True, choices=METRICS)
    parser.add_argument("--value", default=None, choices=config.ALL_VALUES, help="value whose heads are masked / steered")
    parser.add_argument("--values", nargs="+", default=None, choices=config.ALL_VALUES, help="several values (multi-value steering / union mask)")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="lambda per value for --values (default 1/K; sum |lambda| <= 1)")
    parser.add_argument("--eval_value", default=None, help="evaluation value of hh_pairwise (harmless|helpful) / helpsteer2_pairwise")
    parser.add_argument("--intervention", default="none", choices=["none", "mask", "steer"])
    parser.add_argument("--select", default="topk", choices=["topk", "random", "signif"])
    parser.add_argument("--ratio", type=float, default=config.TOP_RATIO)
    parser.add_argument("--k", type=int, default=None, help="override k = int(ratio * L * H)")
    parser.add_argument("--positions", default="all", choices=["all"],
                        help="hooks act at every position (DESIGN 3.1); 'last' is refused here because it would edit only the "
                             "final padded column of a right-padded batch (use eval-generate for the paper semantics)")
    parser.add_argument("--random_sets", type=int, default=None,
                        help="R random head sets (0 = none); default 20 layer-matched sets (DESIGN 5.9), 5 uniform sets (seeds 0-4) in --mode paper")
    parser.add_argument("--eval_split", default="val", choices=["val", "train"],
                        help="pairwise metrics: evaluate the val (default) or the train split (DESIGN 4.1: train pairs for >= 1,000 items)")
    parser.add_argument("--exclude_groups_from", default=None,
                        help="pairwise metrics: drop items whose group id is in <feature dir>/groups.npy (the probing train subsample)")
    parser.add_argument("--eval_groups", default="all", choices=["all", "A", "B"],
                        help="pairwise metrics: only the groups of half A / B of the probe-stats confirmation split of --value's head table")
    parser.add_argument("--rank_by", default="score", choices=["score", "auc_half_A", "auc_half_B"],
                        help="head ranking column: the stored ranking (score) or the half-A / half-B AUC of probe-stats")
    parser.add_argument("--disjoint", action="store_true", help="shorthand for --rank_by auc_half_A --eval_groups B (DESIGN 4.1 / 5.5)")
    parser.add_argument("--random_seed", type=int, default=0,
                        help="seed of the RandomState that draws all R layer-matched sets in sequence (mode paper: set r uses seed + r)")
    parser.add_argument("--context", default="probing", choices=CONTEXT_STYLES,
                        help="tqa_pairwise context: probing = 'Question: {q} Answer:' (default), lmeval = 6-shot primer + 'Q: {q}\\nA:'")
    parser.add_argument("--ppl_min_context", type=int, default=None,
                        help="wikitext: minimal context of every scored token (default window - stride = DESIGN 4.1 item 5; 0 = HF recipe)")
    parser.add_argument("--alpha", nargs="+", type=float, default=None, help=f"steering strengths (default {config.STEER_ALPHAS})")
    parser.add_argument("--eps", type=float, default=config.MASK_EPS)
    parser.add_argument("--std_set", default=None, help="std set of sigma for steering (default: the evaluation set's value)")
    parser.add_argument("--max_items", type=int, default=None, help="subsample of questions / pairs (seeded)")
    parser.add_argument("--min_delta", type=int, default=1, help="helpsteer2: minimal rating difference of a pair")
    parser.add_argument("--max_tokens", type=int, default=None, help="wikitext: token prefix cap (smoke / T4: 50000)")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--split_seed", type=int, default=42, help="TruthfulQA question split seed")
    parser.add_argument("--batch_tokens", type=int, default=4096, help="padded tokens per forward (CPU: 4096, T4: 8192)")
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--bootstrap", type=int, default=1000, help="B for the paired-delta CIs")
    parser.add_argument("--heads_file", default=None, help="override the head-score JSON of --value")
    parser.add_argument("--directions_file", default=None, help="override the directions npz of --value")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--force", action="store_true", help="recompute stored conditions")
    parser.add_argument("--no_progress", action="store_true")


def _eval_value(args):
    if args.metric == "hh_pairwise":
        v = args.eval_value or (args.value if args.value in ("harmless", "helpful", "harmless_pooled") else "harmless")
        if v not in ("harmless", "helpful", "harmless_pooled"):
            raise SystemExit(f"--eval_value for hh_pairwise must be harmless | helpful | harmless_pooled, got {v!r}")
        return v
    if args.metric == "helpsteer2_pairwise":
        v = args.eval_value or (args.value if args.value in config.HELPSTEER_VALUES_EXT else None)
        if v not in config.HELPSTEER_VALUES_EXT:
            raise SystemExit("--eval_value (a HelpSteer2 value) is required for helpsteer2_pairwise")
        return v
    return None


def _default_std_set(args, eval_value, head_values):
    if args.std_set:
        return args.std_set
    if args.metric.startswith("tqa"):
        return "honesty"
    if eval_value is not None:
        return STD_SETS.get(eval_value, eval_value)
    return head_values[0]


def _load_heads(args, p, short, value, override=None):
    """Flat head indices of `value` (TopK by ratio / k, or the significant set) and the JSON they came from."""
    path = override or config.heads_path(p, short, args.revision, value, args.mode)
    if not os.path.exists(path):
        raise SystemExit(f"[eval] head scores not found: {path} (run `probe` first)")
    rank_by = getattr(args, "rank_by", "score")
    if args.select == "signif":
        if rank_by != "score":
            raise SystemExit("[eval] --rank_by only applies to --select topk")
        heads = signif_heads_from_file(path)
    else:
        try:
            heads = top_heads_from_file(path, args.ratio, args.k, score=None if rank_by == "score" else rank_by)
        except ValueError as e:
            raise SystemExit(f"[eval] {e}")
    return heads, path


def _group_filters(args, p, short, head_values):
    """(include_groups, exclude_groups) of the pairwise metrics from --eval_groups / --exclude_groups_from."""
    include = exclude = None
    if args.eval_groups != "all":
        if not head_values:
            raise SystemExit("[eval] --eval_groups needs --value (the head table whose confirmation split defines the halves)")
        from ..probing_stats import confirmation_groups
        path = args.heads_file or config.heads_path(p, short, args.revision, head_values[0], args.mode)
        try:
            a, b = confirmation_groups(path)
        except (ValueError, FileNotFoundError) as e:
            raise SystemExit(f"[eval] {e}")
        include = a if args.eval_groups == "A" else b
    if args.exclude_groups_from:
        gpath = os.path.join(args.exclude_groups_from, "groups.npy")
        if not os.path.exists(gpath):
            raise SystemExit(f"[eval] --exclude_groups_from: {gpath} not found")
        import numpy as np
        exclude = set(int(g) for g in np.load(gpath))
    return include, exclude


def _single_value_vectors(args, p, short, value, heads_flat, std_set, H, D):
    """Steering vectors of one value from its directions npz (sigma of `std_set`, falling back to the value's own)."""
    path = args.directions_file or config.directions_path(p, short, args.revision, value, args.mode)
    if not os.path.exists(path):
        raise SystemExit(f"[eval] directions not found: {path} (run `directions` first)")
    directions, sigmas, meta = load_directions(path)
    if std_set in sigmas:
        sigma, used = sigmas[std_set], std_set
    else:
        d = std_set_dir(p, short, args.revision, std_set, args.mode)
        if is_done(d):
            sigma, used = sigma_along(directions, d), std_set
        else:
            own = meta.get("value", value)
            if own not in sigmas:
                raise SystemExit(f"[eval] no sigma for std set {std_set!r} in {path} and no features at {d}")
            sigma, used = sigmas[own], own
            print(f"[eval] warning: sigma of std set {std_set!r} unavailable; using sigma[{own}] from {path}")
    return steering_vectors_from_directions(directions, sigma, heads_flat, H, D), used, path


def _multi_value_vectors(args, p, short, value_heads, weights, std_set, H, D):
    """Composite steering vectors over several values (directions.steering_vectors; needs the std set's features)."""
    value_dirs = {}
    for v in value_heads:
        path = config.directions_path(p, short, args.revision, v, args.mode)
        if not os.path.exists(path):
            raise SystemExit(f"[eval] directions not found: {path}")
        value_dirs[v] = load_directions(path)[0]
    d = std_set_dir(p, short, args.revision, std_set, args.mode)
    if not is_done(d):
        fallback = std_set_dir(p, short, args.revision, list(value_heads)[0], args.mode)
        if not is_done(fallback):
            raise SystemExit(f"[eval] multi-value steering needs the std set features ({d} or {fallback})")
        print(f"[eval] warning: std set features {d} missing; using {fallback}")
        d, std_set = fallback, list(value_heads)[0]
    return steering_vectors(value_heads, value_dirs, weights, d, H, D), std_set, d


def run(args):
    set_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    p = config.paths(args.root)
    short, hf_id, _ = config.resolve_model(args.model)
    eval_dir = config.eval_dir(p, short, args.revision)
    os.makedirs(eval_dir, exist_ok=True)
    progress = not args.no_progress
    eval_value = _eval_value(args)

    head_values = args.values or ([args.value] if args.value else [])
    if args.intervention != "none" and not head_values:
        raise SystemExit("--value (or --values) is required for --intervention mask | steer")
    if args.disjoint:
        args.rank_by, args.eval_groups = "auc_half_A", "B"
    if args.random_sets is None:
        args.random_sets = 5 if args.mode == "paper" else 20
    include_groups, exclude_groups = _group_filters(args, p, short, head_values)
    weights = None
    if args.intervention == "steer":
        weights = args.weights or [1.0 / len(head_values)] * len(head_values)
        if len(weights) != len(head_values):
            raise SystemExit("--weights must match --values")
        weights = dict(zip(head_values, weights))
    alphas = args.alpha or config.STEER_ALPHAS

    t0 = time.time()
    model, tokenizer = load_model(hf_id, revision=args.revision, device=args.device, dtype=args.dtype, attn_impl=args.attn_impl)
    info = model.vh_info
    L, H, D = model_geometry(model.config)
    print(f"[eval] {info['hf_id']}@{info['revision']} dtype={info['dtype']} attn={info['attn_impl']} device={info['device']} "
          f"L/H/D={(L, H, D)}", flush=True)

    if args.context != "probing" and args.metric != "tqa_pairwise":
        raise SystemExit("--context lmeval is only defined for --metric tqa_pairwise")
    try:
        metric = build_metric(args.metric, tokenizer, eval_value, args.max_items, args.seed, args.split_seed, args.min_delta,
                              args.max_tokens, args.window, args.stride, split=args.eval_split, context=args.context,
                              min_context=args.ppl_min_context, include_groups=include_groups, exclude_groups=exclude_groups)
    except ValueError as e:
        raise SystemExit(f"[eval] {e}")
    if include_groups is not None or exclude_groups is not None:
        print(f"[eval] group filter: eval_groups={args.eval_groups} exclude_from={args.exclude_groups_from or '-'} -> "
              f"{metric.config['n_items']} of {metric.config.get('n_items_unfiltered')} items", flush=True)
    n_req = len(metric.requests) if hasattr(metric, "requests") else len(metric.windows)
    print(f"[eval] metric {metric.tag}: {metric.config['n_items']} items, {n_req} forward requests; data {time.time() - t0:.1f}s", flush=True)

    def stored(path, name, cond):
        rec = load_conditions(path).get(name)
        return rec if (rec and rec.get("key") == condition_key(metric.config, cond) and not args.force) else None

    def evaluate(path, name, cond, hook):
        rec = stored(path, name, cond)
        if rec is not None:
            print(f"[eval] {name}: cached ({path})", flush=True)
            return rec
        res = run_condition(model, tokenizer, hook, metric, args.batch_tokens, args.max_length, progress, desc=name)
        rec = {"condition": name, **cond, "key": condition_key(metric.config, cond), "metric_config": metric.config,
               "model": short, "revision": args.revision or "main", "dtype": info["dtype"], "attn_impl": info["attn_impl"],
               **res}
        append_condition(path, rec)
        st = res["stats"]
        print(f"[eval] {name}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res['summary'].items()})} "
              f"[{st['seconds']}s, {st['n_tokens_scored']} tokens scored, fallback {st['n_fallback']}, truncated {st['n_truncated']}]", flush=True)
        return rec

    # ---- base
    base_path = jsonl_path(eval_dir, metric.tag, "none")
    base_cond = {"intervention": "none", "positions": None, "heads": None, "alpha": None}
    base = evaluate(base_path, "base", base_cond, None)
    meta = {"model": short, "revision": args.revision or "main", "mode": args.mode, "head_values": "+".join(head_values) or None,
            "select": args.select if args.intervention != "none" else None, "ratio": args.ratio, "positions": args.positions,
            "eps": args.eps if args.intervention == "mask" else None, "max_items": args.max_items, "max_tokens": args.max_tokens,
            "min_delta": args.min_delta if args.metric == "helpsteer2_pairwise" else None, "dtype": info["dtype"], "attn_impl": info["attn_impl"],
            "random_seed": args.random_seed if args.intervention != "none" else None,
            "eval_split": args.eval_split, "eval_groups": args.eval_groups if args.eval_groups != "all" else None,
            "rank_by": args.rank_by if (args.rank_by != "score" and args.intervention != "none") else None,
            "exclude_from": args.exclude_groups_from or None}
    summary_path = os.path.join(p["eval"], "summary.csv")
    if args.intervention == "none":
        row = summary_row(metric, base, None, [], "none", args.bootstrap, args.seed, dict(meta, k=None, alpha=None))
        append_summary(summary_path, [row])
        print(format_table(metric, base, None, []))
        print(f"[eval] base only -> {base_path}; summary {summary_path}; {time.time() - t0:.1f}s")
        return 0

    # ---- head selection (union over values)
    value_heads = {}
    for v in head_values:
        value_heads[v], hp = _load_heads(args, p, short, v, args.heads_file if len(head_values) == 1 else None)
        print(f"[eval] {v}: {len(value_heads[v])} heads from {hp} (select={args.select}, rank_by={args.rank_by})")
    union = sorted(set(h for hs in value_heads.values() for h in hs))
    topk_by_layer = group_by_layer(union, H)
    k = len(union)
    if k == 0:
        raise SystemExit("[eval] empty head set (signif selection found no significant heads?)")
    print(f"[eval] intervened set: k={k} heads by layer {topk_by_layer}")
    if args.mode == "paper":
        random_sets = random_sets_uniform(k, L, H, args.random_sets, args.random_seed)
    else:
        random_sets = random_sets_layer_matched(topk_by_layer, L, H, args.random_sets, args.random_seed)

    def random_id(r):
        """Generator seed and index of random set r: mode pythia draws all sets from one RandomState(random_seed)
        (set r is the r-th draw), mode paper seeds random.Random(random_seed + r) per set as the reference repo does."""
        if r is None:
            return {"random_seed": None, "random_index": None}
        return {"random_seed": args.random_seed + r if args.mode == "paper" else args.random_seed, "random_index": r}

    tag = f"{args.intervention}-{'+'.join(head_values)}"
    path = jsonl_path(eval_dir, metric.tag, tag)
    topk_name = "signif" if args.select == "signif" else "topk"
    rows = []

    if args.intervention == "mask":
        def cond(name, heads, r=None):
            return {"intervention": "mask", "select": name, "positions": args.positions, "eps": args.eps, "heads": heads,
                    "k": k, "ratio": args.ratio, **random_id(r), "alpha": None}
        topk = None
        if args.select != "random":
            topk = evaluate(path, topk_name, cond(args.select, topk_by_layer), mask_hook(model, topk_by_layer, args.eps, args.positions))
        randoms = [evaluate(path, f"random{r}", cond("random", s, r), mask_hook(model, s, args.eps, args.positions))
                   for r, s in enumerate(random_sets)]
        rows.append(summary_row(metric, base, topk, randoms, "mask", args.bootstrap, args.seed, dict(meta, k=k, alpha=None)))
        print(format_table(metric, base, topk, randoms))
    else:
        std_set = _default_std_set(args, eval_value, head_values)
        if len(head_values) == 1:
            v = head_values[0]
            vec_topk, std_used, src = _single_value_vectors(args, p, short, v, union, std_set, H, D)
            vec_random = [_single_value_vectors(args, p, short, v, flat_heads(s, H), std_set, H, D)[0] for s in random_sets]
        else:
            vec_topk, std_used, src = _multi_value_vectors(args, p, short, value_heads, weights, std_set, H, D)
            vec_random = [_multi_value_vectors(args, p, short, {v: flat_heads(s, H) for v in head_values}, weights, std_set, H, D)[0]
                          for s in random_sets]
        print(f"[eval] steering vectors: {len(vec_topk)} layers, sigma std set = {std_used} ({src}), weights = {weights}")
        for alpha in alphas:
            def cond(name, heads, r=None):
                return {"intervention": "steer", "select": name, "positions": args.positions, "alpha": alpha, "heads": heads, "k": k,
                        "ratio": args.ratio, "weights": weights, "std_set": std_used, **random_id(r)}
            topk = None
            if args.select != "random":
                topk = evaluate(path, f"{topk_name}_a{alpha:g}", cond(args.select, topk_by_layer), steer_hook(model, vec_topk, alpha, args.positions))
            randoms = [evaluate(path, f"random{r}_a{alpha:g}", cond("random", s, r), steer_hook(model, vr, alpha, args.positions))
                       for r, (s, vr) in enumerate(zip(random_sets, vec_random))]
            rows.append(summary_row(metric, base, topk, randoms, "steer", args.bootstrap, args.seed, dict(meta, k=k, alpha=alpha, std_set=std_used)))
            print(f"--- alpha={alpha:g}")
            print(format_table(metric, base, topk, randoms))

    append_summary(summary_path, rows)
    for row in rows:
        prim = row["primary"]
        if f"topk_{prim}" in row:
            print(f"[eval] {metric.tag} {tag}{'' if row.get('alpha') is None else ' alpha=' + format(row['alpha'], 'g')}: "
                  f"base {prim}={row[f'base_{prim}']:.4f} topk={row[f'topk_{prim}']:.4f} random={row['random_mean']:.4f}+-{row['random_std']:.4f} "
                  f"p_emp({row['p_direction']})={row['p_emp_primary']:.3f}; {row['effect']}={row['delta_mean']:+.4f} "
                  f"[{row['delta_ci_lo']:+.4f}, {row['delta_ci_hi']:+.4f}] p_emp={row['p_emp_delta']:.3f}")
    print(f"[eval] results {path}; summary {summary_path}; {time.time() - t0:.1f}s total")
    return 0
