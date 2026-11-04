"""`eval-generate`: free generation under Base / TopK / Random / Steer -> one jsonl per condition (DESIGN 3.2, 4.2).

    python -m valueheads_pythia.cli eval-generate --model pythia-160m --revision step143000 --eval_set honesty \\
        --prompt_style fewshot_qa --value honesty --intervention mask --select topk [--random_sets 1] [--num_eval 10]
    python -m valueheads_pythia.cli eval-generate ... --eval_set helpfulness --prompt_style qa_plain --value honesty --intervention mask
    python -m valueheads_pythia.cli eval-generate ... --intervention steer --alpha 0.3 0.5 1.0 [--values honesty harmless --weights ...]
    python -m valueheads_pythia.cli eval-generate ... --mode paper --prompt_style paper   (repo template, Llama / Qwen / Yi)

Outputs (idempotent: existing condition files are reused unless --force):
    results/eval/{model}_{revision}/gen-{eval_set}-{prompt_style}_none.jsonl                      base answers
    results/eval/{model}_{revision}/gen-{eval_set}-{prompt_style}_{mask|steer}-{value}_{select}_k{k}[...].jsonl
    results/eval/gen_summary.csv                                                                  one row per condition
Hooks follow the paper semantics by default (positions last, decode_only) -- the prompt forward is untouched and every
decoding step is edited; --positions all also edits the prompt forward. Generation and judging are separate phases:
run `judge` on the jsonl files afterwards. Answers of the harmful prompt sets are never printed.
"""
import csv
import os
import time

import torch

from .. import config
from ..cli import add_common_args
from ..controls import random_sets_layer_matched, random_sets_uniform
from ..data import EVAL_SETS, STD_SETS, eval_set as load_eval_set
from ..eval_generate import (HARMFUL_SETS, PROMPT_STYLE_NAMES, PROMPT_STYLES, condition_tag, default_prompt_style, format_examples,
                             generate_answers, generation_hook, generations_path, judge_values_of, read_generations, same_condition,
                             summarize_records, write_generations)
from ..eval_likelihood import flat_heads
from ..models import load_model, model_geometry, set_seed
from ..probing import group_by_layer
from .eval_likelihood import _load_heads, _multi_value_vectors, _single_value_vectors

SUMMARY_COLUMNS = ["model", "revision", "mode", "eval_set", "prompt_style", "condition", "intervention", "select", "head_values", "k",
                   "alpha", "positions", "decode_only", "random_seed", "n", "mean_new_tokens", "mean_answer_chars", "empty",
                   "truncated_prompts", "refusal_rate", "no_comment_rate", "degenerate_rate", "stop_stop_string", "stop_stop_token",
                   "stop_eos", "stop_max_new_tokens", "seconds", "path"]


def add_args(parser):
    add_common_args(parser)
    parser.add_argument("--eval_set", required=True, choices=list(EVAL_SETS))
    parser.add_argument("--prompt_style", default=None, choices=PROMPT_STYLE_NAMES, help="default: paper in --mode paper, qa_plain for HelpSteer2 sets, else fewshot_qa")
    parser.add_argument("--value", default=None, choices=config.ALL_VALUES, help="value whose heads are masked / steered (default: the eval set's value)")
    parser.add_argument("--values", nargs="+", default=None, choices=config.ALL_VALUES, help="several values (union mask / multi-value steering)")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="lambda per value for --values (default 1/K; sum |lambda| <= 1)")
    parser.add_argument("--intervention", default="none", choices=["none", "mask", "steer"])
    parser.add_argument("--select", default="topk", choices=["topk", "random", "signif"])
    parser.add_argument("--ratio", type=float, default=config.TOP_RATIO)
    parser.add_argument("--k", type=int, default=None, help="override k = int(ratio * L * H)")
    parser.add_argument("--positions", default="last", choices=["last", "all"], help="last = paper semantics (with decode_only)")
    parser.add_argument("--decode_only", default="auto", choices=["auto", "yes", "no"], help="auto: yes for positions=last, no for all")
    parser.add_argument("--random_sets", type=int, default=None,
                        help="R random control sets (default 0; --mode paper: 5 uniform sets with seeds random_seed..random_seed+4 as in the repo)")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--alpha", nargs="+", type=float, default=None, help=f"steering strengths (default {config.STEER_ALPHAS})")
    parser.add_argument("--eps", type=float, default=config.MASK_EPS)
    parser.add_argument("--std_set", default=None, help="std set of sigma for steering (default: STD_SETS[eval_set])")
    parser.add_argument("--num_eval", type=int, default=None, help="first N questions of the eval set (smoke: 10)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=None, help="override the prompt style's default")
    parser.add_argument("--temperature", type=float, default=None, help="sample with top-p 0.9 instead of greedy")
    parser.add_argument("--max_length", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--heads_file", default=None, help="override the head-score JSON of --value")
    parser.add_argument("--directions_file", default=None, help="override the directions npz of --value")
    parser.add_argument("--examples", type=int, default=3, help="example generations printed per condition (never for harmful sets)")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--force", action="store_true", help="regenerate stored conditions")
    parser.add_argument("--no_progress", action="store_true")


def append_gen_summary(path, rows, key=("path",)):
    """Appends rows to gen_summary.csv; an existing row of the same generations file (`path`) is replaced (regeneration
    with --force does not duplicate rows)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keep = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
        new_keys = {tuple(str(r.get(k)) for k in key) for r in rows}
        keep = [r for r in old if tuple(str(r.get(k)) for k in key) not in new_keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in keep + [{c: r.get(c) for c in SUMMARY_COLUMNS} for r in rows]:
            w.writerow({c: r.get(c) for c in SUMMARY_COLUMNS})


def run(args):
    set_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    p = config.paths(args.root)
    short, hf_id, _ = config.resolve_model(args.model)
    eval_dir = config.eval_dir(p, short, args.revision)
    os.makedirs(eval_dir, exist_ok=True)
    progress = not args.no_progress
    style = args.prompt_style or default_prompt_style(args.eval_set, args.mode)
    if style == "paper" and args.mode != "paper":
        print("[gen] warning: prompt_style paper outside --mode paper (the repo template degenerates on Pythia, DESIGN 3.2)")
    harmful = args.eval_set in HARMFUL_SETS
    judge_values = judge_values_of(args.eval_set)
    head_values = args.values or ([args.value] if args.value else ([judge_values[0]] if len(judge_values) == 1 else []))
    if args.intervention != "none" and not head_values:
        raise SystemExit("--value (or --values) is required for --intervention mask | steer on a multi-value eval set")
    weights = None
    if args.intervention == "steer":
        weights = args.weights or [1.0 / len(head_values)] * len(head_values)
        if len(weights) != len(head_values):
            raise SystemExit("--weights must match --values")
        weights = dict(zip(head_values, weights))
    alphas = args.alpha or config.STEER_ALPHAS
    if args.random_sets is None:
        args.random_sets = 5 if args.mode == "paper" else 0
    decode_only = None if args.decode_only == "auto" else args.decode_only == "yes"

    t0 = time.time()
    questions = load_eval_set(args.eval_set, args.num_eval)
    print(f"[gen] eval set {args.eval_set}: {len(questions)} prompts, style {style} "
          f"(max_new_tokens {args.max_new_tokens or PROMPT_STYLES[style]['max_new_tokens']}), judge values {judge_values}"
          f"{' [harmful set: answers are never printed]' if harmful else ''}; data {time.time() - t0:.1f}s", flush=True)
    model, tokenizer = load_model(hf_id, revision=args.revision, device=args.device, dtype=args.dtype, attn_impl=args.attn_impl)
    info = model.vh_info
    L, H, D = model_geometry(model.config)
    print(f"[gen] {info['hf_id']}@{info['revision']} dtype={info['dtype']} attn={info['attn_impl']} device={info['device']} L/H/D={(L, H, D)}", flush=True)

    common = {"eval_set": args.eval_set, "prompt_style": style, "judge_values": judge_values, "harmful": harmful, "template": PROMPT_STYLES[style]["template"],
              "max_new_tokens": args.max_new_tokens or PROMPT_STYLES[style]["max_new_tokens"], "stop_strings": PROMPT_STYLES[style]["stop_strings"],
              "stop_token": PROMPT_STYLES[style]["stop_token"], "temperature": args.temperature, "num_eval": args.num_eval, "n": len(questions),
              "batch_size": args.batch_size, "max_length": args.max_length, "seed": args.seed, "mode": args.mode, "model": short,
              "hf_id": info["hf_id"], "revision": info["revision"], "commit_sha": info.get("commit_sha"), "dtype": info["dtype"],
              "attn_impl": info["attn_impl"], "device": info["device"]}
    conditions, rows = {}, []

    def evaluate(name, cond, hook):
        tag = cond["tag"]
        path = generations_path(eval_dir, args.eval_set, style, tag)
        if os.path.exists(path) and not args.force:
            header, records = read_generations(path)
            if same_condition(header, {**common, **cond}):
                print(f"[gen] {name}: cached ({path})", flush=True)
                conditions[name] = records
                return header, records
            print(f"[gen] {name}: stored file {path} has a different condition; regenerating", flush=True)
        t1 = time.time()
        records = generate_answers(model, tokenizer, questions, style, hook=hook, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, max_length=args.max_length, progress=progress, desc=name)
        header = {**common, "condition": name, **cond, "seconds": round(time.time() - t1, 1)}
        write_generations(path, header, records)
        stats = summarize_records(records)
        rows.append({**header, **stats, "head_values": "+".join(cond.get("head_values") or []) or None, "path": path})
        print(f"[gen] {name}: n={stats['n']} new_tokens={stats['mean_new_tokens']:.1f} chars={stats['mean_answer_chars']:.0f} "
              f"stop={{string {stats['stop_stop_string']}, token {stats['stop_stop_token']}, eos {stats['stop_eos']}, max {stats['stop_max_new_tokens']}}} "
              f"empty={stats['empty']} refusal={stats['refusal_rate']:.2f} no_comment={stats['no_comment_rate']:.2f} degenerate={stats['degenerate_rate']:.2f} "
              f"[{header['seconds']}s] -> {path}", flush=True)
        conditions[name] = records
        return header, records

    base_cond = {"tag": "none", "intervention": "none", "select": None, "head_values": [], "heads": None, "k": None, "ratio": None,
                 "eps": None, "alpha": None, "positions": None, "decode_only": None, "random_seed": None, "std_set": None, "weights": None}
    evaluate("base", base_cond, None)

    if args.intervention != "none":
        value_heads = {}
        for v in head_values:
            value_heads[v], hp = _load_heads(args, p, short, v, args.heads_file if len(head_values) == 1 else None)
            print(f"[gen] {v}: {len(value_heads[v])} heads from {hp} (select={args.select})")
        union = sorted(set(h for hs in value_heads.values() for h in hs))
        topk_by_layer = group_by_layer(union, H)
        k = len(union)
        if k == 0:
            raise SystemExit("[gen] empty head set")
        print(f"[gen] intervened set: k={k} heads by layer {topk_by_layer}")
        if args.mode == "paper":
            random_sets = random_sets_uniform(k, L, H, args.random_sets, args.random_seed)
        else:
            random_sets = random_sets_layer_matched(topk_by_layer, L, H, args.random_sets, args.random_seed)
        sel_name = "signif" if args.select == "signif" else "topk"
        eff_decode_only = (args.positions == "last") if decode_only is None else decode_only

        def cond(select, heads, alpha=None, seed_r=None, std_set=None):
            return {"tag": condition_tag(args.intervention, head_values, select, k, seed_r, alpha, args.positions, eff_decode_only),
                    "intervention": args.intervention,
                    "select": select, "head_values": head_values, "heads": {int(l): hs for l, hs in heads.items()}, "k": k, "ratio": args.ratio,
                    "eps": args.eps if args.intervention == "mask" else None, "alpha": alpha, "positions": args.positions,
                    "decode_only": eff_decode_only, "random_seed": seed_r, "std_set": std_set, "weights": weights}

        if args.intervention == "mask":
            if args.select != "random":
                evaluate(sel_name, cond(args.select, topk_by_layer),
                         generation_hook(model, "mask", heads_by_layer=topk_by_layer, eps=args.eps, positions=args.positions, decode_only=decode_only))
            for r, s in enumerate(random_sets):
                evaluate(f"random{r}", cond("random", s, seed_r=args.random_seed + r),
                         generation_hook(model, "mask", heads_by_layer=s, eps=args.eps, positions=args.positions, decode_only=decode_only))
        else:
            std_set = args.std_set or STD_SETS.get(args.eval_set, head_values[0])
            if len(head_values) == 1:
                v = head_values[0]
                vec_topk, std_used, src = _single_value_vectors(args, p, short, v, union, std_set, H, D)
                vec_random = [_single_value_vectors(args, p, short, v, flat_heads(s, H), std_set, H, D)[0] for s in random_sets]
            else:
                vec_topk, std_used, src = _multi_value_vectors(args, p, short, value_heads, weights, std_set, H, D)
                vec_random = [_multi_value_vectors(args, p, short, {v: flat_heads(s, H) for v in head_values}, weights, std_set, H, D)[0]
                              for s in random_sets]
            print(f"[gen] steering vectors: {len(vec_topk)} layers, sigma std set = {std_used} ({src}), weights = {weights}")
            for alpha in alphas:
                if args.select != "random":
                    evaluate(f"{sel_name}_a{alpha:g}", cond(args.select, topk_by_layer, alpha=alpha, std_set=std_used),
                             generation_hook(model, "steer", vectors=vec_topk, alpha=alpha, positions=args.positions, decode_only=decode_only))
                for r, (s, vr) in enumerate(zip(random_sets, vec_random)):
                    evaluate(f"random{r}_a{alpha:g}", cond("random", s, alpha=alpha, seed_r=args.random_seed + r, std_set=std_used),
                             generation_hook(model, "steer", vectors=vr, alpha=alpha, positions=args.positions, decode_only=decode_only))

    if rows:
        append_gen_summary(os.path.join(p["eval"], "gen_summary.csv"), rows)
    if args.examples and not harmful:
        text = format_examples(conditions, n=args.examples, harmful=harmful)
        if text:
            print("--- examples (question, then one answer per condition)")
            print(text)
    print(f"[gen] {len(conditions)} conditions -> {eval_dir}; {time.time() - t0:.1f}s total")
    return 0
