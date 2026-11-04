"""`overlap`: value heads x ICL-circuit roles (Qwen3; Gao & Gao, arXiv 2611.05344) under layer-matched nulls.

    python -m valueheads_pythia.cli overlap --model Qwen/Qwen3-4B --values honesty harmless helpful helpfulness \\
        complexity verbosity coherence --rankings pythia paper --paper_tag pyfeat --draws 100000 --perms 20000
    python -m valueheads_pythia.cli overlap --model qwen3-0.6b-base --values honesty --draws 10000    # CPU smoke

Reads the head tables results/heads/{model}_{revision}_{value}_{mode}[_{tag}].json written by `probe` (ranking
`pythia` = primary head-only AUC table of mode pythia, ranking `paper` = [head ; layer mean] accuracy table of mode
paper; `--paper_tag pyfeat` = a paper-mode probe on the pythia-mode features, see the overlap notebook). In the paper
ranking `harmless` stands for the reference's harmlessness set, HH-RLHF default config = `harmless_pooled` (own
paper-mode extraction, table without tag; `--paper_harmless harmless` uses harmless-base instead, which is NOT the
reference set); `harmless_pooled` has no pythia ranking. ICL head sets: icl_sets.py (census via VH_ICL_REPO, the
download cache or the pinned GitHub URL; sha256-checked). Post-trained
models (Qwen/Qwen3-4B) get the sets of their base model and are flagged `transferred`. Writes
results/overlap/{model}_{revision}[_{out_tag}]_overlap.{json,md} and ..._overlap_{topk,spearman,shares}.csv; skips when
the outputs exist for the same input tables and settings (idempotent), unless --force.
"""
import json
import os
import time

from .. import config, icl_sets
from .. import overlap as O
from ..cli import add_common_args


def add_args(parser):
    add_common_args(parser, device=False, mode=False, seed=False)
    parser.add_argument("--values", nargs="+", default=["honesty", "harmless", "helpful", "helpfulness", "complexity",
                                                         "verbosity", "coherence"], choices=config.ALL_VALUES)
    parser.add_argument("--rankings", nargs="+", default=["pythia", "paper"], choices=list(O.RANKINGS),
                        help="pythia = head-only AUC table (mode pythia), paper = [head ; layer mean] accuracy table (mode paper)")
    parser.add_argument("--pythia_tag", default=None, help="feature tag of the pythia-mode tables (default: primary table)")
    parser.add_argument("--paper_tag", default=None,
                        help="feature tag of the paper-mode tables, e.g. pyfeat (paper-mode probe on the pythia-mode features)")
    parser.add_argument("--paper_harmless", default="harmless_pooled", choices=["harmless_pooled", "harmless"],
                        help="HH set read for `harmless` in the paper ranking: harmless_pooled = the reference's set (HH-RLHF "
                             "default config, own paper-mode extraction), harmless = harmless-base (not the reference set)")
    parser.add_argument("--pooled_tag", default=None, help="feature tag of the harmless_pooled paper table (default: none)")
    parser.add_argument("--icl_model", default=None, help="ICL head sets (default: base model of --model, e.g. qwen3-4b-base)")
    parser.add_argument("--icl_source", default=None,
                        help="census source: ICL repo checkout, JSON path or URL (default: env VH_ICL_REPO, cache, pinned URL)")
    parser.add_argument("--no_download", action="store_true", help="never download the census (cache / local checkout only)")
    parser.add_argument("--roles", nargs="*", default=None, help="restrict to these ICL roles (default: every role)")
    parser.add_argument("--ratio", type=float, default=config.TOP_RATIO, help="top-k = int(ratio * layers * heads)")
    parser.add_argument("--draws", type=int, default=20000, help="layer-matched null draws per value table (>= 10000)")
    parser.add_argument("--perms", type=int, default=10000, help="within-layer permutations of the Spearman null (0 = skip (b))")
    parser.add_argument("--no_kv_null", action="store_true",
                        help="primary null with exchangeable heads even for GQA (anti-conservative when value heads cluster "
                             "by KV group; comparison runs only)")
    parser.add_argument("--alpha", type=float, default=O.ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_tag", default=None, help="suffix of the output files")
    parser.add_argument("--note", default=None, help="line printed at the top of the markdown summary (e.g. SMOKE)")
    parser.add_argument("--allow_small_draws", action="store_true", help="permit --draws < 10000 (tests only)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def table_path(p, short, revision, value, ranking, tag=None):
    return config.heads_path(p, short, revision, value, O.RANKINGS[ranking]["mode"], tag)


def ranking_values(values, ranking, paper_harmless="harmless_pooled"):
    """Values read for `ranking`: in the paper ranking `harmless` -> `paper_harmless`; harmless_pooled is defined in mode
    paper only, so it has no pythia ranking."""
    out = []
    for v in values:
        if ranking == "paper" and v == "harmless":
            v = paper_harmless
        if ranking == "pythia" and v == "harmless_pooled":
            continue
        if v not in out:
            out.append(v)
    return out


def out_base(p, short, revision, out_tag=None):
    tag = f"_{out_tag}" if out_tag else ""
    return os.path.join(p["results"], "overlap", f"{short}_{revision or 'main'}{tag}_overlap")


def run(args):
    if args.draws < 10000 and not args.allow_small_draws:
        raise SystemExit("[overlap] --draws must be >= 10000 (the null needs at least 10,000 draws)")
    p = config.paths(args.root)
    short, hf_id, info_model = config.resolve_model(args.model)
    try:
        icl_base, _ = icl_sets.resolve_icl_model(args.icl_model or short)
    except KeyError as e:
        raise SystemExit(f"[overlap] {e}")
    transferred = short != icl_base
    spec = icl_sets.CENSUS[icl_base]

    tables, missing = [], []
    tags = {"pythia": args.pythia_tag, "paper": args.paper_tag}
    if "harmless_pooled" in args.values and "pythia" in args.rankings:
        print("[overlap] harmless_pooled: defined in mode paper only -> no pythia ranking")
    for rk in args.rankings:
        for v in ranking_values(args.values, rk, args.paper_harmless):
            tag = args.pooled_tag if (rk == "paper" and v == "harmless_pooled") else tags[rk]
            path = table_path(p, short, args.revision, v, rk, tag)
            if not os.path.exists(path):
                print(f"[overlap] missing {path} (run `probe` first); skipped")
                missing.append(path)
                continue
            tables.append(O.load_value_table(path, v, rk, args.ratio))
    if not tables:
        print("[overlap] no head tables found")
        return 1
    base = out_base(p, short, args.revision, args.out_tag)
    settings = {"draws": args.draws, "perms": args.perms, "seed": args.seed, "ratio": args.ratio, "alpha": args.alpha,
                "roles": args.roles, "kv_null": not args.no_kv_null, "icl_model": icl_base,
                "paper_harmless": args.paper_harmless, "pooled_tag": args.pooled_tag, "stats": "v2.1 kv-primary tarone ties",
                "inputs": {os.path.basename(t.path): O.file_sha256(t.path) for t in tables}}
    if os.path.exists(base + ".json") and not args.force:
        with open(base + ".json") as f:
            old = json.load(f).get("info", {}).get("run_settings")
        if old == json.loads(json.dumps(settings)):
            print(f"[overlap] skip {base}.json: exists for the same tables and settings")
            _print_findings(base + ".json")
            return 0

    t0 = time.time()
    census = icl_sets.census_sets(icl_base, source=args.icl_source, allow_download=not args.no_download)
    role_sets = icl_sets.icl_role_sets(icl_base, census=census)
    heads_per_kv = spec.n_heads // spec.n_kv_heads
    result = O.analyze(tables, role_sets, census, draws=args.draws, perms=args.perms, seed=args.seed,
                       heads_per_kv=heads_per_kv, kv_null=not args.no_kv_null, roles=args.roles, alpha=args.alpha,
                       n_layers=spec.n_layers, n_heads=spec.n_heads, verbose=not args.quiet)
    info = {
        "model": short, "hf_id": hf_id, "revision": args.revision or "main", "ratio": args.ratio,
        "k": int(args.ratio * spec.n_layers * spec.n_heads), "rankings": [r for r in args.rankings if any(t.ranking == r for t in tables)],
        "paper_harmless": args.paper_harmless,
        "note": args.note, "missing": missing, "seconds": round(time.time() - t0, 1), "run_settings": settings,
        "icl": {"model": icl_base, "hf_id": spec.hf_id, "transferred": transferred, "paper": icl_sets.PAPER,
                "repo": icl_sets.ICL_REPO, "commit": icl_sets.ICL_COMMIT, "census_file": spec.path,
                "census_origin": census.origin, "census_sha256": census.sha256, "pinned": census.pinned,
                "screened_layers": list(census.screened_layers),
                "census_counts": {k: len(v) for k, v in census.roles.items()}},
        "tables": [{**t.meta, "value": t.value, "ranking": t.ranking, "score": t.score, "path": t.path, "n_train": t.n_train,
                    "n_val": t.n_val, "k": t.k, "sha256": settings["inputs"][os.path.basename(t.path)]} for t in tables],
    }
    paths = O.write_outputs(result, info, base)
    print(f"[overlap] {short} vs ICL {icl_base}{' (transferred: base-model sets)' if transferred else ''}: "
          f"{len(tables)} tables, {len(result['tested_roles'])} tested roles, {time.time() - t0:.1f}s")
    for k, path in paths.items():
        print(f"[overlap] wrote {path}")
    _print_findings(paths["json"])
    return 0


def _print_findings(json_path):
    with open(json_path) as f:
        payload = json.load(f)
    for rk, fam in payload["families"]["a"].items():
        sig = [r for r in payload["topk"] if r["ranking"] == rk and r["verdict"] in O.VERDICTS_SIGNIFICANT]
        print(f"[overlap] ranking {rk}: null {fam.get('null')}, Holm family m = {fam['m']} after Tarone (K = {fam.get('tarone_K')}; "
              f"{fam.get('m_before_tarone')} tests of {len(fam['values'])} values x {len(fam['roles'])} primary roles); "
              + (f"{len(sig)} significant: " + "; ".join(f"{r['value']} x {r['role']} {r['verdict']} (O={r['overlap']}, "
                                                          f"null {r['null_mean']:.2f}, Holm p={r['p_holm']:.2g})" for r in sig)
                 if sig else "no pair survives Holm"))
