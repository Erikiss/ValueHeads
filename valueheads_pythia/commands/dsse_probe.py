"""`dsse-probe`: DSSE side B -- residual-stream representation of a dataset's side effect (docs/DSSE_SIDEB.md).

    python -m valueheads_pythia.cli dsse-probe --stage download --raw_dir /content/dsse_raw
    python -m valueheads_pythia.cli dsse-probe --stage extract --model pythia-410m --revisions step0 step2000 step16000 step143000
    python -m valueheads_pythia.cli dsse-probe --stage baselines
    python -m valueheads_pythia.cli dsse-probe --stage probe --model pythia-410m --revisions step0 step2000 step16000 step143000
    python -m valueheads_pythia.cli dsse-probe --stage report
    python -m valueheads_pythia.cli dsse-probe --stage all --model pythia-160m --revisions step0 step143000
    python -m valueheads_pythia.cli dsse-probe --stage synthetic --synthetic_out /tmp/dsse_syn      # smoke / tests

Stages: synthetic (write a synthetic DSSE layout; never part of `all`), download (Hub -> raw_dir, then verify),
verify (bytes + sha256 of every file; exit 1 on a mismatch), extract (one feature file per --revisions:
{features}/{model}_{revision}.npz + manifest, skipped when a matching file exists), probe (every task x layer x pooling x
split on the feature files, angles + rank test; idempotent per fingerprint), baselines (TF-IDF / style / metadata once per
root), report (results/dsse/report.md + figures), all (download-or-verify -> extract -> baselines -> probe -> report).

Privacy: nothing here prints dataset text; outputs hold ids, labels, counts, hashes and metrics only. Exit codes: 0 ok,
1 mismatch / missing inputs; programming errors raise.
"""
import glob
import http.client
import json
import os
import time

from .. import config
from .. import dsse_probe as D
from ..cli import add_common_args

STAGES = ("verify", "download", "extract", "probe", "baselines", "report", "synthetic", "all")


def add_args(parser):
    add_common_args(parser, mode=False)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--raw_dir", default=None, help="DSSE raw files (default env VH_DSSE_RAW, else {root}/dsse_raw); Colab: /content/dsse_raw")
    parser.add_argument("--dsse_revision", default=D.DSSE_REVISION, help=f"Hub revision for download (fallback: {D.DSSE_PINNED_COMMIT})")
    parser.add_argument("--revisions", nargs="+", default=None, help="checkpoints for extract / probe (default: --revision or main)")
    parser.add_argument("--models", nargs="+", default=None, help="probe / report: which feature files (default: --model, or every npz)")
    parser.add_argument("--n_rows", type=int, default=D.N_ROWS, help="rows sampled per dataset (seeded, model independent)")
    parser.add_argument("--max_length", type=int, default=D.MAX_LENGTH)
    parser.add_argument("--prompt_cap", type=int, default=D.PROMPT_CAP)
    parser.add_argument("--layer_stride", default="auto", choices=["auto", "1", "2"], help="F7: auto = all layers under the 1.5 GB cap, else every 2nd + last")
    parser.add_argument("--batch_size", type=int, default=None, help="default 64 (GPU) / 16 (CPU)")
    parser.add_argument("--max_tokens_per_batch", type=int, default=None, help="default 16384 (GPU) / 4096 (CPU)")
    parser.add_argument("--stage_dir", default=None, help="local staging directory for the npz write (Colab: /content/dsse_stage)")
    parser.add_argument("--n_pcs", type=int, default=D.N_PCS)
    parser.add_argument("--folds", type=int, default=D.FOLDS)
    parser.add_argument("--boot", type=int, default=D.BOOT)
    parser.add_argument("--perms", type=int, default=D.PERMS)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--tasks", nargs="+", default=list(D.TASKS), choices=list(D.TASKS))
    parser.add_argument("--tiny", action="store_true", help="extract with tiny_model() instead of a download (file name tiny-gptneox_{revision})")
    parser.add_argument("--synthetic_out", default=None, help="stage synthetic: output directory (DSSE layout)")
    parser.add_argument("--n_synthetic", type=int, default=40)
    parser.add_argument("--rows_per_set", type=int, default=24)
    parser.add_argument("--note", default=None, help="line at the top of report.md (e.g. SMOKE)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def on_drive(path):
    """True for a path under a Colab Google-Drive mount (/content/drive/...)."""
    return "/content/drive/" in (os.path.abspath(path) + "/")


def raw_dir_of(args, p):
    """--raw_dir, else VH_DSSE_RAW, else {root}/dsse_raw; when the root is on Google Drive (Colab) the default is the local
    /content/dsse_raw instead: the raw red-teaming rows never go to Drive (stage_download refuses a Drive target)."""
    if args.raw_dir or os.environ.get("VH_DSSE_RAW"):
        return os.path.abspath(args.raw_dir or os.environ["VH_DSSE_RAW"])
    if on_drive(p["root"]) and os.path.isdir("/content"):
        return "/content/dsse_raw"
    return os.path.abspath(os.path.join(p["root"], "dsse_raw"))


def _say(args):
    def say(*a, **k):
        if not args.quiet:
            k.setdefault("flush", True)
            print(*a, **k)
    return say


def _revisions(args):
    return list(args.revisions) if args.revisions else [args.revision or "main"]


def stage_synthetic(args, say):
    if not args.synthetic_out:
        print("[dsse] --synthetic_out is required for --stage synthetic")
        return 1
    info = D.make_synthetic_raw(args.synthetic_out, n_datasets=args.n_synthetic, rows_per_set=args.rows_per_set, seed=args.seed)
    say(f"[dsse] synthetic DSSE layout in {args.synthetic_out}: {info['n']} datasets, behaviours {info['behaviors']}, "
        f"methods {info['methods']}, {info['rows_per_set']} rows per set")
    return 0


def stage_verify(args, raw_dir, say):
    status = D.verify_raw(raw_dir, strict=False)
    ok = True
    for path, s in status.items():
        flag = "missing" if s["missing"] else ("ok" if s["ok"] else "MISMATCH")
        ok &= bool(s["ok"])
        say(f"[dsse] {flag:8s} {path:44s} {s['bytes'] or '-':>10} {(s['sha256'] or '-')[:16]}")
    say(f"[dsse] verify {raw_dir}: {'all ' + str(len(status)) + ' files ok' if ok else 'mismatch or missing files'}")
    return 0 if ok else 1


def stage_download(args, raw_dir, say):
    if on_drive(raw_dir):
        print(f"[dsse] refusing to download the DSSE rows to {raw_dir}: raw data must not be stored on Google Drive "
              f"(use --raw_dir /content/dsse_raw or VH_DSSE_RAW)")
        return 1
    try:
        D.download_raw(raw_dir, revision=args.dsse_revision, say=say)
    except (ValueError, OSError, http.client.HTTPException) as e:
        # OSError covers FileNotFoundError, urllib.error.URLError / HTTPError (wrong revision -> 404), timeouts and
        # connection resets; HTTPException covers IncompleteRead. Exit 1 with a message instead of a traceback.
        print(f"[dsse] download failed: {type(e).__name__}: {e}")
        return 1
    say(f"[dsse] download verified: {len(D.DSSE_FILES)} files in {raw_dir}")
    return 0


def _expected_layers(short, args, n_datasets):
    info = config.MODELS.get(short)
    if args.tiny:
        return D.select_layers(3, 64, n_datasets, stride=args.layer_stride)
    if info:
        return D.select_layers(info["layers"], info["hidden"], n_datasets, stride=args.layer_stride)
    return None


def stage_extract(args, p, raw_dir, say):
    from ..models import choose_weight_files, hub_weight_files, load_model, resolve_device, set_seed, weight_fingerprint
    device = resolve_device(args.device)
    batch_size = args.batch_size or (64 if device.startswith("cuda") else 16)
    max_tokens = args.max_tokens_per_batch or (16384 if device.startswith("cuda") else 4096)
    short, hf_id, _ = (D.TINY_MODEL, None, None) if args.tiny else config.resolve_model(args.model)
    rows = D.build_rows(raw_dir, n_rows=args.n_rows, seed=D.ROW_SEED)
    rows_sha = rows.rows_sha256()
    say(f"[dsse] rows: {len(rows.dataset_ids)} datasets, {len(rows.rows)} sampled rows ({args.n_rows} per set, ROW_SEED {D.ROW_SEED}), "
        f"rows_sha256 {rows_sha[:12]}")
    os.makedirs(p["features"], exist_ok=True)
    for revision in _revisions(args):
        out_path = D.feature_path(p, short, revision)
        files = None if args.tiny else hub_weight_files(hf_id, revision)
        used, ambiguous = choose_weight_files(files)
        expect = {"version": D.VERSION, "n_rows": args.n_rows, "row_seed": D.ROW_SEED, "max_length": args.max_length,
                  "prompt_cap": args.prompt_cap, "hf_id": hf_id, "revision": revision, "rows_sha256": rows_sha,
                  "layers": _expected_layers(short, args, len(rows.dataset_ids)), "weight_fingerprint": weight_fingerprint(used)}
        if ambiguous:
            say(f"[dsse] {hf_id}@{revision}: Hub branch holds model.safetensors AND a sharded index; the shards are used "
                f"({len(used)} files), the single file is ignored")
        try:
            done = D.feature_done(out_path, expect)
        except RuntimeError as e:
            if not args.force:
                raise
            say(f"[dsse] --force: rebuilding ({e})")
            done = False
        if done and D.stale_weight_source(D.feature_manifest(out_path), ambiguous):
            say(f"[dsse] {os.path.basename(out_path)}: extracted before the weight-source check from an ambiguous Hub branch "
                f"(no weight_fingerprint in its manifest): rebuilding, its weights may be main's")
            done = False
        if done and not args.force:
            say(f"[dsse] skip {out_path}: feature file exists for these settings")
            continue
        if not args.force and D.recover_staged(args.stage_dir, out_path, expect, say=say):
            continue                                          # a complete pair whose move to Drive failed last time
        set_seed(args.seed)
        t0 = time.time()
        if args.tiny:
            model, tokenizer = D.tiny_model(seed=args.seed)
            model.vh_info["revision"] = revision
        else:
            model, tokenizer = load_model(hf_id, revision=revision, device=device, dtype=args.dtype, attn_impl=args.attn_impl)
        info = model.vh_info
        L = info["geometry"]["layers"]
        d = info["hidden"]
        layers = D.select_layers(L, d, len(rows.dataset_ids), stride=args.layer_stride)
        say(f"[dsse] weights: {info.get('weight_source')} {[p for p, _ in (info.get('weight_files') or [])]} fingerprint "
            f"{str(info.get('weight_fingerprint'))[:12]}")
        say(f"[dsse] model {info['model']}@{info['revision']} ({info.get('hf_id')}, commit {str(info.get('commit_sha'))[:12]}): "
            f"L/d = {L}/{d}, dtype {info['dtype']}, attn {info['attn_impl']}, device {info['device']}, load {time.time() - t0:.0f}s")
        say(f"[dsse] layers {len(layers)}/{L} ({'all' if len(layers) == L else 'every 2nd + last'}), fp16 file "
            f"{D.feature_bytes(len(rows.dataset_ids), len(layers), d) / 1e9:.3f} GB, batch {batch_size} / {max_tokens} tokens")
        manifest = D.extract_dsse(model, tokenizer, rows, out_path, layers=layers, batch_size=batch_size, max_tokens_per_batch=max_tokens,
                                  max_length=args.max_length, prompt_cap=args.prompt_cap, stage_dir=args.stage_dir,
                                  show_progress=not args.no_progress and not args.quiet, say=say)
        ts = manifest["token_stats"]
        say(f"[dsse] {os.path.basename(out_path)}: {ts['n_rows']} rows, {ts['n_tokens']} tokens, cut rate {ts['cut_rate']:.4f} "
            f"(prompt {ts['n_cut_prompt']}, response {ts['n_cut_response']}, zero-response rows {ts['n_zero_response']}), "
            f"{manifest['tokens_per_second']} tok/s, {manifest['seconds']}s, {os.path.getsize(out_path) / 1e9:.3f} GB")
        del model
    return 0


def _feature_files(args, p):
    if args.models is None and args.revisions is None and args.revision is None and args.stage != "all":
        files = [f for f in sorted(glob.glob(os.path.join(p["features"], "*.npz"))) if not f.endswith(".tmp.npz")]
        skipped = [f for f in files if not os.path.exists(D.manifest_path(f))]
        if skipped:                                           # partial copy (session died during the move) or foreign file
            print(f"[dsse] skipping {len(skipped)} feature file(s) without manifest sidecar: "
                  f"{[os.path.basename(f) for f in skipped]}")
        return [f for f in files if os.path.exists(D.manifest_path(f))]
    models = args.models or [D.TINY_MODEL if args.tiny else config.model_short_name(args.model)]
    return [D.feature_path(p, m, r) for m in models for r in _revisions(args)]


def _settings(args):
    return D.probe_settings(n_pcs=args.n_pcs, folds=args.folds, boot=args.boot, perms=args.perms, seed=args.seed, tasks=args.tasks,
                            n_jobs=args.n_jobs)


def stage_probe(args, p, raw_dir, say):
    from ..models import set_seed
    set_seed(args.seed)
    files = _feature_files(args, p)
    missing = [f for f in files if not os.path.exists(f)]
    if missing or not files:
        print(f"[dsse] missing feature files: {missing or '(none found)'} (run --stage extract first)")
        return 1
    out_dir = p["dsse"]
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        t0 = time.time()
        D.probe_feature_file(f, raw_dir, out_dir, _settings(args), n_jobs=args.n_jobs, force=args.force, say=say)
        say(f"[dsse] probe {os.path.basename(f)}: {time.time() - t0:.0f}s")
    return 0


def _warn_n_rows(args, p):
    """Feature manifests under p['features'] extracted with another --n_rows than the baselines: the report will then
    suspend the paired comparisons (F18), so say so here, where the fix is cheap."""
    for man_path in sorted(glob.glob(os.path.join(p["features"], "*.manifest.json"))):
        try:
            with open(man_path) as f:
                n = json.load(f).get("n_rows")
        except (OSError, ValueError):
            continue
        if n is not None and int(n) != int(args.n_rows):
            print(f"[dsse] WARNING: {os.path.basename(man_path)} was extracted with n_rows = {n}, baselines use --n_rows {args.n_rows}: "
                  f"the report will suspend the paired comparisons for that file (rerun baselines with --n_rows {n} --force)", flush=True)


def stage_baselines(args, p, raw_dir, say):
    from ..models import set_seed
    set_seed(args.seed)
    out_dir = p["dsse"]
    os.makedirs(out_dir, exist_ok=True)
    _warn_n_rows(args, p)
    t0 = time.time()
    D.run_baselines(raw_dir, out_dir, n_rows=args.n_rows, seed_rows=D.ROW_SEED, settings=_settings(args), force=args.force, say=say,
                    n_jobs=max(1, args.n_jobs) if args.n_jobs > 0 else -1)
    say(f"[dsse] baselines: {time.time() - t0:.0f}s")
    return 0


def stage_report(args, p, say):
    out_dir = p["dsse"]
    if not os.path.isdir(out_dir):
        print(f"[dsse] no results in {out_dir}")
        return 1
    results = D.collect_results(out_dir, models=args.models, revisions=args.revisions)
    if not len(results["probe"]) and results["baselines"] is None:
        print(f"[dsse] no probe results in {out_dir} (run --stage probe / baselines first)")
        return 1
    key = D.key_table(results, D.make_oof_loader(out_dir), boot=args.boot, seed=args.seed)
    figures = D.make_figures(out_dir, results)
    path = D.write_report(out_dir, results, key, figures, note=args.note)
    key.to_csv(os.path.join(out_dir, "key_table.csv"), index=False)
    say(f"[dsse] wrote {path} and {sum(len(v) for v in figures.values())} figures")
    if len(key):
        cols = [c for c in ("model", "revision", "behavior_layer", "behavior_pooling", "behavior_random", "behavior_method", "d_tfidf",
                            "d_tfidf_lo", "d_step0", "d_step0_lo", "miss_method", "k95", "mean_angle", "verdict") if c in key.columns]
        say(key[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        notes = [n for n in key.get("pairing_note", []) if isinstance(n, str) and n]
        if notes:
            print("[dsse] WARNING: paired comparisons suspended (different row samples): " + " | ".join(sorted(set(notes))), flush=True)
    return 0


def run(args):
    say = _say(args)
    p = config.paths(args.root)
    raw_dir = raw_dir_of(args, p)
    if args.stage == "synthetic":
        return stage_synthetic(args, say)
    if args.stage == "verify":
        return stage_verify(args, raw_dir, say)
    if args.stage == "download":
        return stage_download(args, raw_dir, say)
    if args.stage in ("extract", "probe", "baselines", "all") and not os.path.exists(os.path.join(raw_dir, "scores.csv")) and args.stage != "all":
        print(f"[dsse] raw directory {raw_dir} has no scores.csv (run --stage download, or --stage synthetic --synthetic_out {raw_dir})")
        return 1
    if args.stage == "extract":
        return stage_extract(args, p, raw_dir, say)
    if args.stage == "probe":
        return stage_probe(args, p, raw_dir, say)
    if args.stage == "baselines":
        return stage_baselines(args, p, raw_dir, say)
    if args.stage == "report":
        return stage_report(args, p, say)
    # all: download (or verify when raw_dir is complete) -> extract -> baselines -> probe -> report
    complete = all(s["ok"] for s in D.verify_raw(raw_dir, strict=False).values())
    if complete:
        rc = stage_verify(args, raw_dir, say)
    elif os.path.exists(os.path.join(raw_dir, "scores.csv")) and not os.path.exists(D.raw_file(raw_dir, "datasets/" + D.BEHAVIORS[0] + ".jsonl.xz")):
        say(f"[dsse] {raw_dir} holds a non-DSSE layout (synthetic?); download skipped")
        rc = 0
    else:
        rc = stage_download(args, raw_dir, say)
    if rc:
        return rc
    for fn in (stage_extract, stage_baselines, stage_probe):
        rc = fn(args, p, raw_dir, say)
        if rc:
            return rc
    return stage_report(args, p, say)
