"""`analyze`: heatmaps, layer profiles, sweep curves, correlations, direction angles, summary and report of a model.

    python -m valueheads_pythia.cli analyze --model pythia-160m                 # everything found under results/
    python -m valueheads_pythia.cli analyze --model pythia-160m --revision step143000 --values honesty

Torch-free: works on the JSON / npz artifacts of probe, probe-stats and directions. Outputs (DESIGN 6.4):
    figures/heatmaps/{model}_{revision}_{value}[_{mode}].png, {model}_{revision}_{mode}_all.png, ..._layers.png
    figures/sweep/{model}_{value}[_{mode}].png, results/sweep/{model}_{value}[_{mode}]_curve.csv
    figures/correlations/*.png, results/correlations_{model}.csv
    figures/angles/{model}_{revision}.png, results/direction_angles_{model}.csv
    results/summary_{model}.csv / .json, results/report_{model}.md
"""
import glob
import json
import os
import re
import time

import pandas as pd

from .. import config
from ..analysis import correlations as corr
from ..analysis import direction_angles as ang
from ..analysis.heatmaps import heatmap_figure, layer_profile_figure
from ..analysis.report import build_summary, load_eval_summaries, load_judge_summary, render_report, write_summary
from ..analysis.sweep_curves import build_sweep, sort_revisions
from ..cli import add_common_args
from ..directions import load_directions
from ..probing import load_head_scores, select_topk
from ..probing_stats import split_stability


def add_args(parser):
    add_common_args(parser, device=False, mode=False, seed=False)
    parser.add_argument("--values", nargs="*", default=None, help="restrict to these values")
    parser.add_argument("--modes", nargs="*", default=None, choices=config.MODES, help="restrict to these modes (default: all found)")
    parser.add_argument("--ratio", type=float, default=config.TOP_RATIO)
    parser.add_argument("--final_revision", default=None, help="reference checkpoint of the sweep curves (default: largest step)")
    parser.add_argument("--no_figures", action="store_true", help="tables and report only")
    parser.add_argument("--quiet", action="store_true")


def discover_tables(p, model, revision=None, values=None, modes=None):
    """{(revision, value, mode, feature_tag): (path, df)} of the model's head-score JSONs under results/heads."""
    out = {}
    for path in sorted(glob.glob(os.path.join(p["heads"], f"{model}_*.json"))):
        try:
            df, meta = load_head_scores(path)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        if meta.get("model") != model:
            continue
        if not {"auc", "acc", "bal_acc", "rank"} <= set(df.columns):
            continue        # derived rankings (e.g. *_meansplits.json: auc_mean over splits) are not probe tables
        key = (meta.get("revision", "main"), meta.get("value"), meta.get("mode", "pythia"), meta.get("feature_tag"))
        if revision and key[0] != revision:
            continue
        if values and key[1] not in values:
            continue
        if modes and key[2] not in modes:
            continue
        out[key] = (path, df)
    return out


def primary_tables(tables):
    """{(revision, mode): {value: df}} of the untagged (primary) tables."""
    out = {}
    for (rev, value, mode, tag), (_, df) in tables.items():
        if tag is None:
            out.setdefault((rev, mode), {})[value] = df
    return out


def run(args):
    t_all = time.time()
    p = config.paths(args.root)
    short, _, _ = config.resolve_model(args.model)
    tables = discover_tables(p, short, args.revision, args.values, args.modes)
    if not tables:
        raise SystemExit(f"[analyze] no head-score tables for {short} under {p['heads']} (run probe / sweep first)")
    say = (lambda *a, **k: None) if args.quiet else print
    say(f"[analyze] {short}: {len(tables)} head tables, revisions {sort_revisions({k[0] for k in tables})}, "
        f"values {sorted({k[1] for k in tables})}, modes {sorted({k[2] for k in tables})}")
    figures, corr_rows, corr_mats, sweep_tables, angle_tables, stability = {}, [], {}, {}, [], {}
    prim = primary_tables(tables)

    # heatmaps + layer profiles per (revision, mode)
    if not args.no_figures:
        for (rev, mode), by_value in prim.items():
            tag = "" if mode == "pythia" else f"_{mode}"
            for value, df in by_value.items():
                path = os.path.join(p["heatmaps"], f"{short}_{rev}_{value}{tag}.png")
                heatmap_figure({value: df}, path, mode=mode, ratio=args.ratio, title=f"{short} @ {rev}: {value} ({mode})")
                figures[("heatmap", rev, f"{value}/{mode}")] = path
            if len(by_value) > 1:
                path = os.path.join(p["heatmaps"], f"{short}_{rev}_{mode}_all.png")
                heatmap_figure(by_value, path, mode=mode, ratio=args.ratio, title=f"{short} @ {rev} ({mode})")
                figures[("heatmap", rev, f"all/{mode}")] = path
            path = os.path.join(p["heatmaps"], f"{short}_{rev}_{mode}_layers.png")
            layer_profile_figure(by_value, path, score="auc", ratio=args.ratio, title=f"{short} @ {rev} ({mode}): Layer-Profile")
            figures[("layers", rev, mode)] = path
        say(f"[analyze] heatmaps + layer profiles for {len(prim)} (revision, mode) pairs -> {p['heatmaps']}")

    # sweep curves per (value, mode)
    by_value_mode = {}
    for (rev, mode), by_value in prim.items():
        for value, df in by_value.items():
            by_value_mode.setdefault((value, mode), {})[rev] = df
    for (value, mode), by_rev in by_value_mode.items():
        if len(by_rev) < 2:
            continue
        tag = "" if mode == "pythia" else f"_{mode}"
        csv = os.path.join(p["sweep"], f"{short}_{value}{tag}_curve.csv")
        fig = None if args.no_figures else os.path.join(p["figures_sweep"], f"{short}_{value}{tag}.png")
        sweep_tables[(value, mode)] = build_sweep(by_rev, short, value, mode, csv, fig, args.ratio, args.final_revision)
        if fig:
            figures[("sweep", value, mode)] = fig
        # correlations across checkpoints
        for kind, mat in corr.across_tables(by_rev).items():
            name = f"{value} ({mode}) über Checkpoints – {kind}"
            corr_mats[name] = mat
            corr_rows.append(corr.tidy(mat, kind).assign(scope="checkpoints", value=value, mode=mode))
            if not args.no_figures:
                path = os.path.join(p["figures"], "correlations", f"{short}_{value}{tag}_checkpoints_{kind}.png")
                corr.plot_correlation(mat, path, title=name)
                figures[("corr", name)] = path
    say(f"[analyze] sweep curves for {sorted(sweep_tables)}")

    # correlations across values + direction angles per (revision, mode)
    for (rev, mode), by_value in prim.items():
        tag = "" if mode == "pythia" else f"_{mode}"
        if len(by_value) >= 2:
            for kind, mat in corr.across_values(by_value, args.ratio).items():
                name = f"{rev} ({mode}) über Werte – {kind}"
                corr_mats[name] = mat
                corr_rows.append(corr.tidy(mat, kind).assign(scope="values", revision=rev, mode=mode))
                if not args.no_figures:
                    path = os.path.join(p["figures"], "correlations", f"{short}_{rev}{tag}_values_{kind}.png")
                    corr.plot_correlation(mat, path, title=name)
                    figures[("corr", name)] = path
        dirs, heads = {}, {}
        for value, df in by_value.items():
            dpath = config.directions_path(p, short, rev, value, mode)
            if os.path.exists(dpath):
                dirs[value] = load_directions(dpath)[0]
                heads[value] = select_topk(df, args.ratio, df.attrs.get("score", "auc"))
        if len(dirs) >= 2:
            table = ang.angle_table(dirs, heads, short, rev)
            table["mode"] = mode
            # reference row per pair over all L * H heads (not only the shared top-k heads)
            ref = ang.all_head_angle_table(dirs, short, rev).rename(
                columns={"n": "all_n", "mean_deg": "all_mean_deg", "median_deg": "all_median_deg", "std_deg": "all_std_deg"})
            table = table.merge(ref[["pair", "all_n", "all_mean_deg", "all_median_deg", "all_std_deg"]], on="pair", how="left")
            table.attrs["angles_by_pair"] = ang.angle_table(dirs, heads, short, rev).attrs["angles_by_pair"]
            angle_tables.append(table)
            if not args.no_figures:
                path = os.path.join(p["figures"], "angles", f"{short}_{rev}{tag}.png")
                ang.plot_angles(table, path, title=f"{short} @ {rev} ({mode}): Richtungswinkel auf geteilten Top-15 %-Heads")
                figures[("angles", f"{rev}/{mode}")] = path

    # split stability (head-only tables tagged split{seed} next to the primary one; a split table without its
    # primary table, e.g. a `probe --train_dir` run alone, is listed in the summary but not in the stability block)
    for (rev, value, mode, tag), (_, df) in tables.items():
        if tag and re.fullmatch(r"split\d+", str(tag)) and value in prim.get((rev, mode), {}):
            key = (rev, value, mode)
            stability.setdefault(key, {"dfs": [prim[(rev, mode)][value]], "names": ["seed42"]})
            stability[key]["dfs"].append(df)
            stability[key]["names"].append(tag.replace("split", "seed"))
    stability = {f"{rev} / {value} ({mode})": split_stability(v["dfs"], args.ratio, names=v["names"]) for (rev, value, mode), v in stability.items()}

    # tables, summary, report
    if corr_rows:
        pd.concat(corr_rows, ignore_index=True).to_csv(os.path.join(p["results"], f"correlations_{short}.csv"), index=False)
    angles = pd.concat(angle_tables, ignore_index=True) if angle_tables else None
    if angles is not None:
        angles.drop(columns=["shared_heads"]).to_csv(os.path.join(p["results"], f"direction_angles_{short}.csv"), index=False)
    summary = build_summary({k: df for k, (_, df) in tables.items()}, args.ratio, {k: path for k, (path, _) in tables.items()})
    csv = write_summary(summary, os.path.join(p["results"], f"summary_{short}.csv"), os.path.join(p["results"], f"summary_{short}.json"))
    manifest = None
    if os.path.exists(p["manifest"]):
        with open(p["manifest"]) as f:
            manifest = json.load(f)
    eval_summaries = load_eval_summaries(p["eval"], short)
    if eval_summaries and not args.no_figures:
        from ..analysis.causal_plots import causal_figures
        for key, path in causal_figures(eval_summaries, short, p["figures_causal"]).items():
            figures[("causal",) + key] = path
        say(f"[analyze] causal figures -> {p['figures_causal']}")
    report = os.path.join(p["results"], f"report_{short}.md")
    render_report(short, summary, sweep_tables, corr_mats, angles, manifest, figures, eval_summaries, stability, report,
                  args.ratio, judge_summary=load_judge_summary(p["judge"], short))
    say(f"[analyze] summary {csv}; report {report}; {len(figures)} figures; {time.time() - t_all:.1f}s")
    say(summary[["revision", "value", "mode", "feature_tag", "top_layer", "top_head", "top_auc", "ci_lo", "ci_hi", "perm_thr", "n_signif", "conf_top1"]]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0
