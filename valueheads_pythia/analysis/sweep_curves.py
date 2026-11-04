"""Checkpoint sweep curves (DESIGN 3.4, 6.4): per value over revisions the top AUC (with CI and permutation
threshold), the number of significant heads and the Spearman / Jaccard agreement with the final checkpoint,
drawn against log(training step). Writes results/sweep/{model}_{value}_curve.csv and figures/sweep/*.png.
"""
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..probing_stats import spearman_jaccard  # noqa: E402
from .heatmaps import CATEGORICAL, geometry_of, layer_profile  # noqa: E402

CURVE_COLUMNS = ["model", "value", "mode", "revision", "step", "top_flat", "top_layer", "top_head", "top_auc", "top_acc",
                 "ci_lo", "ci_hi", "perm_thr", "n_signif", "k", "spearman_to_final", "jaccard_to_final", "conf_top1",
                 "conf_topk", "majority", "pos_rate", "n_train", "n_val", "n_auc_gt_0.7", "n_acc_gt_0.8", "best_layer_max_auc"]


def parse_step(revision):
    """'step143000' -> 143000, 'main' -> None."""
    m = re.fullmatch(r"step(\d+)", str(revision))
    return int(m.group(1)) if m else None


def sort_revisions(revisions):
    """Revisions by training step (non-step names last, alphabetically)."""
    return sorted(revisions, key=lambda r: (parse_step(r) is None, parse_step(r) if parse_step(r) is not None else 0, str(r)))


def final_revision(revisions):
    """The revision with the largest step (reference of the Spearman / Jaccard columns)."""
    ordered = [r for r in sort_revisions(revisions) if parse_step(r) is not None]
    return ordered[-1] if ordered else sort_revisions(revisions)[-1]


def summarize_table(df, ratio=0.15):
    """Per-table summary (top head, CI, threshold, significant set, confirmation) shared by curves and report."""
    score = df.attrs.get("score", "auc")
    best = df.sort_values("rank").iloc[0]
    L, H = geometry_of(df)
    conf = df.attrs.get("confirmation") or {}
    prof = layer_profile(df, "auc", ratio)
    return {"top_flat": int(best.flat_idx), "top_layer": int(best.layer), "top_head": int(best["head"]),
            "top_auc": float(best.auc), "top_acc": float(best.acc), "top_bal_acc": float(best.bal_acc),
            "ci_lo": float(best.ci_lo) if "ci_lo" in df else np.nan, "ci_hi": float(best.ci_hi) if "ci_hi" in df else np.nan,
            "perm_thr": df.attrs.get("perm_threshold", np.nan), "n_signif": int(df.attrs.get("n_signif", 0)) if df.attrs.get("n_signif") is not None else 0,
            "k": int(L * H * ratio), "conf_top1": conf.get("top1_conf_mean", np.nan), "conf_topk": conf.get("topk_conf_mean", np.nan),
            "winners_curse_gap": conf.get("winners_curse_gap", np.nan), "majority": float(df.majority.iloc[0]),
            "pos_rate": float(df.pos_rate.iloc[0]), "n_train": int(df.n_train.iloc[0]), "n_val": int(df.n_val.iloc[0]),
            "n_auc_gt_0.7": int((df.auc > 0.7).sum()), "n_acc_gt_0.8": int((df.acc > 0.8).sum()),
            "best_layer_max_auc": int(prof["auc_max"].idxmax()), "score": score, "p_adj_top": float(best.p_adj) if "p_adj" in df else np.nan}


def sweep_table(dfs_by_revision, model, value, mode="pythia", ratio=0.15, final=None):
    """One row per revision (sorted by step) with the curve statistics; Spearman / Jaccard are against `final`
    (default: the largest step), computed on the score vectors / top-k sets of the two tables."""
    revisions = sort_revisions(dfs_by_revision)
    final = final or final_revision(revisions)
    rows = []
    for rev in revisions:
        df = dfs_by_revision[rev]
        s = summarize_table(df, ratio)
        rho, jac, k = spearman_jaccard(df, dfs_by_revision[final], ratio, df.attrs.get("score", "auc"))
        rows.append({"model": model, "value": value, "mode": mode, "revision": rev, "step": parse_step(rev),
                     **{c: s[c] for c in CURVE_COLUMNS if c in s}, "spearman_to_final": rho, "jaccard_to_final": jac, "k": k})
    table = pd.DataFrame(rows)[CURVE_COLUMNS]
    table.attrs["final"] = final
    return table


def layer_evolution(dfs_by_revision, score="auc"):
    """(L, n_revisions) matrix of the per-layer max score across the sweep (columns sorted by step)."""
    revisions = sort_revisions(dfs_by_revision)
    cols = [layer_profile(dfs_by_revision[r], score)[f"{score}_max"].to_numpy() for r in revisions]
    return np.stack(cols, axis=1), revisions


def write_curve_csv(table, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _x_positions(table):
    """log-scale x positions; step0 is drawn at the position of 'step 1' and labelled explicitly."""
    steps = table["step"].to_numpy(dtype=float)
    steps = np.where(np.isnan(steps), np.nanmax(steps) if np.isfinite(np.nanmax(steps)) else 1.0, steps)
    return np.maximum(steps, 1.0)


def plot_sweep(table, out_path, dfs_by_revision=None, title=None, dpi=150):
    """Four panels: top AUC (CI band, permutation threshold), n_signif, Spearman + Jaccard to the final checkpoint,
    and (when the tables are given) the per-layer max AUC over checkpoints as a layer x checkpoint heatmap."""
    x = _x_positions(table)
    labels = [str(r) for r in table["revision"]]
    with_layers = dfs_by_revision is not None and len(dfs_by_revision) >= 2
    fig, axes = plt.subplots(1, 4 if with_layers else 3, figsize=(17 if with_layers else 13, 3.8))
    c1, c2, c3 = CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2]
    ax = axes[0]
    if table["ci_lo"].notna().any():
        ax.fill_between(x, table["ci_lo"], table["ci_hi"], color=c1, alpha=0.18, lw=0, label="95 %-Bootstrap-CI (Top-Head)")
    ax.plot(x, table["top_auc"], marker="o", ms=5, lw=2, color=c1, label="Top-AUC (Val)")
    if table["perm_thr"].notna().any():
        ax.plot(x, table["perm_thr"], marker="x", ms=5, lw=1.2, ls=":", color="#52514e", label="Permutationsschwelle (95 %)")
    ax.axhline(0.5, color="#9a9994", lw=0.8)
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[1]
    ax.plot(x, table["n_signif"], marker="o", ms=5, lw=2, color=c2, label="signifikante Heads")
    ax.plot(x, table["n_auc_gt_0.7"], marker="s", ms=4, lw=1.2, ls="--", color=c3, label="Heads mit AUC > 0.7")
    ax.axhline(table["k"].iloc[0], color="#9a9994", lw=0.8, ls=":")
    ax.text(x[0], table["k"].iloc[0], f" k = {int(table['k'].iloc[0])} (15 %)", fontsize=7, va="bottom", color="#52514e")
    ax.set_ylabel("Anzahl Heads")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    ax.plot(x, table["spearman_to_final"], marker="o", ms=5, lw=2, color=c1, label="Spearman zu " + str(table.attrs.get("final", "final")))
    ax.plot(x, table["jaccard_to_final"], marker="s", ms=4, lw=2, color=c2, label="Jaccard@k zu " + str(table.attrs.get("final", "final")))
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Rang-Übereinstimmung")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axes[:3]:
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Trainings-Checkpoint (log step; step0 bei 1)")
        ax.grid(color="#e6e5e1", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if with_layers:
        mat, revs = layer_evolution(dfs_by_revision, "auc")
        ax = axes[3]
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.5, vmax=max(0.7, float(np.nanmax(mat))), origin="lower", interpolation="nearest")
        ax.set_xticks(range(len(revs)))
        ax.set_xticklabels(revs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Layer")
        ax.set_title("max AUC je Layer", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_sweep(dfs_by_revision, model, value, mode, csv_path, fig_path=None, ratio=0.15, final=None):
    """sweep_table + csv (+ figure); returns the table."""
    table = sweep_table(dfs_by_revision, model, value, mode, ratio, final)
    write_curve_csv(table, csv_path)
    if fig_path:
        plot_sweep(table, fig_path, dfs_by_revision, title=f"{model} / {value} ({mode}): Checkpoint-Sweep")
    return table
