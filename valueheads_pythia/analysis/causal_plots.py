"""Causal-test figures (figures/causal/, DESIGN 6.4): Base and TopK against the layer-matched Random distribution.

One figure per (revision, metric[-eval_value][-context]) from the rows of results/eval/summary.csv written by
`eval-likelihood`: every run (intervention, alpha, selection, head values) is one horizontal lane showing the Random
mean +- std (thin whisker = min / max over the R sets), the Base value (square) and the TopK value (diamond) of the
primary statistic; the empirical p-value and the paired delta with its bootstrap CI are written beside the lane. One
x-axis per figure (the primary statistic), a legend for the three marks, colors from the project palette
(analysis.heatmaps.CATEGORICAL: Base = slot 1, TopK = slot 2; the Random distribution is neutral ink; validated
palette, see the dataviz reference). Static PNG via matplotlib (Agg).
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .heatmaps import CATEGORICAL  # noqa: E402

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#9a9994", "grid": "#e6e5e1"}
BASE_COLOR, TOPK_COLOR = CATEGORICAL[0], CATEGORICAL[1]


def _is_str(v):
    return isinstance(v, str) and v and v != "nan"


def metric_tag(row):
    """tqa_pairwise[-eval_value][-lmeval]: the metric tag eval-likelihood uses for its jsonl files."""
    tag = str(row["metric"])
    if _is_str(row.get("eval_value")):
        tag += f"-{row['eval_value']}"
    if _is_str(row.get("context")) and row["context"] != "probing":
        tag += f"-{row['context']}"
    return tag


def lane_label(row):
    """'mask [honesty]', 'steer α=0.5 [honesty+harmless]', 'mask (signif) [honesty]', 'mask (disjoint) [honesty]'."""
    lab = str(row["intervention"])
    if lab == "steer" and pd.notna(row.get("alpha")):
        lab += f" α={float(row['alpha']):g}"
    extras = []
    if _is_str(row.get("select")) and row["select"] != "topk":
        extras.append(str(row["select"]))
    if _is_str(row.get("eval_groups")):
        extras.append(f"half {row['eval_groups']}")
    if _is_str(row.get("eval_split")) and row["eval_split"] != "val":
        extras.append(str(row["eval_split"]))
    if extras:
        lab += " (" + ", ".join(extras) + ")"
    if _is_str(row.get("head_values")):
        lab += f" [{row['head_values']}]"
    return lab


def causal_rows(df):
    """Rows of an eval summary that carry a Base and a TopK value of their primary statistic (intervention != none)."""
    if df is None or not {"primary", "intervention", "metric"} <= set(df.columns):
        return pd.DataFrame()
    df = df[df["intervention"].astype(str) != "none"].copy()
    if "run_key" in df.columns:
        df = df.drop_duplicates(subset=["run_key"], keep="last")
    base, topk = [], []
    for _, r in df.iterrows():
        prim = r["primary"]
        base.append(r.get(f"base_{prim}", np.nan))
        topk.append(r.get(f"topk_{prim}", np.nan))
    df["_base"], df["_topk"] = base, topk
    df = df[pd.notna(df["_topk"]) & pd.notna(df["_base"])]
    df["_tag"] = [metric_tag(r) for _, r in df.iterrows()]
    df["_lane"] = [lane_label(r) for _, r in df.iterrows()]
    return df.reset_index(drop=True)


def plot_causal(rows, out_path, title=None, dpi=150):
    """Dot-and-range chart of one metric: one lane per run (Random mean +- std with min / max whisker, Base, TopK)."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.5, 1.3 + 0.6 * n))
    for i, (_, r) in enumerate(rows.iterrows()):
        y = n - 1 - i
        R = int(r.get("R", 0) or 0)
        if R > 0 and pd.notna(r.get("random_mean")):
            if pd.notna(r.get("random_min")) and pd.notna(r.get("random_max")):
                ax.plot([r["random_min"], r["random_max"]], [y, y], color=INK["muted"], lw=1.0, solid_capstyle="butt", zorder=1)
            std = r["random_std"] if pd.notna(r.get("random_std")) else 0.0
            ax.errorbar(r["random_mean"], y, xerr=std, fmt="o", color=INK["secondary"], ms=6, capsize=3, lw=1.5, zorder=2)
        ax.plot(r["_base"], y, marker="s", color=BASE_COLOR, ms=7, ls="none", zorder=3)
        ax.plot(r["_topk"], y, marker="D", color=TOPK_COLOR, ms=7, ls="none", zorder=4)
        note = []
        if pd.notna(r.get("p_emp_primary")):
            note.append(f"p_emp = {float(r['p_emp_primary']):.2f} (R = {R})")
        if pd.notna(r.get("delta_mean")):
            ci = f" [{float(r['delta_ci_lo']):+.3f}, {float(r['delta_ci_hi']):+.3f}]" if pd.notna(r.get("delta_ci_lo")) else ""
            note.append(f"Δ = {float(r['delta_mean']):+.3f}{ci}")
        if note:
            ax.annotate("  ".join(note), xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8, color=INK["secondary"])
    ax.set_yticks(range(n))
    ax.set_yticklabels([rows["_lane"].iloc[n - 1 - i] for i in range(n)], fontsize=9, color=INK["primary"])
    ax.set_xlabel(str(rows["primary"].iloc[0]), fontsize=9, color=INK["secondary"])
    ax.tick_params(axis="x", labelsize=8, colors=INK["secondary"])
    ax.grid(axis="x", color=INK["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK["grid"])
    ax.set_ylim(-0.7, n - 0.3)
    handles = [Line2D([], [], marker="s", color=BASE_COLOR, ls="none", ms=7, label="Base"),
               Line2D([], [], marker="D", color=TOPK_COLOR, ls="none", ms=7, label="TopK"),
               Line2D([], [], marker="o", color=INK["secondary"], ls="-", ms=6, label="Random: Mittel ± Std (dünn: min–max)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10, y=0.995 + 0.03 / max(n, 1), color=INK["primary"])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def causal_figures(eval_summaries, model, out_dir, dpi=150):
    """{(revision, metric tag): path}: one figure per (revision, metric tag) from {revision: eval summary DataFrame}."""
    out = {}
    for rev, df in (eval_summaries or {}).items():
        rows = causal_rows(df)
        if rows.empty:
            continue
        for tag, sub in rows.groupby("_tag", sort=False):
            path = os.path.join(out_dir, f"{model}_{rev}_{tag}.png")
            plot_causal(sub, path, title=f"{model} @ {rev}: {tag}", dpi=dpi)
            out[(str(rev), str(tag))] = path
    return out
