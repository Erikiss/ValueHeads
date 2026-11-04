"""Layer x head heatmaps of head scores (paper Fig. 2 style) and layer profiles.

mode "paper" reproduces analysis/plot_head_heatmaps.py of the reference repo (MIT): the score of each head is
the min-max normalized, squared accuracy, drawn with imshow(vmin=0.6, vmax=1.0, origin="lower") so that 0.60
is the lower clip of the normalized score (the paper's "maximum cap set at 0.60", DESIGN section 2).
mode "pythia" draws the raw val ROC-AUC (chance = 0.5) and marks the significant heads (bootstrap CI lower
bound above the permutation threshold, DESIGN 5.8) when the table carries the statistics.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..probing import scores_vector, select_topk  # noqa: E402

SEQUENTIAL_CMAP = "viridis"      # single perceptually uniform ramp (paper figures use it too)


def geometry_of(df):
    """(L, H) of a head table from df.attrs (falls back to the max layer / head index)."""
    L = int(df.attrs.get("layers") or int(df["layer"].max()) + 1)
    H = int(df.attrs.get("heads") or int(df["head"].max()) + 1)
    return L, H


def score_map_paper(scores_flat, num_layers, num_heads):
    """((s - min) / (max - min)) ** 2 as a (L, H) grid; identical to the repo's score_map (NaN-safe)."""
    arr = np.asarray(scores_flat, dtype=np.float64).reshape(num_layers, num_heads)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi > lo:
        arr = ((arr - lo) / (hi - lo)) ** 2
    else:
        arr = np.zeros_like(arr)
    return arr


def score_grid(df, score=None, mode="pythia"):
    """(L, H) grid: normalized squared accuracy (mode paper) or the raw score (default auc) in mode pythia."""
    L, H = geometry_of(df)
    if mode == "paper":
        return score_map_paper(scores_vector(df, score or "acc"), L, H)
    return scores_vector(df, score or "auc").reshape(L, H)


def mask_grid(df, flat_indices):
    """Boolean (L, H) grid marking the given flat indices."""
    L, H = geometry_of(df)
    m = np.zeros(L * H, dtype=bool)
    m[[int(i) for i in flat_indices]] = True
    return m.reshape(L, H)


def layer_profile(df, score="auc", ratio=0.15):
    """Per-layer summary: max / mean score, number of top-k heads, number of significant heads (if available)."""
    L, H = geometry_of(df)
    topk = set(select_topk(df, ratio, df.attrs.get("score", score)))
    rows = []
    for layer in range(L):
        sub = df[df["layer"] == layer]
        rows.append({"layer": layer, f"{score}_max": float(sub[score].max()) if len(sub) else np.nan,
                     f"{score}_mean": float(sub[score].mean()) if len(sub) else np.nan,
                     "n_topk": int(sub["flat_idx"].isin(topk).sum()),
                     "n_signif": int(sub["signif"].sum()) if "signif" in sub else 0})
    return pd.DataFrame(rows)


def _draw_panel(ax, grid, vmin, vmax, title, cmap=SEQUENTIAL_CMAP, signif=None, topk=None):
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    L, H = grid.shape
    if topk is not None:
        ys, xs = np.nonzero(topk)
        ax.scatter(xs, ys, marker="s", s=140, facecolors="none", edgecolors="white", linewidths=0.9, label="Top-15 %")
    if signif is not None:
        ys, xs = np.nonzero(signif)
        ax.scatter(xs, ys, marker="o", s=28, c="white", edgecolors="black", linewidths=0.6, label="signifikant")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(0, H, max(1, H // 8)))
    ax.set_yticks(range(0, L, max(1, L // 8)))
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def heatmap_figure(dfs_by_value, out_path, mode="pythia", score=None, title=None, ratio=0.15, mark_topk=True, dpi=150):
    """One panel per value (dict value -> head table) into `out_path`; returns the per-value grids.

    mode paper: normalized squared accuracy, viridis, vmin 0.6 / vmax 1.0 (Fig. 2); mode pythia: AUC with a
    colour range shared by all panels (0.5 .. max), significant heads as white dots, top-k as white squares.
    """
    values = list(dfs_by_value)
    grids = {v: score_grid(dfs_by_value[v], score, mode) for v in values}
    if mode == "paper":
        vmin, vmax, label = 0.6, 1.0, "normierter Score ((acc - min)/(max - min))^2"
    else:
        hi = max(np.nanmax(g) for g in grids.values())
        vmin, vmax, label = 0.5, float(max(0.7, np.ceil(hi * 20) / 20)), f"{(score or 'auc').upper()} (Val), Zufall = 0.5"
    n = len(values)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols + 1.2, 3.6 * rows + 0.8), squeeze=False)
    im = None
    for ax, v in zip(axes.flatten(), values):
        df = dfs_by_value[v]
        signif = mask_grid(df, df.attrs.get("signif_heads", [])) if mode != "paper" and df.attrs.get("signif_heads") else None
        topk = mask_grid(df, select_topk(df, ratio, df.attrs.get("score", "auc"))) if mark_topk else None
        thr = df.attrs.get("perm_threshold")
        sub = f"{v}" + (f"  (Perm.-Schwelle {thr:.3f}, n_signif = {df.attrs.get('n_signif', 0)})" if thr is not None and mode != "paper" else "")
        im = _draw_panel(ax, grids[v], vmin, vmax, sub, signif=signif, topk=topk)
    for ax in axes.flatten()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Head")
    for ax in axes[:, 0]:
        ax.set_ylabel("Layer")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(label, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return grids


def layer_profile_figure(dfs_by_value, out_path, score="auc", ratio=0.15, title=None, dpi=150):
    """Max and mean score per layer for every value (one line each) plus the top-k count per layer as bars."""
    values = list(dfs_by_value)
    profiles = {v: layer_profile(dfs_by_value[v], score, ratio) for v in values}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, v in enumerate(values):
        p = profiles[v]
        c = CATEGORICAL[i % len(CATEGORICAL)]
        axes[0].plot(p["layer"], p[f"{score}_max"], marker="o", ms=4, lw=2, color=c, label=f"{v} (max)")
        axes[0].plot(p["layer"], p[f"{score}_mean"], marker="", lw=1.2, ls="--", color=c, alpha=0.8)
        axes[1].bar(p["layer"] + (i - (len(values) - 1) / 2) * 0.8 / max(1, len(values)), p["n_topk"],
                    width=0.8 / max(1, len(values)), color=c, label=v)
    thr = [dfs_by_value[v].attrs.get("perm_threshold") for v in values if dfs_by_value[v].attrs.get("perm_threshold") is not None]
    if thr:
        axes[0].axhline(max(thr), color="#52514e", lw=1, ls=":", label=f"Perm.-Schwelle (max) {max(thr):.3f}")
    axes[0].axhline(0.5, color="#9a9994", lw=0.8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel(f"{score.upper()} (Val); durchgezogen = max, gestrichelt = Mittel")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel(f"Top-{int(ratio * 100)} %-Heads je Layer")
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return profiles


# fixed categorical order (dataviz reference palette; identity never cycles)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
