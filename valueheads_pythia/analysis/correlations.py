"""Correlations of head-score vectors (paper Appendix D; repo analysis/value_correlations.py, MIT).

across values   : within one model / revision, Pearson on binary top-15 % indicator vectors (Fig. 11, the repo's
                  definition) and, additionally, Pearson / Spearman on the raw score vectors.
across checkpoints / models : per value, Pearson / Spearman of the score grids; grids of differing geometry
                  are linearly interpolated onto the first grid's (L, H) axis by axis (Fig. 10, repo `resample`).
"""
import itertools
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..probing import scores_vector, select_topk  # noqa: E402
from .heatmaps import geometry_of  # noqa: E402

DIVERGING_CMAP = "RdBu_r"        # two hues around a neutral midpoint at 0


def resample(grid, rows, cols):
    """Linear interpolation of a 2D grid onto (rows, cols), axis by axis (identical to the repo)."""
    grid = np.asarray(grid, dtype=np.float64)
    tmp = np.stack([np.interp(np.linspace(0, 1, rows), np.linspace(0, 1, grid.shape[0]), grid[:, h])
                    for h in range(grid.shape[1])], axis=1)
    return np.stack([np.interp(np.linspace(0, 1, cols), np.linspace(0, 1, tmp.shape[1]), tmp[l]) for l in range(rows)])


def score_grid(df, score=None):
    """(L, H) raw score grid of a head table (df.attrs['score'] by default)."""
    L, H = geometry_of(df)
    return scores_vector(df, score or df.attrs.get("score", "auc")).reshape(L, H)


def binary_topk_vector(df, ratio=0.15, score=None):
    """(L * H,) 0/1 vector marking the top-k heads (the paper's across-values representation)."""
    L, H = geometry_of(df)
    v = np.zeros(L * H)
    v[select_topk(df, ratio, score or df.attrs.get("score", "auc"))] = 1.0
    return v


def correlation_matrix(vectors, names, method="pearson"):
    """Symmetric DataFrame of pairwise correlations between equally long vectors (NaN entries dropped pairwise)."""
    n = len(vectors)
    m = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        a, b = np.asarray(vectors[i], dtype=float), np.asarray(vectors[j], dtype=float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            r = np.nan
        else:
            r = pearsonr(a[ok], b[ok])[0] if method == "pearson" else spearmanr(a[ok], b[ok]).correlation
        m[i, j] = m[j, i] = r
    return pd.DataFrame(m, index=names, columns=names)


def across_values(dfs_by_value, ratio=0.15):
    """{'pearson_topk': binary top-k vectors (paper), 'pearson_score', 'spearman_score'} over the values of one table set."""
    names = list(dfs_by_value)
    binary = [binary_topk_vector(dfs_by_value[v], ratio) for v in names]
    raw = [scores_vector(dfs_by_value[v], dfs_by_value[v].attrs.get("score", "auc")) for v in names]
    return {"pearson_topk": correlation_matrix(binary, names, "pearson"),
            "pearson_score": correlation_matrix(raw, names, "pearson"),
            "spearman_score": correlation_matrix(raw, names, "spearman")}


def across_tables(dfs_by_name, score=None):
    """Pearson / Spearman of the score grids of several tables (checkpoints or models) of one value.

    Grids with another geometry are resampled onto the first table's (L, H) grid before flattening.
    """
    names = list(dfs_by_name)
    rows, cols = geometry_of(dfs_by_name[names[0]])
    vecs = []
    for n in names:
        g = score_grid(dfs_by_name[n], score)
        if g.shape != (rows, cols):
            g = resample(g, rows, cols)
        vecs.append(g.flatten())
    return {"pearson_score": correlation_matrix(vecs, names, "pearson"),
            "spearman_score": correlation_matrix(vecs, names, "spearman")}


def tidy(matrix, kind):
    """Long format (a, b, kind, r) of the upper triangle of a correlation DataFrame."""
    names = list(matrix.index)
    return pd.DataFrame([{"a": a, "b": b, "kind": kind, "r": float(matrix.loc[a, b])} for a, b in itertools.combinations(names, 2)])


def plot_correlation(matrix, out_path, title=None, dpi=150, annotate=True):
    """Annotated heatmap of a correlation matrix with a diverging colour map centred at 0."""
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.5, 1.0 * n + 1.5))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=DIVERGING_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    if annotate:
        for i in range(n):
            for j in range(n):
                v = matrix.iat[i, j]
                ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("Korrelation", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
