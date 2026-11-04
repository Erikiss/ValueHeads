"""Angles between value directions on the heads shared by two values (paper Appendix K, Table 17, Fig. 12).

Adapted from analysis/direction_angles.py of the reference repo (MIT): for every value pair the mass-mean
directions (results/directions/*.npz) of the heads in both top-15 % sets are compared by their angle in
degrees; n, mean, median and std (population std, ddof=0, as in the repo / Table 17) are reported.
"""
import itertools
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .heatmaps import CATEGORICAL  # noqa: E402


def angle_deg(a, b):
    """Angle in degrees between two vectors (0 for identical directions, 180 for opposite)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))


def shared_head_angles(dir_a, dir_b, heads_a, heads_b):
    """(angles, shared flat indices) over heads_a & heads_b; dir_* are (L * H, D) direction arrays."""
    shared = sorted(set(int(h) for h in heads_a) & set(int(h) for h in heads_b))
    return np.array([angle_deg(dir_a[h], dir_b[h]) for h in shared]), shared


def angle_table(directions_by_value, heads_by_value, model=None, revision=None):
    """DataFrame(pair, value_a, value_b, n_shared, mean_deg, median_deg, std_deg) for every value pair.

    directions_by_value: {value: (L * H, D)}; heads_by_value: {value: [flat_idx, ...]} (top-k sets).
    """
    rows, angles_by_pair = [], {}
    for a, b in itertools.combinations(list(directions_by_value), 2):
        angles, shared = shared_head_angles(directions_by_value[a], directions_by_value[b], heads_by_value[a], heads_by_value[b])
        angles_by_pair[(a, b)] = angles
        rows.append({"model": model, "revision": revision, "pair": f"{a} x {b}", "value_a": a, "value_b": b,
                     "n_shared": int(len(angles)), "mean_deg": float(np.mean(angles)) if len(angles) else np.nan,
                     "median_deg": float(np.median(angles)) if len(angles) else np.nan,
                     "std_deg": float(np.std(angles)) if len(angles) else np.nan, "shared_heads": shared})
    table = pd.DataFrame(rows, columns=["model", "revision", "pair", "value_a", "value_b", "n_shared", "mean_deg", "median_deg", "std_deg", "shared_heads"])
    table.attrs["angles_by_pair"] = angles_by_pair
    return table


def all_head_angle_table(directions_by_value, model=None, revision=None):
    """Same statistics over all L * H heads (not only shared top-k heads): a reference row per pair."""
    rows = []
    for a, b in itertools.combinations(list(directions_by_value), 2):
        n = len(directions_by_value[a])
        angles = np.array([angle_deg(directions_by_value[a][h], directions_by_value[b][h]) for h in range(n)])
        rows.append({"model": model, "revision": revision, "pair": f"{a} x {b}", "n": n, "mean_deg": float(np.nanmean(angles)),
                     "median_deg": float(np.nanmedian(angles)), "std_deg": float(np.nanstd(angles))})
    return pd.DataFrame(rows, columns=["model", "revision", "pair", "n", "mean_deg", "median_deg", "std_deg"])


def plot_angles(table, out_path, title=None, dpi=150):
    """Histograms (0-180 deg, 20 bins) of the shared-head angles per value pair, mean as dashed line (Fig. 12)."""
    angles_by_pair = table.attrs.get("angles_by_pair", {})
    fig, ax = plt.subplots(figsize=(7, 4))
    drawn = 0
    for i, ((a, b), angles) in enumerate(angles_by_pair.items()):
        if len(angles) == 0:
            continue
        c = CATEGORICAL[i % len(CATEGORICAL)]
        ax.hist(angles, bins=20, range=(0, 180), alpha=0.55, color=c, label=f"{a} x {b} (n = {len(angles)}, Mittel {np.mean(angles):.1f} Grad)")
        ax.axvline(np.mean(angles), color=c, ls="--", lw=1.2)
        drawn += 1
    ax.axvline(90, color="#9a9994", lw=0.8)
    ax.set_xlabel("Winkel zwischen den Werterichtungen auf geteilten Top-15 %-Heads (Grad)")
    ax.set_ylabel("Anzahl Heads")
    if drawn:
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    else:
        ax.text(90, 0.5, "keine geteilten Heads", ha="center", fontsize=10, color="#52514e")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
