"""Advanced figures: UMAP patient embeddings + Critical Difference diagram.

Both follow the SciencePlots nature style and stay <= 2000 px.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

from src.figures import setup_style, save, NATURE_PALETTE, _verify_size, DBL_W


COHORT_CLASS_NAMES = {
    "BRCA":     ["Basal", "Her2", "LumA", "LumB", "Normal"],
    "COADREAD": ["Stage I", "Stage II", "Stage III", "Stage IV"],
    "UCEC":     ["EndoGr1", "EndoGr2", "EndoGr3", "MixedGr3", "SerousGr3"],
    "SARC":     ["DDLPS", "MFS", "MPNST", "SS", "STLMS", "ULMS", "UPS"],
    "LUAD":     ["TRU", "prox.-inflam", "prox.-prolif."],
}
COHORT_N = {"BRCA": 875, "COADREAD": 308, "UCEC": 255, "SARC": 210, "LUAD": 185}


# === Figure 5: UMAP grid (refined) ===

def figure_umap_grid(emb_dir: str, out_base: str):
    """Refined UMAP grid: 3x2 layout, 5 cohorts + 1 summary panel
    (per-cohort silhouette score on the learned embedding).
    Per-class convex-hull soft shading for class with >=8 points."""
    import umap
    from scipy.spatial import ConvexHull
    from sklearn.metrics import silhouette_score

    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 7, "axes.titlesize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
    })
    cohorts = ["BRCA", "COADREAD", "UCEC", "SARC", "LUAD"]
    cols = 3
    rows = 2
    fig = plt.figure(figsize=(DBL_W, 2.05 * rows + 0.4))
    gs = fig.add_gridspec(rows, cols, hspace=0.55, wspace=0.30)
    silhouette_per_cohort = {}

    for i, c in enumerate(cohorts):
        path = Path(emb_dir) / c / "embeddings.npz"
        if not path.exists():
            continue
        d = np.load(path)
        X, y = d["X"], d["y"]
        n_neighbors = max(5, min(15, X.shape[0] // 10))
        Z = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.30,
                      n_components=2, random_state=42,
                      metric="euclidean").fit_transform(X)

        # Silhouette on the learned embedding (X) — quantifies how well the
        # learned representation separates the labelled subtypes.
        try:
            sil = silhouette_score(X, y) if len(np.unique(y)) > 1 else float("nan")
        except Exception:
            sil = float("nan")
        silhouette_per_cohort[c] = float(sil)

        ax = fig.add_subplot(gs[i // cols, i % cols])
        names = COHORT_CLASS_NAMES.get(c, [str(k) for k in range(int(y.max()) + 1)])
        unique_y = np.unique(y)

        # 1. Soft convex-hull shading per class (only for classes with >= 8 points)
        for k_idx, cls in enumerate(unique_y):
            mask = (y == cls)
            if mask.sum() < 8:
                continue
            try:
                pts = Z[mask]
                hull = ConvexHull(pts)
                color = NATURE_PALETTE[int(cls) % len(NATURE_PALETTE)]
                ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                        color=color, alpha=0.12, edgecolor=color,
                        linewidth=0.6)
            except Exception:
                continue

        # 2. Scatter on top
        legend_proxies = []
        for k_idx, cls in enumerate(unique_y):
            mask = (y == cls)
            color = NATURE_PALETTE[int(cls) % len(NATURE_PALETTE)]
            ax.scatter(Z[mask, 0], Z[mask, 1], s=7, alpha=0.92,
                       c=color, edgecolors="white", linewidths=0.25)
            label = names[int(cls)] if int(cls) < len(names) else str(int(cls))
            legend_proxies.append(plt.Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.4, markersize=3.4, label=label,
            ))

        n_samp = COHORT_N.get(c, X.shape[0])
        ax.set_title(f"{c}  ($n{{=}}{n_samp}$, $K{{=}}{len(unique_y)}$)",
                     fontsize=8, pad=2)
        ax.set_xlabel("UMAP-1", fontsize=7, labelpad=1)
        ax.set_ylabel("UMAP-2", fontsize=7, labelpad=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.5")
        ax.legend(handles=legend_proxies, fontsize=5.5, frameon=True,
                  facecolor="white", edgecolor="0.7",
                  borderpad=0.25, handletextpad=0.25,
                  labelspacing=0.18, loc="best", ncol=1)

    # Summary panel: per-cohort silhouette score on the learned embedding
    sum_ax = fig.add_subplot(gs[(len(cohorts)) // cols, (len(cohorts)) % cols])
    keys = list(silhouette_per_cohort.keys())
    vals = [silhouette_per_cohort[k] for k in keys]
    colors = [NATURE_PALETTE[i % len(NATURE_PALETTE)] for i in range(len(keys))]
    bars = sum_ax.bar(np.arange(len(keys)), vals, color=colors,
                      edgecolor="white", linewidth=0.5)
    for k, v in enumerate(vals):
        sum_ax.text(k, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6)
    sum_ax.axhline(0, color="black", linewidth=0.4)
    sum_ax.set_xticks(np.arange(len(keys)))
    sum_ax.set_xticklabels(keys, fontsize=6, rotation=0)
    sum_ax.set_ylabel("Silhouette", fontsize=7)
    sum_ax.set_ylim(min(min(vals) - 0.05, -0.05),
                    max(max(vals) + 0.10, 0.10))
    sum_ax.set_title("Class separability (learned repr.)",
                     fontsize=8, pad=2)
    for s in sum_ax.spines.values():
        s.set_linewidth(0.4); s.set_color("0.5")
    sum_ax.tick_params(length=2)

    save(fig, out_base)
    _verify_size(f"{out_base}.png")
    plt.close(fig)


# === Figure 6: Critical Difference diagram (clean Demšar layout) ===

def _friedman_chi2(ranks):
    R = np.asarray(ranks)
    N, k = R.shape
    Rj = R.mean(axis=0)
    chi2 = (12 * N) / (k * (k + 1)) * (np.sum(Rj ** 2) - k * (k + 1) ** 2 / 4)
    return chi2, Rj


def _nemenyi_cd(k: int, N: int, alpha: float = 0.05) -> float:
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164}
    return q05[k] * np.sqrt(k * (k + 1) / (6 * N))


def figure_cd_diagram(results_root: str, out_base: str,
                      cohorts=("BRCA", "COADREAD", "UCEC", "SARC", "LUAD"),
                      alpha: float = 0.05):
    """Clean Demšar layout, no overlapping text. Layout (top -> bottom):
        [CD ruler]  [significance bars]  [rank-axis]  [elbow lines + names]
    """
    setup_style()
    plt.rcParams.update({"axes.labelsize": 8, "axes.titlesize": 8})

    rows = []
    for c in cohorts:
        cv = json.load(open(os.path.join(results_root, c, "cv_results.json")))
        agg = defaultdict(dict)
        for r in cv:
            agg[r.get("fold", 0)][r["model"]] = r["acc"]
        for fold_id, fold_d in agg.items():
            rows.append({"cohort": c, "fold": fold_id, **fold_d})
    df = pd.DataFrame(rows)
    model_order = ["OmicReasonNet", "ORN_FWGCN_head", "MLPConcat",
                   "PlainGCN", "PlainGAT", "rf", "nb"]
    pretty = {"OmicReasonNet": "ORN", "ORN_FWGCN_head": "ORN-FWGCN",
              "MLPConcat": "MLP-concat", "PlainGCN": "GCN", "PlainGAT": "GAT",
              "rf": "Random Forest", "nb": "Naive Bayes"}
    keep = [m for m in model_order if m in df.columns]
    M = df[keep].dropna().to_numpy()
    N, k = M.shape
    ranks = np.array([pd.Series(-row).rank(method="average").values for row in M])
    chi2, mean_ranks = _friedman_chi2(ranks)
    cd = _nemenyi_cd(k, N, alpha)

    order = np.argsort(mean_ranks)
    sorted_names = [pretty[keep[i]] for i in order]
    sorted_ranks = mean_ranks[order]

    # Vertical layout in axis-data coords (y goes 0 at top to 6 at bottom):
    #   y =  0 .. 0.6  : CD ruler band
    #   y =  0.7 .. ~1+0.4*Sbars : significance bars stacked
    #   y =  yaxis     : rank axis line
    #   y >  yaxis     : elbow lines + label rows
    fig = plt.figure(figsize=(5.0, 3.0))
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.84])
    rmin, rmax = 1.0, k
    ax.set_xlim(rmin - 0.1, rmax + 0.1)

    # Compute number of significance bars first to size vertical layout
    intervals = []
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if abs(sorted_ranks[j] - sorted_ranks[i]) <= cd:
                intervals.append((sorted_ranks[i], sorted_ranks[j]))
    keep_int = []
    for g in sorted(intervals):
        a, b = g
        if any(ka <= a and b <= kb and (ka, kb) != (a, b) for ka, kb in keep_int):
            continue
        keep_int = [(ka, kb) for ka, kb in keep_int if not (a <= ka and kb <= b)]
        keep_int.append((a, b))
    keep_int = sorted(set(keep_int))
    n_bars = len(keep_int)

    # Vertical positions (axis data coords; smaller y = higher visually)
    y_cd = 0.0
    y_cd_label = y_cd + 0.35
    y_bars_top = 1.0          # first bar
    y_bars_step = 0.35
    y_axis = y_bars_top + n_bars * y_bars_step + 0.3
    half = (len(sorted_names) + 1) // 2
    label_step = 0.6
    y_label_first = y_axis + 0.5
    y_max = y_label_first + max(half, len(sorted_names) - half) * label_step + 0.4
    ax.set_ylim(y_max, -0.6)   # invert so smaller y is at top

    # Fully blank axes; we draw every visual element ourselves.
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    # CD ruler at top
    ax.plot([rmin, rmin + cd], [y_cd, y_cd], color="black", linewidth=1.0)
    ax.plot([rmin, rmin], [y_cd - 0.10, y_cd + 0.10], color="black", linewidth=1.0)
    ax.plot([rmin + cd, rmin + cd], [y_cd - 0.10, y_cd + 0.10], color="black", linewidth=1.0)
    ax.text(rmin + cd / 2, y_cd_label, f"CD = {cd:.2f}",
            ha="center", va="top", fontsize=7.5)

    # Significance bars (red, thick)
    bar_color = NATURE_PALETTE[0]
    for idx, (a, b) in enumerate(keep_int):
        y_bar = y_bars_top + idx * y_bars_step
        ax.plot([a - 0.04, b + 0.04], [y_bar, y_bar], color=bar_color,
                linewidth=2.4, solid_capstyle="butt")
        ax.plot([a - 0.04, a - 0.04], [y_bar - 0.07, y_bar + 0.07],
                color=bar_color, linewidth=1.0)
        ax.plot([b + 0.04, b + 0.04], [y_bar - 0.07, y_bar + 0.07],
                color=bar_color, linewidth=1.0)

    # Main rank axis (horizontal black line) + ticks above (smaller y values
    # are visually higher because we inverted the y axis at set_ylim).
    ax.plot([rmin, rmax], [y_axis, y_axis], color="black", linewidth=1.0)
    for x in np.arange(rmin, rmax + 0.001, 1.0):
        ax.plot([x, x], [y_axis - 0.10, y_axis + 0.10], color="black", linewidth=1.0)
        ax.text(x, y_axis - 0.16, f"{int(x)}",
                ha="center", va="bottom", fontsize=7)
    for x in np.arange(rmin + 0.5, rmax, 1.0):
        ax.plot([x, x], [y_axis - 0.05, y_axis + 0.05], color="black", linewidth=0.5)
    # (no axis label here -- "lower mean rank = better" is folded into the
    # subtitle below to avoid overlap with the leftmost elbow lines)

    # Elbow lines + labels in two columns
    for i, (name, r) in enumerate(zip(sorted_names, sorted_ranks)):
        if i < half:
            y = y_label_first + i * label_step
            x_lab = rmin - 0.10
            ax.plot([r, r], [y_axis, y], color="black", linewidth=0.6)
            ax.plot([r, x_lab], [y, y], color="black", linewidth=0.6)
            ax.text(x_lab - 0.04, y - 0.06, name,
                    ha="right", va="bottom", fontsize=7.5)
            ax.text(x_lab - 0.04, y + 0.04, f"$\\bar{{R}} = {r:.2f}$",
                    ha="right", va="top", fontsize=6, color="0.35")
        else:
            j = i - half
            y = y_label_first + j * label_step
            x_lab = rmax + 0.10
            ax.plot([r, r], [y_axis, y], color="black", linewidth=0.6)
            ax.plot([r, x_lab], [y, y], color="black", linewidth=0.6)
            ax.text(x_lab + 0.04, y - 0.06, name,
                    ha="left", va="bottom", fontsize=7.5)
            ax.text(x_lab + 0.04, y + 0.04, f"$\\bar{{R}} = {r:.2f}$",
                    ha="left", va="top", fontsize=6, color="0.35")

    # Subtitle inside the axis area, just above the CD ruler
    ax.text((rmin + rmax) / 2, -0.40,
            f"Mean rank (lower is better) on $N{{=}}{N}$ datasets;  "
            f"$\\chi^2{{=}}{chi2:.1f}$, $\\alpha{{=}}{alpha:g}$",
            ha="center", va="top", fontsize=7, color="0.30")

    save(fig, out_base)
    _verify_size(f"{out_base}.png")
    plt.close(fig)
    return {"chi2": float(chi2), "cd": float(cd),
            "mean_ranks": dict(zip(keep, mean_ranks.tolist()))}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["umap", "cd", "both"], default="both")
    ap.add_argument("--results", default="./results")
    ap.add_argument("--emb_dir", default="./results/embeddings")
    ap.add_argument("--out", default="./figures")
    args = ap.parse_args()
    if args.mode in ("cd", "both"):
        info = figure_cd_diagram(args.results, os.path.join(args.out, "fig6_cd_diagram"))
        print("[fig6] CD =", info["cd"], "chi2 =", info["chi2"])
    if args.mode in ("umap", "both"):
        figure_umap_grid(args.emb_dir, os.path.join(args.out, "fig5_umap_grid"))
        print("[fig5] umap grid done")
