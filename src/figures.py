"""Publication-quality figures using SciencePlots nature style.

Nature style: sans-serif, narrow line widths, journal column widths.
- Single column: 89 mm  ≈ 3.5 inch
- Double column: 183 mm ≈ 7.2 inch (we cap at 6.5 inch to stay <=2000 px @ 300 dpi)

Each figure runs pixel-overflow QA on in-plot text annotations.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots  # noqa: F401  (registers styles)

NATURE_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
    "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
]
DPI = 300
COL_W = 3.5
DBL_W = 6.5  # capped 6.5*300 = 1950 px (<2000 limit)


def setup_style():
    plt.style.use(["science", "nature"])
    plt.rcParams.update({
        "text.usetex": False,        # use mathtext (no system LaTeX dependency)
        "mathtext.fontset": "dejavusans",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 0.9,
    })


def qa_check(fig, max_overflow_px: int = 4):
    fig.canvas.draw()
    overflow = []
    for ax in fig.get_axes():
        bb = ax.get_window_extent()
        for txt in ax.texts:
            try:
                tb = txt.get_window_extent()
            except Exception:
                continue
            if tb.width <= 0 or tb.height <= 0:
                continue
            dx = max(0.0, bb.x0 - tb.x0, tb.x1 - bb.x1)
            dy = max(0.0, bb.y0 - tb.y0, tb.y1 - bb.y1)
            if dx > max_overflow_px or dy > max_overflow_px:
                overflow.append((txt.get_text()[:40], round(dx, 1), round(dy, 1)))
    if overflow:
        raise RuntimeError(f"Text overflow: {overflow[:5]}")


def save(fig, out_base: str):
    fig.savefig(f"{out_base}.pdf")
    fig.savefig(f"{out_base}.png", dpi=DPI)


def _verify_size(path: str, limit: int = 2000) -> None:
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if max(w, h) > limit:
            raise RuntimeError(f"{path} {w}x{h} exceeds {limit}px limit")


# === Figure 2: 5-cohort performance violins ===

def figure_perf_violin(results_per_cohort: dict[str, pd.DataFrame], out_base: str):
    """3x2 grid: 5 cohort violin panels + 1 summary panel (cross-cohort mean
    ACC per method, ranked) filling the bottom-right cell (no empty space)."""
    setup_style()
    cohorts = list(results_per_cohort.keys())
    cols, rows = 3, 2
    fig = plt.figure(figsize=(DBL_W, 1.6 * rows + 0.7))
    gs = fig.add_gridspec(rows + 1, cols,
                          height_ratios=[0.4] + [1] * rows,
                          hspace=0.55, wspace=0.45)
    metrics = ["acc", "f1_w", "f1_m", "auc"]
    metric_lbl = ["ACC", "F1$_w$", "F1$_m$", "AUC"]
    axes = []
    for i, c in enumerate(cohorts):
        r = 1 + i // cols
        cc = i % cols
        ax = fig.add_subplot(gs[r, cc])
        axes.append(ax)
        df = results_per_cohort[c]
        long = df.melt(id_vars=["model"], value_vars=metrics,
                       var_name="metric", value_name="score")
        long["metric"] = long["metric"].map(dict(zip(metrics, metric_lbl)))
        n_models = df["model"].nunique()
        sns.violinplot(data=long, x="metric", y="score", hue="model", ax=ax,
                       inner="quartile", cut=0, density_norm="width",
                       linewidth=0.4, legend=False,
                       palette=NATURE_PALETTE[:n_models])
        ax.set_title(c, fontsize=8, pad=2)
        ax.set_xlabel("")
        ax.set_ylabel("Score" if cc == 0 else "")
        ax.set_ylim(0.3, 1.0)
        ax.tick_params(axis="x", labelrotation=0, length=2)
        ax.tick_params(axis="y", length=2)

    # Summary panel: cross-cohort mean ACC per method, ranked descending.
    sum_ax = fig.add_subplot(gs[rows, cols - 1])
    method_means = {}
    for c, df in results_per_cohort.items():
        for m, sub in df.groupby("model"):
            method_means.setdefault(m, []).append(sub["acc"].mean())
    rows_summary = sorted(
        ((m, float(np.mean(v)),
          (1.96 * float(np.std(v, ddof=1)) / np.sqrt(len(v)))
          if len(v) > 1 else 0.0)
         for m, v in method_means.items()),
        key=lambda r: -r[1],
    )
    short = {"OmicReasonNet": "ORN", "ORN_FWGCN_head": "ORN-FWGCN",
             "MLPConcat": "MLP-concat", "PlainGCN": "GCN", "PlainGAT": "GAT",
             "rf": "RF", "nb": "NB"}
    n_methods = len(rows_summary)
    cmap = NATURE_PALETTE[:n_methods]
    y_pos = np.arange(n_methods)[::-1]
    for k, (m, mu, ci) in enumerate(rows_summary):
        sum_ax.barh(y_pos[k], mu, xerr=ci, height=0.7,
                    color=cmap[k], edgecolor="white", linewidth=0.4,
                    error_kw={"linewidth": 0.6, "ecolor": "0.3"})
    sum_ax.set_yticks(y_pos)
    sum_ax.set_yticklabels([short.get(m, m) for m, _, _ in rows_summary],
                           fontsize=6)
    sum_ax.set_xlabel("Mean ACC", fontsize=7)
    sum_ax.set_xlim(0.5, 1.0)
    sum_ax.set_title("All cohorts (mean $\\pm$ 95% CI)", fontsize=8, pad=2)
    sum_ax.tick_params(axis="x", labelsize=6, length=2)
    sum_ax.tick_params(axis="y", length=0)

    # Build the top-row shared legend manually from the model -> palette mapping
    legax = fig.add_subplot(gs[0, :])
    legax.axis("off")
    df0 = next(iter(results_per_cohort.values()))
    model_names = sorted(df0["model"].unique())
    proxies = [plt.Line2D([0], [0], marker="s", linestyle="",
                          markerfacecolor=NATURE_PALETTE[i % len(NATURE_PALETTE)],
                          markeredgecolor="white", markersize=5, label=m)
               for i, m in enumerate(model_names)]
    legax.legend(handles=proxies, loc="center", ncol=min(7, len(proxies)),
                 fontsize=7, columnspacing=0.8, handletextpad=0.3,
                 handlelength=0.8)
    qa_check(fig)
    save(fig, out_base)
    _verify_size(f"{out_base}.png")
    plt.close(fig)


# === Figure 3: ablation bars (5 cohorts × 6 variants) ===

def figure_ablation_bars(ablation_csv: str, out_base: str):
    setup_style()
    df = pd.read_csv(ablation_csv)
    cohorts = df["cohort"].unique().tolist()
    variants = ["full", "no_stat", "no_sem", "no_gate", "no_fwgcn", "no_reasoning"]
    pretty = {"full": "Full", "no_stat": r"$-$Stat", "no_sem": r"$-$Sem",
              "no_gate": r"$-$Gate", "no_fwgcn": r"+MLP head",
              "no_reasoning": r"$-$Reasoning"}
    fig, ax = plt.subplots(figsize=(DBL_W, 2.2), dpi=DPI)
    width = 0.13
    x = np.arange(len(cohorts))
    for j, v in enumerate(variants):
        sub = df[df["variant"] == v].set_index("cohort").reindex(cohorts)
        ax.bar(x + (j - 2.5) * width, sub["acc_mean"], width=width,
               yerr=sub["acc_std"], label=pretty[v],
               color=NATURE_PALETTE[j], capsize=1.5,
               error_kw={"linewidth": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("Accuracy")
    y_min = max(0.0, (df["acc_mean"] - df["acc_std"]).min() - 0.03)
    y_max = min(1.0, (df["acc_mean"] + df["acc_std"]).max() + 0.04)
    ax.set_ylim(y_min, y_max)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              columnspacing=1.0, handletextpad=0.3, handlelength=1.0,
              fontsize=7)
    qa_check(fig)
    save(fig, out_base)
    _verify_size(f"{out_base}.png")
    plt.close(fig)


# === Figure 4: δ sensitivity ===

def figure_delta_sensitivity(delta_csv: str, out_base: str):
    """Delta sensitivity per cohort. 3x2 grid; the bottom-right cell carries
    a cross-cohort summary (mean acc with 95% CI band, plus per-cohort
    light-grey curves) to fill the otherwise-empty cell."""
    setup_style()
    df = pd.read_csv(delta_csv)
    cohorts = df["cohort"].unique().tolist()
    cols, rows = 3, 2
    fig = plt.figure(figsize=(DBL_W, 1.6 * rows + 0.5))
    gs = fig.add_gridspec(rows + 1, cols,
                          height_ratios=[0.3] + [1] * rows,
                          hspace=0.65, wspace=0.30)
    axes = []
    for i, c in enumerate(cohorts):
        r = 1 + i // cols
        cc = i % cols
        ax = fig.add_subplot(gs[r, cc])
        axes.append(ax)
        sub = df[df["cohort"] == c]
        for k, (metric, color) in enumerate(zip(["acc", "f1_w", "f1_m"],
                                                NATURE_PALETTE)):
            ax.errorbar(sub["delta"], sub[f"{metric}_mean"],
                        yerr=sub[f"{metric}_std"], marker="o", ms=2.5,
                        color=color, label=metric.replace("_", "-"),
                        linewidth=0.9, elinewidth=0.5, capsize=1.5)
        ax.set_xscale("log")
        ax.set_title(c, fontsize=8, pad=2)
        ax.set_xlabel(r"$\delta$")
        ax.set_ylabel("Score" if cc == 0 else "")
    # Cross-cohort summary panel filling the bottom-right empty cell
    sum_ax = fig.add_subplot(gs[rows, cols - 1])
    deltas = sorted(df["delta"].unique())
    cohort_curves = []
    for c in cohorts:
        sub = df[df["cohort"] == c].sort_values("delta")
        sum_ax.plot(sub["delta"], sub["acc_mean"], color="0.7",
                    linewidth=0.6, alpha=0.8)
        cohort_curves.append(
            sub.set_index("delta")["acc_mean"].reindex(deltas).values
        )
    arr = np.array(cohort_curves, dtype=float)
    mean_curve = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1)
    n = arr.shape[0]
    ci = 1.96 * sd / np.sqrt(max(n, 1))
    sum_ax.fill_between(deltas, mean_curve - ci, mean_curve + ci,
                        color=NATURE_PALETTE[0], alpha=0.20, linewidth=0)
    sum_ax.plot(deltas, mean_curve, color=NATURE_PALETTE[0],
                linewidth=1.4, marker="o", ms=3)
    sum_ax.set_xscale("log")
    sum_ax.set_xlabel(r"$\delta$")
    sum_ax.set_ylabel("")
    sum_ax.set_title("All cohorts (mean $\\pm$ 95% CI)", fontsize=8, pad=2)
    # Top-row shared legend
    legax = fig.add_subplot(gs[0, :])
    legax.axis("off")
    if axes:
        h, l = axes[0].get_legend_handles_labels()
        legax.legend(h, l, loc="center", ncol=3, fontsize=7,
                     columnspacing=1.0, handletextpad=0.3, handlelength=1.0)
    qa_check(fig)
    save(fig, out_base)
    _verify_size(f"{out_base}.png")
    plt.close(fig)
