"""Real biomarker discovery via reasoning-score ranking + ablation impact.

For each fold:
  1. Rank features within each modality by reasoning score.
  2. Take top-K (configurable). For each, mask its column to 0 (or replace with
     train-set mean) and re-evaluate the model. Record accuracy drop.
  3. Aggregate across folds: a biomarker is reported only if it (a) appears in
     top-K in >= ceil(folds/2) folds and (b) ablation drops accuracy by a
     significant margin (Wilcoxon-signed-rank vs no-ablation, FDR < 0.1).
Avoids cherry-picking by requiring fold reproducibility + statistical test.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score


def aggregate_topk(scores_per_fold: list[list[np.ndarray]], k: int = 30) -> list[list[tuple[int, int, float]]]:
    """Return per-modality list of (feature_idx, n_folds_in_topk, mean_score)."""
    n_folds = len(scores_per_fold)
    if not n_folds:
        return []
    n_views = len(scores_per_fold[0])
    out = []
    for v in range(n_views):
        agg = np.stack([scores_per_fold[f][v] for f in range(n_folds)], axis=0)  # [F, F_v]
        mean = agg.mean(axis=0)
        # rank within each fold
        topk_per_fold = [set(np.argsort(-agg[f])[:k].tolist()) for f in range(n_folds)]
        counts = Counter()
        for s in topk_per_fold:
            counts.update(s)
        ranked = sorted(
            [(idx, cnt, float(mean[idx])) for idx, cnt in counts.items()],
            key=lambda r: (-r[1], -r[2]),
        )
        out.append(ranked)
    return out


def fdr_bh(p: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Benjamini-Hochberg FDR; returns boolean mask of accepted entries."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = np.arange(1, m + 1) * alpha / m
    below = ranked <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    last = np.where(below)[0].max()
    mask = np.zeros(m, dtype=bool)
    mask[order[: last + 1]] = True
    return mask


def report_biomarkers(
    feat_names: list[list[str]],
    scores_per_fold: list[list[np.ndarray]],
    ablation_drops: list[list[dict[int, float]]] | None = None,
    k: int = 30, min_recurrence: int = 3,
) -> dict:
    """Combine top-K recurrence with ablation evidence.

    ablation_drops[fold][view][feat_idx] = acc_drop_when_masked
    """
    out = {"top_features": [], "validated_biomarkers": []}
    agg = aggregate_topk(scores_per_fold, k=k)
    modality_names = ["mRNA", "meth", "miRNA"]
    for v, ranked in enumerate(agg):
        names = feat_names[v]
        for idx, n_in_top, mean_score in ranked:
            if n_in_top < min_recurrence:
                continue
            entry = {
                "modality": modality_names[v],
                "name": names[idx],
                "feat_idx": int(idx),
                "n_folds_in_topk": int(n_in_top),
                "mean_reasoning_score": float(mean_score),
            }
            if ablation_drops is not None:
                drops = [ablation_drops[f][v].get(idx, 0.0) for f in range(len(ablation_drops))]
                drops = np.array(drops, dtype=float)
                if (drops != 0).any():
                    try:
                        w_stat, p_val = wilcoxon(drops, alternative="greater")
                        entry["ablation_mean_drop"] = float(drops.mean())
                        entry["ablation_p"] = float(p_val)
                    except Exception:
                        entry["ablation_mean_drop"] = float(drops.mean())
                        entry["ablation_p"] = float("nan")
            out["top_features"].append(entry)

    # FDR over ablation p-values; mark validated
    if ablation_drops is not None:
        ps = [e.get("ablation_p", 1.0) for e in out["top_features"]]
        if ps:
            mask = fdr_bh(np.array(ps), alpha=0.1)
            for e, ok in zip(out["top_features"], mask):
                if ok:
                    out["validated_biomarkers"].append(e)
    return out
