"""Data loading + graph construction for OmicReasonNet.

Sample graph: cosine similarity kNN on each omics view (sample-as-node).
Feature prior graph: built from prior knowledge files, gene-symbol-aligned to
the omics feature names.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CohortData:
    cohort: str
    X: list[np.ndarray]            # [mRNA, meth, miRNA] -> [N, F_v]
    y: np.ndarray                  # [N]
    feat_names: list[list[str]]    # parallel to X
    sample_ids: np.ndarray | None
    n_classes: int


def _read_csv_no_header(path: str) -> np.ndarray:
    return pd.read_csv(path, header=None).values


def _read_feat_names(path: str) -> list[str]:
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def load_cohort(data_dir: str, cohort: str) -> CohortData:
    """Load a MOGONET-style cohort directory.

    Expected files in {data_dir}/{cohort}/:
        {1,2,3}_tr.csv, {1,2,3}_te.csv, {1,2,3}_featname.csv,
        labels_tr.csv, labels_te.csv
    1=mRNA, 2=meth, 3=miRNA
    """
    root = os.path.join(data_dir, cohort)
    X_tr = [_read_csv_no_header(os.path.join(root, f"{i}_tr.csv")) for i in (1, 2, 3)]
    X_te = [_read_csv_no_header(os.path.join(root, f"{i}_te.csv")) for i in (1, 2, 3)]
    y_tr = _read_csv_no_header(os.path.join(root, "labels_tr.csv")).flatten().astype(int)
    y_te = _read_csv_no_header(os.path.join(root, "labels_te.csv")).flatten().astype(int)

    X = [np.concatenate([tr, te], axis=0).astype(np.float32) for tr, te in zip(X_tr, X_te)]
    y = np.concatenate([y_tr, y_te], axis=0)
    feat_names = [_read_feat_names(os.path.join(root, f"{i}_featname.csv")) for i in (1, 2, 3)]

    if y.min() == 1:
        y = y - 1

    return CohortData(
        cohort=cohort, X=X, y=y, feat_names=feat_names,
        sample_ids=None, n_classes=int(y.max() + 1),
    )


def split_idx_default(n_total: int, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(n_train), np.arange(n_train, n_total)


def cosine_knn_edges(X: np.ndarray, k: int = 10) -> torch.Tensor:
    """Build symmetric kNN edge_index from cosine similarity.

    Returns edge_index shape [2, E].
    """
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]
    knn = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = knn.reshape(-1)
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge = np.stack([src, dst], axis=0)
    edge = np.unique(edge, axis=1)
    return torch.from_numpy(edge.astype(np.int64))


def build_feature_prior_graph(
    feat_names: list[str],
    prior_path: str | None,
    threshold: float = 0.0,
    self_loops: bool = True,
) -> torch.Tensor | None:
    """Build feature-feature prior graph aligned to feat_names.

    Reads prior_path as a square similarity matrix (genes × genes) with header
    matching feature names, OR a (src, dst[, weight]) edge list. Returns
    edge_index aligned to feat_names indices, or None if not aligned.
    """
    if prior_path is None or not os.path.exists(prior_path):
        return None
    name_to_idx = {n.upper(): i for i, n in enumerate(feat_names)}

    try:
        if prior_path.endswith(".xlsx"):
            df = pd.read_excel(prior_path, header=0, index_col=0)
        else:
            df = pd.read_csv(prior_path, sep=None, engine="python", header=0, index_col=0)
    except Exception as e:
        warnings.warn(f"prior {prior_path}: {e}")
        return None

    # Square similarity matrix?
    if df.shape[0] == df.shape[1] and df.shape[0] > 1:
        idxs = []
        for nm in df.index:
            j = name_to_idx.get(str(nm).upper())
            if j is not None:
                idxs.append(j)
            else:
                idxs.append(-1)
        keep = [i for i, v in enumerate(idxs) if v >= 0]
        if not keep:
            return None
        sub = df.iloc[keep, keep].values.astype(np.float32)
        sub_idxs = [idxs[i] for i in keep]
        n_full = len(feat_names)
        rows, cols = np.where(sub > threshold)
        src = np.array([sub_idxs[r] for r in rows], dtype=np.int64)
        dst = np.array([sub_idxs[c] for c in cols], dtype=np.int64)
        if self_loops:
            sl = np.arange(n_full, dtype=np.int64)
            src = np.concatenate([src, sl])
            dst = np.concatenate([dst, sl])
        return torch.from_numpy(np.stack([src, dst], axis=0))

    return None


def gip_kernel(adj_or_profile: np.ndarray, gamma_prime: float = 1.0) -> np.ndarray:
    """Gaussian Interaction Profile similarity from interaction profiles."""
    P = adj_or_profile.astype(np.float64)
    n = P.shape[0]
    norm_sq = (P ** 2).sum(axis=1)
    gamma = gamma_prime / (norm_sq.mean() + 1e-12)
    diff = norm_sq[:, None] + norm_sq[None, :] - 2.0 * (P @ P.T)
    return np.exp(-gamma * np.maximum(diff, 0.0))
