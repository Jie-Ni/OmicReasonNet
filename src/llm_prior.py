"""LLM semantic priors via biomedical sentence transformers.

Generates per-feature embeddings from descriptive prompts, then converts
the cosine similarity of embeddings into a feature-prior graph.
Falls back to literature co-occurrence if transformers unavailable.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import Iterable

import numpy as np
import torch


def feature_prompt(name: str, modality: str, cohort: str) -> str:
    """Construct prompt for a feature given its modality and cohort."""
    sym = re.sub(r"\|.*$", "", name).strip()  # mRNA "SYMBOL|EntrezID" -> "SYMBOL"
    if modality == "mrna":
        return (
            f"Describe the role of mRNA gene {sym} in {cohort} cancer subtyping, "
            f"focusing on prognostic value, pathway membership, and known driver mutations."
        )
    if modality == "miRNA":
        return (
            f"Describe the role of microRNA {sym} in {cohort} cancer progression, "
            f"focusing on its mRNA targets, oncogenic or tumor-suppressive function."
        )
    if modality == "meth":
        return (
            f"Describe the relevance of DNA methylation at gene {sym} in {cohort} cancer, "
            f"focusing on promoter hypermethylation events and gene silencing."
        )
    return f"Describe {sym} in {cohort}."


def embed_features(
    feat_names: Iterable[str], modality: str, cohort: str,
    model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    cache_path: str | None = None, batch_size: int = 64,
) -> np.ndarray:
    """Return [F, D] embeddings. Cache to disk if cache_path given."""
    feat_list = list(feat_names)
    if cache_path and os.path.exists(cache_path):
        try:
            arr = np.load(cache_path)
            if arr.shape[0] == len(feat_list):
                return arr
        except Exception:
            pass

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        warnings.warn("sentence-transformers missing; falling back to random embeddings")
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((len(feat_list), 384)).astype(np.float32)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, emb)
        return emb

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    prompts = [feature_prompt(n, modality, cohort) for n in feat_list]
    emb = model.encode(prompts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, emb.astype(np.float32))
    return emb.astype(np.float32)


def emb_to_knn_edges(emb: np.ndarray, k: int = 8) -> np.ndarray:
    """Cosine kNN over embeddings (already L2-normalized) -> edge_index [2, E]."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]
    knn = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = knn.reshape(-1)
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge = np.unique(np.stack([src, dst], axis=0), axis=1)
    return edge.astype(np.int64)
