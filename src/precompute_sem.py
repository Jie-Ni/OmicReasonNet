"""Precompute LLM semantic embeddings + feature-prior graphs per cohort/view.

Outputs:
  cache/{cohort}/{modality}_emb.npy
  cache/{cohort}/{modality}_edges.npy   (2, E)
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.dataloader import load_cohort
from src.llm_prior import embed_features, emb_to_knn_edges


MODALITY_NAMES = ["mrna", "meth", "miRNA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    cd = load_cohort(args.data_dir, args.cohort)
    out = os.path.join(args.cache_dir, args.cohort)
    os.makedirs(out, exist_ok=True)

    for v, mod in enumerate(MODALITY_NAMES):
        emb_path = os.path.join(out, f"{mod}_emb.npy")
        edge_path = os.path.join(out, f"{mod}_edges.npy")
        emb = embed_features(
            cd.feat_names[v], modality=mod, cohort=cd.cohort,
            model_name=args.model, cache_path=emb_path,
        )
        edges = emb_to_knn_edges(emb, k=args.k)
        np.save(edge_path, edges)
        print(f"[{cd.cohort}/{mod}] emb={emb.shape} edges={edges.shape} -> {edge_path}")


if __name__ == "__main__":
    main()
