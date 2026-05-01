"""Train ORN-MLP-head once per cohort on the full data; save patient
embeddings (post-fusion, pre-classifier) for UMAP/t-SNE visualisation.

Outputs: results/embeddings/{cohort}/embeddings.npz with X (N x D), y (N).

CLI: python -m src.run_embeddings --cohort BRCA
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from src.dataloader import load_cohort, cosine_knn_edges
from src.ablation import AblatedOmicReasonNet
from src.train_cv import load_sample_sem_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--out", default="./results/embeddings")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = load_cohort(args.data_dir, args.cohort)
    sem_edges = load_sample_sem_edges(args.cache_dir, cd.cohort, cd.X, k=10)

    torch.manual_seed(args.seed)
    Xs = [StandardScaler().fit(x).transform(x).astype(np.float32) for x in cd.X]
    Xs_t = [torch.tensor(x, device=device) for x in Xs]
    y_t = torch.tensor(cd.y, dtype=torch.long, device=device)
    full_stat = [cosine_knn_edges(x, k=10).to(device) for x in Xs]
    full_sem = [None if s is None else torch.tensor(s, device=device) for s in sem_edges]
    sample_edge = full_stat[0]

    # Use the no_fwgcn = MLP-head variant (the published ORN default)
    model = AblatedOmicReasonNet(
        feat_dims=[x.shape[1] for x in Xs], n_classes=cd.n_classes,
        variant="no_fwgcn", hidden_dim=64, dropout=0.5, delta=1e-2,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        logits, _ = model(Xs_t, full_stat, full_sem, sample_edge)
        F.cross_entropy(logits, y_t).backward()
        opt.step()

    # Extract pre-classifier representation: re-implement the forward up to
    # the head's first linear, then capture intermediate.
    model.eval()
    with torch.no_grad():
        # Replicate ablation.AblatedOmicReasonNet.forward(no_fwgcn) up to head input
        fused = []
        for v in range(model.n_views):
            x_v = Xs_t[v]
            z_stat = model.stat_enc[v](x_v, full_stat[v])
            sem_e = full_sem[v] if full_sem[v] is not None else full_stat[v]
            z_sem = model.sem_enc[v](x_v, sem_e)
            z = model.fuse[v](z_stat, z_sem)  # variant != no_gate
            s = torch.sigmoid(model.score_head[v](z.mean(0)))
            fused.append(x_v * s.unsqueeze(0) * torch.exp(model.delta * x_v.abs()))
        X_repr = torch.cat(fused, dim=1)
        # head is Sequential: Linear(in,h) -> ReLU -> Dropout -> Linear(h,c)
        # capture after first linear+relu (the "patient embedding")
        h = model.head[0](X_repr)
        h = F.relu(h)
        emb = h.cpu().numpy()

    out_dir = Path(args.out) / cd.cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "embeddings.npz",
                        X=emb.astype(np.float32),
                        y=cd.y.astype(np.int64))
    print(f"[done] {cd.cohort}: emb shape={emb.shape}, n_classes={cd.n_classes}")


if __name__ == "__main__":
    main()
