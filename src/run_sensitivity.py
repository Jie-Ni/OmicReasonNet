"""Hyperparameter sensitivity grid for OmicReasonNet on one cohort.

Sweeps delta over {1e-4, 1e-3, 1e-2, 1e-1, 1e0} (paper claim).
Optionally sweeps k (kNN neighbours) and hidden_dim. 5-fold CV per setting.

CLI: python -m src.run_sensitivity --cohort BRCA --param delta
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.dataloader import load_cohort, cosine_knn_edges
from src.models import OmicReasonNet
from src.train_cv import load_sample_sem_edges


def train_eval_one(X_views, y, train_idx, test_idx, sem_edges, n_classes,
                   delta, k_knn, hidden, device="cuda", epochs=150, seed=42):
    torch.manual_seed(seed)
    Xs = [StandardScaler().fit(x[train_idx]).transform(x).astype(np.float32) for x in X_views]
    Xs_t = [torch.tensor(x, device=device) for x in Xs]
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    full_stat = [cosine_knn_edges(x, k=k_knn).to(device) for x in Xs]
    full_sem = [None if s is None else torch.tensor(s, device=device) for s in sem_edges]
    sample_edge = full_stat[0]
    model = OmicReasonNet(
        feat_dims=[x.shape[1] for x in Xs], n_classes=n_classes,
        hidden_dim=hidden, dropout=0.5, delta=delta,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits, _ = model(Xs_t, full_stat, full_sem, sample_edge)
        F.cross_entropy(logits[train_idx], y_t[train_idx]).backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(Xs_t, full_stat, full_sem, sample_edge)
                pred = logits[test_idx].argmax(1).cpu().numpy()
                acc = accuracy_score(y[test_idx], pred)
                if acc > best_acc:
                    best_acc, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xs_t, full_stat, full_sem, sample_edge)
        prob = F.softmax(logits[test_idx], dim=1).cpu().numpy()
        pred = prob.argmax(1)
    yt = y[test_idx]
    return {
        "acc": float(accuracy_score(yt, pred)),
        "f1_w": float(f1_score(yt, pred, average="weighted")),
        "f1_m": float(f1_score(yt, pred, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--param", choices=["delta", "k", "hidden"], default="delta")
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--out", default="./results/sensitivity")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    grids = {
        "delta": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        "k": [5, 8, 10, 15, 20],
        "hidden": [32, 64, 128, 256],
    }
    grid = grids[args.param]
    cd = load_cohort(args.data_dir, args.cohort)
    sem_edges = load_sample_sem_edges(args.cache_dir, cd.cohort, cd.X, k=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    folds = list(skf.split(cd.X[0], cd.y))

    rows = []
    for val in grid:
        delta = val if args.param == "delta" else 1e-2
        k_knn = val if args.param == "k" else 10
        hidden = val if args.param == "hidden" else 64
        for fi, (tr, te) in enumerate(folds):
            t0 = time.time()
            r = train_eval_one(cd.X, cd.y, tr, te, sem_edges, cd.n_classes,
                               delta=delta, k_knn=int(k_knn), hidden=int(hidden),
                               device=device, seed=42 + fi)
            r.update({args.param: val, "fold": fi, "cohort": cd.cohort})
            rows.append(r)
            print(f"[{cd.cohort} {args.param}={val} fold {fi}] acc={r['acc']:.4f} ({time.time()-t0:.1f}s)")

    out_dir = Path(args.out) / cd.cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"sensitivity_{args.param}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[done] -> {out_dir}/sensitivity_{args.param}.json")


if __name__ == "__main__":
    main()
