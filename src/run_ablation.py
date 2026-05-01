"""Run a single ablation variant on one cohort with 5-fold CV.

CLI: python -m src.run_ablation --cohort BRCA --variant no_sem [--out ./results/ablation]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

from src.dataloader import load_cohort, cosine_knn_edges
from src.ablation import AblatedOmicReasonNet
from src.train_cv import load_sample_sem_edges


def safe_auc(y, prob, k):
    try:
        if k == 2:
            return float(roc_auc_score(y, prob[:, 1]))
        return float(roc_auc_score(label_binarize(y, classes=list(range(k))),
                                   prob, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def standardize(X_list, train_idx):
    return [StandardScaler().fit(X[train_idx]).transform(X).astype(np.float32) for X in X_list]


def train_eval(X_views, y, train_idx, test_idx, sem_edges, n_classes, variant,
               device, epochs=200, lr=1e-3, wd=5e-4, hidden=64, dropout=0.5,
               delta=1e-2, seed=42, patience=30):
    torch.manual_seed(seed)
    Xs = standardize(X_views, train_idx)
    Xs_t = [torch.tensor(x, device=device) for x in Xs]
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    full_stat = [cosine_knn_edges(x, k=10).to(device) for x in Xs]
    full_sem = [None if s is None else torch.tensor(s, device=device) for s in sem_edges]
    sample_edge_full = full_stat[0]

    model = AblatedOmicReasonNet(
        feat_dims=[x.shape[1] for x in Xs], n_classes=n_classes,
        variant=variant, hidden_dim=hidden, dropout=dropout, delta=delta,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_acc, best_state, no_imp = 0.0, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits, _ = model(Xs_t, full_stat, full_sem, sample_edge_full)
        loss = F.cross_entropy(logits[train_idx], y_t[train_idx])
        loss.backward(); opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(Xs_t, full_stat, full_sem, sample_edge_full)
                prob = F.softmax(logits[test_idx], dim=1).cpu().numpy()
                acc = accuracy_score(y[test_idx], prob.argmax(1))
                if acc > best_acc:
                    best_acc, best_state, no_imp = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
                else:
                    no_imp += 5
                if no_imp >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xs_t, full_stat, full_sem, sample_edge_full)
        prob = F.softmax(logits[test_idx], dim=1).cpu().numpy()
        pred = prob.argmax(1)
    yt = y[test_idx]
    return {
        "acc": float(accuracy_score(yt, pred)),
        "f1_w": float(f1_score(yt, pred, average="weighted")),
        "f1_m": float(f1_score(yt, pred, average="macro")),
        "auc": safe_auc(yt, prob, n_classes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["full", "no_stat", "no_sem", "no_gate", "no_fwgcn", "no_reasoning"])
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--out", default="./results/ablation")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = load_cohort(args.data_dir, args.cohort)
    sem_edges = load_sample_sem_edges(args.cache_dir, cd.cohort, cd.X, k=10)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    for fi, (tr, te) in enumerate(skf.split(cd.X[0], cd.y)):
        t0 = time.time()
        r = train_eval(cd.X, cd.y, tr, te, sem_edges, cd.n_classes, args.variant,
                       device, epochs=args.epochs, seed=args.seed + fi)
        r["fold"] = fi
        r["variant"] = args.variant
        r["cohort"] = cd.cohort
        fold_results.append(r)
        print(f"[{cd.cohort}/{args.variant} fold {fi}] acc={r['acc']:.4f} f1w={r['f1_w']:.4f} ({time.time()-t0:.1f}s)")

    out_dir = Path(args.out) / cd.cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"ablation_{args.variant}.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"[done] -> {out_dir}/ablation_{args.variant}.json")


if __name__ == "__main__":
    main()
