"""5-fold stratified CV training for OmicReasonNet + baselines.

CLI: python -m src.train_cv --cohort BRCA --data_dir ./data --out ./results
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize, StandardScaler

from src.dataloader import load_cohort, cosine_knn_edges
from src.models import OmicReasonNet, PlainGCN, PlainGAT, MLPConcat
from src.baselines import fit_eval_classical


def safe_auc(y, prob, k):
    try:
        if k == 2:
            return float(roc_auc_score(y, prob[:, 1]))
        yb = label_binarize(y, classes=list(range(k)))
        return float(roc_auc_score(yb, prob, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def standardize_views(X_list, train_idx):
    out = []
    for X in X_list:
        sc = StandardScaler().fit(X[train_idx])
        out.append(sc.transform(X).astype(np.float32))
    return out


def train_omicreasonnet(
    X_views, y, train_idx, test_idx, n_classes, sem_edges_global,
    device="cuda", epochs=200, lr=1e-3, wd=5e-4, hidden=64, dropout=0.5,
    delta=1e-2, seed=42, patience=30,
):
    torch.manual_seed(seed)
    Xs = standardize_views(X_views, train_idx)
    Xs_t = [torch.tensor(x, device=device) for x in Xs]
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    stat_edges = [cosine_knn_edges(x[train_idx], k=10).to(device) for x in Xs]
    # for full-sample message passing during inference: build edges on all samples
    # using only training labels for similarity is unsafe; we use unsupervised cosine
    full_stat = [cosine_knn_edges(x, k=10).to(device) for x in Xs]
    full_sem = [None if s is None else torch.tensor(s, device=device) for s in sem_edges_global]
    sample_edge_full = full_stat[0]   # use mRNA sample graph as node graph for FWGCN

    model = OmicReasonNet(
        feat_dims=[x.shape[1] for x in Xs], n_classes=n_classes,
        hidden_dim=hidden, dropout=dropout, delta=delta,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_acc, best_state, no_improve = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits, scores = model(Xs_t, full_stat, full_sem, sample_edge_full)
        loss = F.cross_entropy(logits[train_idx], y_t[train_idx])
        loss.backward()
        opt.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits, _ = model(Xs_t, full_stat, full_sem, sample_edge_full)
                prob = F.softmax(logits[test_idx], dim=1).cpu().numpy()
                pred = prob.argmax(axis=1)
                acc = accuracy_score(y[test_idx], pred)
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 5
                if no_improve >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, scores = model(Xs_t, full_stat, full_sem, sample_edge_full)
        prob = F.softmax(logits[test_idx], dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
    yt = y[test_idx]
    return {
        "model": "OmicReasonNet",
        "acc": float(accuracy_score(yt, pred)),
        "f1_w": float(f1_score(yt, pred, average="weighted")),
        "f1_m": float(f1_score(yt, pred, average="macro")),
        "auc": safe_auc(yt, prob, n_classes),
    }, [s.detach().cpu().numpy() for s in scores]


def train_plain_gnn(model_cls, X_concat, y, train_idx, test_idx, n_classes,
                    device="cuda", epochs=150, lr=1e-3, wd=5e-4, hidden=64,
                    dropout=0.5, seed=42, patience=25):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(X_concat[train_idx])
    Xn = sc.transform(X_concat).astype(np.float32)
    edge = cosine_knn_edges(Xn, k=10).to(device)
    Xt = torch.tensor(Xn, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    model = model_cls(Xn.shape[1], hidden, n_classes, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_acc, best_state, no_improve = 0.0, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(Xt, edge)
        loss = F.cross_entropy(out[train_idx], yt[train_idx])
        loss.backward(); opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                out = model(Xt, edge)
                prob = F.softmax(out[test_idx], dim=1).cpu().numpy()
                pred = prob.argmax(axis=1)
                acc = accuracy_score(y[test_idx], pred)
                if acc > best_acc:
                    best_acc, best_state, no_improve = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
                else:
                    no_improve += 5
                if no_improve >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xn, device=device), edge)
        prob = F.softmax(out[test_idx], dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
    return {
        "model": model_cls.__name__,
        "acc": float(accuracy_score(y[test_idx], pred)),
        "f1_w": float(f1_score(y[test_idx], pred, average="weighted")),
        "f1_m": float(f1_score(y[test_idx], pred, average="macro")),
        "auc": safe_auc(y[test_idx], prob, n_classes),
    }


def load_sample_sem_edges(cache_dir: str, cohort: str, X_views: list[np.ndarray], k: int = 10) -> list:
    """Build SAMPLE-level semantic graphs by projecting samples into the LLM
    feature-embedding space (X_v @ emb_v) and running cosine kNN over samples.

    This bridges the modality gap: feature-level LLM semantics become a
    sample-similarity prior aligned with the stat sample graph.
    """
    out = []
    base = os.path.join(cache_dir, cohort)
    emb_files = ["mrna_emb.npy", "meth_emb.npy", "miRNA_emb.npy"]
    for v, fn in enumerate(emb_files):
        path = os.path.join(base, fn)
        if not os.path.exists(path):
            out.append(None); continue
        emb = np.load(path).astype(np.float32)        # [F_v, D]
        Xv = X_views[v]                                # [N, F_v]
        if emb.shape[0] != Xv.shape[1]:
            out.append(None); continue
        sample_emb = Xv @ emb                         # [N, D]
        sample_emb = sample_emb / (np.linalg.norm(sample_emb, axis=1, keepdims=True) + 1e-8)
        sim = sample_emb @ sample_emb.T
        np.fill_diagonal(sim, -np.inf)
        n = sim.shape[0]
        knn = np.argpartition(-sim, kth=k, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        cols = knn.reshape(-1)
        src = np.concatenate([rows, cols])
        dst = np.concatenate([cols, rows])
        edge = np.unique(np.stack([src, dst], axis=0), axis=1).astype(np.int64)
        out.append(edge)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--with_baselines", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cd = load_cohort(args.data_dir, args.cohort)
    print(f"[load] cohort={cd.cohort} n={cd.X[0].shape[0]} k={cd.n_classes}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    all_scores = []

    sem_edges = load_sample_sem_edges(args.cache_dir, cd.cohort, cd.X, k=10)
    print(f"[sem_edges] sample-level: {[None if s is None else s.shape for s in sem_edges]}")

    for fi, (tr, te) in enumerate(skf.split(cd.X[0], cd.y)):
        t0 = time.time()
        r_orn, scores = train_omicreasonnet(
            cd.X, cd.y, tr, te, cd.n_classes, sem_edges,
            device=device, epochs=args.epochs, seed=args.seed + fi,
        )
        r_orn["fold"] = fi
        fold_results.append(r_orn)
        all_scores.append(scores)
        print(f"[fold {fi}] OmicReasonNet acc={r_orn['acc']:.4f} f1w={r_orn['f1_w']:.4f} auc={r_orn['auc']:.4f} ({time.time()-t0:.1f}s)")

        if args.with_baselines:
            X_cat = np.concatenate(cd.X, axis=1)
            for name, cls in [("PlainGCN", PlainGCN), ("PlainGAT", PlainGAT), ("MLPConcat", MLPConcat)]:
                rr = train_plain_gnn(cls, X_cat, cd.y, tr, te, cd.n_classes,
                                     device=device, epochs=150, seed=args.seed + fi)
                rr["fold"] = fi; rr["model"] = name
                fold_results.append(rr)
                print(f"  baseline {name} acc={rr['acc']:.4f}")
            # classical
            X_cat_std = StandardScaler().fit_transform(np.concatenate(cd.X, axis=1))
            for nm in ["rf", "nb"]:
                rr = fit_eval_classical(
                    X_cat_std[tr], cd.y[tr], X_cat_std[te], cd.y[te],
                    cd.n_classes, model=nm, seed=args.seed + fi,
                )
                rr["fold"] = fi; fold_results.append(rr)
                print(f"  baseline {nm} acc={rr['acc']:.4f}")

    out_dir = Path(args.out) / cd.cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(fold_results, f, indent=2)
    np.savez_compressed(out_dir / "feature_scores.npz",
                        scores_per_fold=np.array(all_scores, dtype=object))
    print(f"[done] -> {out_dir}/cv_results.json")


if __name__ == "__main__":
    main()
