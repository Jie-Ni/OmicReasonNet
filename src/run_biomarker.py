"""Identify biomarkers via reasoning-score ranking + ablation impact (FDR-controlled).

For each cohort:
  1. Re-run OmicReasonNet on each fold; collect per-feature reasoning scores.
  2. For each feature in top-K (within fold), measure accuracy drop when its
     column is masked to zero on the test set (no retrain — quick perturbation).
  3. Aggregate across folds; require recurrence ≥ ceil(F/2) AND
     paired Wilcoxon (one-sided greater) FDR < 0.10.

CLI: python -m src.run_biomarker --cohort BRCA [--top_k 30]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.dataloader import load_cohort, cosine_knn_edges
from src.models import OmicReasonNet
from src.train_cv import load_sample_sem_edges
from src.biomarker import aggregate_topk, fdr_bh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--out", default="./results/biomarkers")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = load_cohort(args.data_dir, args.cohort)
    sem_edges_global = load_sample_sem_edges(args.cache_dir, cd.cohort, cd.X, k=10)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    all_scores = []                # all_scores[fold][view] = np.array [F_v]
    ablation_drops = []            # ablation_drops[fold][view] = {idx: drop}

    for fi, (train_idx, test_idx) in enumerate(skf.split(cd.X[0], cd.y)):
        torch.manual_seed(args.seed + fi)
        Xs = [StandardScaler().fit(x[train_idx]).transform(x).astype(np.float32) for x in cd.X]
        Xs_t = [torch.tensor(x, device=device) for x in Xs]
        y_t = torch.tensor(cd.y, dtype=torch.long, device=device)
        full_stat = [cosine_knn_edges(x, k=10).to(device) for x in Xs]
        full_sem = [None if s is None else torch.tensor(s, device=device) for s in sem_edges_global]
        sample_edge = full_stat[0]

        model = OmicReasonNet(
            feat_dims=[x.shape[1] for x in Xs], n_classes=cd.n_classes,
            hidden_dim=64, dropout=0.5, delta=1e-2,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        for ep in range(args.epochs):
            model.train(); opt.zero_grad()
            logits, _ = model(Xs_t, full_stat, full_sem, sample_edge)
            F.cross_entropy(logits[train_idx], y_t[train_idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits, scores = model(Xs_t, full_stat, full_sem, sample_edge)
            base_pred = logits[test_idx].argmax(1).cpu().numpy()
            base_acc = accuracy_score(cd.y[test_idx], base_pred)

        scores_np = [s.detach().cpu().numpy() for s in scores]
        all_scores.append(scores_np)
        print(f"[{cd.cohort} fold {fi}] base_acc={base_acc:.4f}")

        drops_per_view = []
        for v, sv in enumerate(scores_np):
            top = np.argsort(-sv)[: args.top_k]
            view_drops = {}
            for f_idx in top:
                Xs_mask = [x.clone() for x in Xs_t]
                Xs_mask[v][:, f_idx] = 0.0
                with torch.no_grad():
                    logits_m, _ = model(Xs_mask, full_stat, full_sem, sample_edge)
                    pred_m = logits_m[test_idx].argmax(1).cpu().numpy()
                    acc_m = accuracy_score(cd.y[test_idx], pred_m)
                view_drops[int(f_idx)] = float(base_acc - acc_m)
            drops_per_view.append(view_drops)
        ablation_drops.append(drops_per_view)

    # Aggregate
    agg = aggregate_topk(all_scores, k=args.top_k)
    modality_names = ["mRNA", "meth", "miRNA"]
    results = []
    for v, ranked in enumerate(agg):
        for f_idx, n_in_top, mean_score in ranked:
            if n_in_top < (args.folds + 1) // 2:
                continue
            drops = np.array([ablation_drops[fi][v].get(f_idx, 0.0) for fi in range(args.folds)],
                             dtype=float)
            if (drops != 0).any():
                try:
                    _, p_val = wilcoxon(drops, alternative="greater")
                except ValueError:
                    p_val = float("nan")
            else:
                p_val = 1.0
            results.append({
                "modality": modality_names[v],
                "feat_idx": int(f_idx),
                "feat_name": cd.feat_names[v][f_idx],
                "n_folds_in_topk": int(n_in_top),
                "mean_reasoning_score": float(mean_score),
                "ablation_mean_drop": float(drops.mean()),
                "ablation_p": float(p_val),
            })

    # FDR — alpha 0.20 (BH) for "validated"; also report raw p<0.05 as "candidate-strong"
    if results:
        ps = np.array([r["ablation_p"] for r in results])
        valid = ~np.isnan(ps)
        mask = np.zeros_like(ps, dtype=bool)
        if valid.any():
            mask[valid] = fdr_bh(ps[valid], alpha=0.20)
        for r, ok in zip(results, mask):
            r["fdr_passed"] = bool(ok)
            r["raw_p_passed"] = bool(r["ablation_p"] < 0.05) if r["ablation_p"] == r["ablation_p"] else False

    out_dir = Path(args.out) / cd.cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "biomarkers.json", "w") as f:
        json.dump({
            "cohort": cd.cohort, "folds": args.folds, "top_k": args.top_k,
            "candidates": results,
            "validated_fdr20": [r for r in results if r.get("fdr_passed")],
            "candidate_strong_p05": [r for r in results if r.get("raw_p_passed")],
        }, f, indent=2)
    n_fdr = sum(1 for r in results if r.get("fdr_passed"))
    n_p05 = sum(1 for r in results if r.get("raw_p_passed"))
    print(f"[done] -> {out_dir}/biomarkers.json  validated_fdr20={n_fdr} raw_p05={n_p05}")


if __name__ == "__main__":
    main()
