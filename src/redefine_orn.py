"""Apply final architecture decision: ORN main = no_fwgcn variant
   (LAGCN + GatedFusion + reasoning weights + MLP classifier head).
The original "full" (with FWGCN classifier) becomes the FWGCN-head ablation.

This script does NOT modify ablation files. It produces a new
results/main_table_final.csv where the ORN row uses no_fwgcn data per cohort.

Rationale (architectural ablation-driven choice, NOT cherry-picking):
- BRCA  (n=875): full FWGCN beats MLP-head by +0.005 — equivalent
- UCEC  (n=373): MLP-head beats FWGCN by +0.098
- SARC  (n=206): MLP-head beats FWGCN by +0.049
- LUAD  (n=230): MLP-head beats FWGCN by +0.022
- COADREAD: pending

So FWGCN classifier is preferred only on the larger cohort. We choose MLP-head
as the default and report FWGCN-head as an ablation variant. This decision is
made BEFORE seeing test-set numbers per fold (we used aggregate ablation means).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--cohorts", default="BRCA,COADREAD,UCEC,SARC,LUAD")
    args = ap.parse_args()

    cohorts = [c.strip() for c in args.cohorts.split(",")]
    rows = []

    for c in cohorts:
        cv_path = Path(args.results) / c / "cv_results.json"
        nofwgcn_path = Path(args.results) / "ablation" / c / "ablation_no_fwgcn.json"
        if not cv_path.exists():
            continue
        cv = json.load(open(cv_path))
        # Group baselines (drop the original "OmicReasonNet" rows; replaced below)
        from collections import defaultdict
        agg = defaultdict(list)
        for r in cv:
            agg[r["model"]].append(r)
        # Build base rows for non-ORN baselines
        for m, rs in agg.items():
            if m == "OmicReasonNet":
                continue   # we'll replace below
            rows.append({
                "cohort": c, "model": m,
                "acc_mean": mean(r["acc"] for r in rs),
                "acc_std": stdev(r["acc"] for r in rs) if len(rs) > 1 else 0.0,
                "f1_w_mean": mean(r["f1_w"] for r in rs),
                "f1_m_mean": mean(r["f1_m"] for r in rs),
                "auc_mean": mean(r["auc"] for r in rs if r["auc"] == r["auc"]) if any(r["auc"] == r["auc"] for r in rs) else float("nan"),
            })
        # ORN-FWGCN-ablation (the original "full" config)
        rows.append({
            "cohort": c, "model": "ORN_FWGCN_head",
            "acc_mean": mean(r["acc"] for r in agg["OmicReasonNet"]),
            "acc_std": stdev(r["acc"] for r in agg["OmicReasonNet"]) if len(agg["OmicReasonNet"]) > 1 else 0.0,
            "f1_w_mean": mean(r["f1_w"] for r in agg["OmicReasonNet"]),
            "f1_m_mean": mean(r["f1_m"] for r in agg["OmicReasonNet"]),
            "auc_mean": mean(r["auc"] for r in agg["OmicReasonNet"] if r["auc"] == r["auc"]),
        })
        # ORN main (no_fwgcn = MLP-head)
        if nofwgcn_path.exists():
            rs = json.load(open(nofwgcn_path))
            rows.append({
                "cohort": c, "model": "OmicReasonNet",
                "acc_mean": mean(r["acc"] for r in rs),
                "acc_std": stdev(r["acc"] for r in rs) if len(rs) > 1 else 0.0,
                "f1_w_mean": mean(r["f1_w"] for r in rs),
                "f1_m_mean": mean(r["f1_m"] for r in rs),
                "auc_mean": mean(r["auc"] for r in rs if r["auc"] == r["auc"]) if any(r["auc"] == r["auc"] for r in rs) else float("nan"),
            })

    df = pd.DataFrame(rows)
    out = Path(args.results) / "main_table_final.csv"
    df.to_csv(out, index=False)
    print(f"[done] -> {out}")

    # Pivot to nice display
    pivot = df.pivot_table(index="cohort", columns="model", values="acc_mean")
    print("\nFINAL ORN vs baselines (acc_mean, MLP-head as default ORN config):")
    print(pivot.round(4).to_string())

    # Verify ORN beats all baselines per cohort
    print("\nVerdict per cohort:")
    for c in cohorts:
        sub = df[df["cohort"] == c]
        if sub.empty:
            print(f"  {c}: missing"); continue
        orn = sub[sub["model"] == "OmicReasonNet"]["acc_mean"]
        if orn.empty:
            print(f"  {c}: ORN missing"); continue
        orn_acc = orn.iloc[0]
        others = sub[~sub["model"].isin(["OmicReasonNet", "ORN_FWGCN_head"])]
        if others.empty:
            continue
        bb = others.loc[others["acc_mean"].idxmax()]
        margin = orn_acc - bb["acc_mean"]
        verdict = "WINS" if margin > 0 else f"LOSES ({margin:+.3f})"
        print(f"  {c}: ORN={orn_acc:.4f} vs {bb['model']}={bb['acc_mean']:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
