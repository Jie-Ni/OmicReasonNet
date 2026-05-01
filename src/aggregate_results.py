"""Aggregate all results across cohorts into a single summary JSON + CSV.

Outputs:
  results/aggregated.json
  results/main_table.csv (rows=cohorts, cols=models with mean ± std)
  results/ablation_table.csv (rows=cohort×variant)
  results/sensitivity_summary.csv
  results/biomarker_table.csv (validated biomarkers per cohort)

CLI: python -m src.aggregate_results [--results ./results]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_cv(results_dir: str, cohort: str) -> pd.DataFrame:
    p = os.path.join(results_dir, cohort, "cv_results.json")
    return pd.DataFrame(json.load(open(p))) if os.path.exists(p) else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--cohorts", default="BRCA,COADREAD,UCEC,SARC,LUAD")
    args = ap.parse_args()

    cohorts = [c.strip() for c in args.cohorts.split(",")]
    out = {}

    # Main CV table: cohort × model
    main_rows = []
    for c in cohorts:
        df = load_cv(args.results, c)
        if df.empty:
            continue
        for m, sub in df.groupby("model"):
            main_rows.append({
                "cohort": c, "model": m,
                "acc_mean": float(sub["acc"].mean()),
                "acc_std": float(sub["acc"].std()),
                "f1_w_mean": float(sub["f1_w"].mean()),
                "f1_m_mean": float(sub["f1_m"].mean()),
                "auc_mean": float(sub["auc"].mean()),
            })
    main_df = pd.DataFrame(main_rows)
    if not main_df.empty:
        main_df.to_csv(os.path.join(args.results, "main_table.csv"), index=False)
    out["main"] = main_df.to_dict("records")

    # Ablation table
    abl_rows = []
    for c in cohorts:
        cdir = os.path.join(args.results, "ablation", c)
        if not os.path.isdir(cdir):
            continue
        for fp in sorted(os.listdir(cdir)):
            if not (fp.startswith("ablation_") and fp.endswith(".json")):
                continue
            v = fp.replace("ablation_", "").replace(".json", "")
            rs = json.load(open(os.path.join(cdir, fp)))
            accs = [r["acc"] for r in rs]
            abl_rows.append({"cohort": c, "variant": v,
                             "acc_mean": float(np.mean(accs)),
                             "acc_std": float(np.std(accs))})
    abl_df = pd.DataFrame(abl_rows)
    if not abl_df.empty:
        abl_df.to_csv(os.path.join(args.results, "ablation_table.csv"), index=False)
    out["ablation"] = abl_df.to_dict("records")

    # Sensitivity table
    sens_rows = []
    for c in cohorts:
        sdir = os.path.join(args.results, "sensitivity", c)
        if not os.path.isdir(sdir):
            continue
        for fp in sorted(os.listdir(sdir)):
            if not (fp.startswith("sensitivity_") and fp.endswith(".json")):
                continue
            param = fp.replace("sensitivity_", "").replace(".json", "")
            rs = json.load(open(os.path.join(sdir, fp)))
            agg = defaultdict(list)
            for r in rs:
                agg[r[param]].append(r["acc"])
            for v, accs in agg.items():
                sens_rows.append({"cohort": c, "param": param, "value": v,
                                  "acc_mean": float(np.mean(accs)),
                                  "acc_std": float(np.std(accs))})
    sens_df = pd.DataFrame(sens_rows)
    if not sens_df.empty:
        sens_df.to_csv(os.path.join(args.results, "sensitivity_summary.csv"), index=False)
    out["sensitivity"] = sens_df.to_dict("records")

    # Biomarker table
    bio_rows = []
    for c in cohorts:
        p = os.path.join(args.results, "biomarkers", c, "biomarkers.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for r in d.get("validated", []):
            bio_rows.append({
                "cohort": c, "modality": r["modality"], "feat_name": r["feat_name"],
                "mean_reasoning_score": r["mean_reasoning_score"],
                "ablation_mean_drop": r["ablation_mean_drop"],
                "ablation_p": r["ablation_p"],
                "n_folds_in_topk": r["n_folds_in_topk"],
            })
    bio_df = pd.DataFrame(bio_rows)
    if not bio_df.empty:
        bio_df.to_csv(os.path.join(args.results, "biomarker_table.csv"), index=False)
    out["biomarkers"] = bio_df.to_dict("records")

    with open(os.path.join(args.results, "aggregated.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] aggregated {len(main_rows)} main, {len(abl_rows)} ablation, "
          f"{len(sens_rows)} sensitivity, {len(bio_rows)} biomarker rows")


if __name__ == "__main__":
    main()
