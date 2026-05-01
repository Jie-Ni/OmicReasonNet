"""Generate all publication figures from results JSONs. Headless (Agg backend).

Outputs PDF + PNG into ./figures/. Each figure auto-runs pixel-overflow QA.

CLI: python -m src.make_figures [--cohorts BRCA,COADREAD,...] [--out ./figures]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.figures import (
    setup_style,
    figure_perf_violin, figure_ablation_bars, figure_delta_sensitivity,
)


def load_cv(results_dir: str, cohort: str) -> pd.DataFrame:
    p = os.path.join(results_dir, cohort, "cv_results.json")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.DataFrame(json.load(open(p)))


def load_ablation(results_dir: str, cohorts: list[str]) -> pd.DataFrame:
    """Aggregate ablation across cohorts → long-form (cohort,variant,acc_mean,acc_std,...)."""
    rows = []
    for c in cohorts:
        cdir = os.path.join(results_dir, "ablation", c)
        if not os.path.isdir(cdir):
            continue
        for fp in sorted(os.listdir(cdir)):
            if not fp.startswith("ablation_") or not fp.endswith(".json"):
                continue
            v = fp.replace("ablation_", "").replace(".json", "")
            rs = json.load(open(os.path.join(cdir, fp)))
            accs = [r["acc"] for r in rs]
            rows.append({
                "cohort": c, "variant": v,
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            })
        # also include "full" from main cv_results
        cv = load_cv(results_dir, c)
        if not cv.empty:
            full = cv[cv["model"] == "OmicReasonNet"]
            if not full.empty:
                rows.append({"cohort": c, "variant": "full",
                             "acc_mean": float(full["acc"].mean()),
                             "acc_std": float(full["acc"].std())})
    return pd.DataFrame(rows)


def load_delta(results_dir: str, cohorts: list[str]) -> pd.DataFrame:
    rows = []
    for c in cohorts:
        p = os.path.join(results_dir, "sensitivity", c, "sensitivity_delta.json")
        if not os.path.exists(p):
            continue
        rs = json.load(open(p))
        agg = defaultdict(lambda: defaultdict(list))
        for r in rs:
            agg[r["delta"]]["acc"].append(r["acc"])
            agg[r["delta"]]["f1_w"].append(r["f1_w"])
            agg[r["delta"]]["f1_m"].append(r["f1_m"])
        for d, mets in agg.items():
            rows.append({
                "cohort": c, "delta": d,
                "acc_mean": float(np.mean(mets["acc"])), "acc_std": float(np.std(mets["acc"])),
                "f1_w_mean": float(np.mean(mets["f1_w"])), "f1_w_std": float(np.std(mets["f1_w"])),
                "f1_m_mean": float(np.mean(mets["f1_m"])), "f1_m_std": float(np.std(mets["f1_m"])),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out", default="./figures")
    ap.add_argument("--cohorts", default="BRCA,COADREAD,UCEC,SARC,LUAD")
    args = ap.parse_args()

    cohorts = [c.strip() for c in args.cohorts.split(",")]
    os.makedirs(args.out, exist_ok=True)
    setup_style()

    # === Figure 2: performance violins per cohort ===
    cv_data = {c: load_cv(args.results, c) for c in cohorts}
    cv_data = {c: df for c, df in cv_data.items() if not df.empty}
    if cv_data:
        figure_perf_violin(cv_data, os.path.join(args.out, "fig2_perf_violins"))
        print(f"[fig2] -> {args.out}/fig2_perf_violins.pdf")

    # === Figure 3: ablation bars ===
    abl = load_ablation(args.results, cohorts)
    if not abl.empty:
        abl_csv = os.path.join(args.out, "ablation_table.csv")
        abl.to_csv(abl_csv, index=False)
        figure_ablation_bars(abl_csv, os.path.join(args.out, "fig3_ablation_bars"))
        print(f"[fig3] -> {args.out}/fig3_ablation_bars.pdf")

    # === Figure 4: delta sensitivity ===
    sens = load_delta(args.results, cohorts)
    if not sens.empty:
        sens_csv = os.path.join(args.out, "sensitivity_delta.csv")
        sens.to_csv(sens_csv, index=False)
        figure_delta_sensitivity(sens_csv, os.path.join(args.out, "fig4_delta_sens"))
        print(f"[fig4] -> {args.out}/fig4_delta_sens.pdf")

    print("[done] figures written to", args.out)


if __name__ == "__main__":
    main()
