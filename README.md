# OmicReasonNet

**OmicReasonNet: Biomedical Text-Augmented Dual-Graph Fusion for Multi-Omics Classification.**

This repository contains the reference implementation associated with the
OmicReasonNet manuscript, currently being prepared for
*IEEE Transactions on Computational Biology and Bioinformatics (TCBB)*.
OmicReasonNet (ORN) is a dual-prior graph network that couples a
Layer-Attention Graph Convolutional Network (LAGCN) over a heterogeneous
biological topology with a PubMedBERT-derived semantic prior, fuses the
two through a per-feature gate, and produces interpretable reasoning
scores that re-weight the multi-omics input before a two-layer MLP
classifier head.

---

## What is here

```
.
├── src/                   # Python implementation (PyTorch + PyG)
│   ├── dataloader.py          MOGONET-style multi-omics loader + cosine-kNN graph
│   ├── models.py              LAGCN, GatedFusion, FWGCN, OmicReasonNet, baselines
│   ├── ablation.py            AblatedOmicReasonNet (no_stat / no_sem / no_gate / no_fwgcn / no_reasoning)
│   ├── baselines.py           Random Forest / Gaussian NB feature-weighted helpers
│   ├── llm_prior.py           PubMedBERT sentence-transformer prompt + embedding cache
│   ├── precompute_sem.py      One-shot semantic-prior pre-computation per cohort
│   ├── train_cv.py            5-fold stratified CV training (with baselines flag)
│   ├── run_ablation.py        Per-(cohort × variant) ablation runner
│   ├── run_sensitivity.py     δ / k / hidden hyper-parameter grid runner
│   ├── run_biomarker.py       Top-K reasoning + masking-based biomarker test
│   ├── run_embeddings.py      Save post-fusion patient embeddings (for UMAP)
│   ├── biomarker.py           Recurrence + BH-FDR aggregation utilities
│   ├── aggregate_results.py   Per-cohort JSON → main_table.csv / ablation_table.csv / ...
│   ├── redefine_orn.py        Architectural-ablation-driven head choice (MLP-head default)
│   ├── figures.py             Main figures (violins, ablation bars, δ sensitivity)
│   ├── figures_advanced.py    UMAP grid + Critical Difference (Demšar) diagram
│   ├── make_figures.py        Top-level figure generator (calls figures.py)
│   └── utils.py               set_seed / device helpers
│
├── scripts/               # SLURM + R helpers (sanitised)
│   ├── sbatch_brca.sh         Single-cohort BRCA CV
│   ├── sbatch_one_cohort.sh   Per-cohort CV via $COHORT
│   ├── sbatch_full_benchmark.sh  5-cohort job array
│   ├── sbatch_tcga_download.sh   TCGAbiolinks chained download → preprocess → CV
│   ├── sbatch_ablations.sh    25-task array (5 cohorts × 5 ablation variants)
│   ├── sbatch_sensitivity.sh  15-task array (5 cohorts × δ / k / hidden)
│   ├── sbatch_biomarker.sh    5-task array
│   ├── sbatch_embeddings.sh   5-task array (one trained ORN per cohort)
│   ├── download_one_cohort.R  Resumable TCGAbiolinks downloader (mRNA + meth + miRNA + clinical)
│   └── preprocess_cohort.R    MOGONET-style preprocessing (ANOVA F top-N, label collapse)
│
└── data/
    └── BRCA/              # MOGONET-released BRCA preprocessing (5 PAM50-aligned classes,
                           # 1,000 mRNA / 1,000 gene-aggregated meth / 503 miRNA features)
```

> Per-cohort cv_results JSONs, ablation JSONs, sensitivity JSONs and
> biomarker JSONs (the inputs to every figure) are not committed here
> because they are reproducible end-to-end from the scripts on a single
> H100 GPU in <6 GPU-hours. Run
> `python -m src.train_cv --cohort BRCA --with_baselines` to regenerate
> `results/BRCA/cv_results.json`.

---

## Quick start (BRCA, single GPU)

```bash
# 1) create env (or use any python>=3.10 with the listed dependencies)
conda create -n orn python=3.11 -y && conda activate orn
pip install -r requirements.txt

# 2) precompute the LLM semantic prior (downloads PubMedBERT once)
python -m src.precompute_sem --cohort BRCA --data_dir data --cache_dir cache

# 3) 5-fold CV with baselines (3-5 minutes on one H100)
python -m src.train_cv \
    --cohort BRCA --data_dir data --cache_dir cache \
    --out results --folds 5 --epochs 200 --with_baselines
```

The script writes `results/BRCA/cv_results.json` with a row per (model, fold).

## Reproducing the full 5-cohort benchmark

The four other cohorts (COADREAD, UCEC, SARC, LUAD) are not bundled
because they exceed reasonable git-tree sizes. The provided R scripts
download and preprocess them in MOGONET format on the fly:

```bash
# (per cohort, ~5–15 min download + ~1 min preprocess + ~2 min CV on H100)
sbatch --export=ALL,COHORT=COADREAD scripts/sbatch_one_cohort.sh
sbatch --export=ALL,COHORT=UCEC     scripts/sbatch_one_cohort.sh
sbatch --export=ALL,COHORT=SARC     scripts/sbatch_one_cohort.sh
sbatch --export=ALL,COHORT=LUAD     scripts/sbatch_one_cohort.sh
```

After all five cohorts have a `results/$COHORT/cv_results.json`, run:

```bash
python -m src.aggregate_results
python -m src.redefine_orn          # writes results/main_table_final.csv
python -m src.make_figures \
    --cohorts BRCA,COADREAD,UCEC,SARC,LUAD --out figures
python -m src.figures_advanced --mode both \
    --emb_dir results/embeddings --out figures
```

## Architectural ablation and sensitivity

```bash
# 25-task array: 5 cohorts × 5 variants (no_stat / no_sem / no_gate /
# no_fwgcn / no_reasoning)
sbatch scripts/sbatch_ablations.sh

# 15-task array: 5 cohorts × {delta, k, hidden}
sbatch scripts/sbatch_sensitivity.sh

# 5-task biomarker pass (top-K reasoning recurrence + masking p-value)
sbatch scripts/sbatch_biomarker.sh
```

Cluster-specific environment variables in the sbatch headers
(`ORN_ROOT`, `CONDA_ROOT`, `HF_HOME`, partition, QoS) must be set to
your site's values; the version committed here uses placeholder
comments.

---

## Dependencies

Python ≥ 3.10. Tested with PyTorch 2.4 + PyG 2.7 on Linux x86_64 (NVIDIA H100).
See `requirements.txt`. R ≥ 4.3 with TCGAbiolinks ≥ 2.30 and the
sesame methylation-array stack is required for the four downloadable
cohorts; `requirements_R.txt` lists the specific Bioconductor packages.

## Citation

If you use this code, please cite:

> Ni J., Wei Z., Zhang X., Li M., Xie Z., Liu Y., Jatowt A.
> *OmicReasonNet: Biomedical Text-Augmented Dual-Graph Fusion for
> Multi-Omics Classification.*
> Manuscript in preparation for **IEEE Transactions on Computational Biology
> and Bioinformatics (TCBB)**, 2026.

## License

MIT — see `LICENSE`.

## Contact

- Zhuoying Xie — `zyxie@seu.edu.cn` (corresponding)
- Yun Liu — `liuyun@njmu.edu.cn` (corresponding)
- Adam Jatowt — `Adam.Jatowt@uibk.ac.at` (corresponding)
- Jie Ni — `njie@seu.edu.cn` (first author / code maintainer)
