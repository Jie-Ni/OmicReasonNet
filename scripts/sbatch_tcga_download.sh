#!/bin/bash
#SBATCH --job-name=orn-tcga
#SBATCH # --partition=YOUR_GPU_PARTITION
#SBATCH # --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --output=logs/tcga-%A_%a.out
#SBATCH --error=logs/tcga-%A_%a.err

# We use the GPU partition because compute-only partitions may not be
# available; only request 1 GPU placeholder. Heavy work is I/O bound (download).
# Switch to a CPU partition if/when available.

set -euo pipefail
cd ${ORN_ROOT}

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate orn

cohorts=(COADREAD UCEC SARC LUAD)
cohort=${cohorts[$SLURM_ARRAY_TASK_ID]}
out=${ORN_ROOT}/${cohort}

mkdir -p logs $out

echo "[node] $(hostname)"
echo "[$(date)] download start: $cohort"
Rscript scripts/download_one_cohort.R "$cohort" "$out"

echo "[$(date)] preprocess: $cohort"
Rscript scripts/preprocess_cohort.R "$out"

echo "[$(date)] precompute_sem: $cohort"
export HF_HOME=${HF_HOME}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME/sentence-transformers
python -m src.precompute_sem --cohort "$cohort" --data_dir . --cache_dir ./cache

echo "[$(date)] train_cv: $cohort"
python -m src.train_cv --cohort "$cohort" --data_dir . --cache_dir ./cache --out ./results --folds 5 --epochs 200 --with_baselines

echo "DONE-$cohort"
