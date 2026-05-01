#!/bin/bash
#SBATCH --job-name=orn-bench
#SBATCH # --partition=YOUR_GPU_PARTITION
#SBATCH # --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --output=logs/bench-%A_%a.out
#SBATCH --error=logs/bench-%A_%a.err

set -euo pipefail
cd ${ORN_ROOT}

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate orn || conda activate base

cohorts=(BRCA COADREAD UCEC SARC LUAD)
cohort=${cohorts[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs results cache

if [ ! -f "${cohort}/labels_tr.csv" ]; then
  echo "Cohort ${cohort} not yet downloaded; skipping."
  exit 0
fi

python -m src.precompute_sem --cohort "${cohort}" --data_dir . --cache_dir ./cache
python -m src.train_cv --cohort "${cohort}" --data_dir . --out ./results --folds 5 --epochs 200 --with_baselines

echo "DONE-${cohort}"
