#!/bin/bash
#SBATCH --job-name=orn-one
#SBATCH # --partition=YOUR_GPU_PARTITION
#SBATCH # --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/one-%j.out
#SBATCH --error=logs/one-%j.err

# Run precompute_sem + train_cv for ONE cohort.
# Usage: sbatch --export=ALL,COHORT=COADREAD scripts/sbatch_one_cohort.sh

set -euo pipefail
cd ${ORN_ROOT}

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate orn

export HF_HOME=${HF_HOME}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME/sentence-transformers

C=${COHORT:?must export COHORT=...}
echo "[$(date)] $C on $(hostname)"
python -c "import torch; print('cuda', torch.cuda.is_available())"

if [ ! -f "$C/labels_tr.csv" ]; then
  echo "ERROR: $C/labels_tr.csv missing — preprocess not done yet."
  exit 1
fi

python -m src.precompute_sem --cohort "$C" --data_dir . --cache_dir ./cache
python -m src.train_cv --cohort "$C" --data_dir . --out ./results --folds 5 --epochs 200 --with_baselines

echo "DONE-$C"
