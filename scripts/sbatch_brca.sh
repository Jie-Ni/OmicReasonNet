#!/bin/bash
#SBATCH --job-name=orn-brca
#SBATCH # --partition=YOUR_GPU_PARTITION
#SBATCH # --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/brca-%j.out
#SBATCH --error=logs/brca-%j.err

set -euo pipefail
cd ${ORN_ROOT}

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate orn || conda activate base

# Cache HuggingFace models on /scratch (huge quota) instead of $HOME
export HF_HOME=${HF_HOME}
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME/sentence-transformers
mkdir -p $HF_HOME logs results cache

echo "[node] $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
python -c "import torch; print('cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())"

python -m src.precompute_sem --cohort BRCA --data_dir . --cache_dir ./cache
python -m src.train_cv --cohort BRCA --data_dir . --out ./results --folds 5 --epochs 200 --with_baselines

echo "DONE-BRCA"
