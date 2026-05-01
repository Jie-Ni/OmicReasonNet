#!/bin/bash
#SBATCH --job-name=orn-sens
#SBATCH # --partition=YOUR_GPU_PARTITION
#SBATCH # --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-14
#SBATCH --output=logs/sens-%A_%a.out
#SBATCH --error=logs/sens-%A_%a.err

# 5 cohorts x 3 params (delta, k, hidden) = 15 jobs.
set -euo pipefail
cd ${ORN_ROOT}

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate orn
export HF_HOME=${HF_HOME}

cohorts=(BRCA COADREAD UCEC SARC LUAD)
params=(delta k hidden)
c_idx=$((SLURM_ARRAY_TASK_ID / 3))
p_idx=$((SLURM_ARRAY_TASK_ID % 3))
C=${cohorts[$c_idx]}
P=${params[$p_idx]}

echo "[$(date)] $C / $P on $(hostname)"
[ -f "$C/labels_tr.csv" ] || { echo "SKIP $C/$P"; exit 0; }
python -m src.run_sensitivity --cohort "$C" --param "$P" --data_dir . --cache_dir ./cache --out ./results/sensitivity
echo "DONE-SENS-$C-$P"
