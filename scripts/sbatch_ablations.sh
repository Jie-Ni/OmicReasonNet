#!/bin/bash
#SBATCH --job-name=orn-abl
#SBATCH # --partition=YOUR_GPU_PARTITION
#SBATCH # --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-24
#SBATCH --output=logs/abl-%A_%a.out
#SBATCH --error=logs/abl-%A_%a.err

# 5 cohorts x 5 ablation variants = 25 jobs.
# Cohort = idx / 5; variant = idx % 5.
# (variant "full" already covered by main CV → skipped here)
set -euo pipefail
cd ${ORN_ROOT}

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate orn
export HF_HOME=${HF_HOME}

cohorts=(BRCA COADREAD UCEC SARC LUAD)
variants=(no_stat no_sem no_gate no_fwgcn no_reasoning)

c_idx=$((SLURM_ARRAY_TASK_ID / 5))
v_idx=$((SLURM_ARRAY_TASK_ID % 5))
C=${cohorts[$c_idx]}
V=${variants[$v_idx]}

echo "[$(date)] $C / $V on $(hostname)"
if [ ! -f "$C/labels_tr.csv" ]; then
  echo "SKIP $C/$V — data not ready"
  exit 0
fi

python -m src.run_ablation --cohort "$C" --variant "$V" --data_dir . --cache_dir ./cache --out ./results/ablation
echo "DONE-$C-$V"
