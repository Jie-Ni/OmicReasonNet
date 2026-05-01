#!/bin/bash
#SBATCH --job-name=orn-bio
#SBATCH # --partition=YOUR_GPU_PARTITION
#SBATCH # --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --output=logs/bio-%A_%a.out
#SBATCH --error=logs/bio-%A_%a.err

set -euo pipefail
cd ${ORN_ROOT}

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate orn
export HF_HOME=${HF_HOME}

cohorts=(BRCA COADREAD UCEC SARC LUAD)
C=${cohorts[$SLURM_ARRAY_TASK_ID]}

[ -f "$C/labels_tr.csv" ] || { echo "SKIP $C"; exit 0; }
python -m src.run_biomarker --cohort "$C" --data_dir . --cache_dir ./cache --out ./results/biomarkers --top_k 30
echo "DONE-BIO-$C"
