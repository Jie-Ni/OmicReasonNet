#!/bin/bash
#SBATCH --job-name=orn-emb
#SBATCH # --partition=YOUR_GPU_PARTITION
#SBATCH # --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --output=logs/emb-%A_%a.out
#SBATCH --error=logs/emb-%A_%a.err

set -euo pipefail
cd ${ORN_ROOT}
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate orn
export HF_HOME=${HF_HOME}

cohorts=(BRCA COADREAD UCEC SARC LUAD)
C=${cohorts[$SLURM_ARRAY_TASK_ID]}
[ -f "$C/labels_tr.csv" ] || { echo "SKIP $C"; exit 0; }
python -m src.run_embeddings --cohort "$C" --data_dir . --cache_dir ./cache --out ./results/embeddings
echo "DONE-EMB-$C"
