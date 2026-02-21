#!/bin/bash
# =============================================================================
# Smoke test: 5 folds × 1 epoch — validates the full pipeline
# Each fold runs as a separate parallel job via SLURM job array.
# Expected: ~6 min per fold
#
# Usage:  sbatch --array=0-4 slurm/smoke_test.slurm
# =============================================================================

#SBATCH --job-name=petl-smoke
#SBATCH --output=slurm/logs/%x_fold%a_%j.out
#SBATCH --error=slurm/logs/%x_fold%a_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gres=gpu:titan:1

STORAGE="/vol/joberant_nobck/data/NLP_368307701_2526a/yuvalanavi"
VENV_DIR="${STORAGE}/petl-ast/venv"
REPO_DIR="${STORAGE}/petl-ast/PETL-AST-Project"

source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
mkdir -p outputs slurm/logs

FOLD=${SLURM_ARRAY_TASK_ID}
export START_FOLD=${FOLD}
export END_FOLD=$((FOLD + 1))
export EPOCHS_OVERRIDE=1

echo "=== PETL-AST Smoke Test — Fold ${FOLD} ==="
echo "Date:   $(date)"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "CUDA:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

bash download_data.sh

python train.py \
    --data_path 'data' \
    --dataset_name 'ESC-50' \
    --method 'adapter' \
    --adapter_block 'conformer' \
    --adapter_type 'Pfeiffer' \
    --seq_or_par 'parallel' \
    --reduction_rate_adapter 96 \
    --kernel_size 8 \
    --device cuda \
    --num_workers 4 \
    --save_best_ckpt True \
    --output_path '/outputs'

echo ""
echo "=== Fold ${FOLD} Done: $(date) ==="
ls -la outputs/bestmodel_fold${FOLD} 2>/dev/null && echo "Checkpoint OK" || echo "!! No checkpoint"
grep "Trainloss" outputs/training.log 2>/dev/null | tail -1
